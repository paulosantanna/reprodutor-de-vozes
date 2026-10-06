"""Timbre, pitch, and three-band EQ applied to synthesized audio.

Pitch is a duration-preserving phase-vocoder shift. Timbre is a spectral tilt
around 1 kHz (positive is brighter). EQ is a low shelf, a mid peak, and a high
shelf. The same filters run sample by sample so a stream and a full buffer match.
"""

import math
from collections.abc import Iterator

import numpy as np
import torch
from scipy.signal import sosfilt

from pocket_tts.data.voice_profile import VoiceProfile

_EQ_LOW_HZ = 200.0
_EQ_MID_HZ = 1000.0
_EQ_HIGH_HZ = 5000.0
_TIMBRE_PIVOT_HZ = 1000.0
_PITCH_BLOCK = 8192
_PITCH_OVERLAP = 2048


def apply_voice_profile(
    audio: torch.Tensor, sample_rate: int, profile: VoiceProfile
) -> torch.Tensor:
    """Apply `profile` to a waveform. Shape and length are unchanged."""
    if profile.is_neutral():
        return audio
    flat = audio.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    if abs(profile.pitch_semitones) >= 1e-6:
        flat = pitch_shift(flat, sample_rate, profile.pitch_semitones)
    flat = _filter_block(flat, sample_rate, profile)
    return flat.reshape(audio.shape).to(device=audio.device, dtype=audio.dtype)


def stream_with_profile(
    chunks: Iterator[torch.Tensor], sample_rate: int, profile: VoiceProfile
) -> Iterator[torch.Tensor]:
    """Apply `profile` to a stream of 1D chunks, preserving the sample count."""
    if profile.is_neutral():
        yield from chunks
        return
    effect = VoiceEffectStream(sample_rate, profile)
    for chunk in chunks:
        processed = effect.push(chunk)
        if processed.numel():
            yield processed
    tail = effect.flush()
    if tail.numel():
        yield tail


class VoiceEffectStream:
    """Streaming pitch (overlap-add blocks) followed by causal timbre and EQ."""

    def __init__(self, sample_rate: int, profile: VoiceProfile):
        self.sample_rate = sample_rate
        self.profile = profile
        self._pitch = (
            _PitchBlockStream(sample_rate, profile.pitch_semitones)
            if abs(profile.pitch_semitones) >= 1e-6
            else None
        )
        self._sos = design_sos(sample_rate, profile)
        self._zi = (
            np.zeros((self._sos.shape[0], 2), dtype=np.float64) if self._sos is not None else None
        )
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    def push(self, chunk: torch.Tensor) -> torch.Tensor:
        self._device = chunk.device
        self._dtype = chunk.dtype
        samples = chunk.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
        if self._pitch is not None:
            samples = self._pitch.push(samples)
        return self._finish(samples)

    def flush(self) -> torch.Tensor:
        samples = self._pitch.flush() if self._pitch is not None else torch.zeros(0)
        return self._finish(samples)

    def _finish(self, samples: torch.Tensor) -> torch.Tensor:
        filtered = _sos_filter(samples, self._sos, self._zi)
        if self._zi is not None and filtered[1] is not None:
            self._zi = filtered[1]
        out = filtered[0]
        if out.numel() == 0:
            return out
        return out.to(device=self._device, dtype=self._dtype)


class _PitchBlockStream:
    """Pitch-shift fixed blocks and crossfade the overlap so the length matches."""

    def __init__(self, sample_rate: int, semitones: float):
        self.sample_rate = sample_rate
        self.semitones = semitones
        self._pending = torch.zeros(0, dtype=torch.float32)
        self._tail: torch.Tensor | None = None

    def push(self, samples: torch.Tensor) -> torch.Tensor:
        self._pending = torch.cat([self._pending, samples.reshape(-1)])
        pieces: list[torch.Tensor] = []
        hop = _PITCH_BLOCK - _PITCH_OVERLAP
        while self._pending.numel() >= _PITCH_BLOCK:
            block = self._pending[:_PITCH_BLOCK]
            shifted = pitch_shift(block, self.sample_rate, self.semitones)
            if self._tail is None:
                pieces.append(shifted[:hop])
            else:
                pieces.append(_crossfade(self._tail, shifted[:_PITCH_OVERLAP]))
                pieces.append(shifted[_PITCH_OVERLAP:hop])
            self._tail = shifted[hop:]
            self._pending = self._pending[hop:]
        return _cat(pieces)

    def flush(self) -> torch.Tensor:
        rest = self._pending
        self._pending = torch.zeros(0, dtype=torch.float32)
        if rest.numel() == 0:
            self._tail = None
            return rest
        shifted = (
            pitch_shift(rest, self.sample_rate, self.semitones) if rest.numel() >= 64 else rest
        )
        tail = self._tail
        self._tail = None
        if tail is None:
            return shifted
        overlap = min(_PITCH_OVERLAP, tail.numel(), shifted.numel())
        return _cat([_crossfade(tail[:overlap], shifted[:overlap]), shifted[overlap:]])


def pitch_shift(audio: torch.Tensor, sample_rate: int, semitones: float) -> torch.Tensor:
    """Shift pitch by `semitones` and keep the number of samples.

    A phase vocoder changes duration without changing pitch, then resampling
    restores the original length and leaves the pitch shift behind.
    """
    if abs(semitones) < 1e-6:
        return audio
    source = audio.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    if source.numel() < 64:
        return source
    # sample_rate selects the FFT size so the same hop covers a similar time span.
    n_fft = 2048 if sample_rate >= 16000 else 1024
    hop = n_fft // 4
    window = torch.hann_window(n_fft, device=source.device, dtype=source.dtype)
    rate = 2.0 ** (-float(semitones) / 12.0)
    spec = torch.stft(
        source[None, :],
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    phase_advance = torch.linspace(0, math.pi * hop, spec.shape[-2], device=spec.device)[..., None]
    stretched_spec = _phase_vocoder(spec, rate, phase_advance)
    stretched_len = max(1, int(round(source.numel() / rate)))
    stretched = torch.istft(
        stretched_spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        length=stretched_len,
    )[0]
    return _resample_1d(stretched, source.numel())


def _phase_vocoder(spec: torch.Tensor, rate: float, phase_advance: torch.Tensor) -> torch.Tensor:
    """Speed a complex spectrogram up by `rate` without changing its pitch.

    `rate` below 1 lengthens the spectrogram. This follows the phase-vocoder
    used by torchaudio: interpolate magnitude, and accumulate the phase deviation
    from the hop's expected advance.
    """
    if rate == 1.0:
        return spec
    shape = spec.size()
    spec = spec.reshape([-1] + list(shape[-2:]))
    real_dtype = torch.real(spec).dtype
    time_steps = torch.arange(0, spec.size(-1), rate, device=spec.device, dtype=real_dtype)
    alphas = time_steps % 1.0
    phase_0 = spec[..., :1].angle()
    spec = torch.nn.functional.pad(spec, [0, 2])
    spec_0 = spec.index_select(-1, time_steps.long())
    spec_1 = spec.index_select(-1, (time_steps + 1).long())
    phase = spec_1.angle() - spec_0.angle() - phase_advance
    phase = phase - 2 * math.pi * torch.round(phase / (2 * math.pi))
    phase = phase + phase_advance
    phase = torch.cat([phase_0, phase[..., :-1]], dim=-1)
    phase_acc = torch.cumsum(phase, -1)
    magnitude = alphas * spec_1.abs() + (1 - alphas) * spec_0.abs()
    stretched = torch.polar(magnitude, phase_acc)
    return stretched.reshape(shape[:-2] + stretched.shape[1:])


def design_sos(sample_rate: int, profile: VoiceProfile) -> np.ndarray | None:
    """Biquad sections for timbre tilt and the three EQ bands. None if all gains are 0."""
    sections: list[np.ndarray] = []
    if abs(profile.timbre) >= 1e-6:
        half = profile.timbre / 2
        sections.append(_biquad("highshelf", sample_rate, _TIMBRE_PIVOT_HZ, half))
        sections.append(_biquad("lowshelf", sample_rate, _TIMBRE_PIVOT_HZ, -half))
    if abs(profile.eq_low_db) >= 1e-6:
        sections.append(_biquad("lowshelf", sample_rate, _EQ_LOW_HZ, profile.eq_low_db))
    if abs(profile.eq_mid_db) >= 1e-6:
        sections.append(_biquad("peaking", sample_rate, _EQ_MID_HZ, profile.eq_mid_db, q=0.8))
    if abs(profile.eq_high_db) >= 1e-6:
        sections.append(_biquad("highshelf", sample_rate, _EQ_HIGH_HZ, profile.eq_high_db))
    if not sections:
        return None
    return np.stack(sections, axis=0)


def _filter_block(audio: torch.Tensor, sample_rate: int, profile: VoiceProfile) -> torch.Tensor:
    sos = design_sos(sample_rate, profile)
    if sos is None:
        return audio
    zi = np.zeros((sos.shape[0], 2), dtype=np.float64)
    filtered, _zi = _sos_filter(audio, sos, zi)
    return filtered


def _sos_filter(
    audio: torch.Tensor, sos: np.ndarray | None, zi: np.ndarray | None
) -> tuple[torch.Tensor, np.ndarray | None]:
    if audio.numel() == 0 or sos is None or zi is None:
        return audio, zi
    filtered, next_zi = sosfilt(sos, audio.detach().cpu().numpy().astype(np.float64), zi=zi)
    return torch.from_numpy(np.ascontiguousarray(filtered, dtype=np.float32)), next_zi


def _biquad(
    kind: str, sample_rate: int, freq_hz: float, gain_db: float, q: float = 0.707
) -> np.ndarray:
    nyquist = sample_rate / 2
    freq = min(max(freq_hz, 20.0), nyquist * 0.9)
    amp = 10 ** (gain_db / 40)
    omega = 2 * math.pi * freq / sample_rate
    cos_w = math.cos(omega)
    sin_w = math.sin(omega)
    alpha = sin_w / (2 * q)
    if kind == "peaking":
        b0 = 1 + alpha * amp
        b1 = -2 * cos_w
        b2 = 1 - alpha * amp
        a0 = 1 + alpha / amp
        a1 = -2 * cos_w
        a2 = 1 - alpha / amp
    elif kind == "lowshelf":
        sqrt_amp = math.sqrt(amp)
        b0 = amp * ((amp + 1) - (amp - 1) * cos_w + 2 * sqrt_amp * alpha)
        b1 = 2 * amp * ((amp - 1) - (amp + 1) * cos_w)
        b2 = amp * ((amp + 1) - (amp - 1) * cos_w - 2 * sqrt_amp * alpha)
        a0 = (amp + 1) + (amp - 1) * cos_w + 2 * sqrt_amp * alpha
        a1 = -2 * ((amp - 1) + (amp + 1) * cos_w)
        a2 = (amp + 1) + (amp - 1) * cos_w - 2 * sqrt_amp * alpha
    elif kind == "highshelf":
        sqrt_amp = math.sqrt(amp)
        b0 = amp * ((amp + 1) + (amp - 1) * cos_w + 2 * sqrt_amp * alpha)
        b1 = -2 * amp * ((amp - 1) + (amp + 1) * cos_w)
        b2 = amp * ((amp + 1) + (amp - 1) * cos_w - 2 * sqrt_amp * alpha)
        a0 = (amp + 1) - (amp - 1) * cos_w + 2 * sqrt_amp * alpha
        a1 = 2 * ((amp - 1) - (amp + 1) * cos_w)
        a2 = (amp + 1) - (amp - 1) * cos_w - 2 * sqrt_amp * alpha
    else:
        raise ValueError(f"unknown biquad kind {kind}")
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0], dtype=np.float64)


def _resample_1d(audio: torch.Tensor, new_len: int) -> torch.Tensor:
    if audio.numel() == new_len:
        return audio
    return torch.nn.functional.interpolate(
        audio.view(1, 1, -1), size=new_len, mode="linear", align_corners=False
    ).view(-1)


def _crossfade(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    count = min(left.numel(), right.numel())
    if count == 0:
        return left.new_zeros(0)
    fade = torch.linspace(0, 1, count, dtype=left.dtype)
    return left[:count] * (1 - fade) + right[:count] * fade


def _cat(pieces: list[torch.Tensor]) -> torch.Tensor:
    pieces = [piece for piece in pieces if piece.numel()]
    if not pieces:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(pieces)
