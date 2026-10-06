import math
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from pocket_tts.data.audio_effects import apply_voice_profile, pitch_shift, stream_with_profile
from pocket_tts.data.voice_profile import VoiceProfile, profile_from_state, write_profile
from pocket_tts.main import cli_app
from pocket_tts.models.model_state import (
    _import_model_state,
    export_model_state,
    retune_saved_voice,
)

runner = CliRunner()
SAMPLE_RATE = 24000


def _tone(freq: float, seconds: float = 1.5) -> torch.Tensor:
    time = torch.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return torch.sin(2 * math.pi * freq * time)


def _rms(audio: torch.Tensor) -> float:
    body = audio[SAMPLE_RATE // 5 :]
    return float(torch.sqrt(torch.mean(body.square())).item())


def _peak_freq(audio: torch.Tensor) -> float:
    body = audio[audio.numel() // 10 :]
    window = torch.hann_window(body.numel())
    spectrum = torch.fft.rfft(body * window).abs()
    spectrum[0] = 0
    bin_index = int(torch.argmax(spectrum).item())
    return bin_index * SAMPLE_RATE / body.numel()


def test_neutral_profile_keeps_the_waveform():
    audio = _tone(220)
    processed = apply_voice_profile(audio, SAMPLE_RATE, VoiceProfile())

    assert torch.equal(processed, audio)


def test_pitch_shift_raises_and_lowers_a_tone_without_changing_length():
    audio = _tone(220, seconds=2.0)
    higher = pitch_shift(audio, SAMPLE_RATE, 12)
    lower = pitch_shift(audio, SAMPLE_RATE, -12)

    assert higher.numel() == audio.numel()
    assert lower.numel() == audio.numel()
    assert _peak_freq(higher) == pytest.approx(440, abs=15)
    assert _peak_freq(lower) == pytest.approx(110, abs=8)


def test_eq_bands_change_the_matching_frequencies():
    low = apply_voice_profile(_tone(60), SAMPLE_RATE, VoiceProfile(eq_low_db=12))
    mid = apply_voice_profile(_tone(1000), SAMPLE_RATE, VoiceProfile(eq_mid_db=12))
    high = apply_voice_profile(_tone(8000), SAMPLE_RATE, VoiceProfile(eq_high_db=12))
    low_untouched = apply_voice_profile(_tone(8000), SAMPLE_RATE, VoiceProfile(eq_low_db=12))

    assert _rms(low) / _rms(_tone(60)) > 2.5
    assert _rms(mid) / _rms(_tone(1000)) > 2.5
    assert _rms(high) / _rms(_tone(8000)) > 1.8
    assert _rms(low_untouched) / _rms(_tone(8000)) < 1.4


def test_timbre_brightens_highs_relative_to_lows():
    low = _tone(150)
    high = _tone(6000)
    brighter = VoiceProfile(timbre=12)
    darker = VoiceProfile(timbre=-12)

    bright_ratio = _rms(apply_voice_profile(high, SAMPLE_RATE, brighter)) / _rms(
        apply_voice_profile(low, SAMPLE_RATE, brighter)
    )
    dark_ratio = _rms(apply_voice_profile(high, SAMPLE_RATE, darker)) / _rms(
        apply_voice_profile(low, SAMPLE_RATE, darker)
    )

    assert bright_ratio > dark_ratio * 2


def test_streaming_profile_preserves_length_and_pitch():
    audio = _tone(220, seconds=2.0)
    chunks = list(audio.split(1000))
    profile = VoiceProfile(pitch_semitones=12, eq_high_db=3)
    processed = torch.cat(list(stream_with_profile(iter(chunks), SAMPLE_RATE, profile)))

    assert processed.numel() == audio.numel()
    assert _peak_freq(processed) == pytest.approx(440, abs=20)


def test_profile_roundtrip_keeps_the_voice_tensors(tmp_path: Path):
    state = {"layer": {"cache": torch.tensor([1.25, -0.5, 3.0])}}
    profile = VoiceProfile(timbre=2.5, pitch_semitones=-3, eq_low_db=1, eq_mid_db=-2, eq_high_db=4)
    destination = tmp_path / "voice.safetensors"

    export_model_state(state, destination, voice_profile=profile)
    loaded = _import_model_state(destination, torch.device("cpu"))

    assert torch.equal(loaded["layer"]["cache"], state["layer"]["cache"])
    assert profile_from_state(loaded).as_dict() == profile.as_dict()


def test_retune_updates_only_the_controls_that_were_passed(tmp_path: Path):
    state = {"layer": {"cache": torch.tensor([4.0])}}
    destination = tmp_path / "voice.safetensors"
    write_profile(state, VoiceProfile(timbre=1, pitch_semitones=2, eq_low_db=3))
    export_model_state(state, destination)

    updated = retune_saved_voice(destination, pitch_semitones=-4, eq_high_db=5)
    loaded = _import_model_state(destination, torch.device("cpu"))

    assert updated.as_dict() == {
        "timbre": 1,
        "pitch_semitones": -4,
        "eq_low_db": 3,
        "eq_mid_db": 0,
        "eq_high_db": 5,
    }
    assert torch.equal(loaded["layer"]["cache"], torch.tensor([4.0]))
    assert profile_from_state(loaded).as_dict() == updated.as_dict()


def test_tune_voice_command_saves_the_profile_back(tmp_path: Path):
    voice = tmp_path / "voice.safetensors"
    export_model_state({"layer": {"cache": torch.ones(2)}}, voice)

    result = runner.invoke(
        cli_app, ["tune-voice", str(voice), "--pitch", "5", "--eq-mid", "-3", "-q"]
    )

    assert result.exit_code == 0, result.output
    loaded = profile_from_state(_import_model_state(voice, torch.device("cpu")))
    assert loaded.pitch_semitones == 5
    assert loaded.eq_mid_db == -3
    assert loaded.timbre == 0


def test_tune_voice_rejects_an_audio_file(tmp_path: Path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"not a voice state")

    result = runner.invoke(cli_app, ["tune-voice", str(audio), "--timbre", "1"])

    assert result.exit_code != 0


def test_voice_profile_rejects_values_outside_the_supported_range():
    with pytest.raises(ValueError, match="pitch_semitones"):
        VoiceProfile(pitch_semitones=30)
