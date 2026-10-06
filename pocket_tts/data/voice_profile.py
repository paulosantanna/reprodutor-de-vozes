"""Timbre, pitch, and EQ settings stored on a cloned voice.

A cloned voice is the model's streaming state (see `export_model_state`). These
settings ride along in that state under `__voice_profile__` and are applied to
later synthesis, so they can be changed without encoding the prompt again.
"""

import math

import torch

PROFILE_MODULE = "__voice_profile__"
PROFILE_FIELDS = ("timbre", "pitch_semitones", "eq_low_db", "eq_mid_db", "eq_high_db")
MAX_TIMBRE_DB = 24.0
MAX_PITCH_SEMITONES = 24.0
MAX_EQ_DB = 24.0

_LIMITS = {
    "timbre": (MAX_TIMBRE_DB, "dB"),
    "pitch_semitones": (MAX_PITCH_SEMITONES, "semitones"),
    "eq_low_db": (MAX_EQ_DB, "dB"),
    "eq_mid_db": (MAX_EQ_DB, "dB"),
    "eq_high_db": (MAX_EQ_DB, "dB"),
}


class VoiceProfile:
    """Post-synthesis controls for one voice.

    `timbre` is spectral brightness in dB (positive is brighter). `pitch_semitones`
    raises or lowers pitch without changing duration. The EQ fields are low / mid /
    high gains in dB. Zero leaves that control unchanged.
    """

    def __init__(
        self,
        timbre: float = 0.0,
        pitch_semitones: float = 0.0,
        eq_low_db: float = 0.0,
        eq_mid_db: float = 0.0,
        eq_high_db: float = 0.0,
    ):
        self.timbre = float(timbre)
        self.pitch_semitones = float(pitch_semitones)
        self.eq_low_db = float(eq_low_db)
        self.eq_mid_db = float(eq_mid_db)
        self.eq_high_db = float(eq_high_db)
        for name, (limit, unit) in _LIMITS.items():
            value = getattr(self, name)
            if not math.isfinite(value) or abs(value) > limit:
                raise ValueError(
                    f"{name} must be between {-limit:g} and {limit:g} {unit}, got {value}"
                )

    def is_neutral(self) -> bool:
        return all(abs(getattr(self, name)) < 1e-6 for name in PROFILE_FIELDS)

    def merged(
        self,
        *,
        timbre: float | None = None,
        pitch_semitones: float | None = None,
        eq_low_db: float | None = None,
        eq_mid_db: float | None = None,
        eq_high_db: float | None = None,
    ) -> "VoiceProfile":
        """Replace only the controls that were actually passed."""
        return VoiceProfile(
            timbre=self.timbre if timbre is None else timbre,
            pitch_semitones=self.pitch_semitones if pitch_semitones is None else pitch_semitones,
            eq_low_db=self.eq_low_db if eq_low_db is None else eq_low_db,
            eq_mid_db=self.eq_mid_db if eq_mid_db is None else eq_mid_db,
            eq_high_db=self.eq_high_db if eq_high_db is None else eq_high_db,
        )

    def to_metadata(self) -> dict[str, str]:
        return {f"voice_profile.{name}": f"{getattr(self, name):.6g}" for name in PROFILE_FIELDS}

    @classmethod
    def from_metadata(cls, metadata: dict[str, str] | None) -> "VoiceProfile":
        if not metadata:
            return cls()
        values: dict[str, float] = {}
        for name in PROFILE_FIELDS:
            raw = metadata.get(f"voice_profile.{name}")
            if raw is not None:
                values[name] = float(raw)
        return cls(**values)

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in PROFILE_FIELDS}


def profile_from_state(state: dict[str, dict[str, torch.Tensor]]) -> VoiceProfile:
    block = state.get(PROFILE_MODULE)
    if not block:
        return VoiceProfile()
    values: dict[str, float] = {}
    for name in PROFILE_FIELDS:
        tensor = block.get(name)
        if tensor is not None:
            values[name] = float(tensor.detach().float().reshape(-1)[0].item())
    return VoiceProfile(**values)


def write_profile(state: dict[str, dict[str, torch.Tensor]], profile: VoiceProfile) -> None:
    """Store `profile` on a voice state so a later export reloads it."""
    state[PROFILE_MODULE] = {
        name: torch.tensor([getattr(profile, name)], dtype=torch.float32) for name in PROFILE_FIELDS
    }


def controls_profile(
    state: dict[str, dict[str, torch.Tensor]],
    *,
    timbre: float | None = None,
    pitch: float | None = None,
    eq_low: float | None = None,
    eq_mid: float | None = None,
    eq_high: float | None = None,
) -> VoiceProfile:
    """Merge explicit controls onto the profile already stored on `state`."""
    return profile_from_state(state).merged(
        timbre=timbre, pitch_semitones=pitch, eq_low_db=eq_low, eq_mid_db=eq_mid, eq_high_db=eq_high
    )
