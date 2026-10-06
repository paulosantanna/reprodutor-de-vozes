"""Reading and writing the voice-conditioning state as safetensors.

A voice prompt is expensive to encode, so `pocket-tts export-voice` saves the
resulting per-module streaming state and generation reloads it. A voice profile
(timbre, pitch, EQ) is stored in the same file and reapplied at synthesis.
"""

from pathlib import Path
from urllib.parse import urlsplit

import safetensors
import safetensors.torch
import torch

from pocket_tts.data.voice_profile import (
    PROFILE_MODULE,
    VoiceProfile,
    profile_from_state,
    write_profile,
)


def _flatten_state(
    model_state: dict[str, dict[str, torch.Tensor]], voice_profile: VoiceProfile | None
) -> tuple[dict[str, torch.Tensor], dict[str, str] | None]:
    if voice_profile is not None:
        model_state = dict(model_state)
        write_profile(model_state, voice_profile)
    flat: dict[str, torch.Tensor] = {}
    for module_name, module_state in model_state.items():
        for key, tensor_value in module_state.items():
            flat[f"{module_name}/{key}"] = tensor_value
    metadata = (
        profile_from_state(model_state).to_metadata() if PROFILE_MODULE in model_state else None
    )
    return flat, metadata


def _save_kwargs(metadata: dict[str, str] | None) -> dict[str, dict[str, str]]:
    return {"metadata": metadata} if metadata is not None else {}


def export_model_state(
    model_state: dict[str, dict[str, torch.Tensor]],
    dest: str | Path,
    voice_profile: VoiceProfile | None = None,
):
    flat, metadata = _flatten_state(model_state, voice_profile)
    safetensors.torch.save_file(flat, dest, **_save_kwargs(metadata))


def export_model_state_bytes(
    model_state: dict[str, dict[str, torch.Tensor]], voice_profile: VoiceProfile | None = None
) -> bytes:
    flat, metadata = _flatten_state(model_state, voice_profile)
    return safetensors.torch.save(flat, **_save_kwargs(metadata))


def retune_saved_voice(
    source: str | Path,
    dest: str | Path | None = None,
    *,
    timbre: float | None = None,
    pitch_semitones: float | None = None,
    eq_low_db: float | None = None,
    eq_mid_db: float | None = None,
    eq_high_db: float | None = None,
) -> VoiceProfile:
    """Update the profile stored on an already cloned `.safetensors` voice.

    Unspecified controls keep the value already saved on the voice. The model
    state itself is not re-encoded.
    """
    source_path = Path(source)
    if not _is_safetensors_source(source_path):
        raise ValueError(
            "tune-voice updates a .safetensors voice produced by export-voice. "
            "Pass an audio file to export-voice instead."
        )
    state = _import_model_state(source_path, torch.device("cpu"))
    profile = profile_from_state(state).merged(
        timbre=timbre,
        pitch_semitones=pitch_semitones,
        eq_low_db=eq_low_db,
        eq_mid_db=eq_mid_db,
        eq_high_db=eq_high_db,
    )
    destination = source_path if dest is None else Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        export_model_state(state, temporary, voice_profile=profile)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return profile


def _is_safetensors_source(source: str | Path) -> bool:
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        source_text = urlsplit(source_text).path
    elif source_text.startswith("hf://"):
        source_text = source_text.rsplit("@", 1)[0]
    return source_text.endswith(".safetensors")


def _import_model_state(
    source: str | Path, device: torch.device
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    with safetensors.safe_open(source, framework="pt") as f:
        for key in f.keys():
            module_name, tensor_key = key.split("/")
            result.setdefault(module_name, {})
            if tensor_key == "current_end":
                # we used the shape[0] as step index before for torch.compile() compatibility,
                # but it's not needed anymore
                tensor = f.get_tensor(key)
                result[module_name]["offset"] = torch.full(
                    (1,), fill_value=tensor.shape[0], dtype=torch.long, device=device
                )
            else:
                result[module_name][tensor_key] = f.get_tensor(key).to(device)
        if PROFILE_MODULE not in result:
            profile = VoiceProfile.from_metadata(f.metadata())
            if not profile.is_neutral():
                write_profile(result, profile)
                for name, tensor in result[PROFILE_MODULE].items():
                    result[PROFILE_MODULE][name] = tensor.to(device)
    return result
