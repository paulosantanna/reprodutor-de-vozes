"""HTTP API tests for timbre, pitch, and EQ on cloned voices."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from pocket_tts import main
from pocket_tts.data.voice_profile import VoiceProfile, profile_from_state
from pocket_tts.models.model_state import _import_model_state
from pocket_tts.modules.stateful_module import ModelState


class RecordingModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(mimi=SimpleNamespace(sample_rate=24000))
        self.profiles: list[VoiceProfile | None] = []

    def get_state_for_audio_prompt(
        self, audio_conditioning: str, truncate: bool = False
    ) -> ModelState:
        return {"layer": {"cache": torch.tensor([1.0, 2.0])}}

    def _cached_get_state_for_audio_prompt(
        self, audio_conditioning: str, truncate: bool = False
    ) -> ModelState:
        return self.get_state_for_audio_prompt(audio_conditioning, truncate)

    def generate_audio_stream(
        self,
        model_state: ModelState,
        text_to_generate: str,
        stop: object = None,
        voice_profile: VoiceProfile | None = None,
    ) -> Iterator[torch.Tensor]:
        self.profiles.append(voice_profile)
        yield torch.zeros(2400)


def _client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, RecordingModel]:
    model = RecordingModel()
    monkeypatch.setattr(main, "tts_model", model)
    monkeypatch.setattr(main, "default_voice_state", {"layer": {"cache": torch.zeros(1)}})
    monkeypatch.setattr(main, "saved_voice_states", {})
    return TestClient(main.web_app), model


def test_tts_applies_request_controls(monkeypatch: pytest.MonkeyPatch):
    client, model = _client(monkeypatch)

    response = client.post(
        "/tts", data={"text": "Hello.", "timbre": "4", "pitch": "-2", "eq_low": "3"}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    profile = model.profiles[-1]
    assert profile.timbre == 4
    assert profile.pitch_semitones == -2
    assert profile.eq_low_db == 3


def test_saved_voice_profile_is_used_on_later_synthesis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    client, model = _client(monkeypatch)

    saved = client.post(
        "/voice/profile", data={"voice_url": "alba", "pitch": "3", "eq_high": "-6", "timbre": "1.5"}
    )
    assert saved.status_code == 200, saved.text
    voice_id = saved.json()["voice_id"]

    later = client.post("/tts", data={"text": "Again.", "voice_id": voice_id})
    assert later.status_code == 200
    profile = model.profiles[-1]
    assert profile.pitch_semitones == 3
    assert profile.eq_high_db == -6
    assert profile.timbre == 1.5

    downloaded = client.get(f"/voice/profile/{voice_id}")
    assert downloaded.status_code == 200
    path = tmp_path / "voice.safetensors"
    path.write_bytes(downloaded.content)
    loaded = _import_model_state(path, torch.device("cpu"))
    assert torch.equal(loaded["layer"]["cache"], torch.tensor([1.0, 2.0]))
    assert profile_from_state(loaded).pitch_semitones == 3

    changed = client.post("/voice/profile", data={"voice_id": voice_id, "pitch": "-1"})
    assert changed.status_code == 200
    assert changed.json()["pitch"] == -1
    assert changed.json()["eq_high"] == -6

    client.post("/tts", data={"text": "After the change.", "voice_id": voice_id})
    assert model.profiles[-1].pitch_semitones == -1
    assert model.profiles[-1].eq_high_db == -6


def test_tts_rejects_an_out_of_range_control(monkeypatch: pytest.MonkeyPatch):
    client, _model = _client(monkeypatch)

    response = client.post("/tts", data={"text": "Hello.", "eq_mid": "40"})

    assert response.status_code == 400
    assert "eq_mid_db" in response.json()["detail"]


def test_saving_the_default_voice_updates_later_requests(monkeypatch: pytest.MonkeyPatch):
    client, model = _client(monkeypatch)

    saved = client.post("/voice/profile", data={"eq_low": "5"})
    assert saved.status_code == 200

    response = client.post("/tts", data={"text": "Using the default voice."})
    assert response.status_code == 200
    assert model.profiles[-1].eq_low_db == 5
