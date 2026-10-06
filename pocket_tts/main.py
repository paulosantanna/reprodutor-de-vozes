import copy
import io
import logging
import os
import sys
import tempfile
import threading
import uuid
from collections.abc import Generator
from pathlib import Path
from queue import Queue
from typing import Annotated, BinaryIO, cast

import typer
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.data.voice_profile import VoiceProfile, controls_profile, write_profile
from pocket_tts.default_parameters import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_FRAMES_AFTER_EOS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_SAMPLER_DECODE_STEPS,
    MAX_TOKEN_PER_CHUNK,
    get_default_text_for_language,
    get_default_voice_for_language,
)
from pocket_tts.models.model_state import (
    export_model_state,
    export_model_state_bytes,
    retune_saved_voice,
)
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import ModelState
from pocket_tts.utils.logging_utils import enable_logging
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

logger = logging.getLogger(__name__)

cli_app = typer.Typer(
    help="Kyutai Pocket TTS - Text-to-Speech generation tool", pretty_exceptions_show_locals=False
)


# ------------------------------------------------------
# The pocket-tts server implementation
# ------------------------------------------------------

# Global model instance
tts_model: TTSModel | None = None
# State of the voice served when a request doesn't specify one. It is resolved once from the
# `serve` options, so that requests never pay for the encoding of the default voice.
default_voice_state: ModelState | None = None
# Cloned voices whose timbre, pitch, and EQ were saved during this process.
saved_voice_states: dict[str, ModelState] = {}

web_app = FastAPI(
    title="Kyutai Pocket TTS API", description="Text-to-Speech generation API", version="1.0.0"
)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://pod1-10007.internal.kyutai.org",
        "https://kyutai.org",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _loaded_model() -> TTSModel:
    if tts_model is None:
        raise RuntimeError("no model loaded: `pocket-tts serve` loads it before serving requests")
    return tts_model


@web_app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """Serve the frontend."""
    static_path = Path(__file__).parent / "static" / "index.html"
    content = static_path.read_text()
    # Replace the placeholder with the actual default text prompt
    origin = str(_loaded_model().origin)
    print(origin)
    content = content.replace("DEFAULT_TEXT_PROMPT", get_default_text_for_language(origin))
    return content


@web_app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


def write_to_queue(
    queue: Queue[bytes | None],
    text_to_generate: str,
    model_state: ModelState,
    stop: threading.Event,
    voice_profile: VoiceProfile | None,
):
    """Allows writing to the StreamingResponse as if it were a file."""

    class FileLikeToQueue(io.IOBase):
        def __init__(self, queue: Queue[bytes | None]):
            self.queue = queue

        def write(self, data: bytes):
            self.queue.put(data)

        def flush(self):
            pass

        def close(self):
            self.queue.put(None)

    model = _loaded_model()
    audio_chunks = model.generate_audio_stream(
        model_state=model_state,
        text_to_generate=text_to_generate,
        stop=stop,
        voice_profile=voice_profile,
    )
    # FileLikeToQueue only implements the write/close subset that StreamingWAVWriter uses.
    stream_audio_chunks(
        cast(BinaryIO, FileLikeToQueue(queue)), audio_chunks, model.config.mimi.sample_rate
    )


def generate_data_with_state(
    text_to_generate: str, model_state: ModelState, voice_profile: VoiceProfile | None
) -> Generator[bytes, None, None]:
    queue: Queue[bytes | None] = Queue()
    stop = threading.Event()

    # Run your function in a thread
    thread = threading.Thread(
        target=write_to_queue, args=(queue, text_to_generate, model_state, stop, voice_profile)
    )
    thread.start()

    try:
        # Yield data as it becomes available
        while True:
            data = queue.get()
            if data is None:
                break
            yield data
    finally:
        # Also runs when the client disconnects: stop the generation instead of
        # finishing it for nobody, and make sure the worker is done with the model.
        stop.set()
        thread.join()


def _profile_or_400(
    model_state: ModelState,
    timbre: float | None,
    pitch: float | None,
    eq_low: float | None,
    eq_mid: float | None,
    eq_high: float | None,
) -> VoiceProfile:
    try:
        return controls_profile(
            model_state, timbre=timbre, pitch=pitch, eq_low=eq_low, eq_mid=eq_mid, eq_high=eq_high
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _state_from_upload(voice_wav: UploadFile) -> ModelState:
    # Preserve the extension so safetensors voices and audio files both load.
    suffix = Path(voice_wav.filename).suffix if voice_wav.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        content = voice_wav.file.read()
        temp_file.write(content)
        temp_file.flush()
        temp_file_path = temp_file.name
    try:
        return _loaded_model().get_state_for_audio_prompt(Path(temp_file_path), truncate=True)
    finally:
        os.unlink(temp_file_path)


def _state_for_request(
    voice_url: str | None, voice_wav: UploadFile | None, voice_id: str | None
) -> tuple[ModelState, bool]:
    """Resolve the voice for a request. The bool is true when the default voice was used."""
    if voice_url is not None and voice_wav is not None:
        raise HTTPException(status_code=400, detail="Cannot provide both voice_url and voice_wav")
    if voice_id is not None and (voice_url is not None or voice_wav is not None):
        raise HTTPException(
            status_code=400, detail="Cannot combine voice_id with voice_url or voice_wav"
        )
    if voice_id is not None:
        try:
            return saved_voice_states[voice_id], False
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown voice_id") from exc
    if voice_url is not None:
        if not (
            voice_url.startswith("http://")
            or voice_url.startswith("https://")
            or voice_url.startswith("hf://")
            or voice_url in _ORIGINS_OF_PREDEFINED_VOICES
        ):
            raise HTTPException(
                status_code=400, detail="voice_url must start with http://, https://, or hf://"
            )
        logging.warning("Using voice from URL: %s", voice_url)
        return _loaded_model()._cached_get_state_for_audio_prompt(voice_url), False
    if voice_wav is not None:
        return _state_from_upload(voice_wav), False
    if default_voice_state is not None:
        return default_voice_state, True
    raise HTTPException(status_code=500, detail="The server has no default voice loaded.")


def _profile_payload(voice_id: str, profile: VoiceProfile) -> dict[str, float | str]:
    return {
        "voice_id": voice_id,
        "timbre": profile.timbre,
        "pitch": profile.pitch_semitones,
        "eq_low": profile.eq_low_db,
        "eq_mid": profile.eq_mid_db,
        "eq_high": profile.eq_high_db,
    }


@web_app.post("/tts")
def text_to_speech(
    text: str = Form(...),
    voice_url: str | None = Form(None),
    voice_wav: UploadFile | None = File(None),
    voice_id: str | None = Form(None),
    timbre: float | None = Form(None),
    pitch: float | None = Form(None),
    eq_low: float | None = Form(None),
    eq_mid: float | None = Form(None),
    eq_high: float | None = Form(None),
) -> StreamingResponse:
    """
    Generate speech from text using the pre-loaded voice prompt or a custom voice.

    Args:
        text: Text to convert to speech
        voice_url: Optional built-in voice name (e.g., "alba"), or voice URL (http://, https://, or hf://)
        voice_wav: Optional uploaded voice file (mutually exclusive with voice_url)
        voice_id: Optional id returned by POST /voice/profile
        timbre: Spectral brightness in dB. Omitted values keep the voice profile.
        pitch: Pitch shift in semitones.
        eq_low: Low-band EQ gain in dB.
        eq_mid: Mid-band EQ gain in dB.
        eq_high: High-band EQ gain in dB.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    model_state, _used_default = _state_for_request(voice_url, voice_wav, voice_id)
    profile = _profile_or_400(model_state, timbre, pitch, eq_low, eq_mid, eq_high)
    return StreamingResponse(
        generate_data_with_state(text, model_state, profile),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


@web_app.post("/voice/profile")
def save_voice_profile(
    voice_url: str | None = Form(None),
    voice_wav: UploadFile | None = File(None),
    voice_id: str | None = Form(None),
    timbre: float | None = Form(None),
    pitch: float | None = Form(None),
    eq_low: float | None = Form(None),
    eq_mid: float | None = Form(None),
    eq_high: float | None = Form(None),
) -> dict[str, float | str]:
    """Store timbre, pitch, and EQ on a cloned voice for later synthesis."""
    global default_voice_state
    model_state, used_default = _state_for_request(voice_url, voice_wav, voice_id)
    # The resolved state may be cached or shared. Copy before writing the profile back.
    model_state = copy.deepcopy(model_state)
    profile = _profile_or_400(model_state, timbre, pitch, eq_low, eq_mid, eq_high)
    write_profile(model_state, profile)
    stored_id = voice_id if voice_id is not None else uuid.uuid4().hex
    saved_voice_states[stored_id] = model_state
    if used_default:
        default_voice_state = model_state
    return _profile_payload(stored_id, profile)


@web_app.get("/voice/profile/{voice_id}")
def download_voice_profile(voice_id: str) -> Response:
    """Download a saved voice, including its timbre, pitch, and EQ, as safetensors."""
    try:
        model_state = saved_voice_states[voice_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown voice_id") from exc
    payload = export_model_state_bytes(model_state)
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{voice_id}.safetensors"'},
    )


@cli_app.command()
def serve(
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help="Language for the TTS model. "
            "'english_2026-01', 'english_2026-04', 'english_2026-09', 'english_drifting_26-09', 'english', 'french', 'french_24l', 'german', 'german_24l', 'portuguese', 'portuguese_24l', 'italian', 'italian_24l', 'spanish', 'spanish_24l', 'dutch', 'dutch_24l'."
            " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-09'.",
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to a model config .yaml file: a local path, an https:// URL, or an hf:// path. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
    default_voice: Annotated[
        str | None,
        typer.Option(
            help="Voice used by requests that don't ask for one: a built-in voice name, "
            "a local path to an audio file or to a .safetensors voice, an https:// URL, "
            "or an hf:// path. Defaults to the built-in voice of the language.",
            show_default=False,
        ),
    ] = None,
    quantize: Annotated[
        bool, typer.Option(help="Apply int8 quantization to reduce memory usage")
    ] = False,
    timbre: Annotated[
        float | None,
        typer.Option(
            help="Spectral brightness in dB stored on the default voice. Positive is brighter."
        ),
    ] = None,
    pitch: Annotated[
        float | None, typer.Option(help="Pitch shift in semitones stored on the default voice.")
    ] = None,
    eq_low: Annotated[
        float | None, typer.Option(help="Low-band EQ gain in dB stored on the default voice.")
    ] = None,
    eq_mid: Annotated[
        float | None, typer.Option(help="Mid-band EQ gain in dB stored on the default voice.")
    ] = None,
    eq_high: Annotated[
        float | None, typer.Option(help="High-band EQ gain in dB stored on the default voice.")
    ] = None,
):
    """Start the FastAPI server."""

    global tts_model, default_voice_state
    tts_model = TTSModel.load_model(language=language, config=config, quantize=quantize)
    if default_voice is None:
        default_voice = get_default_voice_for_language(language, config)
    # Resolved before serving: a voice that cannot be loaded fails at startup instead of on
    # the first request, which would otherwise pay for the encoding of the audio file.
    default_voice_state = tts_model.get_state_for_audio_prompt(default_voice)
    profile = _cli_profile(default_voice_state, timbre, pitch, eq_low, eq_mid, eq_high)
    if profile is not None:
        write_profile(default_voice_state, profile)

    uvicorn.run("pocket_tts.main:web_app", host=host, port=port, reload=reload)


# ------------------------------------------------------
# The pocket-tts single generation CLI implementation
# ------------------------------------------------------


@cli_app.command()
def generate(
    text: Annotated[str | None, typer.Option(help="Text to generate")] = None,
    voice: Annotated[
        str | None,
        typer.Option(
            help=(
                "Path to audio conditioning file (voice to clone). "
                "Defaults to a built-in voice chosen from the language: "
                "'giovanni' for italian, 'lola' for spanish, 'juergen' for german, "
                "'rafael' for portuguese, 'estelle' for french, 'alba' otherwise. "
                "With the config or checkpoint argument, defaults to alba's audio file, "
                "which any model can clone."
            ),
            show_default=False,
        ),
    ] = None,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help=(
                "Language for the TTS model. "
                "'english_2026-01', 'english_2026-04', 'english_2026-09', 'english_drifting_26-09', 'english', 'french', 'french_24l', 'german', 'german_24l', 'portuguese', 'portuguese_24l', 'italian', 'italian_24l', 'spanish', 'spanish_24l', 'dutch', 'dutch_24l'."
                " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-09'. "
                "The '24l' variants are bigger models, "
                "not distilled yet and here only as preview. They're not the final "
                "models for those languages."
            ),
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to a model config .yaml file: a local path, an https:// URL, or an hf:// path. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
    checkpoint: Annotated[
        str | None,
        typer.Option(help="Training checkpoint (.pt) to load instead of the config's weights"),
    ] = None,
    sampler_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_SAMPLER_DECODE_STEPS,
    lsd_decode_steps: Annotated[
        int | None, typer.Option(hidden=True, help="Deprecated: use --sampler-decode-steps")
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(
            help="Temperature for generation. Defaults to the model's recommended "
            "value from its config (0.3)."
        ),
    ] = None,
    noise_clamp: Annotated[
        float | None, typer.Option(help="Noise clamp value")
    ] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[float, typer.Option(help="EOS threshold")] = DEFAULT_EOS_THRESHOLD,
    frames_after_eos: Annotated[
        int | None, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    output_path: Annotated[
        str, typer.Option(help="Output path for generated audio")
    ] = "./tts_output.wav",
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
    max_tokens: Annotated[
        int, typer.Option(help="Maximum number of tokens per chunk.")
    ] = MAX_TOKEN_PER_CHUNK,
    quantize: Annotated[
        bool, typer.Option(help="Apply int8 quantization to reduce memory usage")
    ] = False,
    timbre: Annotated[
        float | None,
        typer.Option(
            help="Spectral brightness in dB. Positive is brighter. "
            "Omitted values keep the settings saved on the voice."
        ),
    ] = None,
    pitch: Annotated[
        float | None,
        typer.Option(
            help="Pitch shift in semitones. Positive is higher. "
            "Omitted values keep the settings saved on the voice."
        ),
    ] = None,
    eq_low: Annotated[
        float | None, typer.Option(help="Low-band EQ gain in dB (shelf around 200 Hz).")
    ] = None,
    eq_mid: Annotated[
        float | None, typer.Option(help="Mid-band EQ gain in dB (peak around 1 kHz).")
    ] = None,
    eq_high: Annotated[
        float | None, typer.Option(help="High-band EQ gain in dB (shelf around 5 kHz).")
    ] = None,
):
    """Generate speech using Kyutai Pocket TTS."""
    if lsd_decode_steps is not None:
        logger.warning("--lsd-decode-steps is deprecated, use --sampler-decode-steps")
        sampler_decode_steps = lsd_decode_steps
    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        if text is None:
            text = get_default_text_for_language(language)
        if text == "-":
            # Read text from stdin
            text = sys.stdin.read()

        if not text.strip():
            logger.error("No input received from stdin.")
            raise typer.Exit(code=1)
        tts_model = TTSModel.load_model(
            language=language,
            config=config,
            temp=temperature,
            sampler_decode_steps=sampler_decode_steps,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            quantize=quantize,
            checkpoint=checkpoint,
        )
        tts_model.to(device)

        if voice is None:
            voice = get_default_voice_for_language(language, config, checkpoint)
        model_state_for_voice = tts_model.get_state_for_audio_prompt(voice)
        voice_profile = _cli_profile(model_state_for_voice, timbre, pitch, eq_low, eq_mid, eq_high)
        # Stream audio generation directly to file or stdout
        audio_chunks = tts_model.generate_audio_stream(
            model_state=model_state_for_voice,
            text_to_generate=text,
            frames_after_eos=frames_after_eos,
            max_tokens=max_tokens,
            voice_profile=voice_profile,
        )

        stream_audio_chunks(output_path, audio_chunks, tts_model.config.mimi.sample_rate)

        # Only print the result message if not writing to stdout
        if output_path != "-":
            logger.info("Results written in %s", output_path)
        logger.info("-" * 20)
        logger.info(
            "If you want to try multiple voices and prompts quickly, try the `serve` command."
        )
        logger.info(
            "If you like Kyutai projects, comment, like, subscribe at https://x.com/kyutai_labs"
        )


# ----------------------------------------------
# export audio to safetensors CLI implementation
# ----------------------------------------------


@cli_app.command()
def export_voice(
    audio_path: Annotated[
        str, typer.Argument(help="Audio file or directory to convert and export")
    ],
    export_path: Annotated[str, typer.Argument(help="Output file or directory")],
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help=(
                "Language for the TTS model. "
                "'english_2026-01', 'english_2026-04', 'english_2026-09', 'english_drifting_26-09', 'english', 'french', 'french_24l', 'german', 'german_24l', 'portuguese', 'portuguese_24l', 'italian', 'italian_24l', 'spanish', 'spanish_24l', 'dutch', 'dutch_24l'."
                " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-09'. "
                "The '24l' variants are bigger models, "
                "not distilled yet and here only as preview."
            ),
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to a model config .yaml file: a local path, an https:// URL, or an hf:// path. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
    timbre: Annotated[
        float | None, typer.Option(help="Spectral brightness in dB saved on the exported voice.")
    ] = None,
    pitch: Annotated[
        float | None, typer.Option(help="Pitch shift in semitones saved on the exported voice.")
    ] = None,
    eq_low: Annotated[
        float | None, typer.Option(help="Low-band EQ gain in dB saved on the exported voice.")
    ] = None,
    eq_mid: Annotated[
        float | None, typer.Option(help="Mid-band EQ gain in dB saved on the exported voice.")
    ] = None,
    eq_high: Annotated[
        float | None, typer.Option(help="High-band EQ gain in dB saved on the exported voice.")
    ] = None,
):
    """Convert and save audio to .safetensors file"""

    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        tts_model = TTSModel.load_model(language=language, config=config)
        voice_profile = _cli_profile({}, timbre, pitch, eq_low, eq_mid, eq_high)
        model_state = tts_model.get_state_for_audio_prompt(
            audio_conditioning=audio_path, truncate=True, voice_profile=voice_profile
        )
        export_model_state(model_state, export_path)


def _cli_profile(
    state: ModelState,
    timbre: float | None,
    pitch: float | None,
    eq_low: float | None,
    eq_mid: float | None,
    eq_high: float | None,
) -> VoiceProfile | None:
    if all(value is None for value in (timbre, pitch, eq_low, eq_mid, eq_high)):
        return None
    try:
        return controls_profile(
            state, timbre=timbre, pitch=pitch, eq_low=eq_low, eq_mid=eq_mid, eq_high=eq_high
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@cli_app.command()
def tune_voice(
    voice_path: Annotated[str, typer.Argument(help="Already cloned .safetensors voice to update")],
    output_path: Annotated[
        str | None,
        typer.Option(help="Where to write the voice. Defaults to overwriting voice_path."),
    ] = None,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    timbre: Annotated[
        float | None,
        typer.Option(help="Spectral brightness in dB. Omitted values keep the saved setting."),
    ] = None,
    pitch: Annotated[
        float | None,
        typer.Option(help="Pitch shift in semitones. Omitted values keep the saved setting."),
    ] = None,
    eq_low: Annotated[
        float | None, typer.Option(help="Low-band EQ gain in dB. Omitted values are kept.")
    ] = None,
    eq_mid: Annotated[
        float | None, typer.Option(help="Mid-band EQ gain in dB. Omitted values are kept.")
    ] = None,
    eq_high: Annotated[
        float | None, typer.Option(help="High-band EQ gain in dB. Omitted values are kept.")
    ] = None,
):
    """Change timbre, pitch, and EQ on an already cloned voice and save them back."""
    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        if all(value is None for value in (timbre, pitch, eq_low, eq_mid, eq_high)):
            raise typer.BadParameter(
                "Pass at least one of --timbre, --pitch, --eq-low, --eq-mid, or --eq-high."
            )
        try:
            profile = retune_saved_voice(
                voice_path,
                output_path,
                timbre=timbre,
                pitch_semitones=pitch,
                eq_low_db=eq_low,
                eq_mid_db=eq_mid,
                eq_high_db=eq_high,
            )
        except (ValueError, OSError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        destination = output_path or voice_path
        logger.info(
            "Saved voice profile to %s (timbre=%.2f dB, pitch=%.2f st, eq=%.2f/%.2f/%.2f dB)",
            destination,
            profile.timbre,
            profile.pitch_semitones,
            profile.eq_low_db,
            profile.eq_mid_db,
            profile.eq_high_db,
        )


if __name__ == "__main__":
    cli_app()
