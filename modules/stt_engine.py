"""
VEYRA AI — Speech-to-Text Engine
Uses OpenAI Whisper for high-accuracy offline transcription.

IMPORTANT: Audio is passed as a numpy array directly to Whisper,
completely bypassing the ffmpeg requirement.
"""

import math
import traceback
import numpy as np
from scipy.signal import resample_poly
import whisper


class STTEngine:
    """
    Wraps OpenAI Whisper for single-utterance transcription.

    Audio is always passed as a float32 numpy array at 16 kHz so that
    Whisper never shells out to ffmpeg (which may not be installed).
    """

    def __init__(self, model_name: str = "base"):
        print(f"    -> Downloading / loading Whisper '{model_name}' model ...")
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Transcribe an audio waveform to text.

        Args:
            audio_data:  1-D float32 numpy array (mono).
            sample_rate: Sample rate in Hz.

        Returns:
            dict with ``success``, ``text``, and ``language``.
        """
        try:
            # ── Guard: empty / None ──────────────────────────────
            if audio_data is None or audio_data.size == 0:
                return {"success": True, "text": "", "language": "en"}

            # ── Stereo → mono ────────────────────────────────────
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            audio_data = audio_data.astype(np.float32)

            # ── Sanitize NaN / Inf ───────────────────────────────
            if not np.isfinite(audio_data).all():
                audio_data = np.nan_to_num(
                    audio_data, nan=0.0, posinf=1.0, neginf=-1.0
                )

            # ── Resample to 16 kHz (Whisper's native rate) ───────
            target_sr = 16000
            if sample_rate != target_sr and audio_data.size > 0:
                gcd = math.gcd(int(sample_rate), target_sr)
                up = target_sr // gcd
                down = int(sample_rate) // gcd
                audio_data = resample_poly(audio_data, up, down).astype(np.float32)

            # ── Normalize if clipping ────────────────────────────
            peak = float(np.max(np.abs(audio_data))) if audio_data.size > 0 else 0.0
            if peak > 1.0:
                audio_data = audio_data / peak

            # ── Too short (< 0.1 s at 16 kHz) ───────────────────
            if audio_data.size < 1600:
                return {"success": True, "text": "", "language": "en"}

            # ── Transcribe via numpy array — no ffmpeg needed ────
            result = self.model.transcribe(
                audio_data,       # numpy array, NOT a file path
                fp16=False,
                language="en",
            )

            text = result.get("text", "").strip()
            language = result.get("language", "en")

            return {"success": True, "text": text, "language": language}

        except Exception as exc:
            traceback.print_exc()
            return {"success": False, "text": "", "error": str(exc)}
