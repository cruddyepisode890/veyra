"""
VEYRA AI — Text-to-Speech Engine
Uses pyttsx3 for offline, zero-latency speech synthesis.
"""

import os
import tempfile
import threading
import pyttsx3


class TTSEngine:
    """
    Offline text-to-speech using the system's SAPI5 (Windows) or
    espeak (Linux) voices.  Thread-safe via a lock because pyttsx3
    maintains internal state that isn't re-entrant.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "static", "audio", "tts"
        )
        os.makedirs(self._output_dir, exist_ok=True)

    def synthesize(self, text: str, output_path: str | None = None) -> dict:
        """
        Render *text* to a WAV file.

        Args:
            text:        The sentence to speak.
            output_path: Where to write the WAV.  Auto-generated if omitted.

        Returns:
            dict with ``success``, ``audio_path``, and ``text``.
        """
        if not output_path:
            fd, output_path = tempfile.mkstemp(
                suffix=".wav", dir=self._output_dir
            )
            os.close(fd)

        try:
            with self._lock:
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)       # words per minute
                engine.setProperty("volume", 0.9)

                # Pick a pleasant voice when available
                voices = engine.getProperty("voices")
                for voice in voices:
                    lower = voice.name.lower()
                    if "zira" in lower or "female" in lower or "hazel" in lower:
                        engine.setProperty("voice", voice.id)
                        break

                engine.save_to_file(text, output_path)
                engine.runAndWait()
                engine.stop()

            return {"success": True, "audio_path": output_path, "text": text}
        except Exception as exc:
            return {"success": False, "error": str(exc), "text": text}
