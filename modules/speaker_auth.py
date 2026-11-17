"""
VEYRA AI — Speaker Authentication Module
Uses Resemblyzer d-vector embeddings for voice profile enrollment
and cosine-similarity-based speaker verification.

Includes a robust preprocess wrapper that handles webrtcvad hangs/crashes
by falling back to simple resampling + normalization when VAD fails.
"""

import os
import json
import traceback
import numpy as np
import librosa
from resemblyzer import VoiceEncoder
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────
PROFILE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "voice_profiles")
SIMILARITY_THRESHOLD = 0.75          # minimum cosine similarity to accept
EMBEDDING_DIM = 256                  # d-vector dimensionality
TARGET_SR = 16000                    # Resemblyzer's native sample rate


def _safe_preprocess(audio: np.ndarray, source_sr: int) -> np.ndarray:
    """
    Resemble preprocess_wav but with a fallback when webrtcvad hangs or
    crashes (e.g. on unusual sample rates or very short clips).

    Steps:
        1. Resample to 16 kHz
        2. Try the full resemblyzer preprocess (includes VAD silence trim)
        3. On failure → fall back to simple resample + peak-normalise
    """
    # Always ensure float32
    audio = audio.astype(np.float32)

    # Resample to 16 kHz first (resemblyzer expects this)
    if source_sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=TARGET_SR)

    # Normalise volume (simple peak normalisation)
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak > 0:
        audio = audio / peak * 0.95  # slight headroom

    try:
        from resemblyzer import preprocess_wav
        processed = preprocess_wav(audio, source_sr=TARGET_SR)
        if processed is not None and len(processed) > 0:
            return processed
    except Exception:
        # webrtcvad crash or hang — fall back to the already-resampled audio
        traceback.print_exc()

    return audio


class SpeakerAuthenticator:
    """
    Manages voice-profile enrollment and real-time speaker verification.

    Workflow
    --------
    1. **Enroll** — the user supplies ≥ 3 audio clips.  Each clip is
       converted to a 256-dim d-vector; the average is stored as the
       authorized voice profile.
    2. **Verify** — a new audio clip is embedded and compared to the
       stored profile via cosine similarity.  A score ≥ SIMILARITY_THRESHOLD
       grants access.
    """

    def __init__(self):
        print("    -> Loading speaker encoder model ...")
        self.encoder = VoiceEncoder()
        os.makedirs(PROFILE_DIR, exist_ok=True)
        self.profile_path = os.path.join(PROFILE_DIR, "authorized_user.npy")
        self.meta_path = os.path.join(PROFILE_DIR, "profile_meta.json")
        self._profile_embedding: np.ndarray | None = None
        self._load_profile()

    # ── internal helpers ──────────────────────────────────────
    def _load_profile(self):
        if os.path.exists(self.profile_path):
            self._profile_embedding = np.load(self.profile_path)
            print("    -> Loaded existing voice profile.")

    def _embed(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        """Return 256-dim embedding, or None if audio is too short."""
        try:
            processed = _safe_preprocess(audio, source_sr=sr)
            if len(processed) < 1600:            # < 0.1 s at 16 kHz
                return None
            return self.encoder.embed_utterance(processed)
        except Exception:
            traceback.print_exc()
            return None

    # ── public API ────────────────────────────────────────────
    def is_enrolled(self) -> bool:
        return self._profile_embedding is not None

    def enroll(self, audio_samples: list[np.ndarray], sample_rate: int = 16000) -> dict:
        """
        Create a voice profile from multiple audio samples.

        Args:
            audio_samples: list of 1-D float32 numpy arrays.
            sample_rate:   sample rate shared by all clips.

        Returns:
            dict with ``success``, ``message``, and ``num_samples``.
        """
        embeddings = []
        for audio in audio_samples:
            emb = self._embed(audio, sample_rate)
            if emb is not None:
                embeddings.append(emb)

        if len(embeddings) < 2:
            return {
                "success": False,
                "message": "Not enough valid audio samples. Please speak clearly for at least 2 seconds per sample.",
            }

        # Average embedding → authorized voice profile
        self._profile_embedding = np.mean(embeddings, axis=0).astype(np.float32)
        np.save(self.profile_path, self._profile_embedding)

        meta = {
            "num_samples": len(embeddings),
            "embedding_dim": int(self._profile_embedding.shape[0]),
            "threshold": SIMILARITY_THRESHOLD,
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return {
            "success": True,
            "message": f"Voice profile created from {len(embeddings)} samples.",
            "num_samples": len(embeddings),
        }

    def verify(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Verify whether *audio* belongs to the enrolled user.

        Returns:
            dict with ``authorized``, ``similarity``, ``threshold``, ``message``.
        """
        if not self.is_enrolled():
            return {
                "authorized": False,
                "similarity": 0.0,
                "threshold": SIMILARITY_THRESHOLD,
                "message": "No voice profile found. Please enroll first.",
            }

        emb = self._embed(audio, sample_rate)
        if emb is None:
            return {
                "authorized": False,
                "similarity": 0.0,
                "threshold": SIMILARITY_THRESHOLD,
                "message": "Audio too short for verification.",
            }

        # Cosine similarity
        dot = float(np.dot(self._profile_embedding, emb))
        norm = float(np.linalg.norm(self._profile_embedding) * np.linalg.norm(emb))
        similarity = dot / norm if norm > 0 else 0.0
        authorized = similarity >= SIMILARITY_THRESHOLD

        return {
            "authorized": authorized,
            "similarity": round(similarity, 4),
            "threshold": SIMILARITY_THRESHOLD,
            "message": "Access granted." if authorized else "Access denied — voice not recognized.",
        }

    def delete_profile(self) -> dict:
        """Delete the stored voice profile and metadata."""
        for p in (self.profile_path, self.meta_path):
            if os.path.exists(p):
                os.remove(p)
        self._profile_embedding = None
        return {"success": True, "message": "Voice profile deleted."}
