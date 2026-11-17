"""
VEYRA AI — Noise Suppression Module
Uses spectral gating via noisereduce to clean audio signals
before speaker verification and transcription.
"""

import numpy as np
import noisereduce as nr


def suppress_noise(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    prop_decrease: float = 0.8,
) -> np.ndarray:
    """
    Apply noise suppression to audio data using spectral gating.

    The algorithm estimates a noise profile from the signal itself
    (assuming stationary background noise) and subtracts the noise
    spectrum while preserving speech frequencies.

    Args:
        audio_data:    1-D numpy array of audio samples (float32).
        sample_rate:   Sample rate of the audio in Hz.
        prop_decrease: Proportion of noise to remove (0.0–1.0).

    Returns:
        Denoised audio as a numpy float32 array.
    """
    if audio_data.size == 0 or audio_data.size < 2048:
        return audio_data.astype(np.float32)

    try:
        reduced = nr.reduce_noise(
            y=audio_data.astype(np.float32),
            sr=sample_rate,
            prop_decrease=prop_decrease,
            stationary=True,
            n_fft=2048,
            hop_length=512,
        )
        return reduced.astype(np.float32)
    except Exception:
        return audio_data.astype(np.float32)


def estimate_noise_level(audio_data: np.ndarray) -> float:
    """
    Estimate the RMS noise level of an audio signal.

    Returns:
        RMS level in dB (dBFS).  Returns -infinity for silence.
    """
    if audio_data.size == 0:
        return float("-inf")
    rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
    if rms == 0:
        return float("-inf")
    return float(20.0 * np.log10(rms))
