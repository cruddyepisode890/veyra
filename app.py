"""
VEYRA AI — Flask Application Server
=========================================
Main entry-point.  Exposes REST endpoints consumed by the browser UI:

    GET  /                   → Dashboard
    GET  /api/status         → Enrollment check
    POST /api/enroll         → Multi-sample voice enrollment
    POST /api/process        → Full pipeline (denoise → auth → STT → NLP → TTS)
    GET  /api/tts-audio      → Stream a TTS WAV file
    DELETE /api/delete-profile → Remove stored voice profile
"""

import io
import os

import numpy as np
import soundfile as sf
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

from modules.speaker_auth import SpeakerAuthenticator
from modules.noise_suppression import suppress_noise, estimate_noise_level
from modules.stt_engine import STTEngine
from modules.nlp_engine import recognize_intent
from modules.tts_engine import TTSEngine

# ── Bootstrap ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

print("\n==========================================")
print("          Veyra AI -- Starting            ")
print("==========================================\n")

print("Initializing Speaker Authenticator ...")
authenticator = SpeakerAuthenticator()

print("Loading Whisper STT model ...")
stt_engine = STTEngine(model_name="base")

print("Initializing TTS Engine ...")
tts_engine = TTSEngine()

print("\nAll modules loaded -- Veyra AI is ready!\n")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "audio")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────
def _read_audio(file_storage) -> tuple[np.ndarray | None, int, str | None]:
    """Read a WAV from a Flask FileStorage object."""
    try:
        data, sr = sf.read(io.BytesIO(file_storage.read()))
        if data.ndim > 1:                       # stereo → mono
            data = np.mean(data, axis=1)
        return data.astype(np.float32), sr, None
    except Exception as exc:
        return None, 16000, str(exc)


# ── Global Error Handlers ─────────────────────────────────────
@app.errorhandler(500)
def handle_500(e):
    return jsonify({"success": False, "message": "Internal server error", "error": str(e)}), 500

@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "message": "API route not found", "error": str(e)}), 404
    return render_template("index.html")


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    try:
        return jsonify({"enrolled": authenticator.is_enrolled(), "status": "ready"})
    except Exception as exc:
        return jsonify({"enrolled": False, "status": "error", "error": str(exc)}), 500


@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    """Accept multiple audio files and build a voice profile."""
    try:
        samples: list[np.ndarray] = []
        rate = 16000

        for key in sorted(request.files):
            audio, sr, err = _read_audio(request.files[key])
            if audio is not None and audio.size > 0:
                audio = suppress_noise(audio, sr)
                samples.append(audio)
                rate = sr

        if len(samples) < 2:
            return jsonify({
                "success": False,
                "message": "Please provide at least 2 clear audio samples.",
            }), 400

        result = authenticator.enroll(samples, rate)
        status = 200 if result["success"] else 400
        return jsonify(result), status
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Server error: {str(exc)}"}), 500


@app.route("/api/process", methods=["POST"])
def api_process():
    """
    Full voice-command pipeline:
        1. Noise suppression
        2. Speaker verification
        3. Speech-to-text  (Whisper)
        4. NLP / intent recognition
        5. Text-to-speech  (response)
    """
    try:
        if "audio" not in request.files:
            return jsonify({"success": False, "error": "No audio file provided."}), 400

        audio, sr, err = _read_audio(request.files["audio"])
        if err or audio is None:
            return jsonify({"success": False, "error": err or "Invalid audio data."}), 400

        # 1 — Noise suppression
        db_before = estimate_noise_level(audio)
        cleaned = suppress_noise(audio, sr)
        db_after = estimate_noise_level(cleaned)

        # 2 — Speaker verification
        auth = authenticator.verify(cleaned, sr)

        if not auth["authorized"]:
            return jsonify({
                "success": True,
                "authorized": False,
                "similarity": auth["similarity"],
                "threshold": auth["threshold"],
                "message": auth["message"],
                "pipeline": {
                    "noise_suppression": {
                        "before_db": round(db_before, 2),
                        "after_db": round(db_after, 2),
                    },
                    "speaker_auth": auth,
                },
            })

        # 3 — Speech-to-text
        stt = stt_engine.transcribe(cleaned, sr)
        if not stt["success"]:
            err_msg = stt.get("error", "Unknown error")
            print(f"[STT Error] {err_msg}")
            return jsonify({"success": False, "error": f"Speech recognition failed: {err_msg}"}), 500

        # 4 — NLP
        nlp = recognize_intent(stt["text"])

        # 5 — TTS
        tts = tts_engine.synthesize(nlp["response"])

        response = {
            "success": True,
            "authorized": True,
            "pipeline": {
                "noise_suppression": {
                    "before_db": round(db_before, 2),
                    "after_db": round(db_after, 2),
                },
                "speaker_auth": {
                    "similarity": auth["similarity"],
                    "threshold": auth["threshold"],
                    "authorized": True,
                },
                "stt": {
                    "text": stt["text"],
                    "language": stt.get("language", "en"),
                },
                "nlp": {
                    "intent": nlp["intent"],
                    "confidence": nlp["confidence"],
                    "response": nlp["response"],
                },
                "tts": {"success": tts["success"]},
            },
        }

        if tts["success"] and tts.get("audio_path"):
            filename = os.path.basename(tts["audio_path"])
            response["audio_url"] = f"/static/audio/tts/{filename}"

        return jsonify(response)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Processing error: {str(exc)}"}), 500


@app.route("/api/tts-audio", methods=["GET"])
def api_tts_audio():
    path = request.args.get("path", "")
    if not path:
        filename = request.args.get("filename", "")
        if filename:
            path = os.path.join(UPLOAD_DIR, "tts", filename)
    if path and os.path.isfile(path):
        return send_file(path, mimetype="audio/wav")
    return jsonify({"error": "Audio not found."}), 404


@app.route("/api/delete-profile", methods=["DELETE"])
def api_delete_profile():
    try:
        return jsonify(authenticator.delete_profile())
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


# ── Entry-point ───────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=5000)

