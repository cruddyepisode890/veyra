/**
 * VEYRA AI — Frontend Application
 * =====================================
 * Handles microphone recording (raw PCM → WAV), API communication,
 * enrollment flow, and UI state management.
 */

// ── API Base ──────────────────────────────────────────────────
// Automatically connects to Flask on port 5000 if HTML is opened via Live Preview or file://
const API_BASE = (window.location.protocol === "file:" || (window.location.port && window.location.port !== "5000"))
  ? "http://127.0.0.1:5000"
  : "";

// ── State ────────────────────────────────────────────────────
const state = {
  enrolled: false,
  recording: false,
  processing: false,

  // Audio recording
  audioCtx: null,
  mediaStream: null,
  sourceNode: null,
  processorNode: null,
  recordedChunks: [],
  sampleRate: 16000,

  // Enrollment
  enrollSamples: [],
  enrollStep: 0,
  enrollRecording: false,

  // Timer
  timerStart: 0,
  timerInterval: null,
};

const ENROLL_PROMPTS = [
  '"Hello, my name is, and I am the authorized user."',
  '"The quick brown fox jumps over the lazy dog."',
  '"I enjoy using Veyra, my personal assistant."',
  '"Every day brings new opportunities and challenges."',
  '"Technology makes our lives easier and more connected."',
];

const MIN_SAMPLES = 3;

// ── DOM refs ─────────────────────────────────────────────────
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// ── Init ─────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  await checkStatus();
  setupEventListeners();
  log("info", "Veyra AI interface loaded.");
});

async function checkStatus() {
  const dot = $("#status-dot");
  const text = $("#status-text");
  try {
    const res = await fetch(`${API_BASE}/api/status`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.enrolled = data.enrolled;
    if (dot) dot.classList.remove("offline");
    if (text) text.textContent = "System Ready";
    updateProfileUI();
    updateMicUI();
  } catch (err) {
    if (dot) dot.classList.add("offline");
    if (text) text.textContent = "Backend Offline";
    log("error", "Cannot reach backend. Make sure python app.py is running.");
  }
}

function setupEventListeners() {
  // Main mic button
  $("#mic-btn").addEventListener("click", toggleRecording);

  // Enrollment
  $("#btn-enroll").addEventListener("click", openEnrollModal);
  $("#btn-delete-profile").addEventListener("click", deleteProfile);
  $("#modal-close-btn").addEventListener("click", closeEnrollModal);
  $("#enroll-mic-btn").addEventListener("click", toggleEnrollRecording);

  // Close modal on overlay click
  $("#enroll-modal").addEventListener("click", (e) => {
    if (e.target === $("#enroll-modal")) closeEnrollModal();
  });
}

// ══════════════════════════════════════════════════════════════
//  Audio Recording (Raw PCM → WAV)
// ══════════════════════════════════════════════════════════════
async function startRecording() {
  try {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: false },
    });
  } catch (err) {
    log("error", "Microphone access denied.");
    return false;
  }

  state.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: 16000,
  });
  state.sampleRate = state.audioCtx.sampleRate;
  state.sourceNode = state.audioCtx.createMediaStreamSource(state.mediaStream);

  // ScriptProcessor captures raw PCM
  state.processorNode = state.audioCtx.createScriptProcessor(4096, 1, 1);
  state.recordedChunks = [];

  state.processorNode.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    state.recordedChunks.push(new Float32Array(input));
  };

  state.sourceNode.connect(state.processorNode);
  state.processorNode.connect(state.audioCtx.destination);

  state.recording = true;
  return true;
}

function stopRecording() {
  if (state.processorNode) {
    state.processorNode.disconnect();
    state.processorNode = null;
  }
  if (state.sourceNode) {
    state.sourceNode.disconnect();
    state.sourceNode = null;
  }
  if (state.audioCtx) {
    state.audioCtx.close();
    state.audioCtx = null;
  }
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((t) => t.stop());
    state.mediaStream = null;
  }
  state.recording = false;

  // Merge chunks into single Float32Array
  const totalLength = state.recordedChunks.reduce((s, c) => s + c.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of state.recordedChunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

/**
 * Encode Float32 PCM samples into a WAV Blob.
 */
function encodeWAV(samples, sampleRate) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = samples.length * (bitsPerSample / 8);
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeStr = (off, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);          // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  // Float32 → Int16
  let idx = 44;
  for (let i = 0; i < samples.length; i++, idx += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(idx, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([view], { type: "audio/wav" });
}

// ══════════════════════════════════════════════════════════════
//  Main interaction
// ══════════════════════════════════════════════════════════════
async function toggleRecording() {
  if (state.processing) return;

  if (!state.enrolled) {
    log("warn", "Please enroll your voice first.");
    return;
  }

  if (!state.recording) {
    // Start
    const ok = await startRecording();
    if (!ok) return;
    startTimer();
    updateMicUI();
    hideResults();
    log("info", "Recording started — speak your command.");
  } else {
    // Stop and process
    const samples = stopRecording();
    stopTimer();
    updateMicUI();

    if (samples.length < state.sampleRate * 0.5) {
      log("warn", "Recording too short. Please hold for at least 1 second.");
      return;
    }

    log("info", "Processing audio …");
    await processAudio(samples);
  }
}

async function processAudio(samples) {
  state.processing = true;
  showSpinner("Running AI pipeline …");
  setPipelineState("all", "");
  setPipelineState(1, "active");

  const wav = encodeWAV(samples, state.sampleRate);
  const form = new FormData();
  form.append("audio", wav, "command.wav");

  try {
    const res = await fetch(`${API_BASE}/api/process`, { method: "POST", body: form });
    const contentType = res.headers.get("content-type");

    let data;
    if (contentType && contentType.includes("application/json")) {
      data = await res.json();
    } else {
      const text = await res.text();
      throw new Error(`Server returned non-JSON error (${res.status}): ${text.slice(0, 100)}`);
    }

    hideSpinner();
    state.processing = false;

    if (!data.success && data.error) {
      log("error", data.error);
      return;
    }

    // Animate pipeline steps
    const pipeline = data.pipeline || {};
    if (pipeline.noise_suppression) {
      setPipelineState(1, "done");
      setPipelineStatus(1, `${pipeline.noise_suppression.before_db}→${pipeline.noise_suppression.after_db} dB`);
      log("success", `Noise suppressed: ${pipeline.noise_suppression.before_db} dB → ${pipeline.noise_suppression.after_db} dB`);
    }

    if (pipeline.speaker_auth) {
      const auth = pipeline.speaker_auth;
      if (auth.authorized) {
        setPipelineState(2, "done");
        setPipelineStatus(2, `sim: ${auth.similarity}`);
        log("success", `Speaker verified — similarity: ${auth.similarity}`);
      } else {
        setPipelineState(2, "failed");
        setPipelineStatus(2, `sim: ${auth.similarity}`);
        log("error", `Access denied — similarity: ${auth.similarity} (threshold: ${auth.threshold})`);
        showDenied(auth.similarity, auth.threshold);
        return;
      }
    }

    if (!data.authorized) {
      showDenied(data.similarity || 0, data.threshold || 0.75);
      return;
    }

    if (pipeline.stt) {
      setPipelineState(3, "done");
      setPipelineStatus(3, `"${truncate(pipeline.stt.text, 20)}"`);
      log("info", `Transcribed: "${pipeline.stt.text}"`);
      showTranscript(pipeline.stt.text);
    }

    if (pipeline.nlp) {
      setPipelineState(4, "done");
      setPipelineStatus(4, pipeline.nlp.intent);
      log("info", `Intent: ${pipeline.nlp.intent} (${pipeline.nlp.confidence})`);
    }

    if (pipeline.tts) {
      setPipelineState(5, pipeline.tts.success ? "done" : "failed");
    }

    // Show response
    if (pipeline.nlp && pipeline.nlp.response) {
      showResponse(pipeline.nlp, data.audio_url);
    }
  } catch (err) {
    hideSpinner();
    state.processing = false;
    log("error", `Request failed: ${err.message}`);
  }
}

// ══════════════════════════════════════════════════════════════
//  Enrollment flow
// ══════════════════════════════════════════════════════════════
function openEnrollModal() {
  state.enrollSamples = [];
  state.enrollStep = 0;
  state.enrollRecording = false;
  updateEnrollUI();
  $("#enroll-modal").classList.add("visible");
}

function closeEnrollModal() {
  if (state.enrollRecording) {
    stopRecording();
    state.enrollRecording = false;
  }
  $("#enroll-modal").classList.remove("visible");
}

async function toggleEnrollRecording() {
  if (!state.enrollRecording) {
    const ok = await startRecording();
    if (!ok) return;
    state.enrollRecording = true;
    $("#enroll-mic-btn").classList.add("recording");
    $("#enroll-mic-btn").textContent = "⏹";
    setEnrollStatus("Recording … speak now.", "var(--red)");
  } else {
    const samples = stopRecording();
    state.enrollRecording = false;
    $("#enroll-mic-btn").classList.remove("recording");
    $("#enroll-mic-btn").textContent = "🎤";

    if (samples.length < state.sampleRate * 1) {
      setEnrollStatus("Too short — please record for at least 2 seconds.", "var(--red)");
      return;
    }

    state.enrollSamples.push(encodeWAV(samples, state.sampleRate));
    state.enrollStep++;

    if (state.enrollStep >= ENROLL_PROMPTS.length) {
      // All samples collected — submit
      await submitEnrollment();
    } else {
      setEnrollStatus(`Sample ${state.enrollStep}/${ENROLL_PROMPTS.length} recorded ✓`, "var(--green)");
      updateEnrollUI();
    }
  }
}

async function submitEnrollment() {
  setEnrollStatus("Creating voice profile …", "var(--accent)");
  const form = new FormData();
  state.enrollSamples.forEach((blob, i) => form.append(`sample_${i}`, blob, `sample_${i}.wav`));

  try {
    const res = await fetch(`${API_BASE}/api/enroll`, { method: "POST", body: form });
    const contentType = res.headers.get("content-type");

    let data;
    if (contentType && contentType.includes("application/json")) {
      data = await res.json();
    } else {
      const text = await res.text();
      throw new Error(`Server returned non-JSON response (${res.status}): ${text.slice(0, 100)}`);
    }

    if (res.ok && data.success) {
      state.enrolled = true;
      updateProfileUI();
      updateMicUI();
      log("success", `Voice profile created from ${data.num_samples} samples.`);
      setEnrollStatus("✅ Voice profile created successfully!", "var(--green)");
      setTimeout(closeEnrollModal, 1500);
    } else {
      setEnrollStatus(`❌ ${data.message || data.error || "Enrollment failed"}`, "var(--red)");
      state.enrollSamples = [];
      state.enrollStep = 0;
      updateEnrollUI();
    }
  } catch (err) {
    setEnrollStatus(`❌ Enrollment failed: ${err.message}`, "var(--red)");
  }
}

async function deleteProfile() {
  if (!confirm("Delete your voice profile? You will need to re-enroll.")) return;
  try {
    await fetch(`${API_BASE}/api/delete-profile`, { method: "DELETE" });
    state.enrolled = false;
    updateProfileUI();
    updateMicUI();
    log("warn", "Voice profile deleted.");
  } catch {
    log("error", "Failed to delete profile.");
  }
}

// ══════════════════════════════════════════════════════════════
//  UI updaters
// ══════════════════════════════════════════════════════════════
function updateProfileUI() {
  const ring = $(".profile-ring");
  const label = $(".profile-label");
  const detail = $(".profile-detail");
  const enrollBtn = $("#btn-enroll");
  const deleteBtn = $("#btn-delete-profile");

  if (state.enrolled) {
    ring.className = "profile-ring enrolled";
    $(".profile-icon").textContent = "🔓";
    label.textContent = "Voice Enrolled";
    detail.textContent = "Your unique voice profile is active.";
    enrollBtn.style.display = "none";
    deleteBtn.style.display = "inline-flex";
  } else {
    ring.className = "profile-ring not-enrolled";
    $(".profile-icon").textContent = "🔒";
    label.textContent = "Not Enrolled";
    detail.textContent = "Record your voice to get started.";
    enrollBtn.style.display = "inline-flex";
    deleteBtn.style.display = "none";
  }
}

function updateMicUI() {
  const btn = $("#mic-btn");
  const label = $(".mic-label");
  const wrap = $(".mic-btn-wrap");
  const waveform = $(".waveform-container");

  if (!state.enrolled) {
    btn.disabled = true;
    label.textContent = "Enroll your voice first";
    btn.textContent = "🎤";
    btn.classList.remove("recording");
    wrap.classList.remove("recording");
    waveform.classList.remove("active");
  } else if (state.recording) {
    btn.disabled = false;
    btn.textContent = "⏹";
    btn.classList.add("recording");
    wrap.classList.add("recording");
    waveform.classList.add("active");
    label.textContent = "Recording — click to stop";
  } else {
    btn.disabled = false;
    btn.textContent = "🎤";
    btn.classList.remove("recording");
    wrap.classList.remove("recording");
    waveform.classList.remove("active");
    label.textContent = "Click to speak a command";
  }
}

function updateEnrollUI() {
  // Update dots
  const dots = $$(".enroll-dot");
  dots.forEach((dot, i) => {
    dot.className = "enroll-dot";
    if (i < state.enrollStep) dot.classList.add("done");
    else if (i === state.enrollStep) dot.classList.add("current");
  });

  // Update prompt
  const prompt = $(".enroll-prompt");
  if (state.enrollStep < ENROLL_PROMPTS.length) {
    prompt.textContent = `Say: ${ENROLL_PROMPTS[state.enrollStep]}`;
  } else {
    prompt.textContent = "All samples recorded!";
  }

  $(".enroll-instruction").textContent = `Sample ${state.enrollStep + 1} of ${ENROLL_PROMPTS.length}`;
}

function setEnrollStatus(msg, color) {
  const el = $(".enroll-status-msg");
  el.textContent = msg;
  el.style.color = color || "var(--text-secondary)";
}

// ── Pipeline UI ──────────────────────────────────────────────
function setPipelineState(step, state) {
  if (step === "all") {
    $$(".pipeline-step").forEach((el) => {
      el.className = "pipeline-step";
      el.querySelector(".step-status").textContent = "—";
    });
    return;
  }
  const el = $(`#pipeline-step-${step}`);
  if (el) {
    el.className = `pipeline-step ${state}`;
  }
}

function setPipelineStatus(step, text) {
  const el = $(`#pipeline-step-${step} .step-status`);
  if (el) el.textContent = text;
}

// ── Results UI ───────────────────────────────────────────────
function showResponse(nlp, audioUrl) {
  $(".response-card").classList.add("visible");
  $(".denied-banner").classList.remove("visible");
  $(".response-text").textContent = nlp.response;

  // Meta
  $(".response-meta").innerHTML = `
    <div class="response-meta-item"><span class="label">Intent</span><span class="value">${nlp.intent}</span></div>
    <div class="response-meta-item"><span class="label">Confidence</span><span class="value">${nlp.confidence}</span></div>
  `;

  // Audio
  const audioDiv = $(".response-audio");
  if (audioUrl) {
    const fullAudioUrl = (audioUrl.startsWith("http://") || audioUrl.startsWith("https://"))
      ? audioUrl
      : `${API_BASE}${audioUrl}`;
    audioDiv.innerHTML = `<audio controls autoplay src="${fullAudioUrl}"></audio>`;
  } else {
    audioDiv.innerHTML = "";
  }
}

function showDenied(similarity, threshold) {
  $(".response-card").classList.remove("visible");
  $(".denied-banner").classList.add("visible");
  $(".denied-banner .denied-sim").textContent = `Similarity: ${similarity} / Threshold: ${threshold}`;
}

function showTranscript(text) {
  const el = $(".transcript-text");
  el.textContent = `"${text}"`;
  el.classList.add("visible");
}

function hideResults() {
  $(".response-card").classList.remove("visible");
  $(".denied-banner").classList.remove("visible");
  $(".transcript-text").classList.remove("visible");
}

// ── Spinner ──────────────────────────────────────────────────
function showSpinner(text) {
  $(".spinner-text").textContent = text || "Processing …";
  $(".spinner-overlay").classList.add("visible");
}
function hideSpinner() {
  $(".spinner-overlay").classList.remove("visible");
}

// ── Timer ────────────────────────────────────────────────────
function startTimer() {
  state.timerStart = Date.now();
  const el = $(".mic-timer");
  el.classList.add("visible");
  state.timerInterval = setInterval(() => {
    const elapsed = (Date.now() - state.timerStart) / 1000;
    const m = Math.floor(elapsed / 60).toString().padStart(2, "0");
    const s = Math.floor(elapsed % 60).toString().padStart(2, "0");
    el.textContent = `${m}:${s}`;
  }, 200);
}

function stopTimer() {
  clearInterval(state.timerInterval);
  $(".mic-timer").classList.remove("visible");
}

// ── Activity log ─────────────────────────────────────────────
function log(type, message) {
  const container = $(".log-container");
  const now = new Date();
  const time = now.toLocaleTimeString("en-US", { hour12: false });

  const entry = document.createElement("div");
  entry.className = `log-entry ${type}`;
  entry.innerHTML = `<span class="log-time">${time}</span><span class="log-msg">${message}</span>`;

  container.prepend(entry);

  // Cap at 50 entries
  while (container.children.length > 50) container.lastChild.remove();
}

// ── Helpers ──────────────────────────────────────────────────
function truncate(str, len) {
  return str.length > len ? str.slice(0, len) + "…" : str;
}
