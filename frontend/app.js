const wsProtocol = location.protocol === "https:" ? "wss" : "ws";
const runtimeConfig = window.__INTERVIEW_CONFIG__ || {};

function stripTrailingSlash(value) {
  return String(value || "").trim().replace(/\/+$/, "");
}

function toWsUrlBase(httpBase) {
  return String(httpBase || "").replace(/^http:/i, "ws:").replace(/^https:/i, "wss:");
}

const configuredBackendHttp = stripTrailingSlash(runtimeConfig.backendUrl || runtimeConfig.apiBase);
const configuredBackendWs = stripTrailingSlash(runtimeConfig.wsUrl || runtimeConfig.websocketUrl);

function apiUrl(path) {
  const p = String(path || "");
  if (/^https?:\/\//i.test(p)) return p;
  return configuredBackendHttp ? `${configuredBackendHttp}${p}` : p;
}

function deriveBasePath() {
  const path = window.location.pathname || "/";
  if (path === "/") return "";

  const segments = path.split("/").filter(Boolean);
  if (segments.length && segments[segments.length - 1].includes(".")) {
    segments.pop();
  }
  return segments.length ? `/${segments.join("/")}` : "";
}

const basePath = deriveBasePath();
const wsCandidatesRaw = [];
if (configuredBackendWs) {
  wsCandidatesRaw.push(`${configuredBackendWs}/ws/interview`);
}
if (configuredBackendHttp && !configuredBackendWs) {
  wsCandidatesRaw.push(`${toWsUrlBase(configuredBackendHttp)}/ws/interview`);
}
wsCandidatesRaw.push(`${wsProtocol}://${location.host}${basePath}/ws/interview`);
wsCandidatesRaw.push(`${wsProtocol}://${location.host}/ws/interview`);
const wsCandidates = [...new Set(wsCandidatesRaw)];

let ws = null;
let wsConnected = false;
let sessionId = null;
let manualSessionEnded = false;
let pendingStartAfterReconnect = false;
let wsReconnectAttempts = 0;
let wsReconnectTimer = null;
let wsConnecting = false;

const startBtn = document.getElementById("startBtn");
const nextBtn = document.getElementById("nextBtn");
const stopBtn = document.getElementById("stopBtn");

const videoEl = document.getElementById("video");
const frameCanvas = document.getElementById("frameCanvas");
const transcriptEl = document.getElementById("transcript");
const responseEl = document.getElementById("response");
const idealAnswerEl = document.getElementById("idealAnswer");
const alertsEl = document.getElementById("alerts");
const interviewerPresence = document.getElementById("interviewerPresence");
const candidateTile = document.getElementById("candidateTile");
const interviewerTile = document.getElementById("interviewerTile");
const videoGrid = document.getElementById("videoGrid");
const selfViewVideo = document.getElementById("selfViewVideo");
const micBtn = document.getElementById("micBtn");
const camBtn = document.getElementById("camBtn");
const candidateMicIcon = document.getElementById("candidateMicIcon");
const participantCandidateState = document.getElementById("participantCandidateState");
const rightSidebar = document.getElementById("rightSidebar");
const sidebarTitle = document.getElementById("sidebarTitle");
const insightsPanel = document.getElementById("insightsPanel");
const participantsPanel = document.getElementById("participantsPanel");
const chatPanel = document.getElementById("chatPanel");
const toggleInsights = document.getElementById("toggleInsights");
const toggleParticipants = document.getElementById("toggleParticipants");
const toggleChat = document.getElementById("toggleChat");
const toggleSidebar = document.getElementById("toggleSidebar");
const chatStream = document.getElementById("chatStream");
const chatInput = document.getElementById("chatInput");
const chatSend = document.getElementById("chatSend");

const stateText = document.getElementById("stateText");
const speakingState = document.getElementById("speakingState");
const speakingDuration = document.getElementById("speakingDuration");
const vadScore = document.getElementById("vadScore");
const eyeScore = document.getElementById("eyeScore");
const headScore = document.getElementById("headScore");
const visualSpeaking = document.getElementById("visualSpeaking");
const mouthScore = document.getElementById("mouthScore");
const gazeDirection = document.getElementById("gazeDirection");
const objectsText = document.getElementById("objectsText");
const baselineObjectsEl = document.getElementById("baselineObjects");
const pauseCountEl = document.getElementById("pauseCount");
const longSilenceEl = document.getElementById("longSilence");
const timelineEl = document.getElementById("timeline");
const answerQualityEl = document.getElementById("answerQuality");
const confidenceScoreEl = document.getElementById("confidenceScore");
const cheatingRiskEl = document.getElementById("cheatingRisk");
const finalScoreEl = document.getElementById("finalScore");
const clipsCountEl = document.getElementById("clipsCount");
const clipFrameworkEl = document.getElementById("clipFramework");
const wsStatus = document.getElementById("wsStatus");
const sttModeEl = document.getElementById("sttMode");
const modelStatus = document.getElementById("modelStatus");
const geminiApiKeyEl = document.getElementById("geminiApiKey");
const saveGeminiKeyBtn = document.getElementById("saveGeminiKey");
const geminiStatusEl = document.getElementById("geminiStatus");
const analyzeTranscriptFileBtn = document.getElementById("analyzeTranscriptFile");
const analysisSummaryEl = document.getElementById("analysisSummary");
const accuracyChart = document.getElementById("accuracyChart");

let mediaStream = null;
let audioContext = null;
let scriptProcessor = null;
let sourceNode = null;
let isRunning = false;
let videoTimer = null;
let hasInitializedMedia = false;
let localRecognizer = null;
let localRecognizerRunning = false;
let localFinalText = "";
let transcriptSyncTimer = null;
let useBrowserTranscript = true;
let lastQuestionSpoken = "";
let lastQuestionSpokenAt = 0;
let lastAlertsSignature = "";
let lastTimelineSignature = "";
let lastTimelineRenderAt = 0;
let currentTranscriptLogFile = "";
let currentAccuracySeries = [];
let micEnabled = true;
let camEnabled = true;

const ICON_MIC_ON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 15a3.5 3.5 0 0 0 3.5-3.5v-5a3.5 3.5 0 1 0-7 0v5A3.5 3.5 0 0 0 12 15Zm6-3.5a1 1 0 1 0-2 0a4 4 0 1 1-8 0a1 1 0 1 0-2 0a6 6 0 0 0 5 5.91V20H9.5a1 1 0 1 0 0 2h5a1 1 0 1 0 0-2H13v-2.59A6 6 0 0 0 18 11.5Z"/></svg>';
const ICON_MIC_OFF = '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M4.71 3.29a1 1 0 0 0-1.42 1.42L8 9.41v2.09a3.5 3.5 0 0 0 5.55 2.82l1.1 1.1A5.95 5.95 0 0 1 12 16a4 4 0 0 1-4-4a1 1 0 1 0-2 0a5.97 5.97 0 0 0 5 5.91V20H9.5a1 1 0 1 0 0 2h5a1 1 0 1 0 0-2H13v-2.09a5.8 5.8 0 0 0 3.06-1.26l3.23 3.23a1 1 0 1 0 1.42-1.42L4.71 3.29ZM12 4a3.5 3.5 0 0 1 3.5 3.5v3.09l-2-2V7.5a1.5 1.5 0 0 0-3 0v.09l-2-2V7.5A3.5 3.5 0 0 1 12 4Z"/></svg>';
const ICON_CAM_ON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M14 8.5A2.5 2.5 0 0 0 11.5 6h-7A2.5 2.5 0 0 0 2 8.5v7A2.5 2.5 0 0 0 4.5 18h7a2.5 2.5 0 0 0 2.5-2.5v-1.43l4.2 2.75A1.2 1.2 0 0 0 20 15.81V8.19a1.2 1.2 0 0 0-1.8-1.01L14 9.93V8.5Z"/></svg>';
const ICON_CAM_OFF = '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M3.2 2.9a1 1 0 1 0-1.4 1.4l2.2 2.2v9a2.5 2.5 0 0 0 2.5 2.5h8.9l2.4 2.4a1 1 0 0 0 1.4-1.4L3.2 2.9Zm10.8 9.6l4.2 2.8a1.2 1.2 0 0 0 1.8-1V8.2a1.2 1.2 0 0 0-1.8-1l-2.5 1.6L14 7.1v5.4ZM7.5 6h4a2.5 2.5 0 0 1 2.2 1.3l-1.5.9L7.5 3.5V6Z"/></svg>';

function sendJson(payload) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
  }
}

function setAlerts(messages) {
  const unique = [...new Set(messages)].slice(0, 5);
  const signature = unique.join("||");
  if (signature === lastAlertsSignature) {
    return;
  }
  lastAlertsSignature = signature;

  alertsEl.innerHTML = "";
  unique.forEach((msg) => {
    const li = document.createElement("li");
    li.textContent = msg;
    alertsEl.appendChild(li);
  });
}

function renderTimeline(items) {
  if (!timelineEl) return;
  const rows = (items || []).slice(-12);
  const signature = rows.map((entry) => `${entry.timestamp}|${entry.event}|${entry.confidence}`).join("||");
  const now = Date.now();
  if (signature === lastTimelineSignature) {
    return;
  }
  if (now - lastTimelineRenderAt < 220) {
    return;
  }
  lastTimelineSignature = signature;
  lastTimelineRenderAt = now;

  if (!rows.length) {
    timelineEl.textContent = "Waiting for events...";
    return;
  }
  timelineEl.innerHTML = rows
    .map((entry) => {
      const ts = Number(entry.timestamp || 0).toFixed(1);
      const ev = String(entry.event || "unknown");
      const conf = Number(entry.confidence || 0).toFixed(2);
      return `<div class="timeline-row">t=${ts}s | ${ev} | conf=${conf}</div>`;
    })
    .join("");
}

function renderFinalReport(report) {
  if (!report || !report.scores) return;
  answerQualityEl.textContent = Number(report.scores.answer_quality || 0).toFixed(1);
  confidenceScoreEl.textContent = Number(report.scores.confidence || 0).toFixed(1);
  cheatingRiskEl.textContent = Number(report.scores.cheating_risk || 0).toFixed(1);
  finalScoreEl.textContent = Number(report.scores.final_score || 0).toFixed(1);
  clipsCountEl.textContent = String((report.clips || []).length);

  if (clipFrameworkEl && report.clip_analysis_framework) {
    const model = report.clip_analysis_framework.recommended_model || "YOLOv8m/YOLOv8l";
    const size = report.clip_analysis_framework.recommended_imgsz || 640;
    clipFrameworkEl.textContent = `Post-processing: ${model} at imgsz ${size} for refined suspicious clip verification.`;
  }
  renderTimeline(report.timeline || []);
  if (report.ai_accuracy_series) {
    currentAccuracySeries = report.ai_accuracy_series;
    renderAccuracyGraph(currentAccuracySeries);
  }
}

function renderAccuracyGraph(points) {
  if (!accuracyChart) return;
  const ctx = accuracyChart.getContext("2d");
  if (!ctx) return;

  const w = accuracyChart.width;
  const h = accuracyChart.height;
  const pad = { left: 32, right: 16, top: 14, bottom: 28 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(10, 12, 20, 0.65)";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(255,255,255,0.14)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, h - pad.bottom);
  ctx.lineTo(w - pad.right, h - pad.bottom);
  ctx.stroke();

  for (let y = 0; y <= 100; y += 25) {
    const py = pad.top + plotH - (y / 100) * plotH;
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.beginPath();
    ctx.moveTo(pad.left, py);
    ctx.lineTo(w - pad.right, py);
    ctx.stroke();
    ctx.fillStyle = "rgba(220,230,255,0.8)";
    ctx.font = "10px JetBrains Mono";
    ctx.fillText(String(y), 6, py + 3);
  }

  const safe = Array.isArray(points) ? points : [];
  if (!safe.length) {
    ctx.fillStyle = "rgba(230,240,255,0.72)";
    ctx.font = "12px Sora";
    ctx.fillText("No answer accuracy points yet", pad.left + 8, pad.top + 22);
    return;
  }

  const step = safe.length > 1 ? plotW / (safe.length - 1) : plotW / 2;
  ctx.strokeStyle = "#36d597";
  ctx.lineWidth = 2;
  ctx.beginPath();

  safe.forEach((p, i) => {
    const yVal = Math.max(0, Math.min(100, Number(p.y || 0)));
    const x = pad.left + (safe.length > 1 ? i * step : plotW / 2);
    const y = pad.top + plotH - (yVal / 100) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  safe.forEach((p, i) => {
    const yVal = Math.max(0, Math.min(100, Number(p.y || 0)));
    const x = pad.left + (safe.length > 1 ? i * step : plotW / 2);
    const y = pad.top + plotH - (yVal / 100) * plotH;
    ctx.fillStyle = "#2ab3ff";
    ctx.beginPath();
    ctx.arc(x, y, 3.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(235,245,255,0.9)";
    ctx.font = "10px JetBrains Mono";
    ctx.fillText(String(Math.round(yVal)), x - 8, y - 8);
  });
}

async function saveGeminiApiKey() {
  const key = (geminiApiKeyEl?.value || "").trim();
  if (!key) {
    if (geminiStatusEl) geminiStatusEl.textContent = "Enter API key first.";
    return;
  }
  try {
    const res = await fetch(apiUrl("/ai/gemini/config"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: key }),
    });
    const data = await res.json();
    if (geminiStatusEl) {
      geminiStatusEl.textContent = data.configured
        ? `Gemini configured (${data.model || "model"})`
        : "Gemini key rejected";
    }
    if (geminiApiKeyEl) {
      geminiApiKeyEl.value = "";
    }
  } catch (_) {
    if (geminiStatusEl) geminiStatusEl.textContent = "Failed to configure Gemini key.";
  }
}

async function analyzeTranscriptFile() {
  if (!currentTranscriptLogFile) {
    if (analysisSummaryEl) analysisSummaryEl.textContent = "No transcript log file path from backend yet.";
    return;
  }
  try {
    const res = await fetch(apiUrl("/analysis/transcript-file"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript_path: currentTranscriptLogFile, session_id: sessionId }),
    });
    const data = await res.json();
    if (!data.ok) {
      if (analysisSummaryEl) analysisSummaryEl.textContent = "No analyzable transcript lines found.";
      return;
    }
    currentAccuracySeries = data.accuracy_series || [];
    renderAccuracyGraph(currentAccuracySeries);
    const avg = currentAccuracySeries.length
      ? (currentAccuracySeries.reduce((a, b) => a + Number(b.y || 0), 0) / currentAccuracySeries.length).toFixed(1)
      : "0.0";
    if (analysisSummaryEl) {
      analysisSummaryEl.textContent = `Analyzed ${currentAccuracySeries.length} answers. Average accuracy: ${avg}%`;
    }
  } catch (_) {
    if (analysisSummaryEl) analysisSummaryEl.textContent = "Transcript analysis request failed.";
  }
}

function setInterviewerPresence(text) {
  if (interviewerPresence && text) {
    interviewerPresence.textContent = text;
  }
}

function showSidebarPanel(name) {
  if (!rightSidebar) return;
  insightsPanel.classList.remove("active");
  participantsPanel.classList.remove("active");
  chatPanel.classList.remove("active");

  if (name === "participants") {
    participantsPanel.classList.add("active");
    sidebarTitle.textContent = "People";
  } else if (name === "chat") {
    chatPanel.classList.add("active");
    sidebarTitle.textContent = "Chat";
  } else {
    insightsPanel.classList.add("active");
    sidebarTitle.textContent = "Insights";
  }
  rightSidebar.classList.remove("hidden");
}

function toggleSidebarVisibility() {
  if (!rightSidebar) return;
  rightSidebar.classList.toggle("hidden");
}

function appendChat(author, text) {
  if (!chatStream || !text) return;
  const row = document.createElement("div");
  row.className = "chat-row";
  row.textContent = `${author}: ${text}`;
  chatStream.appendChild(row);
  chatStream.scrollTop = chatStream.scrollHeight;
}

function applyMicState(enabled) {
  micEnabled = enabled;
  if (mediaStream) {
    mediaStream.getAudioTracks().forEach((track) => {
      track.enabled = enabled;
    });
  }
  if (micBtn) {
    micBtn.classList.toggle("off", !enabled);
    const iconEl = document.getElementById("micBtnIcon");
    const labelEl = document.getElementById("micBtnLabel");
    if (iconEl) iconEl.innerHTML = enabled ? ICON_MIC_ON : ICON_MIC_OFF;
    if (labelEl) labelEl.textContent = enabled ? "Mic" : "Muted";
  }
  if (candidateMicIcon) {
    candidateMicIcon.innerHTML = `${enabled ? ICON_MIC_ON : ICON_MIC_OFF}<span id="candidateMicLabel">${enabled ? "On" : "Muted"}</span>`;
    candidateMicIcon.classList.toggle("muted", !enabled);
  }
  if (participantCandidateState) {
    participantCandidateState.textContent = enabled ? "mic on" : "mic muted";
  }
}

function applyCamState(enabled) {
  camEnabled = enabled;
  if (mediaStream) {
    mediaStream.getVideoTracks().forEach((track) => {
      track.enabled = enabled;
    });
  }
  if (camBtn) {
    camBtn.classList.toggle("off", !enabled);
    const iconEl = document.getElementById("camBtnIcon");
    const labelEl = document.getElementById("camBtnLabel");
    if (iconEl) iconEl.innerHTML = enabled ? ICON_CAM_ON : ICON_CAM_OFF;
    if (labelEl) labelEl.textContent = enabled ? "Cam" : "Off";
  }
}

function setupFocusMode() {
  if (!videoGrid) return;
  [candidateTile, interviewerTile].forEach((tile) => {
    if (!tile) return;
    tile.addEventListener("click", () => {
      const isPinned = tile.classList.contains("pinned");
      [candidateTile, interviewerTile].forEach((x) => x && x.classList.remove("pinned"));
      if (isPinned) {
        videoGrid.classList.remove("pinned");
      } else {
        tile.classList.add("pinned");
        videoGrid.classList.add("pinned");
      }
    });
  });
}

function updateSpeakingVisuals(speaking) {
  if (candidateTile) {
    candidateTile.classList.toggle("speaking", Boolean(speaking));
  }
  if (interviewerTile) {
    interviewerTile.classList.toggle("speaking", !Boolean(speaking) && isRunning);
  }
}

function speakText(text) {
  if (!window.speechSynthesis || !text) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.03;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

function speakQuestionOnce(questionText) {
  const q = (questionText || "").trim();
  if (!q) return;
  const now = Date.now();
  if (q === lastQuestionSpoken && (now - lastQuestionSpokenAt) < 3000) {
    return;
  }
  lastQuestionSpoken = q;
  lastQuestionSpokenAt = now;
  speakText(q);
}

function downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
  if (outputSampleRate === inputSampleRate) {
    return buffer;
  }
  const sampleRateRatio = inputSampleRate / outputSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i += 1) {
      accum += buffer[i];
      count += 1;
    }
    result[offsetResult] = accum / count;
    offsetResult += 1;
    offsetBuffer = nextOffsetBuffer;
  }

  return result;
}

function floatTo16BitPCM(float32Array) {
  const output = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i += 1) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output;
}

function int16ToBase64(int16Array) {
  const bytes = new Uint8Array(int16Array.buffer);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

async function setupMedia() {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
    video: {
      facingMode: "user",
      width: { ideal: 1280, max: 1920 },
      height: { ideal: 720, max: 1080 },
      frameRate: { ideal: 24, max: 30 },
    },
  });

  videoEl.srcObject = mediaStream;
  if (selfViewVideo) {
    selfViewVideo.srcObject = mediaStream;
  }

  audioContext = new AudioContext();
  if (audioContext.state !== "running") {
    try {
      await audioContext.resume();
    } catch (_) {
      // Will retry resume when user clicks Start.
    }
  }
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

  scriptProcessor.onaudioprocess = (event) => {
    if (!isRunning) return;

    const input = event.inputBuffer.getChannelData(0);
    const downsampled = downsampleBuffer(input, audioContext.sampleRate, 16000);
    const pcm16 = floatTo16BitPCM(downsampled);
    const b64 = int16ToBase64(pcm16);

    sendJson({
      type: "audio",
      pcm16: b64,
      sample_rate: 16000,
      channels: 1,
    });
  };

  sourceNode.connect(scriptProcessor);
  scriptProcessor.connect(audioContext.destination);

  const ctx = frameCanvas.getContext("2d", { willReadFrequently: false });
  videoTimer = window.setInterval(() => {
    if (!isRunning || ws.readyState !== WebSocket.OPEN || videoEl.readyState < 2) return;

    ctx.drawImage(videoEl, 0, 0, frameCanvas.width, frameCanvas.height);
    const dataUrl = frameCanvas.toDataURL("image/jpeg", 0.85);
    const b64 = dataUrl.split(",")[1];

    sendJson({
      type: "video",
      jpeg: b64,
      ts: Date.now(),
    });
  }, 90);
}

function cleanupMedia() {
  if (videoTimer) {
    clearInterval(videoTimer);
    videoTimer = null;
  }

  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor.onaudioprocess = null;
    scriptProcessor = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  videoEl.srcObject = null;
  if (selfViewVideo) {
    selfViewVideo.srcObject = null;
  }
  stopLocalRecognizer();
  if (transcriptSyncTimer) {
    clearInterval(transcriptSyncTimer);
    transcriptSyncTimer = null;
  }
  hasInitializedMedia = false;
}

function initLocalRecognizer() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return;

  localRecognizer = new SR();
  localRecognizer.lang = "en-US";
  localRecognizer.continuous = true;
  localRecognizer.interimResults = true;

  localRecognizer.onresult = (event) => {
    if (!useBrowserTranscript) return;
    let interim = "";
    let finalFromEvent = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const segment = event.results[i][0].transcript || "";
      if (event.results[i].isFinal) {
        localFinalText = `${localFinalText} ${segment}`.trim();
        finalFromEvent = `${finalFromEvent} ${segment}`.trim();
      } else {
        interim += segment;
      }
    }

    const liveText = `${localFinalText} ${interim}`.trim();
    if (liveText) {
      transcriptEl.textContent = liveText;
    }

    if (interim.trim()) {
      sendJson({ type: "client_transcript", text: interim.trim(), final: false });
    }
    if (finalFromEvent.trim()) {
      sendJson({ type: "client_transcript", text: finalFromEvent.trim(), final: true });
    }
  };

  localRecognizer.onerror = () => {
    localRecognizerRunning = false;
  };

  localRecognizer.onend = () => {
    if (!isRunning) {
      localRecognizerRunning = false;
      return;
    }
    try {
      localRecognizer.start();
      localRecognizerRunning = true;
    } catch (_) {
      localRecognizerRunning = false;
    }
  };
}

function startLocalRecognizer() {
  if (!useBrowserTranscript) return;
  if (!localRecognizer) {
    initLocalRecognizer();
  }
  if (!localRecognizer || localRecognizerRunning) return;

  try {
    localRecognizer.start();
    localRecognizerRunning = true;
  } catch (_) {
    localRecognizerRunning = false;
  }
}

function stopLocalRecognizer() {
  if (!localRecognizer) return;
  try {
    localRecognizer.stop();
  } catch (_) {
    // no-op
  }
  localRecognizerRunning = false;
}

function startTranscriptSync() {
  if (transcriptSyncTimer) {
    clearInterval(transcriptSyncTimer);
  }
  transcriptSyncTimer = window.setInterval(() => {
    if (!isRunning) return;
    if (!useBrowserTranscript) return;
    const liveText = (transcriptEl.textContent || "").trim();
    if (!liveText || liveText === "Waiting for speech...") return;

    // Primary sync path: websocket.
    if (wsConnected) {
      sendJson({ type: "client_transcript_sync", text: liveText });
    }

    // Fallback sync path: HTTP, keeps backend transcript visible even if WS path is flaky.
    fetch(apiUrl("/debug/ingest/transcript"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: liveText, final: false, source: "frontend_sync_timer", session_id: sessionId }),
      keepalive: true,
    }).catch(() => {
      // Silent failure; websocket path may still be active.
    });
  }, 1200);
}

async function startInterviewFlow() {
  if (isRunning) return;
  if (!wsConnected) {
    if (manualSessionEnded) {
      pendingStartAfterReconnect = true;
      manualSessionEnded = false;
      connectSocket();
      setAlerts(["Reconnecting... starting interview when ready."]);
      return;
    }
    pendingStartAfterReconnect = true;
    connectSocket();
    setAlerts(["Waiting for server websocket..."]);
    return;
  }

  try {
    if (!hasInitializedMedia) {
      await setupMedia();
      hasInitializedMedia = true;
    }

    if (audioContext && audioContext.state !== "running") {
      await audioContext.resume();
    }
    if (audioContext && audioContext.state !== "running") {
      throw new Error("Microphone is blocked. Click page and allow mic permission.");
    }

    startLocalRecognizer();
    isRunning = true;
    setInterviewerPresence("Listening to your answer");
    startTranscriptSync();
    sendJson({ type: "control", action: "start" });
  } catch (error) {
    const msg = error && error.message ? error.message : "Camera/microphone permission denied";
    setAlerts([`Media error: ${msg}`]);
  }
}

function attachSocketHandlers(socket, candidateIndex) {
  socket.onopen = () => {
    wsConnecting = false;
    wsReconnectAttempts = 0;
    if (wsReconnectTimer) {
      clearTimeout(wsReconnectTimer);
      wsReconnectTimer = null;
    }
    wsConnected = true;
    wsStatus.textContent = "connected";
    if (!isRunning) {
      setAlerts([]);
    }
    // Prepare media early so camera preview can appear before interview starts.
    if (!hasInitializedMedia) {
      setupMedia()
        .then(() => {
          hasInitializedMedia = true;
        })
        .catch(() => {
          setAlerts(["Allow camera/microphone permissions, then click Start Interview."]);
        });
    }

    if (pendingStartAfterReconnect) {
      pendingStartAfterReconnect = false;
      startInterviewFlow();
    }
  };

  socket.onerror = () => {
    wsStatus.textContent = "connecting";
  };

  socket.onmessage = (event) => {
    let payload = null;
    try {
      payload = JSON.parse(event.data);
    } catch (_) {
      return;
    }

    if (payload.type === "state") {
      stateText.textContent = payload.state || "IDLE";
      const questionText = payload.question || payload.prompt;
      if (idealAnswerEl) {
        idealAnswerEl.textContent = payload.ideal_answer || "Ideal answer will appear here.";
      }
      if (questionText) {
        responseEl.textContent = questionText;
        setInterviewerPresence("Asking the next question");
        speakQuestionOnce(questionText);
        window.setTimeout(() => setInterviewerPresence("Listening to your answer"), 1400);
      } else if ((payload.state || "").toUpperCase() === "IDLE") {
        setInterviewerPresence("Ready to start interview");
      }
      return;
    }

    if (payload.type === "audio_metrics") {
      speakingState.textContent = payload.speaking_state;
      speakingDuration.textContent = `${Number(payload.speaking_duration || 0).toFixed(1)}s`;
      vadScore.textContent = Number(payload.vad_confidence || 0).toFixed(2);
      pauseCountEl.textContent = String(payload.pause_count || 0);
      longSilenceEl.textContent = payload.long_silence ? "yes" : "no";
      updateSpeakingVisuals(payload.speaking_state === "speaking");
      return;
    }

    if (payload.type === "transcript") {
      const incoming = payload.text || "";
      // Prefer backend transcript (Whisper when available) as source of truth.
      if (incoming) {
        transcriptEl.textContent = incoming;
      }
      return;
    }

    if (payload.type === "vision_metrics") {
      eyeScore.textContent = Number(payload.eye_contact_score || 0).toFixed(2);
      headScore.textContent = Number(payload.head_movement_score || 0).toFixed(2);
      visualSpeaking.textContent = payload.visually_speaking ? "yes" : "no";
      mouthScore.textContent = Number(payload.mouth_movement_score || 0).toFixed(2);
      gazeDirection.textContent = payload.gaze_direction || "unknown";

      const objects = payload.detected_objects || [];
      objectsText.textContent = objects.length ? objects.join(", ") : "none";
      const baseline = payload.baseline_objects || [];
      baselineObjectsEl.textContent = baseline.length ? baseline.join(", ") : "none";

      pauseCountEl.textContent = String(payload.pause_count || 0);
      longSilenceEl.textContent = payload.long_silence ? "yes" : "no";
      if (payload.alerts && payload.alerts.length) {
        setAlerts(payload.alerts);
      }
      return;
    }

    if (payload.type === "timeline_update") {
      if (payload.timeline) {
        renderTimeline(payload.timeline);
      }
      return;
    }

    if (payload.type === "feedback") {
      setAlerts(payload.messages || []);
      return;
    }

    if (payload.type === "response") {
      // Keep backend responses internal; only question text is shown in UI.
      return;
    }

    if (payload.type === "system") {
      if (payload.session_id != null) {
        sessionId = payload.session_id;
      }
      if (payload.transcript_log_file) {
        currentTranscriptLogFile = payload.transcript_log_file;
      }
      if (payload.ended === true) {
        manualSessionEnded = true;
      }
      if (sttModeEl && payload.transcript_mode) {
        sttModeEl.textContent = payload.transcript_mode;
      }
      if (typeof payload.stt_available === "boolean") {
        useBrowserTranscript = !payload.stt_available;
        if (!useBrowserTranscript) {
          stopLocalRecognizer();
        } else if (isRunning) {
          startLocalRecognizer();
        }
      }
      if (typeof payload.model_available === "boolean") {
        modelStatus.textContent = payload.model_available ? "loaded" : "not loaded";
      }
      if (typeof payload.gemini_configured === "boolean" && geminiStatusEl) {
        geminiStatusEl.textContent = payload.gemini_configured
          ? "Gemini configured"
          : "Gemini not configured";
      }
      if (typeof payload.vad_available === "boolean" && !payload.vad_available) {
        setAlerts(["WebRTC VAD package not available, using RMS fallback VAD."]);
      }
      return;
    }

    if (payload.type === "qa_analysis_update") {
      if (payload.series) {
        currentAccuracySeries = payload.series;
        renderAccuracyGraph(currentAccuracySeries);
      }
      if (analysisSummaryEl && payload.latest) {
        const score = Number(payload.latest.accuracy || 0).toFixed(1);
        const idx = payload.latest.index || currentAccuracySeries.length;
        analysisSummaryEl.textContent = `Latest evaluation: Q${idx} accuracy ${score}%`;
      }
      return;
    }

    if (payload.type === "final_report") {
      renderFinalReport(payload.report || {});
      return;
    }
  };

  socket.onclose = () => {
    wsConnecting = false;
    wsConnected = false;
    wsStatus.textContent = manualSessionEnded ? "ended" : "disconnected";
    if (manualSessionEnded) {
      isRunning = false;
      cleanupMedia();
      setInterviewerPresence("Interview ended");
      setAlerts(["Interview ended. Click Start to begin a new session."]);
      return;
    }
    if (candidateIndex + 1 < wsCandidates.length) {
      connectSocket(candidateIndex + 1);
    } else {
      const delayMs = Math.min(20000, 1200 * Math.pow(1.8, Math.min(wsReconnectAttempts, 8)));
      wsReconnectAttempts += 1;
      wsStatus.textContent = "reconnecting";
      setAlerts([
        `WebSocket disconnected. Retrying in ${Math.max(1, Math.round(delayMs / 1000))}s...`,
      ]);
      if (wsReconnectTimer) {
        clearTimeout(wsReconnectTimer);
      }
      wsReconnectTimer = window.setTimeout(() => {
        wsReconnectTimer = null;
        if (!manualSessionEnded && !wsConnected) {
          connectSocket(0);
        }
      }, delayMs);
    }
  };
}

function connectSocket(candidateIndex = 0) {
  if (wsConnected || wsConnecting) return;
  wsStatus.textContent = "connecting";
  wsConnecting = true;
  try {
    ws = new WebSocket(wsCandidates[candidateIndex]);
    attachSocketHandlers(ws, candidateIndex);
  } catch (_) {
    wsConnecting = false;
    wsStatus.textContent = "disconnected";
  }
}

async function refreshGeminiStatus() {
  try {
    const res = await fetch(apiUrl("/ai/gemini/status"));
    const data = await res.json();
    if (geminiStatusEl) {
      geminiStatusEl.textContent = data.configured
        ? `Gemini configured (${data.model || "model"})`
        : "Gemini not configured";
    }
  } catch (_) {
    if (geminiStatusEl) geminiStatusEl.textContent = "Gemini status unavailable";
  }
}

connectSocket();
setupFocusMode();
renderAccuracyGraph([]);
refreshGeminiStatus();

if (toggleInsights) {
  toggleInsights.addEventListener("click", () => showSidebarPanel("insights"));
}
if (toggleParticipants) {
  toggleParticipants.addEventListener("click", () => showSidebarPanel("participants"));
}
if (toggleChat) {
  toggleChat.addEventListener("click", () => showSidebarPanel("chat"));
}
if (toggleSidebar) {
  toggleSidebar.addEventListener("click", () => toggleSidebarVisibility());
}

if (micBtn) {
  micBtn.addEventListener("click", () => applyMicState(!micEnabled));
}
if (camBtn) {
  camBtn.addEventListener("click", () => applyCamState(!camEnabled));
}

if (chatSend) {
  chatSend.addEventListener("click", () => {
    const msg = (chatInput?.value || "").trim();
    if (!msg) return;
    appendChat("You", msg);
    if (chatInput) chatInput.value = "";
    window.setTimeout(() => appendChat("AI", "Message received."), 250);
  });
}

if (chatInput) {
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      chatSend?.click();
    }
  });
}

if (saveGeminiKeyBtn) {
  saveGeminiKeyBtn.addEventListener("click", () => {
    saveGeminiApiKey();
  });
}

if (analyzeTranscriptFileBtn) {
  analyzeTranscriptFileBtn.addEventListener("click", () => {
    analyzeTranscriptFile();
  });
}

startBtn.addEventListener("click", async () => {
  localFinalText = "";
  transcriptEl.textContent = "Waiting for speech...";
  lastQuestionSpoken = "";
  lastQuestionSpokenAt = 0;
  updateSpeakingVisuals(false);
  renderTimeline([]);
  applyMicState(true);
  applyCamState(true);
  if (manualSessionEnded && !wsConnected) {
    pendingStartAfterReconnect = true;
    manualSessionEnded = false;
    connectSocket();
    setAlerts(["Starting new session..."]);
    return;
  }
  await startInterviewFlow();
});

nextBtn.addEventListener("click", () => {
  sendJson({ type: "control", action: "next" });
});

stopBtn.addEventListener("click", () => {
  manualSessionEnded = true;
  pendingStartAfterReconnect = false;
  if (wsConnected) {
    sendJson({ type: "control", action: "stop" });
  }
  isRunning = false;
  cleanupMedia();
  localFinalText = "";
  speakingState.textContent = "paused";
  speakingDuration.textContent = "0.0s";
  vadScore.textContent = "0.00";
  longSilenceEl.textContent = "no";
  applyMicState(true);
  applyCamState(true);
  setInterviewerPresence("Interview paused");
  updateSpeakingVisuals(false);
});
