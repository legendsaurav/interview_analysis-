const authCard = document.getElementById("authCard");
const interviewSection = document.getElementById("interviewSection");
const guestLoginBtn = document.getElementById("guestLoginBtn");
const interviewForm = document.getElementById("interviewForm");
const interviewText = document.getElementById("interviewText");
const requestSizeInfo = document.getElementById("requestSizeInfo");
const statusText = document.getElementById("statusText");
const audioWrap = document.getElementById("audioWrap");
const resultAudio = document.getElementById("resultAudio");
const generateBtn = document.getElementById("generateBtn");

const VOICERSS_MAX_REQUEST_BYTES = 100 * 1024;
const VOICERSS_API_KEY_LENGTH = 32;

guestLoginBtn.addEventListener("click", () => {
  authCard.classList.add("hidden");
  interviewSection.classList.remove("hidden");
  interviewSection.scrollIntoView({ behavior: "smooth", block: "start" });
  updateRequestSizeInfo(interviewText.value.trim());
});

interviewText.addEventListener("input", () => {
  updateRequestSizeInfo(interviewText.value.trim());
});

interviewForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const text = interviewText.value.trim();
  if (!text) {
    setStatus("Please type a message first.", true);
    return;
  }

  const estimate = estimateVoiceRssRequestSize(text);
  if (estimate.totalBytes > VOICERSS_MAX_REQUEST_BYTES) {
    setStatus(
      "Text is too long for VoiceRSS 100KB request limit. Shorten your message and try again.",
      true
    );
    updateRequestSizeInfo(text);
    return;
  }

  generateBtn.disabled = true;
  setStatus("Generating interview audio...");
  audioWrap.classList.add("hidden");
  resultAudio.removeAttribute("src");

  try {
    const response = await fetch("/api/interview/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "Failed to generate audio.");
    }

    const cacheBusted = `${payload.audioUrl}?t=${Date.now()}`;
    resultAudio.src = cacheBusted;
    resultAudio.load();

    audioWrap.classList.remove("hidden");
    setStatus("Interview audio generated successfully.");
  } catch (error) {
    setStatus(error.message || "Something went wrong.", true);
  } finally {
    generateBtn.disabled = false;
  }
});

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", Boolean(isError));
}

function estimateVoiceRssRequestSize(text) {
  const params = new URLSearchParams({
    key: "x".repeat(VOICERSS_API_KEY_LENGTH),
    hl: "en-us",
    src: text,
    c: "MP3",
    f: "44khz_16bit_stereo",
  });

  const fullQuery = params.toString();
  const totalBytes = new TextEncoder().encode(fullQuery).length;
  return { totalBytes };
}

function formatBytes(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(2)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(3)} MB`;
}

function updateRequestSizeInfo(text) {
  const { totalBytes } = estimateVoiceRssRequestSize(text || "");
  const remaining = VOICERSS_MAX_REQUEST_BYTES - totalBytes;
  const withinLimit = remaining >= 0;

  const used = formatBytes(totalBytes);
  const limit = formatBytes(VOICERSS_MAX_REQUEST_BYTES);
  const remainingText = withinLimit
    ? `${formatBytes(remaining)} remaining`
    : `${formatBytes(Math.abs(remaining))} over limit`;

  requestSizeInfo.textContent = `VoiceRSS request estimate: ${used} / ${limit} (${remainingText}).`;
  requestSizeInfo.classList.toggle("warning", !withinLimit);
  generateBtn.disabled = !withinLimit;
}

updateRequestSizeInfo("");
