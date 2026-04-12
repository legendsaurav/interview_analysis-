const turnOnBtn = document.getElementById("turnOnBtn");
const statusText = document.getElementById("statusText");
const detailText = document.getElementById("detailText");
const spinner = document.getElementById("spinner");
const studioText = document.getElementById("studioText");
const resultSection = document.getElementById("resultSection");
const resultLink = document.getElementById("resultLink");
const studioUrlInput = document.getElementById("studioUrlInput");
const projectNameInput = document.getElementById("projectNameInput");
const commandsInput = document.getElementById("commandsInput");

const DEFAULT_STUDIO_URL = "https://lightning.ai/2024epb1279/financial-llm-training-project/studios/proud-aquamarine-362uf/code";
const DEFAULT_STUDIO_COMMANDS = "python app.py";
const MESSAGE_TIMEOUT_MS = 20000;

function mapStatusToUi(status, resultUrl, lastError) {
  if (lastError) {
    return {
      label: "Error",
      detail: lastError,
      busy: true,
      showResult: false,
      resultUrl: ""
    };
  }

  switch (status) {
    case "pending":
      return {
        label: "Processing",
        detail: "Waiting for GPU...",
        busy: true,
        showResult: false,
        resultUrl: ""
      };
    case "processing":
      return {
        label: "Processing",
        detail: "Generating interview...",
        busy: true,
        showResult: false,
        resultUrl: ""
      };
    case "done":
      return {
        label: "Completed",
        detail: "Interview Ready",
        busy: false,
        showResult: Boolean(resultUrl),
        resultUrl: resultUrl || ""
      };
    default:
      return {
        label: "Idle",
        detail: "Waiting for action.",
        busy: false,
        showResult: false,
        resultUrl: ""
      };
  }
}

function isStudioBusy(studioStatusMessage) {
  if (!studioStatusMessage) {
    return false;
  }

  const msg = studioStatusMessage.toLowerCase();
  return (
    msg.includes("starting studio automation") ||
    msg.includes("setting up") ||
    msg.includes("waiting") ||
    msg.includes("submitted")
  );
}

function renderState(state) {
  let effectiveStatus = state.status;
  if ((effectiveStatus === "idle" || !effectiveStatus) && isStudioBusy(state.studioStatusMessage)) {
    effectiveStatus = "processing";
  }

  const ui = mapStatusToUi(effectiveStatus, state.resultUrl, state.lastError);

  statusText.textContent = ui.label;
  detailText.textContent = ui.detail;
  spinner.classList.toggle("hidden", !ui.busy);
  resultSection.classList.toggle("hidden", !ui.showResult);

  if (ui.showResult) {
    resultLink.href = ui.resultUrl;
  }

  studioText.textContent = `Studio automation: ${state.studioStatusMessage || "Idle"}`;

  const isBusy = effectiveStatus === "pending" || effectiveStatus === "processing";
  turnOnBtn.disabled = isBusy;
  turnOnBtn.textContent = isBusy ? "RUNNING" : "TURN ON";
}

function withTimeout(promise, timeoutMs, timeoutMessage) {
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs);
    })
  ]);
}

function getUserId(existingUserId) {
  if (existingUserId) {
    return existingUserId;
  }

  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }

  return `user-${Date.now()}-${Math.floor(Math.random() * 10000)}`;
}

async function loadState() {
  const data = await chrome.storage.local.get([
    "userId",
    "jobId",
    "status",
    "resultUrl",
    "lastError",
    "studioStatusMessage",
    "studioUrl",
    "studioProjectName",
    "studioCommands"
  ]);

  studioUrlInput.value = data.studioUrl || DEFAULT_STUDIO_URL;
  projectNameInput.value = data.studioProjectName || "";
  commandsInput.value = data.studioCommands || DEFAULT_STUDIO_COMMANDS;

  renderState({
    status: data.status || "idle",
    resultUrl: data.resultUrl || "",
    lastError: data.lastError || "",
    studioStatusMessage: data.studioStatusMessage || "Idle"
  });
}

turnOnBtn.addEventListener("click", async () => {
  turnOnBtn.disabled = true;
  detailText.textContent = "Starting job...";

  try {
    const data = await chrome.storage.local.get(["userId"]);
    const userId = getUserId(data.userId);
    const studioUrl = studioUrlInput.value.trim() || DEFAULT_STUDIO_URL;
    const studioProjectName = projectNameInput.value.trim();
    const studioCommands = commandsInput.value;
      
    const commands = studioCommands
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    await chrome.storage.local.set({
      userId,
      studioUrl,
      studioProjectName,
      studioCommands,
      status: "processing",
      lastError: "",
      studioStatusMessage: "Starting Studio automation..."
    });

    await loadState();

    const response = await withTimeout(
      chrome.runtime.sendMessage({
        type: "START_JOB",
        userId,
        studioUrl,
        studioProjectName,
        commands
      }),
      MESSAGE_TIMEOUT_MS,
      "Studio automation is still running. Please wait."
    );

    if (!response || !response.ok) {
      throw new Error(response?.error || "Unable to start job.");
    }

    await loadState();
  } catch (_error) {
    const message = _error?.message || "Server not reachable";
    const isTransient = message.includes("still running");

    await chrome.storage.local.set(
      isTransient
        ? {
            status: "processing",
            lastError: "",
            studioStatusMessage: "Starting Studio automation..."
          }
        : {
            status: "error",
            lastError: message
          }
    );
    await loadState();
  } finally {
    const state = await chrome.storage.local.get(["status"]);
    const isBusy = state.status === "pending" || state.status === "processing";
    turnOnBtn.disabled = isBusy;
  }
});

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName !== "local") {
    return;
  }

  if (changes.status || changes.resultUrl || changes.lastError || changes.studioStatusMessage) {
    loadState();
  }
});

loadState();
