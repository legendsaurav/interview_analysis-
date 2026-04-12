const API_BASE = "http://localhost:3000";
const POLL_INTERVAL_MS = 7000;
const DEFAULT_STUDIO_URL = "https://lightning.ai/2024epb1279/financial-llm-training-project/studios/proud-aquamarine-362uf/code";
const AUTOMATION_TIMEOUT_MS = 180000;

let pollIntervalId = null;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function attachDebugger(tabId) {
  return new Promise((resolve, reject) => {
    chrome.debugger.attach({ tabId }, "1.3", () => {
      const err = chrome.runtime.lastError;
      if (err) {
        reject(new Error(err.message));
        return;
      }
      resolve();
    });
  });
}

function detachDebugger(tabId) {
  return new Promise((resolve) => {
    chrome.debugger.detach({ tabId }, () => {
      resolve();
    });
  });
}

function sendDebuggerCommand(tabId, method, params = {}) {
  return new Promise((resolve, reject) => {
    chrome.debugger.sendCommand({ tabId }, method, params, (result) => {
      const err = chrome.runtime.lastError;
      if (err) {
        reject(new Error(err.message));
        return;
      }
      resolve(result);
    });
  });
}

async function inspectStudioFocus(tabId) {
  const injected = await chrome.scripting.executeScript({
    target: { tabId },
    func: () => {
      const active = document.activeElement;
      const activeTag = active?.tagName || "";

      const editorFocused = Boolean(
        document.querySelector(".monaco-editor.focused") ||
        (active && active.closest && active.closest(".monaco-editor"))
      );

      const terminalFocused = Boolean(
        (active && active.matches && active.matches("textarea.xterm-helper-textarea")) ||
        (active && active.closest && active.closest(".xterm, [class*='terminal' i], [data-testid*='terminal' i]"))
      );

      return {
        editorFocused,
        terminalFocused,
        activeTag
      };
    }
  });

  return injected[0]?.result || { editorFocused: false, terminalFocused: false, activeTag: "" };
}

async function locateTerminalCoordinates(tabId) {
  const injected = await chrome.scripting.executeScript({
    target: { tabId },
    func: async () => {
      const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

      function openTerminalPanel() {
        const candidates = Array.from(document.querySelectorAll("button, [role='tab'], [role='button'], .monaco-text-button, .action-label"));
        const terminalTab = candidates.find((el) => {
          const text = (el.textContent || "").trim().toLowerCase();
          const aria = (el.getAttribute("aria-label") || "").trim().toLowerCase();
          return text === "terminal" || aria.includes("terminal");
        });

        if (terminalTab) {
          terminalTab.click();
          return true;
        }

        return false;
      }

      function centerOf(el) {
        const rect = el.getBoundingClientRect();
        return {
          x: Math.floor(rect.left + rect.width / 2),
          y: Math.floor(rect.top + rect.height / 2)
        };
      }

      function clampPoint(point) {
        return {
          x: Math.max(2, Math.min(window.innerWidth - 2, point.x)),
          y: Math.max(2, Math.min(window.innerHeight - 2, point.y))
        };
      }

      function findTerminalTab() {
        const candidates = Array.from(document.querySelectorAll("button, [role='tab'], [role='button'], .monaco-text-button, .action-label"));
        return candidates.find((el) => {
          const text = (el.textContent || "").trim().toLowerCase();
          const aria = (el.getAttribute("aria-label") || "").trim().toLowerCase();
          return text === "terminal" || aria === "terminal" || aria.includes("terminal");
        }) || null;
      }

      function findTarget() {
        const strictSelectors = [
          "textarea.xterm-helper-textarea",
          ".xterm-screen",
          ".xterm"
        ];

        for (const selector of strictSelectors) {
          const node = document.querySelector(selector);
          if (node) {
            return node;
          }
        }

        // Fallback to area below TERMINAL tab in VS Code-like layout.
        const terminalTab = findTerminalTab();
        if (terminalTab) {
          const tabRect = terminalTab.getBoundingClientRect();
          return {
            __virtualTarget: true,
            point: clampPoint({
              x: Math.floor(tabRect.left + Math.max(80, tabRect.width / 2)),
              y: Math.floor(tabRect.bottom + 70)
            })
          };
        }

        return null;
      }

      for (let i = 0; i < 35; i += 1) {
        if (i % 2 === 0) {
          openTerminalPanel();
        }

        const target = findTarget();
        if (target) {
          if (target.__virtualTarget) {
            const terminalTab = findTerminalTab();
            return {
              ok: true,
              mode: "panel-fallback",
              tabPoint: terminalTab ? clampPoint(centerOf(terminalTab)) : null,
              inputPoint: target.point
            };
          }

          const terminalTab = findTerminalTab();
          return {
            ok: true,
            mode: "xterm-target",
            tabPoint: terminalTab ? clampPoint(centerOf(terminalTab)) : null,
            inputPoint: clampPoint(centerOf(target))
          };
        }

        await wait(1200);
      }

      return { ok: false };
    }
  });

  return injected[0]?.result || { ok: false };
}

async function sendCommandsWithDebugger(tabId, commands) {
  const loc = await locateTerminalCoordinates(tabId);
  if (!loc.ok) {
    return { ok: false, error: "Debugger fallback: terminal not located" };
  }

  await attachDebugger(tabId);

  try {
    async function clickPoint(x, y, clickCount = 1) {
      await sendDebuggerCommand(tabId, "Input.dispatchMouseEvent", {
        type: "mouseMoved",
        x,
        y,
        button: "left"
      });
      await sendDebuggerCommand(tabId, "Input.dispatchMouseEvent", {
        type: "mousePressed",
        x,
        y,
        button: "left",
        clickCount
      });
      await sendDebuggerCommand(tabId, "Input.dispatchMouseEvent", {
        type: "mouseReleased",
        x,
        y,
        button: "left",
        clickCount
      });
    }

    async function pressCtrlBackquote() {
      await sendDebuggerCommand(tabId, "Input.dispatchKeyEvent", {
        type: "keyDown",
        windowsVirtualKeyCode: 17,
        code: "ControlLeft",
        key: "Control"
      });

      await sendDebuggerCommand(tabId, "Input.dispatchKeyEvent", {
        type: "keyDown",
        windowsVirtualKeyCode: 192,
        code: "Backquote",
        key: "`",
        modifiers: 2
      });

      await sendDebuggerCommand(tabId, "Input.dispatchKeyEvent", {
        type: "keyUp",
        windowsVirtualKeyCode: 192,
        code: "Backquote",
        key: "`",
        modifiers: 2
      });

      await sendDebuggerCommand(tabId, "Input.dispatchKeyEvent", {
        type: "keyUp",
        windowsVirtualKeyCode: 17,
        code: "ControlLeft",
        key: "Control"
      });
    }

    async function focusTerminalViaCommandPalette() {
      // Ctrl+Shift+P
      await sendDebuggerCommand(tabId, "Input.dispatchKeyEvent", {
        type: "keyDown",
        windowsVirtualKeyCode: 17,
        code: "ControlLeft",
        key: "Control"
      });
      await sendDebuggerCommand(tabId, "Input.dispatchKeyEvent", {
        type: "keyDown",
        windowsVirtualKeyCode: 16,
        code: "ShiftLeft",
        key: "Shift"
      });
      await sendDebuggerCommand(tabId, "Input.dispatchKeyEvent", {
        type: "keyDown",
        windowsVirtualKeyCode: 80,
        code: "KeyP",
        key: "P",
        modifiers: 10
      });
      await sendDebuggerCommand(tabId, "Input.dispatchKeyEvent", {
        type: "keyUp",
        windowsVirtualKeyCode: 80,
        code: "KeyP",
        key: "P",
        modifiers: 10
      });
      await sendDebuggerCommand(tabId, "Input.dispatchKeyEvent", {
        type: "keyUp",
        windowsVirtualKeyCode: 16,
        code: "ShiftLeft",
        key: "Shift"
      });
      await sendDebuggerCommand(tabId, "Input.dispatchKeyEvent", {
        type: "keyUp",
        windowsVirtualKeyCode: 17,
        code: "ControlLeft",
        key: "Control"
      });

      await sleep(250);
      await sendDebuggerCommand(tabId, "Input.insertText", { text: "Terminal: Focus Terminal" });
      await sleep(150);
      await sendDebuggerCommand(tabId, "Input.dispatchKeyEvent", {
        type: "keyDown",
        windowsVirtualKeyCode: 13,
        code: "Enter",
        key: "Enter"
      });
      await sendDebuggerCommand(tabId, "Input.dispatchKeyEvent", {
        type: "keyUp",
        windowsVirtualKeyCode: 13,
        code: "Enter",
        key: "Enter"
      });
      await sleep(350);
    }

    async function focusTerminalStrict() {
      for (let i = 0; i < 10; i += 1) {
        // Escape helps move focus out of the editor widget before terminal click.
        await sendDebuggerCommand(tabId, "Input.dispatchKeyEvent", {
          type: "keyDown",
          windowsVirtualKeyCode: 27,
          code: "Escape",
          key: "Escape"
        });
        await sendDebuggerCommand(tabId, "Input.dispatchKeyEvent", {
          type: "keyUp",
          windowsVirtualKeyCode: 27,
          code: "Escape",
          key: "Escape"
        });

        if (i % 2 === 0) {
          await focusTerminalViaCommandPalette();
        }

        await pressCtrlBackquote();
        await sleep(120);

        if (loc.tabPoint) {
          await clickPoint(loc.tabPoint.x, loc.tabPoint.y, 1);
          await sleep(180);
        }

        await clickPoint(loc.inputPoint.x, loc.inputPoint.y, 2);
        await sleep(140);
        await clickPoint(loc.inputPoint.x, loc.inputPoint.y, 1);
        await sleep(260);

        // Fallback click zones in lower pane where terminal usually lives.
        const fallbackY = [0.78, 0.84, 0.9];
        const fallbackX = [0.2, 0.35, 0.5];
        const tabInfo = await chrome.tabs.get(tabId);
        const width = tabInfo.width || 1400;
        const height = tabInfo.height || 900;
        for (const yFactor of fallbackY) {
          for (const xFactor of fallbackX) {
            await clickPoint(Math.floor(width * xFactor), Math.floor(height * yFactor), 1);
            await sleep(70);
          }
        }

        const focus = await inspectStudioFocus(tabId);
        if (focus.terminalFocused && !focus.editorFocused) {
          return { ok: true, focus };
        }
      }

      const finalFocus = await inspectStudioFocus(tabId);
      return {
        ok: false,
        focus: finalFocus
      };
    }

    const focusResult = await focusTerminalStrict();
    if (!focusResult.ok) {
      return {
        ok: false,
        error: `Debugger fallback: cursor did not move to terminal (editorFocused=${focusResult.focus.editorFocused}, active=${focusResult.focus.activeTag || "unknown"})`
      };
    }

    for (const command of commands) {
      // Insert command + newline to avoid focus races between insert and enter.
      await sendDebuggerCommand(tabId, "Input.insertText", { text: `${command}\n` });
      await sleep(900);
    }

    return {
      ok: true,
      details: [
        `Debugger mode: ${loc.mode || "unknown"}`,
        "Terminal focus acquired before command send"
      ]
    };
  } catch (error) {
    return { ok: false, error: `Debugger fallback failed: ${error?.message || "unknown"}` };
  } finally {
    await detachDebugger(tabId);
  }
}

async function withTimeout(taskPromise, timeoutMs, timeoutMessage) {
  let timeoutId;

  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error(timeoutMessage));
    }, timeoutMs);
  });

  try {
    return await Promise.race([taskPromise, timeoutPromise]);
  } finally {
    clearTimeout(timeoutId);
  }
}

async function storeState(partialState) {
  await chrome.storage.local.set({
    ...partialState,
    updatedAt: Date.now()
  });
}

function stopPolling() {
  if (pollIntervalId) {
    clearInterval(pollIntervalId);
    pollIntervalId = null;
  }
}

function startPolling() {
  stopPolling();

  pollIntervalId = setInterval(() => {
    pollJobStatus().catch(() => {
      // polling errors are handled in pollJobStatus
    });
  }, POLL_INTERVAL_MS);
}

async function notifyCompletion(resultUrl) {
  await chrome.notifications.create({
    type: "basic",
    iconUrl: chrome.runtime.getURL("icon-128.png"),
    title: "AI Interview Generator Controller",
    message: "Your interview is ready"
  });

  if (resultUrl) {
    // keep result URL available for popup UI
    await storeState({ resultUrl });
  }
}

function isStudioUrl(url) {
  if (!url) {
    return false;
  }

  return url.includes("lightning.ai");
}

async function waitForTabComplete(tabId, timeoutMs = 35000) {
  const initialTab = await chrome.tabs.get(tabId);
  if (initialTab.status === "complete") {
    return;
  }

  await new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      chrome.tabs.onUpdated.removeListener(handleUpdate);
      reject(new Error("Timed out waiting for Lightning Studio tab load"));
    }, timeoutMs);

    function handleUpdate(updatedTabId, changeInfo) {
      if (updatedTabId === tabId && changeInfo.status === "complete") {
        clearTimeout(timeoutId);
        chrome.tabs.onUpdated.removeListener(handleUpdate);
        resolve();
      }
    }

    chrome.tabs.onUpdated.addListener(handleUpdate);
  });
}

async function findOrCreateStudioTab(studioUrl) {
  const tabs = await chrome.tabs.query({});
  const existingTab = tabs.find((tab) => isStudioUrl(tab.url));

  if (existingTab?.id) {
    return existingTab.id;
  }

  const createdTab = await chrome.tabs.create({
    url: studioUrl || DEFAULT_STUDIO_URL,
    active: false
  });

  if (!createdTab.id) {
    throw new Error("Unable to open Lightning Studio tab");
  }

  return createdTab.id;
}

async function openProjectFromHome(tabId, preferredProjectName) {
  const injected = await chrome.scripting.executeScript({
    target: { tabId },
    args: [preferredProjectName || ""],
    func: async (projectName) => {
      const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

      function isVisible(element) {
        if (!element) {
          return false;
        }

        const style = window.getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        return style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
      }

      function getText(el) {
        return (el?.textContent || "").replace(/\s+/g, " ").trim();
      }

      function normalize(text) {
        return (text || "").toLowerCase().replace(/\s+/g, " ").trim();
      }

      function findClickable(el) {
        if (!el) {
          return null;
        }

        const closestClickable = el.closest("a[href], button, [role='button'], [role='link'], div[tabindex='0']");
        if (closestClickable && isVisible(closestClickable)) {
          return closestClickable;
        }

        const nested = el.querySelector("a[href], button, [role='button'], [role='link'], div[tabindex='0']");
        if (nested && isVisible(nested)) {
          return nested;
        }

        return null;
      }

      function dispatchClick(target) {
        const events = ["pointerdown", "mousedown", "pointerup", "mouseup", "click"];
        for (const type of events) {
          target.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, view: window }));
        }
      }

      function findProjectTarget(preferred) {
        const allElements = Array.from(document.querySelectorAll("a, button, [role='button'], [role='link'], div, li, article, span, p"));
        const visible = allElements.filter((el) => isVisible(el));

        if (preferred) {
          const exactTextEl = visible.find((el) => normalize(getText(el)) === preferred);
          const exactTarget = findClickable(exactTextEl);
          if (exactTarget) {
            return exactTarget;
          }

          const includesTextEl = visible.find((el) => normalize(getText(el)).includes(preferred));
          const includesTarget = findClickable(includesTextEl);
          if (includesTarget) {
            return includesTarget;
          }
        }

        const recentCandidates = visible.filter((el) => {
          const text = normalize(getText(el));
          if (text.length < 3 || text.length > 180) {
            return false;
          }

          return text.includes("sleeping") || text.includes("hour ago") || text.includes("minutes ago");
        });

        for (const row of recentCandidates) {
          const target = findClickable(row);
          if (target) {
            return target;
          }
        }

        const studioLinks = Array.from(document.querySelectorAll("a[href]"))
          .filter((el) => isVisible(el))
          .filter((el) => /studio|home|machine|app/i.test((el.getAttribute("href") || "") + " " + getText(el)));

        if (studioLinks.length > 0) {
          return studioLinks[0];
        }

        return null;
      }

      const preferred = normalize(projectName);

      let selected = null;
      for (let i = 0; i < 20; i += 1) {
        selected = findProjectTarget(preferred);
        if (selected) {
          break;
        }
        await wait(500);
      }

      if (!selected) {
        const nearbyLabels = Array.from(document.querySelectorAll("a, button, [role='button'], h1, h2, h3, span"))
          .map((el) => getText(el))
          .filter((text) => text.length > 2)
          .slice(0, 25);

        return {
          ok: false,
          reason: "No project card found on home page",
          labels: nearbyLabels
        };
      }

      const href = selected.getAttribute && selected.getAttribute("href") ? selected.getAttribute("href") : "";
      dispatchClick(selected);

      return {
        ok: true,
        clickedText: getText(selected),
        href
      };
    }
  });

  return injected[0]?.result || { ok: false, reason: "Project click injection failed" };
}

async function automateLightningStudio(studioUrl, commands, preferredProjectName) {
  const tabId = await findOrCreateStudioTab(studioUrl);
  await chrome.tabs.update(tabId, { active: true, url: studioUrl || DEFAULT_STUDIO_URL });
  await waitForTabComplete(tabId);
  await sleep(1500);

  let currentTab = await chrome.tabs.get(tabId);
  if (currentTab.url && /\/home(\?|$|\/)/i.test(currentTab.url)) {
    const openResult = await openProjectFromHome(tabId, preferredProjectName);
    if (!openResult.ok) {
      return {
        ok: false,
        error: openResult.reason || "Unable to open project from home",
        details: openResult.labels || []
      };
    }

    if (openResult.href) {
      const absoluteUrl = openResult.href.startsWith("http")
        ? openResult.href
        : new URL(openResult.href, currentTab.url || "https://lightning.ai/").toString();

      await chrome.tabs.update(tabId, { url: absoluteUrl, active: true });
    }

    await sleep(800);
    await waitForTabComplete(tabId, 40000);
    await sleep(1800);
    currentTab = await chrome.tabs.get(tabId);
  }

  const wakeStep = await chrome.scripting.executeScript({
    target: { tabId },
    args: [],
    func: () => {
      const candidates = Array.from(document.querySelectorAll("button, [role='button']"));
      const wakeButton = candidates.find((el) => /(turn\s*on|wake)/i.test((el.textContent || "").trim()));
      const pageText = (document.body?.textContent || "").toLowerCase();

      if (wakeButton) {
        wakeButton.dispatchEvent(new MouseEvent("mousedown", { bubbles: true, cancelable: true, view: window }));
        wakeButton.dispatchEvent(new MouseEvent("mouseup", { bubbles: true, cancelable: true, view: window }));
        wakeButton.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: window }));
      }

      return {
        clickedWake: Boolean(wakeButton),
        sleepingPage: pageText.includes("studio is sleeping")
      };
    }
  });

  if (wakeStep[0]?.result?.clickedWake) {
    await sleep(5000);
  }

  if (!Array.isArray(commands) || commands.length === 0) {
    return {
      ok: true,
      details: ["No terminal commands provided"],
      ran: 0
    };
  }

  const debuggerResult = await sendCommandsWithDebugger(tabId, commands);
  if (debuggerResult.ok) {
    return {
      ok: true,
      details: debuggerResult.details || ["Commands sent via debugger input"],
      ran: commands.length
    };
  }

  return {
    ok: false,
    error: debuggerResult.error || "Failed to send commands to terminal",
    details: debuggerResult.details || []
  };
}

async function startRemoteJob(userId) {
  const response = await fetch(`${API_BASE}/start-job`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      userId,
      jobType: "interview_generation"
    })
  });

  if (!response.ok) {
    let message = `Backend /start-job failed with ${response.status}`;
    try {
      const errorPayload = await response.json();
      if (errorPayload?.error) {
        message = errorPayload.error;
      }
    } catch (_err) {
      // keep fallback message
    }
    throw new Error(message);
  }

  const payload = await response.json();

  if (!payload.jobId) {
    throw new Error("Missing jobId from backend");
  }

  await storeState({
    jobId: payload.jobId,
    status: "pending",
    resultUrl: "",
    lastError: ""
  });

  startPolling();
  await pollJobStatus();

  return payload.jobId;
}

async function pollJobStatus() {
  const { jobId, status } = await chrome.storage.local.get(["jobId", "status"]);

  if (!jobId) {
    stopPolling();
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/job-status/${encodeURIComponent(jobId)}`, {
      method: "GET"
    });

    if (!response.ok) {
      throw new Error(`Backend /job-status failed with ${response.status}`);
    }

    const payload = await response.json();
    const nextStatus = payload.status || "pending";
    const resultUrl = payload.resultUrl || "";

    await storeState({
      status: nextStatus,
      resultUrl,
      lastError: ""
    });

    if (nextStatus === "done") {
      stopPolling();
      await notifyCompletion(resultUrl);
    }
  } catch (_error) {
    // Keep polling and surface outage to popup.
    await storeState({
      status: status === "done" ? "done" : "processing",
      lastError: _error?.message || "Server not reachable"
    });
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "START_JOB") {
    const userId = message.userId;
    const studioUrl = message.studioUrl || DEFAULT_STUDIO_URL;
    const studioProjectName = message.studioProjectName || "";
    const commands = Array.isArray(message.commands) ? message.commands : [];

    (async () => {
      await storeState({
        status: "processing",
        lastError: "",
        studioStatusMessage: "Starting Studio automation..."
      });

      let studioResult;
      try {
        studioResult = await withTimeout(
          automateLightningStudio(studioUrl, commands, studioProjectName),
          AUTOMATION_TIMEOUT_MS,
          "Studio automation timed out"
        );
        if (studioResult.ok) {
          await storeState({
            studioStatusMessage: `Studio commands submitted (${commands.length})`
          });
        } else {
          await storeState({
            studioStatusMessage: `Studio automation failed: ${studioResult.error || studioResult.reason || "unknown reason"}`
          });
          throw new Error(studioResult.error || studioResult.reason || "Studio automation failed");
        }
      } catch (_error) {
        const errorMessage = _error?.message || "Studio automation failed";
        studioResult = { ok: false, error: errorMessage };
        await storeState({
          studioStatusMessage: `Studio automation failed: ${errorMessage}`
        });
        throw _error;
      }

      const jobId = await startRemoteJob(userId);
      return { jobId, studioResult };
    })()
      .then(({ jobId, studioResult }) => {
        sendResponse({ ok: true, jobId, studioResult });
      })
      .catch((error) => {
        storeState({
          status: "error",
          lastError: error?.message || "Server not reachable",
          studioStatusMessage: "Studio automation or backend failed"
        }).finally(() => {
          sendResponse({ ok: false, error: error?.message || "Server not reachable" });
        });
      });

    return true;
  }

  if (message?.type === "GET_STATE") {
    chrome.storage.local
      .get(["jobId", "status", "resultUrl", "lastError"])
      .then((state) => sendResponse({ ok: true, state }))
      .catch(() => sendResponse({ ok: false, error: "Unable to fetch state" }));

    return true;
  }

  return false;
});

async function resumePollingIfNeeded() {
  const state = await chrome.storage.local.get(["jobId", "status"]);

  if (state.jobId && state.status !== "done") {
    startPolling();
    await pollJobStatus();
  }
}

chrome.runtime.onInstalled.addListener(() => {
  resumePollingIfNeeded().catch(() => {
    // no-op
  });
});

chrome.runtime.onStartup.addListener(() => {
  resumePollingIfNeeded().catch(() => {
    // no-op
  });
});
