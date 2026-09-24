/* Share text between the Linux desktop and the device clipboard. */
const modifierKeys = [
  [0xffe1, "ShiftLeft"], [0xffe2, "ShiftRight"],
  [0xffe3, "ControlLeft"], [0xffe4, "ControlRight"],
  [0xffe9, "AltLeft"], [0xffea, "AltRight"],
  [0xffeb, "MetaLeft"], [0xffec, "MetaRight"],
];

function message(clipboard, text) {
  clipboard.status.textContent = text;
}

function openClipboard(clipboard, hint = "") {
  if (!clipboard.dialog.open) {
    clipboard.text.value = clipboard.remoteText;
    clipboard.dialog.showModal();
  }
  clipboard.text.focus();
  clipboard.text.select();
  clipboard.hint.textContent = hint || "Paste text below, then choose Paste into desktop. On a phone, touch and hold the text box to show Paste.";
}

function pasteText(clipboard, text) {
  if (!clipboard.connected || clipboard.rfb.viewOnly) {
    message(clipboard, "Connect to the desktop before pasting.");
    return;
  }
  if (!text) {
    openClipboard(clipboard, "The clipboard contains no text. Paste or type text below.");
    return;
  }
  clipboard.rfb.clipboardPasteFrom(text);
  // Release the held shortcut modifiers before sending the Linux paste shortcut.
  for (const [keysym, code] of modifierKeys) clipboard.rfb.sendKey(keysym, code, false);
  // Shift+Insert pastes in Linux editors and terminals without sending Enter.
  clipboard.rfb.sendKey(0xffe1, "ShiftLeft", true);
  clipboard.rfb.sendKey(0xff63, "Insert");
  clipboard.rfb.sendKey(0xffe1, "ShiftLeft", false);
  if (clipboard.dialog.open) clipboard.dialog.close();
  clipboard.rfb.focus();
  message(clipboard, "Paste sent to the selected desktop app.");
}

async function pasteFromDevice(clipboard) {
  if (clipboard.reading) return;
  clipboard.reading = true;
  try {
    const text = await navigator.clipboard.readText();
    pasteText(clipboard, text);
  } catch {
    openClipboard(clipboard, "Your browser needs a manual paste. Paste into this text box, then choose Paste into desktop.");
  } finally {
    clipboard.reading = false;
  }
}

async function copyToDevice(clipboard) {
  try {
    await navigator.clipboard.writeText(clipboard.text.value);
    clipboard.hint.textContent = "Copied to this device.";
  } catch {
    clipboard.text.focus();
    clipboard.text.select();
    clipboard.hint.textContent = "Use your device’s Copy command on the selected text.";
  }
}

async function writeDeviceClipboard(clipboard, content) {
  const text = Promise.resolve(content);
  try {
    if (typeof ClipboardItem === "function" && navigator.clipboard?.write) {
      // Start within the copy gesture; Safari lets the remote text arrive later.
      const payload = text.then(value => new Blob([value], {type: "text/plain"}));
      payload.catch(() => {});
      await navigator.clipboard.write([new ClipboardItem({"text/plain": payload})]);
    } else {
      await navigator.clipboard.writeText(await text);
    }
    message(clipboard, "Copied to this device. Paste into any app.");
    return true;
  } catch {
    try {
      await text;
      message(clipboard, "Desktop text is ready. Tap Copy to device, or use Clipboard to copy manually.");
    } catch {
      message(clipboard, "No new copied text received. Select text and copy in the desktop app.");
    }
    return false;
  }
}

function beginCopy(clipboard) {
  if (clipboard.pendingCopy) return;
  let resolve, reject;
  const text = new Promise((accept, decline) => { resolve = accept; reject = decline; });
  const timer = setTimeout(() => {
    clipboard.pendingCopy = null;
    reject(new Error("No desktop clipboard update"));
  }, 4000);
  // VNC can suppress notifications for unchanged text. Ask again after the key
  // reaches the desktop, rather than copying a potentially stale local cache.
  const refresh = setTimeout(() => clipboard.rfb.requestClipboard(), 150);
  clipboard.pendingCopy = {resolve, reject, timer, refresh};
  void writeDeviceClipboard(clipboard, text);
}

function remoteCopyShortcut(clipboard) {
  for (const [keysym, code] of modifierKeys) clipboard.rfb.sendKey(keysym, code, false);
  // Ctrl+Insert copies in Linux editors and terminals without interrupting a job.
  clipboard.rfb.sendKey(0xffe3, "ControlLeft", true);
  clipboard.rfb.sendKey(0xff63, "Insert");
  clipboard.rfb.sendKey(0xffe3, "ControlLeft", false);
}

function receiveClipboard(clipboard, text) {
  clipboard.remoteText = text;
  document.querySelector("#copy-device").disabled = !clipboard.connected || !text;
  if (!clipboard.dialog.open) clipboard.text.value = text;
  if (clipboard.pendingCopy) {
    clearTimeout(clipboard.pendingCopy.timer);
    clearTimeout(clipboard.pendingCopy.refresh);
    clipboard.pendingCopy.resolve(text);
    clipboard.pendingCopy = null;
  } else if (text && clipboard.connected && !clipboard.dialog.open && document.hasFocus() && document.visibilityState === "visible") {
    void writeDeviceClipboard(clipboard, text);
  } else if (text) {
    message(clipboard, "Desktop text is ready. Tap Copy to device.");
  }
}

function desktopFocused(clipboard) {
  return clipboard.screen.contains(document.activeElement) && !clipboard.dialog.open;
}

function nativePaste(clipboard, event) {
  if (!desktopFocused(clipboard)) return;
  const text = event.clipboardData?.getData("text/plain");
  event.preventDefault();
  event.stopImmediatePropagation();
  if (event.clipboardData?.files.length && !text) {
    openClipboard(clipboard, "This desktop clipboard accepts text. Use the Codex chat attachment button for images.");
    return;
  }
  pasteText(clipboard, text);
}

function bindKeyboard(clipboard) {
  document.addEventListener("keydown", (event) => {
    if (!desktopFocused(clipboard) || event.altKey || !(event.metaKey || event.ctrlKey)) return;
    if (event.key.toLowerCase() === "c" && clipboard.connected && !clipboard.rfb.viewOnly) {
      if (!event.repeat) beginCopy(clipboard);
      if (event.metaKey) {
        event.preventDefault(); event.stopImmediatePropagation();
        if (!event.repeat) remoteCopyShortcut(clipboard);
      }
      return;
    }
    if (event.key.toLowerCase() !== "v") return;
    // Keep the browser's native paste event; noVNC otherwise cancels it.
    event.stopImmediatePropagation();
  }, true);
  document.addEventListener("paste", (event) => nativePaste(clipboard, event), true);
}

function bindConnection(clipboard) {
  clipboard.rfb.addEventListener("connect", () => {
    clipboard.connected = true;
    document.querySelector("#paste-device").disabled = false;
    document.querySelector("#send-clipboard").disabled = false;
    document.querySelector("#copy-device").disabled = !clipboard.remoteText;
  });
  clipboard.rfb.addEventListener("disconnect", () => {
    clipboard.connected = false;
    document.querySelector("#paste-device").disabled = true;
    document.querySelector("#send-clipboard").disabled = true;
    document.querySelector("#copy-device").disabled = true;
    if (clipboard.pendingCopy) {
      clearTimeout(clipboard.pendingCopy.timer);
      clearTimeout(clipboard.pendingCopy.refresh);
      clipboard.pendingCopy.reject(new Error("Desktop disconnected"));
      clipboard.pendingCopy = null;
    }
  });
  clipboard.rfb.addEventListener("clipboard", (event) => {
    receiveClipboard(clipboard, event.detail.text);
  });
}

/**
 * Install clipboard controls without reading the device clipboard in the background.
 * Args: rfb — the current noVNC connection.
 * Returns: None.
 * Raises: An error if the viewer's required controls are missing.
 */
export function installClipboard(rfb) {
  const clipboard = {
    rfb, connected: false, reading: false, remoteText: "", pendingCopy: null,
    screen: document.querySelector("#screen"),
    dialog: document.querySelector("#clipboard"),
    text: document.querySelector("#clipboard-text"),
    hint: document.querySelector("#clipboard-hint"),
    status: document.querySelector("#clipboard-status"),
  };
  document.querySelector("#paste-device").onclick = () => pasteFromDevice(clipboard);
  document.querySelector("#copy-device").onclick = async () => {
    if (!await writeDeviceClipboard(clipboard, clipboard.remoteText))
      openClipboard(clipboard, "Use your device’s Copy command on the selected desktop text.");
  };
  document.querySelector("#open-clipboard").onclick = () => openClipboard(clipboard);
  document.querySelector("#send-clipboard").onclick = () => pasteText(clipboard, clipboard.text.value);
  document.querySelector("#copy-clipboard").onclick = () => copyToDevice(clipboard);
  document.querySelector("#close-clipboard").onclick = () => clipboard.dialog.close();
  clipboard.dialog.addEventListener("close", () => rfb.focus());
  bindKeyboard(clipboard);
  bindConnection(clipboard);
}
