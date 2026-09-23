/* Bridge deliberate browser paste gestures to the selected Linux application. */
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
  // noVNC maps Mac Command to Alt. Release the original shortcut before pasting.
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
  });
  clipboard.rfb.addEventListener("disconnect", () => {
    clipboard.connected = false;
    document.querySelector("#paste-device").disabled = true;
    document.querySelector("#send-clipboard").disabled = true;
  });
  clipboard.rfb.addEventListener("clipboard", (event) => {
    clipboard.remoteText = event.detail.text;
    // Incoming clipboard changes must not overwrite a local paste draft.
    if (!clipboard.dialog.open) clipboard.text.value = clipboard.remoteText;
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
    rfb, connected: false, reading: false, remoteText: "",
    screen: document.querySelector("#screen"),
    dialog: document.querySelector("#clipboard"),
    text: document.querySelector("#clipboard-text"),
    hint: document.querySelector("#clipboard-hint"),
    status: document.querySelector("#clipboard-status"),
  };
  document.querySelector("#paste-device").onclick = () => pasteFromDevice(clipboard);
  document.querySelector("#open-clipboard").onclick = () => openClipboard(clipboard);
  document.querySelector("#send-clipboard").onclick = () => pasteText(clipboard, clipboard.text.value);
  document.querySelector("#copy-clipboard").onclick = () => copyToDevice(clipboard);
  document.querySelector("#close-clipboard").onclick = () => clipboard.dialog.close();
  clipboard.dialog.addEventListener("close", () => rfb.focus());
  bindKeyboard(clipboard);
  bindConnection(clipboard);
}
