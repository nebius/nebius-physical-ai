// Exercise both clipboard directions with the real viewer and a recording VNC connection.
const fakeConnection = `
export default class RFB extends EventTarget {
  constructor(screen) {
    super(); this.calls=[]; this.viewOnly=false;
    this.canvas=document.createElement('canvas'); this.canvas.tabIndex=0;
    screen.append(this.canvas);
    this.canvas.addEventListener('keydown', event=>{
      this.calls.push(['nativeKey',event.key]); event.preventDefault();
    });
    setTimeout(()=>this.dispatchEvent(new Event('connect')),0);
  }
  focus(){this.canvas.focus();}
  sendKey(...args){this.calls.push(['key',...args]);}
  clipboardPasteFrom(text){this.calls.push(['clipboard',text]);}
  requestClipboard(){this.calls.push(['requestClipboard']);}
}
`;

function startViewer() {
  cy.readFile("../../src/npa/tools/desktop/remote.py").then(source => {
    const html = source.match(/_VIEWER = """([\s\S]*?)"""/)[1];
    cy.intercept("GET", "/desktop.html", {body: html, headers: {"content-type": "text/html"}});
  });
  cy.readFile("../../src/npa/tools/desktop/desktop_clipboard.js").then(body => {
    cy.intercept("GET", "/desktop_clipboard.js*", {body, headers: {"content-type": "text/javascript"}});
  });
  cy.intercept("GET", "/core/rfb.js*", {body: fakeConnection, headers: {"content-type": "text/javascript"}});
  cy.visit("/desktop.html", {onBeforeLoad(win) {
    Object.defineProperty(win.navigator, "clipboard", {value: {
      readText: cy.stub().resolves("Device text\nUnicode: café 日本語 🧪").as("readClipboard"),
      writeText: cy.stub().resolves().as("writeClipboard"),
      write: cy.stub().callsFake(async items => {
        win.deviceClipboard = await (await items[0].getType("text/plain")).text();
      }).as("writeClipboardItem"),
    }});
  }});
  cy.get("#paste-device").should("be.enabled");
}

function clipboardCalls() {
  return cy.window().then(win => win.desktopRfb.calls.filter(call => call[0] === "clipboard"));
}

describe("Cloud desktop clipboard", () => {
  beforeEach(() => { cy.viewport(390, 844); startViewer(); });

  it("reads only after a tap and pastes multiline Unicode without pressing Enter", () => {
    cy.get("@readClipboard").should("not.have.been.called");
    cy.get("#paste-device").click();
    cy.get("@readClipboard").should("have.been.calledOnce");
    clipboardCalls().should("deep.equal", [["clipboard", "Device text\nUnicode: café 日本語 🧪"]]);
    cy.window().then(win => {
      const keys = win.desktopRfb.calls.filter(call => call[0] === "key");
      expect(keys.filter(call => call[2] === "Insert")).to.have.length(1);
      expect(keys.some(call => call[2] === "Enter")).to.equal(false);
      expect(keys.findIndex(call => call[2] === "AltLeft" && call[3] === false))
        .to.be.lessThan(keys.findIndex(call => call[2] === "Insert"));
    });
    cy.get("#clipboard-status").should("contain", "Paste sent");
  });

  it("lets Mac Command-V reach the native paste event exactly once", () => {
    cy.window().then(win => {
      const canvas=win.document.querySelector("canvas"); canvas.focus();
      const key=new win.KeyboardEvent("keydown", {key:"v",code:"KeyV",metaKey:true,bubbles:true,cancelable:true});
      canvas.dispatchEvent(key);
      expect(key.defaultPrevented).to.equal(false);
      const data=new win.DataTransfer(); data.setData("text/plain", "Command-V content");
      canvas.dispatchEvent(new win.ClipboardEvent("paste", {clipboardData:data,bubbles:true,cancelable:true}));
      expect(win.desktopRfb.calls.some(call => call[0] === "nativeKey")).to.equal(false);
    });
    clipboardCalls().should("deep.equal", [["clipboard", "Command-V content"]]);
    cy.get("@readClipboard").should("not.have.been.called");
  });

  it("offers a phone-sized manual paste when permission is denied", () => {
    cy.get("@readClipboard").then(stub => stub.rejects(new Error("NotAllowedError")));
    cy.get("#paste-device").click();
    cy.get("#clipboard").should("be.visible");
    cy.get("#clipboard-hint").should("contain", "manual paste");
    cy.get("#clipboard-text").type("Phone pasted text\nSecond line");
    cy.window().then(win => {
      const rect=win.document.querySelector("#clipboard").getBoundingClientRect();
      expect(rect.left).to.be.at.least(0); expect(rect.right).to.be.at.most(win.innerWidth);
    });
    clipboardCalls().should("have.length",0);
    cy.get("#send-clipboard").click();
    clipboardCalls().should("deep.equal", [["clipboard", "Phone pasted text\nSecond line"]]);
    cy.get("#clipboard").should("not.be.visible");
  });

  it("preserves a local draft when the desktop clipboard changes", () => {
    cy.get("#open-clipboard").click();
    cy.get("#clipboard-text").type("Draft on this device");
    cy.window().then(win => win.desktopRfb.dispatchEvent(new win.CustomEvent("clipboard", {detail:{text:"Remote copy"}})));
    cy.get("#clipboard-text").should("have.value", "Draft on this device");
    cy.get("@writeClipboard").should("not.have.been.called");
    cy.get("#close-clipboard").click();
    cy.get("#open-clipboard").click();
    cy.get("#clipboard-text").should("have.value", "Remote copy");
    cy.get("#copy-clipboard").click();
    cy.get("@writeClipboard").should("have.been.calledOnceWith", "Remote copy");
  });

  it("does not intercept paste into the local clipboard editor", () => {
    cy.get("#open-clipboard").click();
    cy.get("#clipboard-text").focus();
    cy.window().then(win => {
      const data=new win.DataTransfer(); data.setData("text/plain", "Local editing");
      const event=new win.ClipboardEvent("paste", {clipboardData:data,bubbles:true,cancelable:true});
      win.document.querySelector("#clipboard-text").dispatchEvent(event);
      expect(event.defaultPrevented).to.equal(false);
    });
    clipboardCalls().should("have.length",0);
  });

  it("keeps paste disabled after disconnect", () => {
    cy.window().then(win => win.desktopRfb.dispatchEvent(new win.Event("disconnect")));
    cy.get("#paste-device").should("be.disabled");
    cy.get("#open-clipboard").click();
    cy.get("#send-clipboard").should("be.disabled");
    clipboardCalls().should("have.length",0);
  });

  it("starts Mac copy during the gesture and waits for fresh remote Unicode text", () => {
    cy.window().then(win => {
      win.deviceClipboard = "Existing local text";
      const canvas=win.document.querySelector("canvas"); canvas.focus();
      canvas.dispatchEvent(new win.KeyboardEvent("keydown", {
        key:"c", code:"KeyC", metaKey:true, bubbles:true, cancelable:true,
      }));
      expect(win.navigator.clipboard.write).to.have.been.calledOnce;
      expect(win.deviceClipboard).to.equal("Existing local text");
      expect(win.desktopRfb.calls.some(call => call[0] === "nativeKey")).to.equal(false);
      expect(win.desktopRfb.calls.filter(call => call[2] === "Insert")).to.have.length(1);
      win.desktopRfb.dispatchEvent(new win.CustomEvent("clipboard", {detail:{text:"Remote\n日本語 🧪"}}));
    });
    cy.window().its("deviceClipboard").should("equal", "Remote\n日本語 🧪");
    cy.get("#clipboard-status").should("contain", "Copied to this device");
    cy.get("@readClipboard").should("not.have.been.called");
  });

  it("preserves Linux copy shortcuts while exporting their result", () => {
    cy.window().then(win => {
      const canvas=win.document.querySelector("canvas"); canvas.focus();
      canvas.dispatchEvent(new win.KeyboardEvent("keydown", {
        key:"C", code:"KeyC", ctrlKey:true, shiftKey:true, bubbles:true, cancelable:true,
      }));
      expect(win.desktopRfb.calls).to.deep.equal([["nativeKey", "C"]]);
      win.desktopRfb.dispatchEvent(new win.CustomEvent("clipboard", {detail:{text:"Terminal selection"}}));
    });
    cy.window().its("deviceClipboard").should("equal", "Terminal selection");
  });

  it("offers one-tap copy when automatic clipboard access is denied", () => {
    cy.get("@writeClipboardItem").then(stub => stub.onFirstCall().rejects(new Error("NotAllowedError")));
    cy.window().then(win => {
      cy.stub(win.document, "hasFocus").returns(true);
      win.desktopRfb.dispatchEvent(new win.CustomEvent("clipboard", {detail:{text:"Copied in the desktop menu"}}));
    });
    cy.get("#clipboard-status").should("contain", "Tap Copy to device");
    cy.get("#copy-device").should("be.enabled").click();
    cy.window().its("deviceClipboard").should("equal", "Copied in the desktop menu");
  });

  it("does not replace the local clipboard while the desktop tab is unfocused", () => {
    cy.window().then(win => {
      cy.stub(win.document, "hasFocus").returns(false);
      win.desktopRfb.dispatchEvent(new win.CustomEvent("clipboard", {detail:{text:"Background desktop text"}}));
    });
    cy.get("@writeClipboardItem").should("not.have.been.called");
    cy.get("#copy-device").should("be.enabled").click();
    cy.window().its("deviceClipboard").should("equal", "Background desktop text");
  });

  it("selects the remote text for manual copy when the toolbar is denied access", () => {
    cy.get("@writeClipboardItem").then(stub => stub.rejects(new Error("NotAllowedError")));
    cy.window().then(win => {
      win.desktopRfb.dispatchEvent(new win.CustomEvent("clipboard", {detail:{text:"Manual remote copy"}}));
    });
    cy.get("#copy-device").click();
    cy.get("#clipboard").should("be.visible");
    cy.get("#clipboard-text").should("have.value", "Manual remote copy").then(field => {
      expect(field[0].selectionStart).to.equal(0);
      expect(field[0].selectionEnd).to.equal("Manual remote copy".length);
    });
    cy.get("#clipboard-hint").should("contain", "device’s Copy command");
  });

  it("does not copy stale text when a shortcut receives no remote update", () => {
    cy.clock();
    cy.window().then(win => {
      win.deviceClipboard = "Keep this local text";
      const canvas=win.document.querySelector("canvas"); canvas.focus();
      canvas.dispatchEvent(new win.KeyboardEvent("keydown", {
        key:"c", code:"KeyC", metaKey:true, bubbles:true, cancelable:true,
      }));
    });
    cy.tick(4000);
    cy.get("#clipboard-status").should("contain", "No new copied text");
    cy.window().its("deviceClipboard").should("equal", "Keep this local text");
  });

  it("requests the current VNC clipboard when unchanged text produces no notification", () => {
    cy.clock();
    cy.window().then(win => {
      cy.stub(win.document, "hasFocus").returns(false);
      win.desktopRfb.dispatchEvent(new win.CustomEvent("clipboard", {detail:{text:"Repeated selection"}}));
      win.deviceClipboard = "Another app's local text";
      const canvas=win.document.querySelector("canvas"); canvas.focus();
      canvas.dispatchEvent(new win.KeyboardEvent("keydown", {
        key:"c", code:"KeyC", metaKey:true, bubbles:true, cancelable:true,
      }));
    });
    cy.tick(150);
    cy.window().then(win => {
      expect(win.desktopRfb.calls.filter(call=>call[0]==="requestClipboard")).to.have.length(1);
      expect(win.deviceClipboard).to.equal("Another app's local text");
      win.desktopRfb.dispatchEvent(new win.CustomEvent("clipboard", {detail:{text:"Repeated selection"}}));
    });
    cy.window().its("deviceClipboard").should("equal", "Repeated selection");
    cy.get("#clipboard-status").should("contain", "Copied to this device");
  });
});
