// Exercise paste gestures with the real viewer and a recording VNC connection.
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
}
`;

function startViewer() {
  cy.readFile("../../src/npa/tools/desktop/remote.py").then(source => {
    const html = source.match(/_VIEWER = """([\s\S]*?)"""/)[1];
    cy.intercept("GET", "/desktop.html", {body: html, headers: {"content-type": "text/html"}});
  });
  cy.readFile("../../src/npa/tools/desktop/desktop_clipboard.js").then(body => {
    cy.intercept("GET", "/desktop_clipboard.js", {body, headers: {"content-type": "text/javascript"}});
  });
  cy.intercept("GET", "/core/rfb.js*", {body: fakeConnection, headers: {"content-type": "text/javascript"}});
  cy.visit("/desktop.html", {onBeforeLoad(win) {
    Object.defineProperty(win.navigator, "clipboard", {value: {
      readText: cy.stub().resolves("Device text\nUnicode: café 日本語 🧪").as("readClipboard"),
      writeText: cy.stub().resolves().as("writeClipboard"),
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
});
