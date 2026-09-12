// Browser regression coverage for reaching the footer across Agent tabs.
function expectFooterInViewport() {
  cy.get("#statusBar").should(($footer) => {
    const box = $footer[0].getBoundingClientRect();
    const viewport = $footer[0].ownerDocument.defaultView.innerHeight;
    expect(box.top, "footer top").to.be.at.least(0);
    expect(box.bottom, "footer bottom").to.be.at.most(viewport);
  });
}

function wheelOver(selector) {
  cy.get(selector).then(($element) => {
    const element = $element[0];
    const win = element.ownerDocument.defaultView;
    const frame = window.top.document.querySelector("iframe.aut-iframe").getBoundingClientRect();
    const box = element.getBoundingClientRect();
    const scale = frame.width / win.innerWidth;
    return Cypress.automation("remote:debugger:protocol", {
      command: "Input.dispatchMouseEvent",
      params: {
        type: "mouseWheel", deltaX: 0, deltaY: 2000,
        x: frame.left + (box.left + 10) * scale,
        y: frame.top + Math.min(box.top + 30, win.innerHeight - 30) * scale,
      },
    });
  });
}

describe("Agent page scrolling", () => {
  it("continues scrolling the page after the View sidebar reaches its end", () => {
    cy.viewport(1600, 800);
    cy.visitMockAgent();
    cy.wait("@session");
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
    cy.get("#tabRerun").click();
    cy.get(".rerun-rail").scrollTo("bottom", { ensureScrollable: false });
    cy.scrollTo("top");
    wheelOver(".rerun-rail");
    expectFooterInViewport();
  });

  for (const [width, height] of [[1600, 800], [1024, 600], [375, 667]]) {
    it(`reaches the bottom on Main and View at ${width}x${height}`, () => {
      cy.viewport(width, height);
      cy.visitMockAgent();
      cy.wait("@session");
      cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
      for (const tab of ["#tabMain", "#tabRerun", "#tabMain"]) {
        cy.get(tab).click();
        cy.scrollTo("bottom");
        expectFooterInViewport();
      }
    });
  }
});
