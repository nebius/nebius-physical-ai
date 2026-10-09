// Exercise the production offline renderer using explicitly synthetic media.
describe("Workflow evidence preview", () => {
  beforeEach(() => cy.visit("/workflow-preview/"));

  it("keeps engineering measurements accessible without changing the result", () => {
    cy.get(".metrics > div:visible").should("have.length", 4);
    cy.contains("button", "Engineering").click();
    cy.get(".metrics > div:visible").should("have.length", 6);
    cy.get("#measured-evidence").should("have.attr", "open");
    cy.get("#measured-evidence pre").should(
      "contain",
      '"quality_passed": false',
    );
    cy.get("#measured-evidence script").should("not.exist");
    cy.contains("button", "Walkthrough").click();
    cy.get("#measured-evidence").should("not.have.attr", "open");
    cy.get("#metrics-toggle")
      .click()
      .should("have.attr", "aria-expanded", "true");
  });

  it("steps synchronized views and focuses one camera", () => {
    cy.get("#timeline-0 .images img").should("have.length", 4);
    cy.get("#timeline-0 .images img")
      .first()
      .invoke("attr", "src")
      .then((original) => {
        cy.get('[aria-label="Next frame: Camera rig"]').click();
        cy.get("#timeline-0 output").should(
          "have.text",
          "Recorded fixture step 1",
        );
        cy.get("#timeline-0 .images img")
          .first()
          .should("not.have.attr", "src", original);
      });
    cy.get("#view-0").select("Front depth");
    cy.get("#timeline-0 .images img")
      .should("have.length", 1)
      .and("have.attr", "alt", "Front depth");
    cy.get("#timeline-1 output").should("have.text", "Recorded fixture step 0");
  });

  it("runs one slideshow at a time and stops playback when scrubbing", () => {
    cy.clock();
    cy.get("#timeline-0 .play").click();
    cy.tick(500);
    cy.get("#timeline-0 output").should("have.text", "Recorded fixture step 1");
    cy.get("#timeline-1 .play").click();
    cy.get("#timeline-0 .play").should("have.text", "Play");
    cy.tick(500);
    cy.get("#timeline-1 output").should("have.text", "Recorded fixture step 1");
    cy.get("#timeline-1 input[type=range]").invoke("val", 2).trigger("input");
    cy.tick(1000);
    cy.get("#timeline-1 output").should("have.text", "Recorded fixture step 2");
    cy.get("#timeline-1 .play").should("have.attr", "aria-pressed", "false");
  });

  it("supports keyboard point rotation and fits a phone viewport", () => {
    cy.get("#timeline-0 canvas").then((canvas) => {
      const before = canvas[0].toDataURL();
      cy.wrap(canvas).focus().trigger("keydown", { key: "ArrowRight" });
      cy.wrap(canvas).should((rotated) =>
        expect(rotated[0].toDataURL()).not.to.equal(before),
      );
      cy.get("#timeline-0 .point-panel button").click();
      cy.wrap(canvas).should((reset) =>
        expect(reset[0].toDataURL()).to.equal(before),
      );
    });
    cy.viewport(390, 844);
    cy.document().should((document) => {
      expect(document.documentElement.scrollWidth).to.be.at.most(390);
    });
    cy.get("#timeline-0 .images img").each((image) => {
      expect(image[0].complete && image[0].naturalWidth > 0).to.equal(true);
    });
  });
});
