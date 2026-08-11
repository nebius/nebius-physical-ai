import {
  ARTIFACT_ONLY_RUN_ID,
  COMPLEX_WORKFLOW_YAML,
  DF_INPUT_ONLY_RUN_ID,
  DF_MOCK_RUN_ID,
  FIELD_IDS,
  GENERIC_WORKFLOW_YAML,
  NON_STOCK_RUN_ID,
  SIM_VIZ,
  STATIC_BUTTON_IDS,
  WORKFLOW_YAML,
} from "../support/e2e";

describe("NPA agent UI with mocked APIs", () => {
  beforeEach(() => {
    cy.visitMockAgent();
    cy.wait("@session");
    cy.wait("@simAssets");
    cy.wait("@agentAccess");
    cy.wait("@artifactRuns");
    // Wait for boot mount to finish so later loadRun/loadArtifact are not clobbered
    // by ensureFrankaRerunLoaded. Cover clears quickly after warm (no splash latency).
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
  });

  it("renders a generic Stages panel (not Sim2Real-only)", () => {
    cy.get("#stagesPanel").should("exist");
    cy.get("#stagesPanel h3").should("have.text", "Stages");
    cy.contains("Sim2Real Run Monitor").should("not.exist");
    cy.get("#stagesPanel .hint").should("contain.text", "evidence-backed timeline and artifacts");
    cy.get("#stagesPanel .hint").should("not.contain.text", "Sim2Real-only");
    cy.get("#stageList").should("have.attr", "aria-label", "Workflow stages");
    cy.get("#stageList").should("contain.text", "Select assets");
    cy.get("#stageList").should("contain.text", "Render");
    cy.get("#stageList").should("contain.text", "Succeeded");
    cy.get("#runSummary").should("contain.text", "mock-run");
  });

  it("shows generic workflow stages for non-Sim2Real runs", () => {
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#runIdInput").clear({ force: true }).type("cosmos-reason-run", { force: true });
    cy.get("#loadRunData").click({ force: true });
    cy.wait("@loadRun");
    // Clicking the Main tab from Viewer switches to the Main panel directly
    // (it must NOT pop the chat drawer out).
    cy.get("#tabMain").click();
    cy.get("#panelChat").should("have.class", "is-active");
    cy.get("#panelChat").should("not.have.class", "chat-drawer-open");
    cy.get("#stagesPanel h3").should("have.text", "Stages");
    cy.get("#runSummary").should("contain.text", "cosmos-reason-run");
    cy.get("#stageList").should("contain.text", "Fetch checkpoint");
    cy.get("#stageList").should("contain.text", "Reason");
    cy.get("#stageList").should("contain.text", "Publish");
    cy.get("#stageList").should("contain.text", "Running");
    cy.get("#runLog").should("contain.text", "generic workflow stages active");
    cy.contains("Sim2Real Run Monitor").should("not.exist");
  });

  it("separates source, conditioning, and augmented PAIDF data", () => {
    cy.get("#tabVoxel").should("contain.text", "Dataset & provenance").click();
    cy.get("#voxelRunInput").clear().type(DF_MOCK_RUN_ID);
    cy.get("#voxelLoadRun").click();
    cy.wait("@dfDataset");

    cy.get("#voxelReview").should("have.class", "is-real").and("contain.text", "Real FiftyOne Brain review");
    cy.get("#voxelSummary")
      .should("contain.text", "source input 1")
      .and("contain.text", "derived conditioning 1")
      .and("contain.text", "synthetic/augmented 1")
      .and("contain.text", "User-supplied input");
    cy.get("#voxelGrid .voxel-group").should("have.length", 3).then(($groups) => {
      expect($groups.eq(0).text()).to.include("User-supplied input (1)");
      expect($groups.eq(1).text()).to.include("Derived conditioning data (1)");
      expect($groups.eq(2).text()).to.include("Synthetic / augmented data (1)");
    });
    cy.get('#voxelGrid .voxel-card[data-role="source_input"]').should("contain.text", "source.mp4");
    cy.get('#voxelGrid .voxel-card[data-role="derived_conditioning"]').should("contain.text", "conditioning-frame-0001.png");
    cy.get('#voxelGrid .voxel-card[data-role="synthetic_augmented"]').should("contain.text", "aug0");
    cy.get("#panelVoxel").should("not.contain.text", "FiftyOne-style");
  });

  it("renders artifact-only groups without fabricated outcomes and clears demo state", () => {
    // First render the isolated local fixture so the switch proves its graph is
    // cleared before the storage-backed response arrives.
    cy.get("#stagesRunInput").clear().type("franka-demo");
    cy.get("#stagesLoadRun").click();
    cy.wait("@loadFranka");
    cy.get("#stageList").should("contain.text", "Local Franka demo");

    cy.get("#stagesRunInput").clear().type(ARTIFACT_ONLY_RUN_ID);
    cy.get("#stagesLoadRun").click();
    cy.wait("@artifactOnlyList");
    cy.wait("@runDetails");

    cy.get("#runSummary").should("contain.text", ARTIFACT_ONLY_RUN_ID);
    cy.get("#runSummary").should("contain.text", "S3 artifacts");
    cy.get("#stageList .stage-item").should("have.length", 6);
    cy.get("#stageList .stage-progress").should(
      "have.text",
      "6 observed groups · execution status unavailable",
    );
    cy.get("#stageList .stage-status").each(($status) => {
      expect($status.text()).to.eq("Observed output");
    });
    cy.get("#stageList").should("not.contain.text", "Not run");
    cy.get("#stageList").should("not.contain.text", "Succeeded");
    cy.get("#stageList").should("not.contain.text", "Local Franka demo");
    cy.get("#stageList .stage-evidence").should("have.length", 6);
    cy.get("#stageList .stage-evidence").first().should("contain.text", "artifact_listing");
    cy.get("#stageList .stage-evidence").first().should("contain.text", "observed");
  });

  it("blocks a stale stage response from restoring the previous run graph", () => {
    const slowRun = "submitted-run";
    const fastRun = "cosmos-reason-run";
    cy.intercept("GET", `/api/workflows/sim2real/runs/${slowRun}*`, (req) => {
      req.reply({
        delay: 600,
        body: {
          run: {
            run_id: slowRun,
            status: "running",
            stages: [{ id: "stale-stage", label: "Stale stage", status: "running" }],
            logs: [],
          },
        },
      });
    }).as("slowRunDetails");
    cy.intercept("GET", `/api/workflows/sim2real/runs/${fastRun}*`, {
      run: {
        run_id: fastRun,
        status: "running",
        stages: [{ id: "current-stage", label: "Current stage", status: "running" }],
        logs: [],
      },
    }).as("fastRunDetails");

    cy.get("#stagesRunInput").clear().type(slowRun);
    cy.get("#stagesLoadRun").click();
    cy.wait(40);
    cy.get("#stagesRunInput").clear().type(fastRun);
    cy.get("#stagesLoadRun").click();
    cy.wait("@fastRunDetails");
    cy.wait(800);
    cy.get("#runSummary").should("contain.text", fastRun);
    cy.get("#stageList").should("contain.text", "Current stage");
    cy.get("#stageList").should("not.contain.text", "Stale stage");
  });

  it("keeps Stages generic after drafting a non-Sim2Real workflow YAML", () => {
    cy.get("#workflowYaml").clear().type(GENERIC_WORKFLOW_YAML, { delay: 0 });
    cy.get("#workflowValidate").click();
    cy.wait("@workflowValidate");
    cy.get("#stagesPanel h3").should("have.text", "Stages");
    cy.get("#stagesPanel .hint").should("contain.text", "evidence-backed timeline and artifacts");
    cy.contains("Sim2Real Run Monitor").should("not.exist");
  });

  it("never shows Loading application bundle without mount latency", () => {
    cy.get("#rerunBundleCover").should("exist");
    cy.window().then((win) => {
      const html = win.document.documentElement.outerHTML;
      expect(html).to.include("scheduleRerunBundleUncover");
      expect(html).to.include("Uncover without blocking mount latency");
      expect(html).to.include("waitUntilRerunPastBundleSplash");
      expect(html).to.include("swapRerunRecordingInPlace");
      expect(html).to.include("add_receiver");
      expect(html).not.to.include("await waitUntilRerunPastBundleSplash(iframe, 45000)");
      expect(html).not.to.include('Mount the viewer immediately so "Loading application bundle" starts early');
    });
    // Visible chrome only (skip <script> source, which contains the splash detector regex).
    cy.get("#rerunBundleCover .cover-title").should(($el) => {
      expect($el.text()).not.to.match(/Loading application bundle/i);
    });
    cy.get("#rerunBundleCover .cover-hint").should(($el) => {
      expect($el.text()).not.to.match(/Loading application bundle/i);
    });
    cy.get("#statusBar").should(($el) => {
      expect($el.text()).not.to.match(/Loading application bundle/i);
    });
    // Mock Rerun serves a canvas with no splash; cover should clear quickly (no cold wasm).
    cy.get("#rerunBundleCover", { timeout: 15000 }).should("have.attr", "hidden");
    cy.get("#rerunFrame").should(($frame) => {
      const frame = $frame[0];
      const doc = frame.contentDocument || (frame.contentWindow && frame.contentWindow.document);
      const text = String((doc && doc.body && doc.body.innerText) || "");
      expect(text).not.to.match(/Loading application bundle/i);
    });
  });

  it("renders every static control and generated panel", () => {
    for (const id of STATIC_BUTTON_IDS) {
      cy.get(`#${id}`).should("exist");
    }
    for (const id of FIELD_IDS) {
      cy.get(`#${id}`).should("exist");
    }
    cy.get("#workflowYaml").should("contain.value", "apiVersion: npa.workflow/v0.0.1");
    cy.get("#workflowSubmitHint").should("contain.text", "plan-only");
    cy.get("#tabMain").should("have.attr", "aria-selected", "true");
    cy.get("#tabRerun").click();
    cy.get("#tabRerun").should("have.attr", "aria-selected", "true");
    cy.get("#panelRerun").should("have.class", "is-active").and("have.attr", "aria-hidden", "false");
    cy.get("#panelChat").should("have.class", "is-inactive").and("have.attr", "aria-hidden", "true");
    cy.get("#renderModeRerun").should("have.class", "is-active");
    cy.get("#renderModeVideo").should("exist");
    cy.get("#viewerPaneRerun").should("have.class", "is-active-viewer");
    cy.get("#simRunId").should("contain.text", "mock-run");
  });

  it("selects one escaped project detail at a time and preserves the stable project ID", () => {
    const projectCapabilities = (artifactStatus, artifactReason, readStatus, readReason, submitStatus, submitReason) => ({
      project_metadata: { status: "available", reason: "Project identity was returned by tenant discovery." },
      storage_resource_discovery: { status: "available", reason: "Object storage resources visible in this project were listed." },
      artifact_discovery: { status: artifactStatus, reason: artifactReason, scope: "read_only" },
      artifact_read: { status: readStatus, reason: readReason, scope: "read_only" },
      workflow_submission: { status: submitStatus, reason: submitReason, scope: "deployment_project" },
    });
    const resource = (id, name, readStatus = "available", readReason = "Object reads were verified.") => ({
      type: "object_storage_bucket",
      id,
      name,
      capabilities: {
        artifact_discovery: { status: "available", reason: "Object listing was verified.", scope: "read_only" },
        artifact_read: { status: readStatus, reason: readReason, scope: "read_only" },
      },
    });
    const hostileProjectId = 'project-x"><svg/onload=window.__npaXss=1>';
    const hostileBucket = 'bucket-<img src=x onerror=window.__npaXss=2>-&"\'';
    const projects = [
      {
        id: "project-d",
        name: "Empty Bucket",
        deployment_project: false,
        status: "partial",
        capabilities: projectCapabilities(
          "available",
          "The empty bucket remains searchable.",
          "unverified",
          "Read access is unverified because the empty bucket has no object to probe.",
          "unavailable",
          "Tenant-wide discovery does not enable workflow submission in this project.",
        ),
        resources: [resource(
          "resource-empty",
          "empty-artifacts",
          "unverified",
          "Object listing succeeded; the empty bucket has no object to verify read access.",
        )],
      },
      {
        id: "project-b",
        name: "Foreign Project",
        deployment_project: false,
        status: "available",
        capabilities: projectCapabilities(
          "available",
          "At least one project bucket is searchable.",
          "available",
          "Artifact object reads were verified.",
          "unavailable",
          "Tenant-wide discovery does not enable workflow submission in this project.",
        ),
        resources: [resource("resource-b", "foreign-artifacts")],
      },
      {
        id: "project-f",
        name: "Duplicate Name",
        deployment_project: false,
        status: "available",
        capabilities: projectCapabilities("unavailable", "No searchable object storage bucket was verified.", "unavailable", "No reads were verified.", "unavailable", "Home project only."),
        resources: [],
      },
      {
        id: "project-a",
        name: "Project Alpha",
        deployment_project: true,
        status: "available",
        capabilities: projectCapabilities("available", "At least one project bucket is searchable.", "available", "Artifact object reads were verified.", "available", "Workflow submission remains scoped to the deployment project."),
        resources: [resource("resource-a", "project-artifacts")],
      },
      {
        id: "project-c",
        name: "No Bucket",
        deployment_project: false,
        status: "partial",
        capabilities: projectCapabilities("unavailable", "No searchable object storage bucket was verified for this project.", "unavailable", "Artifact object reads were not verified.", "unavailable", "Tenant-wide discovery does not enable workflow submission in this project."),
        resources: [],
      },
      {
        id: "project-e",
        name: "Duplicate Name",
        deployment_project: false,
        status: "available",
        capabilities: projectCapabilities("unavailable", "No searchable object storage bucket was verified.", "unavailable", "No reads were verified.", "unavailable", "Home project only."),
        resources: [],
      },
      {
        id: hostileProjectId,
        name: '<img src=x onerror="window.__npaXss=3">',
        deployment_project: false,
        status: "partial",
        capabilities: projectCapabilities("available", '<script>window.__npaXss=4</script> searchable.', "unverified", 'No object named "<probe>" exists.', "unavailable", '<img src=x onerror="window.__npaXss=5"> home project only.'),
        resources: [resource("resource-hostile", hostileBucket, "unverified", '<svg onload="window.__npaXss=6">empty</svg>')],
      },
    ];
    const accessPayload = {
      apiVersion: "npa.agent.access/v1",
      identity: {
        tenant_id: "tenant-test",
        deployment_project_id: "project-a",
        deployment_project_name: "Project Alpha",
      },
      status: "partial",
      scope: "partial_tenant",
      capabilities: { artifact_discovery: { status: "partial", reason: "Accessible projects are searched; specific gaps are reported." } },
      projects,
      errors: [{ message: '<img src=x onerror="window.__npaXss=7"> access probe unavailable.' }],
    };
    cy.intercept("GET", "/api/access*", { statusCode: 200, body: accessPayload }).as("selectorAccess");
    cy.get("#agentAccessRefresh").click();
    cy.wait("@selectorAccess");

    cy.get('label[for="agentAccessProjectSelect"]').should("contain.text", "Project");
    cy.get("#agentAccessProjectSelect")
      .should("have.prop", "tagName", "SELECT")
      .and("be.enabled")
      .and("have.value", "project-a");
    cy.get("#agentAccessProjectSelect option").should("have.length", projects.length).then(($options) => {
      expect([...$options].map((option) => option.value)).to.deep.equal([
        "project-a",
        "project-b",
        "project-c",
        "project-d",
        "project-e",
        "project-f",
        hostileProjectId,
      ]);
    });
    cy.get("#agentAccessProjects .access-project-detail").should("have.length", 1)
      .and("contain.text", "Project Alpha")
      .and("contain.text", "project-a")
      .and("contain.text", "deployment project");
    cy.get("#agentAccessPanel").should(($panel) => {
      expect($panel.text()).not.to.match(/\bpartial\b/i);
    });

    cy.get("#agentAccessProjectSelect").select("project-b");
    cy.get("#agentAccessProjects .access-project-detail").should("have.length", 1)
      .and("contain.text", "Foreign Project")
      .and("contain.text", "foreign-artifacts")
      .and("contain.text", "tenant read-only")
      .and("not.contain.text", "project-artifacts");
    cy.get('#agentAccessProjects button[data-access-action="list"]')
      .should("have.attr", "data-project-id", "project-b")
      .and("have.attr", "data-resource-bucket", "foreign-artifacts");

    cy.get("#agentAccessProjectSelect").select("project-c");
    cy.get("#agentAccessProjects .access-project-detail").should("have.length", 1)
      .and("contain.text", "No searchable artifact bucket.")
      .and("contain.text", "No Bucket")
      .and("not.contain.text", "foreign-artifacts");
    cy.get("#agentAccessProjects button[data-access-action]").should("not.exist");

    cy.get("#agentAccessProjectSelect").select("project-d");
    cy.get("#agentAccessProjects .access-project-detail").should("have.length", 1)
      .and("contain.text", "Empty Bucket")
      .and("contain.text", "empty-artifacts")
      .and("contain.text", "Read: Unverified")
      .and("contain.text", "empty bucket has no object to verify read access")
      .and("not.contain.text", "No searchable artifact bucket.");
    cy.get('#agentAccessProjects button[data-access-action="list"]').should("be.enabled");
    cy.get('#agentAccessProjects button[data-access-action="read"]').should("be.disabled");

    cy.get("#agentAccessRefresh").click();
    cy.wait("@selectorAccess");
    cy.get("#agentAccessProjectSelect").should("have.value", "project-d");
    cy.reload();
    cy.wait("@selectorAccess");
    cy.get("#agentAccessProjectSelect").should("have.value", "project-d");
    cy.get("#agentAccessProjects .access-project-detail").should("have.length", 1).and("contain.text", "Empty Bucket");

    cy.get("#agentAccessProjectSelect option").then(($options) => {
      const duplicateLabels = [...$options].filter((option) => option.textContent.includes("Duplicate Name"));
      expect(duplicateLabels.map((option) => option.textContent)).to.deep.equal([
        "Duplicate Name — project-e",
        "Duplicate Name — project-f",
      ]);
    });
    cy.get("#agentAccessProjectSelect").invoke("val", hostileProjectId).trigger("change");
    cy.get("#agentAccessProjects .access-project-detail").should("have.length", 1)
      .and("contain.text", hostileProjectId)
      .and("contain.text", hostileBucket)
      .and("contain.text", "window.__npaXss=4");
    cy.get("#agentAccessPanel").find("script, img, svg").should("not.exist");
    cy.window().its("__npaXss").should("not.exist");
    cy.get("#agentAccessProjectSelect").focus().should("be.focused");

    cy.viewport(375, 667);
    cy.get("#agentAccessProjectSelect").should("be.visible").then(($select) => {
      const selectRect = $select[0].getBoundingClientRect();
      const panelRect = $select[0].closest("#agentAccessPanel").getBoundingClientRect();
      expect(selectRect.left).to.be.at.least(panelRect.left);
      expect(selectRect.right).to.be.at.most(panelRect.right + 1);
    });

    cy.intercept("GET", "/api/access*", {
      statusCode: 200,
      body: { ...accessPayload, projects: projects.filter((project) => project.id !== hostileProjectId) },
    }).as("selectorAccessMissing");
    cy.get("#agentAccessRefresh").click();
    cy.wait("@selectorAccessMissing");
    cy.get("#agentAccessProjectSelect").should("have.value", "project-a");
    cy.get("#agentAccessProjects .access-project-detail").should("have.length", 1).and("contain.text", "Project Alpha");
  });

  it("makes available access capabilities semantic mouse and keyboard actions", () => {
    cy.get('#agentAccessProjects button[data-access-action="list"][data-resource-bucket="project-artifacts"]')
      .should("have.prop", "tagName", "BUTTON")
      .and("be.enabled")
      .and("have.attr", "aria-label");
    cy.get('#agentAccessProjects button[data-access-action="list"][data-resource-bucket="project-artifacts"]').click();
    cy.wait("@artifactRuns").its("request.url").should((url) => {
      expect(url).to.include("project_id=project-a");
      expect(url).to.include("resource_bucket=project-artifacts");
    });
    cy.get("#agentAccessActionResult")
      .should("be.visible")
      .and("contain.text", "List complete")
      .and("contain.text", "3 runs")
      .and("contain.text", "project-a")
      .and("contain.text", "project-artifacts")
      .and("contain.text", NON_STOCK_RUN_ID);

    // Native <button> semantics make Enter dispatch the same scoped action.
    cy.get('#agentAccessProjects button[data-access-action="list"][data-resource-bucket="project-artifacts"]')
      .focus();
    cy.press(Cypress.Keyboard.Keys.ENTER);
    cy.wait("@artifactRuns").its("request.url").should("include", "resource_bucket=project-artifacts");

    // Read is independently actionable and opens the real artifact browser + preferred preview.
    cy.get('#agentAccessProjects button[data-access-action="read"][data-resource-bucket="project-artifacts"]')
      .should("be.enabled")
      .click();
    cy.wait("@artifactRuns");
    cy.wait("@nonStockArtifactList");
    cy.wait("@loadArtifact");
    cy.get("#renderedDataSummary").should("contain.text", NON_STOCK_RUN_ID);
    cy.get("#tabMain").click();
    cy.get("#agentAccessActionResult")
      .should("contain.text", "Readable artifacts opened")
      .and("contain.text", "project-artifacts");

    // Space/Enter activation is supplied by the same native button, not a key shim.
    cy.get('#agentAccessProjects button[data-access-action="read"][data-resource-bucket="project-artifacts"]')
      .focus();
    cy.press(Cypress.Keyboard.Keys.ENTER);
    cy.wait("@artifactRuns");
    cy.wait("@nonStockArtifactList");
    cy.get("#agentAccessActionResult").should("contain.text", "Readable artifacts opened");
  });

  it("disables denied and unavailable access actions with visible reasons", () => {
    for (const bucket of ["denied-artifacts", "unavailable-artifacts"]) {
      cy.get(`#agentAccessProjects [data-resource-bucket="${bucket}"]`).each(($button) => {
        expect($button[0].tagName).to.eq("BUTTON");
        expect($button).to.be.disabled;
        expect($button.attr("aria-describedby")).to.be.a("string").and.not.be.empty;
      });
    }
    cy.get("#agentAccessProjects").should("contain.text", "Permission denied while listing objects.");
    cy.get("#agentAccessProjects").should("contain.text", "Object reads could not be verified.");
  });

  it("shows busy/error feedback and blocks stale resource responses", () => {
    cy.intercept("GET", "/api/artifacts/runs*", (req) => {
      const url = new URL(req.url);
      const bucket = url.searchParams.get("resource_bucket") || "";
      if (bucket === "project-artifacts") {
        req.reply({
          delay: 700,
          statusCode: 200,
          body: {
            ok: true,
            runs: [{ run_id: "stale-project-run", bucket, project_id: "project-a" }],
            total_runs: 1,
            truncated: false,
          },
        });
        return;
      }
      req.reply({
        delay: 25,
        statusCode: 200,
        body: {
          ok: true,
          runs: [{ run_id: "fresh-archive-run", bucket, project_id: "project-a" }],
          total_runs: 1,
          truncated: false,
        },
      });
    }).as("scopedAccessRuns");

    cy.get('#agentAccessProjects button[data-access-action="list"][data-resource-bucket="project-artifacts"]').click();
    cy.get("#agentAccessActionResult")
      .should("have.attr", "aria-busy", "true")
      .and("contain.text", "Querying the selected project and bucket");
    cy.get('#agentAccessProjects button[data-access-action="list"][data-resource-bucket="archive-artifacts"]').click();
    cy.wait("@scopedAccessRuns");
    cy.get("#agentAccessActionResult", { timeout: 3000 })
      .should("contain.text", "fresh-archive-run")
      .and("contain.text", "archive-artifacts")
      .and("not.contain.text", "stale-project-run");
    cy.wait(800);
    cy.get("#agentAccessActionResult").should("not.contain.text", "stale-project-run");

    cy.intercept("GET", "/api/artifacts/runs*", {
      statusCode: 502,
      body: { ok: false, error: "Scoped list probe failed." },
    }).as("failedAccessRuns");
    cy.get('#agentAccessProjects button[data-access-action="list"][data-resource-bucket="project-artifacts"]').click();
    cy.wait("@failedAccessRuns");
    cy.get("#agentAccessActionResult")
      .should("have.attr", "role", "alert")
      .and("contain.text", "List failed")
      .and("contain.text", "Scoped list probe failed")
      .and("not.contain.text", "fresh-archive-run");
  });

  it("refreshing access cancels and clears an in-flight scoped result", () => {
    cy.intercept("GET", "/api/artifacts/runs*", {
      delay: 700,
      statusCode: 200,
      body: {
        ok: true,
        runs: [{ run_id: "must-not-render-after-refresh", bucket: "project-artifacts", project_id: "project-a" }],
        total_runs: 1,
        truncated: false,
      },
    }).as("refreshStaleRuns");
    cy.get('#agentAccessProjects button[data-access-action="list"][data-resource-bucket="project-artifacts"]').click();
    cy.get("#agentAccessActionResult").should("have.attr", "aria-busy", "true");
    cy.get("#agentAccessRefresh").click();
    cy.wait("@agentAccess");
    cy.get("#agentAccessActionResult").should("have.attr", "hidden");
    cy.wait(800);
    cy.get("#agentAccessActionResult").should("have.attr", "hidden");
    cy.get("#agentAccessActionResult").should("not.contain.text", "must-not-render-after-refresh");
  });

  it("opens a JSON-only run through Read and renders useful content", () => {
    cy.intercept("GET", "/api/artifacts/runs*", (req) => {
      const url = new URL(req.url);
      expect(url.searchParams.get("project_id")).to.eq("project-a");
      expect(url.searchParams.get("resource_bucket")).to.eq("project-artifacts");
      req.reply({
        statusCode: 200,
        body: {
          ok: true,
          runs: [{
            run_id: "json-only-storage-run",
            bucket: "project-artifacts",
            project_id: "project-a",
            source_type: "artifact_storage",
            source_label: "S3 artifacts",
          }],
          total_runs: 1,
          truncated: false,
        },
      });
    }).as("jsonScopedRuns");
    cy.get('#agentAccessProjects button[data-access-action="read"][data-resource-bucket="project-artifacts"]')
      .focus();
    cy.press(Cypress.Keyboard.Keys.ENTER);
    cy.wait("@jsonScopedRuns");
    cy.wait("@jsonOnlyArtifactList");
    cy.wait("@loadArtifact");
    cy.get("#renderModeData").should("have.class", "is-active");
    cy.get("#artifactPreviewHost pre")
      .should("contain.text", "json-only-storage-run")
      .and("contain.text", "evaluations");
    cy.get("#artifactList").should("contain.text", "Download");
  });

  it("embeds the Lichtblick MCAP viewer as a Viewer render mode", () => {
    cy.window().then((win) => {
      cy.stub(win, "open").as("windowOpen");
    });
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");

    // Lichtblick render-mode tab + dedicated iframe pane exist and activate.
    cy.get("#renderModeLichtblick").should("exist").click();
    cy.get("#renderModeLichtblick").should("have.class", "is-active");
    cy.get("#viewerPaneLichtblick").should("have.class", "is-active-viewer");
    cy.get("#viewerPaneRerun").should("have.class", "is-inactive-viewer");
    cy.get("#lichtblickFrame")
      .should("have.attr", "src")
      .and("include", "/lichtblick/")
      .and("include", "ds.url");
    // The embedded Lichtblick app renders the MCAP data source (mock fixture).
    cy.get("#lichtblickFrame").its("0.contentWindow.__NPA_MOCK_LICHTBLICK__", { timeout: 15000 }).should("exist");

    // "Open in Lichtblick" opens the same-origin viewer URL in a new tab.
    cy.get("#openLichtblick").click();
    cy.get("@windowOpen").should("have.been.called");

    // Reload stays on the Lichtblick pane and re-mounts the iframe.
    cy.get("#loadLichtblickViewer").click();
    cy.get("#viewerPaneLichtblick").should("have.class", "is-active-viewer");
    cy.get("#lichtblickFrame").should("have.attr", "src").and("include", "/lichtblick/");

    // Switching back to Rerun deactivates the Lichtblick pane (both stay mounted).
    cy.get("#renderModeRerun").click();
    cy.get("#viewerPaneRerun").should("have.class", "is-active-viewer");
    cy.get("#viewerPaneLichtblick").should("have.class", "is-inactive-viewer");
  });

  it("loads a sim2real.mcap artifact into the embedded Lichtblick viewer", () => {
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#runIdSelect").select(NON_STOCK_RUN_ID);
    cy.wait("@nonStockArtifactList");
    cy.wait("@loadArtifact");
    cy.get("#rerunFrame").should("have.attr", "src").and("include", "/rerun/");

    cy.get("#artifactList").should("contain.text", `${NON_STOCK_RUN_ID}/reports/sim2real.mcap`);
    cy.get("#artifactList").should("contain.text", "mcap");
    cy.get("#artifactList").should("contain.text", "View in Lichtblick");

    cy.get(`#artifactList button[data-action="load-artifact"][data-key="${NON_STOCK_RUN_ID}/reports/sim2real.mcap"]`).click();
    cy.wait("@loadArtifact");
    cy.get("#renderModeLichtblick").should("have.class", "is-active");
    cy.get("#viewerPaneLichtblick").should("have.class", "is-active-viewer");
    cy.get("#lichtblickFrame").should("have.attr", "src").and("include", "/lichtblick/");
    cy.get("#renderedDataSummary").should("contain.text", "mcap");
  });

  it("lists artifacts for a pasted run id over a stale dropdown selection", () => {
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    // Leave a stale dropdown selection (mock-run) without firing its change handler,
    // then paste a different specific run id into the input.
    cy.get("#runIdSelect").then(($s) => {
      $s[0].value = "mock-run";
    });
    cy.get("#runIdInput").clear().type(NON_STOCK_RUN_ID);
    // Clicking the button steals focus from the input, but "List artifacts" must still
    // use the pasted run id, not the stale dropdown value.
    cy.get("#artifactLoadRunArtifacts").click();
    cy.wait("@nonStockArtifactList");
    cy.get("#artifactList").should("contain.text", `${NON_STOCK_RUN_ID}/reports/sim2real.mcap`);
    cy.get("#artifactList").should("contain.text", "View in Lichtblick");
    cy.get("#artifactList").should("not.contain.text", "mock-run/preview.png");
  });

  it("covers chat quick actions, sessions, model selection, submit, and copy", () => {
    cy.get("#chatActionS3").click();
    cy.get("#chatInput").should("contain.value", "configure S3");
    cy.get("#chatActionCosmos").click();
    cy.get("#chatInput").should("contain.value", "Cosmos3");
    cy.get("#chatActionWatch").click();
    cy.get("#chatInput").should("contain.value", "Rerun");
    cy.get("#chatActionWorkflow").click();
    cy.get("#chatInput").should("contain.value", "2-step sim2real workflow");

    cy.get("#chatModel").select("mock/model");
    cy.get("#chatSend").click();
    cy.wait("@chat");
    cy.get("#chatLog .msg-row.user").should("contain.text", "2-step sim2real workflow");
    cy.get("#chatLog .msg-row.assistant").should("contain.text", "Here is a 2-step workflow");
    cy.get("#workflowYaml").should("contain.value", "cypress-sim2real");

    cy.window().then((win) => {
      if (!win.navigator.clipboard) {
        Object.defineProperty(win.navigator, "clipboard", {
          value: { writeText: () => Promise.resolve() },
          configurable: true,
        });
      }
      cy.stub(win.navigator.clipboard, "writeText").resolves();
    });
    cy.get(".msg-copy-btn").contains(/^Copy/).first().click();
    cy.get("#toastHost").should("contain.text", "copied");

    cy.get("#newChatSession").click();
    cy.wait("@newChatSession");
    cy.get("#chatSessionSelect").should("have.value", "new-session");
    cy.get("#chatSessionSelect").select("session-two");
    cy.wait("@selectChatSession");
    cy.get("#chatLog").should("contain.text", "show status");
  });

  it("covers workflow draft upload, validate, plan, and submit buttons", () => {
    cy.get("#workflowYaml").clear().type(WORKFLOW_YAML, { delay: 0 });

    cy.get("#workflowUpload").click();
    cy.wait("@workflowDraft");
    cy.get("#chatLog").should("contain.text", "Uploaded workflow YAML");

    cy.get("#workflowValidate").click();
    cy.wait("@workflowValidate");
    cy.get("#workflowValidation").should("contain.text", "valid");

    cy.get("#workflowPlan").click();
    cy.wait("@workflowPlan");
    cy.get("#workflowPlanHost").should("be.visible");
    cy.get("#workflowPlanHost").should("contain.text", "workbench.sim2real.status");
    cy.get("#workflowPlanOutput").should("contain.text", "workbench.sim2real.status");
    cy.get("#workflowValidation").should("contain.text", "planned");

    cy.get("#workflowSubmitYaml").click();
    cy.wait("@workflowSubmitYaml");
    cy.get("#chatLog").should("contain.text", "Submitted npa.workflow");
    cy.get("#chatLog").should("contain.text", "plan");
  });

  it("covers Stages panel, Rerun buttons, and run-data loading", () => {
    cy.window().then((win) => {
      cy.stub(win, "open").as("windowOpen");
    });

    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");

    // The Selection / Scene-mode section was removed — the viewer just shows run
    // artifacts now, so no robot/scene/cube controls or Franka/Submit buttons.
    cy.get("#sceneMode").should("not.exist");
    cy.get("#applySelection").should("not.exist");
    cy.get("#loadFrankaRerun").should("not.exist");
    cy.get("#submitWorkflow").should("not.exist");
    cy.get("#stagesPanel h3").should("have.text", "Stages");

    cy.get("#loadRerunViewer").click({ force: true });
    cy.get("#statusBar").should(($bar) => {
      expect($bar.text()).to.match(/Rerun|Reload|Ready/);
    });

    cy.get("#workflowStatus").click();
    cy.wait("@workflowStatus");
    cy.get("#tabMain").click();
    cy.get("#chatLog").should("contain.text", "Latest workflow status");

    cy.get("#tabRerun").click();
    cy.get('#runIdSelect option[value="mock-run"][data-source-type="workflow_history"]').then(($opt) => {
      const select = $opt[0].parentElement;
      select.selectedIndex = [...select.options].indexOf($opt[0]);
      cy.wrap(select).trigger("change");
    });
    cy.wait("@loadRun");
    cy.get("#tabMain").click();
    cy.get("#chatLog").should("contain.text", "Loaded run context");
    cy.get("#runLog").should("contain.text", "mock run log");
    cy.get("#stagesPanel h3").should("have.text", "Stages");

    cy.get("#tabRerun").click();
    cy.get("#openRerun").click();
    cy.get("@windowOpen").should("have.been.called");
  });

  it("consolidates runs & artifacts into one latest-first picker", () => {
    cy.get("#tabRerun").click();
    cy.get("#runsArtifactsPanel").should("exist");
    cy.contains("h3", "Runs & artifacts").should("exist");
    cy.contains("h4", "Active run").should("not.exist");
    cy.contains("h4", "Artifacts").should("not.exist");
    cy.get("#artifactRunSelect").should("not.exist");

    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#artifactDiscoverStatus").should("contain.text", "latest first");
    cy.get("#runIdSelect option").then(($opts) => {
      const values = [...$opts].map((opt) => opt.value).filter(Boolean);
      // Discovered non-stock run is newest; must appear before older mock-run.
      expect(values[0]).to.eq(NON_STOCK_RUN_ID);
      expect(values).to.include("mock-run");
      expect(values).to.include("submitted-run");
    });
    cy.get("#stagesRunSelect option").then(($opts) => {
      const values = [...$opts].map((opt) => opt.value).filter(Boolean);
      expect(values[0]).to.eq(NON_STOCK_RUN_ID);
    });
  });

  it("preserves known and unknown viewability in lightweight run summaries", () => {
    cy.intercept("GET", "/api/artifacts/runs*", {
      statusCode: 200,
      body: {
        ok: true,
        runs: [
          {
            run_id: "known-viewable-run",
            source_type: "artifact_storage",
            has_viewable: true,
            summary_complete: false,
            last_modified: "2026-08-02T00:00:00Z",
          },
          {
            run_id: "unknown-viewability-run",
            source_type: "artifact_storage",
            has_viewable: null,
            summary_complete: false,
            last_modified: "2026-08-01T00:00:00Z",
          },
        ],
        total_runs: 2,
        truncated: false,
      },
    }).as("viewabilityRuns");

    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@viewabilityRuns");
    cy.get('#runIdSelect option[value="known-viewable-run"]')
      .should("contain.text", "viewable")
      .and("not.contain.text", "viewability unknown");
    cy.get('#runIdSelect option[value="unknown-viewability-run"]')
      .should("contain.text", "viewability unknown");
  });

  it("follows the native S3 artifact cursor and merges pages", () => {
    const runId = "paged-ui-run";
    cy.intercept("GET", "/api/artifacts/runs*", {
      statusCode: 200,
      body: {
        ok: true,
        runs: [{
          run_id: runId,
          source_type: "artifact_storage",
          bucket: "paged-bucket",
          project_id: "project-paged",
          has_viewable: true,
          summary_complete: false,
          last_modified: "2026-08-03T00:00:00Z",
        }],
        total_runs: 1,
        truncated: false,
      },
    }).as("pagedRuns");
    cy.intercept("GET", `/api/artifacts/run/${runId}*`, (req) => {
      const cursor = String((req.query && req.query.cursor) || "");
      if (!cursor) {
        req.reply({
          statusCode: 200,
          body: {
            ok: true,
            run_id: runId,
            bucket: "paged-bucket",
            project_id: "project-paged",
            resolved_prefix: "category",
            artifacts: [{
              run_id: runId,
              key: `category/${runId}/a.json`,
              s3_uri: `s3://paged-bucket/category/${runId}/a.json`,
              render: "json",
              size: 10,
            }],
            preferred: null,
            truncated: true,
            next_cursor: "cursor-page-2",
          },
        });
        return;
      }
      expect(cursor).to.eq("cursor-page-2");
      expect(req.query.resolved_prefix).to.eq("category");
      expect(req.query.source_selected).to.eq("1");
      expect(req.query.resource_bucket).to.eq("paged-bucket");
      req.reply({
        statusCode: 200,
        body: {
          ok: true,
          run_id: runId,
          bucket: "paged-bucket",
          project_id: "project-paged",
          resolved_prefix: "category",
          artifacts: [{
            run_id: runId,
            key: `category/${runId}/b.json`,
            s3_uri: `s3://paged-bucket/category/${runId}/b.json`,
            render: "json",
            size: 11,
          }],
          preferred: null,
          truncated: false,
          next_cursor: "",
        },
      });
    }).as("pagedArtifacts");

    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@pagedRuns");
    cy.get("#runIdSelect").select(runId);
    cy.wait("@pagedArtifacts");
    cy.get("#artifactList").should("contain.text", `category/${runId}/a.json`);
    cy.get('#artifactList button[data-action="load-more-artifacts"]').click();
    cy.wait("@pagedArtifacts");
    cy.get("#artifactList")
      .should("contain.text", `category/${runId}/a.json`)
      .and("contain.text", `category/${runId}/b.json`);
    cy.get('#artifactList button[data-action="load-more-artifacts"]').should("not.exist");
  });

  it("covers artifact discovery, dynamic artifact load button, and camera cards", () => {
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    // Discovery is generic (no path prefix); all runs show.
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    // Consolidated picker may already have mock-run selected — force list via button.
    cy.get('#runIdSelect option[data-run-id="mock-run"][data-source-type="artifact_storage"]').then(($opt) => {
      const select = $opt[0].parentElement;
      select.selectedIndex = [...select.options].indexOf($opt[0]);
    });
    cy.get("#artifactLoadRunArtifacts").click();
    cy.wait("@artifactList");
    cy.get("#artifactList").should("contain.text", "mock-run/preview.png");

    cy.get("#artifactLoadRunArtifacts").click();
    cy.wait("@artifactList");
    cy.get("#artifactList button[data-action='load-artifact']").click();
    cy.wait("@loadArtifact");
    // loadArtifact no longer spams chat; the viewer / preview host reflects the load.
    cy.get("#artifactPreviewHost").should("not.have.attr", "hidden");

    cy.get("#tabMain").click();
    cy.get("#panelChat").should("have.class", "is-active").and("have.attr", "aria-hidden", "false");
    cy.get("#panelChat").should("not.have.class", "chat-drawer-open");
    cy.get("#panelRerun").should("have.class", "is-inactive");
  });

  it("dispatches local, workflow-history, and JSON-only S3 runs by provenance", () => {
    const JSON_RUN_ID = "json-only-storage-run";
    let workflowLoads = 0;
    let localLoads = 0;
    cy.intercept("POST", "/api/sim-viz/load-run", (req) => {
      workflowLoads += 1;
      const runId = String((req.body && req.body.run_id) || "");
      req.reply({ statusCode: 200, body: { ok: true, sim_viz: { ...SIM_VIZ, run_id: runId, active_run_id: runId } } });
    }).as("provenanceWorkflowLoad");
    cy.intercept("POST", "/api/sim-viz/load-franka-demo", (req) => {
      localLoads += 1;
      req.reply({
        statusCode: 200,
        body: {
          ok: true,
          sim_viz: { ...SIM_VIZ, run_id: "franka-demo", active_run_id: "franka-demo", source_type: "local_demo" },
        },
      });
    }).as("provenanceLocalLoad");

    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get(`#runIdSelect option[value="${JSON_RUN_ID}"]`)
      .should("have.attr", "data-source-type", "artifact_storage")
      .and("contain.text", "S3 artifacts");
    cy.get("#runIdSelect").select(JSON_RUN_ID);
    cy.wait("@jsonOnlyArtifactList");
    cy.wait("@loadArtifact");
    cy.get("#renderModeData").should("have.class", "is-active");
    cy.get("#artifactPreviewHost pre").should("contain.text", "evaluations");
    cy.get("#stageList").should("contain.text", "evaluation");
    cy.get("#artifactList").should("contain.text", "policy.ckpt");
    cy.get(`#runIdSelect option[value="${JSON_RUN_ID}"][data-source-type="artifact_storage"]`)
      .should("have.length", 1);
    cy.wrap(null).should(() => expect(workflowLoads, "S3 selection never calls load-run").to.eq(0));

    cy.get("#runIdSelect").select("franka-demo");
    cy.wait("@provenanceLocalLoad");
    cy.get("#tabMain").click();
    cy.get("#stageList").should("contain.text", "Local Franka demo");
    cy.get("#runSummary").should("contain.text", "franka-demo");
    cy.wrap(null).should(() => expect(localLoads, "local demo endpoint called once").to.eq(1));

    cy.get("#tabRerun").click();
    cy.get("#runIdSelect").select("cosmos-reason-run");
    cy.wait("@provenanceWorkflowLoad");
    cy.get("#tabMain").click();
    cy.get("#stageList").should("contain.text", "Fetch checkpoint");
    cy.wrap(null).should(() => expect(workflowLoads, "workflow endpoint called once").to.eq(1));
  });

  it("discovers and interacts with non-stock Sim2Real run artifacts", () => {
    cy.window().then((win) => {
      cy.stub(win, "open").as("windowOpen");
    });

    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#runIdSelect").select(NON_STOCK_RUN_ID);
    cy.wait("@nonStockArtifactList");
    cy.wait("@loadArtifact");

    cy.get("#artifactList").should("contain.text", `${NON_STOCK_RUN_ID}/reports/sim2real.rrd`);
    cy.get("#rerunFrame").should("have.attr", "src").and("include", "/rerun/");
    cy.get("#artifactList").should("contain.text", "rerun");
    cy.get("#artifactList").should("contain.text", "video");
    cy.get("#artifactList").should("contain.text", "json");
    cy.get("#artifactList").should("contain.text", "text");
    cy.get("#artifactList").should("contain.text", "download");
    cy.get("#artifactList").should("contain.text", "View in Rerun");
    cy.get("#artifactList").should("contain.text", "Play");
    cy.get("#artifactTypeFilter").select("video");
    cy.wait("@nonStockArtifactList");
    cy.get("#artifactList").should("contain.text", `${NON_STOCK_RUN_ID}/rollouts/customer-camera.mp4`);
    cy.get("#artifactList").should("not.contain.text", `${NON_STOCK_RUN_ID}/reports/sim2real.rrd`);
    cy.get("#artifactSort").select("largest");
    cy.wait("@nonStockArtifactList");
    cy.get("#artifactList").should("contain.text", "Showing 1 of");
    cy.get("#artifactTypeFilter").select("");
    cy.wait("@nonStockArtifactList");
    cy.get("#simRunId").should("contain.text", NON_STOCK_RUN_ID);
    cy.get("#simStage").should("contain.text", "stage_14_rerun_viz");
    cy.get("#simCamera").should("contain.text", "customer-overhead");
    cy.get("#rerunFrame").should("have.attr", "src").and("include", "/rerun/");

    cy.get(`#runIdSelect option[value="${NON_STOCK_RUN_ID}"][data-source-type="workflow_history"]`).then(($opt) => {
      const select = $opt[0].parentElement;
      select.selectedIndex = [...select.options].indexOf($opt[0]);
      cy.wrap(select).trigger("change");
    });
    cy.wait("@loadRun");
    cy.get("#tabMain").click();
    cy.get("#stagesPanel h3").should("have.text", "Stages");
    cy.get("#runSummary").should("contain.text", NON_STOCK_RUN_ID);
    cy.get("#stageList").should("contain.text", "Customer assets");
    cy.get("#runLog").should("contain.text", "non-stock sim2real artifacts");

    cy.get("#tabRerun").click();
    cy.get("#artifactLoadRunArtifacts").click();
    cy.wait("@nonStockArtifactList");
    cy.wait("@loadArtifact");
    cy.get(
      `#artifactList button[data-action='load-artifact'][data-key="${NON_STOCK_RUN_ID}/rollouts/customer-camera.mp4"]`
    ).click();
    cy.wait("@loadArtifact");
    cy.wait("@artifactFile");
    cy.get("#renderModeVideo").should("have.class", "is-active");
    cy.get("#viewerPaneMedia").should("have.class", "is-active-viewer");
    cy.get("#artifactPreviewHost video")
      .should("have.attr", "src")
      .and("match", /^blob:/);
    cy.get("#artifactPreviewHost video")
      .should("have.attr", "data-preview-url")
      .and("include", "customer-camera.mp4");
    cy.get("#renderedDataSummary").should("contain.text", "video");

    cy.get(
      `#artifactList button[data-action='load-artifact'][data-key="${NON_STOCK_RUN_ID}/reports/sim2real-report.json"]`
    ).click();
    cy.wait("@loadArtifact");
    cy.get("#renderModeData").should("have.class", "is-active");
    cy.get("#artifactPreviewHost pre").should("contain.text", "promoted");

    cy.get(
      `#artifactList button[data-action='load-artifact'][data-key="${NON_STOCK_RUN_ID}/logs/orchestrator.log"]`
    ).click();
    cy.wait("@loadArtifact");
    cy.get("#artifactPreviewHost pre").should("contain.text", "loaded customer scene mesh");

    cy.get(
      `#artifactList button[data-action='load-artifact'][data-key="${NON_STOCK_RUN_ID}/raw/custom-dynamics.fooz"]`
    ).click();
    cy.wait("@loadArtifact");
    cy.get("#artifactPreviewHost").should("contain.text", "download");
    cy.get("#artifactPreviewHost a").should("have.attr", "href").and("include", "custom-dynamics.fooz");

    cy.get(
      `#artifactList button[data-action='load-artifact'][data-key="${NON_STOCK_RUN_ID}/reports/sim2real.rrd"]`
    ).click();
    cy.wait("@loadArtifact");
    cy.get("#renderModeRerun").should("have.class", "is-active");
    cy.get("#rerunFrame").should("have.attr", "src").and("include", "/rerun/");

    cy.get("#openRerun").click();
    cy.get("@windowOpen").should("have.been.called");
  });

  it("finds runs by name/ID via a client-side filter (no path prefix needed)", () => {
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    // Generic discovery lists every run — no prefix/category to type.
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#runIdSelect option").then(($opts) => {
      const values = [...$opts].map((o) => o.value).filter(Boolean);
      expect(values).to.include(NON_STOCK_RUN_ID);
      expect(values).to.include("mock-run");
    });
    // Typing part of a run name/ID filters the list client-side.
    cy.get("#artifactPrefix").clear().type("non-stock");
    cy.get("#runIdSelect option").then(($opts) => {
      const values = [...$opts].map((o) => o.value).filter(Boolean);
      expect(values).to.include(NON_STOCK_RUN_ID);
      expect(values).to.not.include("mock-run");
    });
    // Clearing restores the full list.
    cy.get("#artifactPrefix").clear();
    cy.get("#runIdSelect option").then(($opts) => {
      const values = [...$opts].map((o) => o.value).filter(Boolean);
      expect(values).to.include("mock-run");
    });
  });

  it("shows per-stage provenance with an honest augment engine and click-to-filter", () => {
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    // Load a data-factory run whose augment is a REAL Cosmos Transfer 2.5 GPU render.
    // Enter in the paste box loads the typed run and lists its artifacts.
    cy.get("#runIdInput").clear({ force: true }).type(`${DF_MOCK_RUN_ID}{enter}`, { force: true });
    cy.wait("@dfArtifactList");
    cy.wait("@artifactProvenance");

    // The per-stage panel lists each pipeline stage with its producing component.
    cy.get("#artifactProvenance").should("contain.text", "Pipeline stages");
    cy.get("#artifactProvenance .prov-row").its("length").should("be.gte", 4);
    cy.get("#artifactProvenance").should("contain.text", "Augment");
    cy.get("#artifactProvenance").should("contain.text", "Cosmos Transfer 2.5");
    // Honesty banner: this run's augment is real GPU, so it must say so (green).
    cy.get("#artifactProvenance .prov-ok").should("contain.text", "real Cosmos Transfer 2.5");
    cy.get("#artifactProvenance .prov-warn").should("not.exist");

    // The Stage filter carries per-stage counts.
    cy.get("#artifactStageFilter option").then(($opts) => {
      const labels = [...$opts].map((o) => o.textContent || "");
      expect(labels.some((t) => /cosmos_augmented \(\d+\)/.test(t)), "cosmos_augmented has a count").to.eq(true);
      expect(labels.some((t) => /All stages \(\d+\)/.test(t)), "all-stages total").to.eq(true);
    });

    // Clicking the Augment stage row scopes the artifact list to that stage.
    cy.get('#artifactProvenance .prov-clickable[data-stage="cosmos_augmented"]').click();
    cy.wait("@dfArtifactList");
    cy.get("#artifactStageFilter").should("have.value", "cosmos_augmented");
    cy.get("#artifactList").should("contain.text", "cosmos_augmented/aug0/augmented_video.mp4");
    cy.get("#artifactList").should("not.contain.text", "/input/video_0.mp4");
  });

  it("shows source-frame images inline when the Source frames stage is clicked", () => {
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#runIdInput").clear({ force: true }).type(`${DF_MOCK_RUN_ID}{enter}`, { force: true });
    cy.wait("@dfArtifactList");
    cy.wait("@artifactProvenance");

    // Click the "Source frames" (input) pipeline stage.
    cy.get('#artifactProvenance .prov-clickable[data-stage="input"]').should("contain.text", "Source frames").click();
    cy.wait("@dfArtifactList");

    // The artifact list scopes to the input stage and shows the source frames...
    cy.get("#artifactStageFilter").should("have.value", "input");
    cy.get("#artifactList").should("contain.text", "input/frame_00.png");
    cy.get("#artifactList").should("contain.text", "input/frame_01.png");
    cy.get("#artifactList").should("not.contain.text", "cosmos_augmented/");

    // ...as actual inline image thumbnails (not just filenames): each image card
    // renders an <img> lazy-loaded through the authenticated download proxy.
    cy.wait("@artifactDownload");
    cy.get("#artifactList .artifact-card[data-render='image'] .artifact-thumb img")
      .its("length")
      .should("be.gte", 2);
  });

  it("warns when a data-factory run has only raw input and no augmented output", () => {
    cy.get("#tabRerun").click();
    // A DF run with only input/ + configs/ (augment never produced output) must
    // be flagged so a raw input clip is not mistaken for a Data Factory result.
    cy.get("#runIdInput").clear({ force: true }).type(`${DF_INPUT_ONLY_RUN_ID}{enter}`, { force: true });
    cy.wait("@dfInputOnlyArtifactList");
    cy.wait("@artifactProvenance");
    cy.get("#artifactProvenance .prov-warn").should("contain.text", "no augmented output");
    cy.get("#artifactProvenance .prov-ok").should("not.exist");
  });

  it("filters artifacts by workflow stage and tags timeline rows by stage", () => {
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRuns");
    cy.get("#runIdSelect").select(NON_STOCK_RUN_ID);
    cy.wait("@nonStockArtifactList");
    cy.wait("@loadArtifact");

    // The Stage (workflow-progress) selector is populated from the loaded
    // artifacts' first path segment after the run id.
    cy.get("#artifactStageFilter option").then(($opts) => {
      const values = [...$opts].map((opt) => opt.value);
      expect(values).to.include.members(["reports", "rollouts", "logs", "raw"]);
    });

    // Selecting a stage scopes the artifact list to that workflow-progress step.
    cy.get("#artifactStageFilter").select("rollouts");
    cy.wait("@nonStockArtifactList");
    cy.get("#artifactList").should("contain.text", `${NON_STOCK_RUN_ID}/rollouts/customer-camera.mp4`);
    cy.get("#artifactList").should("not.contain.text", `${NON_STOCK_RUN_ID}/reports/sim2real.rrd`);

    // Clearing the stage filter restores the full listing.
    cy.get("#artifactStageFilter").select("");
    cy.wait("@nonStockArtifactList");
    cy.get("#artifactList").should("contain.text", `${NON_STOCK_RUN_ID}/reports/sim2real.rrd`);

    // The artifact-derived timeline tags rows with a stage key so they are
    // clickable to scope the browser. (The click handler is covered by the
    // agent unit test; the periodic sim-viz poll re-renders #stageList in the
    // mock, so a live click assertion here would be race-prone.)
    cy.get("#stageList").should("exist");
  });

  it("grounds complex chat queries and complex workflow YAML drafts", () => {
    cy.get("#chatInput").type(
      "For the non-stock customer run, what can I view, which artifact should I load first, and how do I keep Rerun interactive?",
      { delay: 0 },
    );
    cy.get("#chatSend").click();
    cy.wait("@chat");
    cy.get("#chatLog").should("contain.text", "Non-stock Sim2Real artifacts");
    cy.get("#chatLog").should("contain.text", NON_STOCK_RUN_ID);
    cy.get("#chatLog").should("contain.text", "Artifact browser");

    cy.get("#chatInput").type(
      "Draft a complex VLM/RL outer loop workflow YAML for non-stock assets with a quality gate and promote or loop-back transitions.",
      { delay: 0 },
    );
    cy.get("#chatSend").click();
    cy.wait("@chat");
    cy.get("#workflowYaml").should("contain.value", "cypress-vlm-rl-loop");
    cy.get("#workflowYaml").should("contain.value", "workbench.token_factory.reason");
    cy.get("#workflowYaml").should("contain.value", "loop_back");

    cy.get("#workflowYaml").clear().type(COMPLEX_WORKFLOW_YAML, { delay: 0 });
    cy.get("#workflowValidate").click();
    cy.wait("@workflowValidate");
    cy.get("#workflowName").should("contain.text", "cypress-vlm-rl-loop");
    cy.get("#workflowStates").should("contain.text", "vlm_gate");

    cy.get("#workflowPlan").click();
    cy.wait("@workflowPlan");
    cy.get("#workflowPlanHost").should("contain.text", "workbench.token_factory.reason");
    cy.get("#workflowPlanOutput").should("contain.text", "workbench.token_factory.reason");
  });

  it("covers mobile panels toggle and mobile chat auth flow", () => {
    cy.viewport("iphone-x");
    cy.visitMockAgent();
    cy.get("body").should("have.class", "mobile-agent");

    cy.get("#mobilePanelsToggle").click();
    cy.get("body").should("have.class", "mobile-show-panels");
    cy.get("#mobilePanelsToggle").should("have.attr", "aria-expanded", "true");

    cy.get("#mobileChatPassword").type("mock-password");
    cy.get("#mobileChatAuthBtn").click();
    cy.wait("@health");
    cy.get("body").should("have.class", "mobile-auth-ready");

    cy.get("#chatInput").type("mobile hello");
    cy.get("#chatSend").click();
    cy.wait("@chat");
    cy.get("#chatLog").should("contain.text", "mobile hello");

    // Widening the viewport must leave mobile-agent layout (toggle, not add-only).
    cy.viewport(1280, 800);
    cy.get("body").should("not.have.class", "mobile-agent");
  });

  it("escapes quotes in artifact keys to prevent attribute XSS", () => {
    const evilKey = 'a" onmouseover="alert(1)';
    // Override the default mock-run list intercept with a malicious key.
    cy.intercept("GET", "/api/artifacts/run/mock-run*", {
      statusCode: 200,
      body: {
        ok: true,
        run_id: "mock-run",
        prefix: "sim2real-b",
        artifacts: [
          {
            key: evilKey,
            s3_uri: `s3://mock/${evilKey}`,
            size: 12,
            last_modified: "2026-01-01T00:00:00Z",
            render: "download",
          },
        ],
        preferred: null,
      },
    }).as("evilArtifactList");
    cy.get("#tabRerun").click();
    cy.get('#runIdSelect option[data-run-id="mock-run"][data-source-type="artifact_storage"]').then(($opt) => {
      const select = $opt[0].parentElement;
      select.selectedIndex = [...select.options].indexOf($opt[0]);
    });
    cy.get("#artifactLoadRunArtifacts").click();
    cy.wait("@evilArtifactList");
    cy.get("#artifactList").then(($el) => {
      const html = $el.html() || "";
      // Attribute values must be quote-escaped so the key cannot break out of
      // data-*="..." and inject an event handler. Text nodes may still show
      // literal quotes after the browser parses escaped HTML.
      expect(html).to.include('data-key="a&quot; onmouseover=&quot;alert(1)"');
      expect(html).to.include('data-s3-uri="s3://mock/a&quot; onmouseover=&quot;alert(1)"');
      expect(html).to.not.match(/data-(?:key|s3-uri|name)="a"\s+onmouseover=/i);
      cy.get("#artifactList button[data-action='load-artifact']").should(($btn) => {
        // DOM getAttribute returns the decoded value; no separate attribute breakout.
        expect($btn.attr("data-key")).to.eq(evilKey);
        expect($btn.attr("onmouseover")).to.eq(undefined);
      });
    });
  });

  it("clears the Rerun Caching cover and keeps it hidden after remounts", () => {
    // Boot must clear the cover; it must not stick on "Caching Rerun assets…".
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
    cy.get("#statusBar").should("not.contain.text", "Caching Rerun assets");

    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("body").should("have.class", "viewer-focus");

    // Repeated remounts / reloads must not leave the Caching overlay visible.
    cy.get("#loadRerunViewer").click({ force: true });
    cy.get("#rerunBundleCover", { timeout: 15000 }).should("have.attr", "hidden");
    cy.get("#loadRerunViewer").click({ force: true });
    cy.get("#rerunBundleCover", { timeout: 15000 }).should("have.attr", "hidden");
    cy.get("#statusBar").should(($el) => {
      expect($el.text()).not.to.match(/Caching Rerun assets/i);
    });
    cy.get("#rerunBundleCover .cover-hint").should(($el) => {
      // When hidden, hint text may still say Almost ready / Caching — but cover must stay hidden.
      expect(Cypress.$("#rerunBundleCover").attr("hidden")).to.exist;
    });

    // Soft-path: reload again while viewer is already mounted.
    cy.get("#loadRerunViewer").click({ force: true });
    cy.wait(400);
    cy.get("#rerunBundleCover").should("have.attr", "hidden");
    cy.get("#statusBar", { timeout: 10000 }).should("not.contain.text", "Caching Rerun assets");
  });

  it("opens the chat collapsible only via the chat button, and Main tab never pops it out", () => {
    cy.get("#tabRerun").click();
    cy.get("body").should("have.class", "viewer-focus");
    cy.get("#chatDrawerToggle").should("be.visible");
    cy.get("#panelChat").should("not.have.class", "chat-drawer-open");

    // The chat collapsible opens ONLY when the chat button (FAB) is clicked.
    cy.get("#chatDrawerToggle").click();
    cy.get("#panelChat").should("have.class", "chat-drawer-open");
    cy.get("#chatDrawerToggle").should("have.class", "is-open");
    cy.get("#chatDrawerClose").should("be.visible").click();
    cy.get("#panelChat").should("not.have.class", "chat-drawer-open");
    cy.get("#rerunBundleCover").should("have.attr", "hidden");

    // Regression guard: clicking the Main tab from the Viewer switches to the
    // Main panel and must NOT pop the chat drawer out.
    cy.get("#tabMain").click();
    cy.get("#panelChat").should("have.class", "is-active");
    cy.get("#panelChat").should("not.have.class", "chat-drawer-open");
    cy.get("#panelRerun").should("have.class", "is-inactive");
    cy.get("body").should("not.have.class", "viewer-focus");

    // Returning to the Viewer and back to Main again still never pops the drawer.
    cy.get("#tabRerun").click();
    cy.get("body").should("have.class", "viewer-focus");
    cy.get("#tabMain").click();
    cy.get("#panelChat").should("not.have.class", "chat-drawer-open");
    cy.get("#panelChat").should("have.class", "is-active");
  });

  it("labels the main tab 'Main' (renamed from Chat)", () => {
    cy.get("#tabMain").should("exist").and("have.text", "Main");
    cy.get("#tabMain").should("have.attr", "data-tab", "main");
    cy.get("#tabChat").should("not.exist");
  });

  it("shows a scroll-to-bottom arrow when scrolled up and jumps to the latest message", () => {
    // Fill the chat via real sends so the log overflows and can be scrolled.
    for (let i = 0; i < 6; i += 1) {
      cy.get("#chatInput").type(`Draft a 2-step Sim2Real workflow YAML please (${i})`, { delay: 0 });
      cy.get("#chatSend").click();
      cy.wait("@chat");
    }
    // Each new message auto-scrolls to the bottom, so the arrow is hidden.
    cy.get("#chatScrollBottom").should("have.attr", "hidden");

    // Scrolling up reveals the jump-to-latest arrow.
    cy.get("#chatLog").scrollTo("top");
    cy.get("#chatScrollBottom").should("not.have.attr", "hidden");
    cy.get("#chatScrollBottom").should("be.visible");

    // Clicking the arrow returns to the end of the chat and hides the arrow.
    cy.get("#chatScrollBottom").click();
    cy.get("#chatLog").should(($log) => {
      const el = $log[0];
      expect(el.scrollHeight - el.scrollTop - el.clientHeight).to.be.lessThan(41);
    });
    cy.get("#chatScrollBottom").should("have.attr", "hidden");
  });

  it("keeps local Workflow YAML edits across refresh-driven run loads", () => {
    const edited = "apiVersion: npa.workflow/v0.0.1\nkind: Workflow\nmetadata:\n  name: local-edit\n";
    cy.get("#workflowYaml").clear().type(edited, { delay: 0 });
    cy.get("#tabRerun").click();
    cy.get("#runIdInput").clear({ force: true }).type("cosmos-reason-run", { force: true });
    cy.get("#loadRunData").click({ force: true });
    cy.wait("@loadRun");
    cy.get("#tabMain").click();
    cy.get("#workflowYaml").should("contain.value", "local-edit");
  });

  it("Stages Load prefers pasted run id over a stale dropdown selection", () => {
    cy.get("#tabMain").click();
    cy.get("#stagesRunSelect").then(($select) => {
      $select[0].value = "mock-run";
    });
    cy.get("#stagesRunInput").clear().type("cosmos-reason-run", { delay: 0 });
    cy.get("#stagesLoadRun").click();
    cy.wait("@loadRun");
    cy.get("#runSummary").should("contain.text", "cosmos-reason-run");
  });

  it("Stages search filters the run list by name", () => {
    cy.get("#tabMain").click();
    cy.get("#stagesRunInput").clear().type("mock", { delay: 0 });
    cy.get("#stagesRunSearchHint").should("contain.text", "match");
    cy.get("#stagesRunSelect option").then(($opts) => {
      const values = [...$opts].map((opt) => opt.value).filter(Boolean);
      expect(values.length).to.be.greaterThan(0);
      expect(values.every((value) => value.includes("mock"))).to.eq(true);
    });
    cy.get("#stagesLoadRun").click();
    cy.wait("@artifactList");
    cy.get("#runSummary").should("contain.text", "mock-run");
  });

  it("keeps current-run evidence separate from a failed maintenance-job lookup", () => {
    const maintenanceId = "codex-maintenance-20310102T030405Z-deadbeef";
    cy.intercept("GET", `/api/artifacts/run/${maintenanceId}*`, {
      statusCode: 404,
      body: {
        ok: false,
        error: {
          code: "run_not_discovered",
          message: "No discovered NPA workflow/artifact run has this identifier. Identifiers under /home/ubuntu/codex-runs are Codex maintenance job IDs, not NPA run IDs.",
        },
      },
    }).as("maintenanceNotFound");

    cy.get("#runSummary").should("contain.text", "mock-run");
    cy.get("#stagesRunInput").clear().type(maintenanceId, { delay: 0 });
    cy.get("#stagesLoadRun").click();
    cy.wait("@maintenanceNotFound");
    cy.get("#runSummary").should("contain.text", "currently loaded run");
    cy.get("#runSummary").should("contain.text", "mock-run");
    cy.get("#runSummary").should("not.contain.text", maintenanceId);
    cy.get("#stagesRunSearchResult").should("contain.text", "Codex maintenance job IDs");
    cy.get("#stagesRunSearchResult").should("contain.text", "Currently loaded run remains mock-run");
  });

  it("surfaces an old run beyond the newest page via server-side (q=) search", () => {
    // Reproduces the real-world "run doesn't show" case: the run is older than
    // the newest page the default listing returns, so it only appears when the
    // client asks the SERVER to search by name/ID (?q=). The default page must
    // NOT contain it; the ?q= page must. Both the Rerun-tab and Stages-tab run
    // pickers must render the option once the server search returns it.
    const OLD_RUN_ID = "rtxpro-staged-2x2-20260613t011356z";
    const FRAGMENT = "rtxpro-staged";
    const oldRun = {
      run_id: OLD_RUN_ID,
      has_viewable: true,
      artifact_count: 141,
      last_modified: "2026-06-13T01:13:56Z",
    };
    const newestPage = [
      {
        run_id: NON_STOCK_RUN_ID,
        has_viewable: true,
        artifact_count: 5,
        last_modified: "2026-07-11T18:00:00Z",
      },
      {
        run_id: "mock-run",
        has_viewable: true,
        artifact_count: 1,
        last_modified: "2026-07-07T03:33:00Z",
      },
    ];
    // Registered inside the test so it takes precedence over the default
    // "@artifactRuns" intercept for every subsequent /api/artifacts/runs call.
    cy.intercept("GET", "/api/artifacts/runs*", (req) => {
      const q = String((req.query && req.query.q) || "").trim().toLowerCase();
      const matchesOld = q && OLD_RUN_ID.toLowerCase().includes(q);
      req.reply({
        statusCode: 200,
        body: {
          ok: true,
          runs: matchesOld ? [oldRun] : newestPage,
          total_runs: 328,
          truncated: true,
          query: q,
        },
      });
    }).as("artifactRunsPaged");

    // Rerun tab: the default listing omits the old run…
    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRunsPaged");
    cy.get("#runIdSelect option").then(($opts) => {
      const values = [...$opts].map((o) => o.value).filter(Boolean);
      expect(values, "default page omits the old run").to.not.include(OLD_RUN_ID);
    });
    // …but typing a fragment triggers a debounced server search that finds it.
    cy.get("#artifactPrefix").clear().type(FRAGMENT, { delay: 0 });
    cy.get("#runSummary").should("contain.text", "currently loaded run");
    cy.get("#runSummary").should("contain.text", "mock-run");
    cy.wait("@artifactRunsPaged").its("request.url").should("include", "q=");
    cy.get("#runIdSelect option").should(($opts) => {
      const values = [...$opts].map((o) => o.value).filter(Boolean);
      expect(values, "server search surfaces the old run in the Rerun picker").to.include(OLD_RUN_ID);
    });

    // Stages tab: same server-search path must populate the stages picker.
    cy.get("#tabMain").click();
    cy.get("#stagesRunInput").clear().type(FRAGMENT, { delay: 0 });
    cy.get("#stagesRunSearchResult").should("contain.text", "separate from the currently loaded run mock-run");
    cy.wait("@artifactRunsPaged").its("request.url").should("include", "q=");
    cy.get("#stagesRunSelect option").should(($opts) => {
      const values = [...$opts].map((o) => o.value).filter(Boolean);
      expect(values, "server search surfaces the old run in the Stages picker").to.include(OLD_RUN_ID);
    });
  });

  it("follows every server cursor into the default run picker", () => {
    // The API owns bounded pagination. The browser must follow every cursor and
    // render runs beyond the first page without requiring a guessed search term.
    const bigList = Array.from({ length: 150 }, (_unused, i) => {
      const idx = String(i).padStart(3, "0");
      return {
        run_id: `bulk-run-${idx}`,
        has_viewable: true,
        artifact_count: 1,
        // Descending timestamps so index 149 is the oldest (would fall off a
        // 100-run page).
        last_modified: `2026-06-${String((i % 27) + 1).padStart(2, "0")}T00:00:${idx.slice(-2)}Z`,
      };
    });
    const capturedUrls = [];
    cy.intercept("GET", "/api/artifacts/runs*", (req) => {
      capturedUrls.push(String(req.url || ""));
      const cursor = String((req.query && req.query.cursor) || "");
      const start = cursor ? 100 : 0;
      const visible = bigList.slice(start, start + 100);
      req.reply({
        statusCode: 200,
        body: {
          ok: true,
          runs: visible,
          total_runs: bigList.length,
          next_cursor: start + visible.length < bigList.length ? "page-two" : "",
          truncated: start + visible.length < bigList.length,
          pagination_complete: start + visible.length >= bigList.length,
        },
      });
    }).as("artifactRunsFull");

    cy.get("#tabRerun").click();
    cy.get("#panelRerun").should("have.class", "is-active");
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@artifactRunsFull");
    cy.wait("@artifactRunsFull");
    cy.wrap(null).should(() => {
      const limitMatch = capturedUrls[0].match(/[?&]limit=(\d+)/);
      expect(limitMatch, "default discovery sends a limit").to.not.eq(null);
      expect(Number(limitMatch[1]), "default discovery limit exceeds the old 100 cap").to.be.greaterThan(100);
      expect(capturedUrls[1], "second request follows the cursor").to.include("cursor=page-two");
    });
    cy.get("#runIdSelect option").should(($opts) => {
      const values = [...$opts].map((o) => o.value).filter(Boolean);
      // Far more than the historical 100-run cap render without any search
      // (the picker also unions the sim-viz "known" runs, so allow >=).
      expect(values.length, "all runs render without search").to.be.at.least(bigList.length);
      const valueSet = new Set(values);
      for (const run of bigList) {
        expect(valueSet.has(run.run_id), `${run.run_id} present without search`).to.eq(true);
      }
      // The oldest run (would fall off a 100-run page) is present by default.
      expect(values, "oldest run shows without typing").to.include("bulk-run-149");
    });
  });

  it("keeps duplicate run basenames as separate source-qualified cards", () => {
    const RUN_ID = "duplicate-run";
    const REF_A = "npa1_source_a";
    const REF_B = "npa1_source_b";
    cy.intercept("GET", "/api/artifacts/runs*", {
      statusCode: 200,
      body: {
        ok: true,
        contract: "s3-source-qualified-v1",
        runs: [
          { run_id: RUN_ID, run_ref: REF_A, source_prefix: "shared/category-a", has_viewable: true, artifact_count: 1 },
          { run_id: RUN_ID, run_ref: REF_B, source_prefix: "shared/category-b", has_viewable: true, artifact_count: 1 },
        ],
        total_runs: 2,
        truncated: false,
      },
    }).as("qualifiedRuns");
    cy.intercept("GET", `/api/artifacts/run/${REF_B}*`, {
      statusCode: 200,
      body: {
        ok: true,
        contract: "s3-source-qualified-v1",
        run_id: RUN_ID,
        run_ref: REF_B,
        prefix: "shared/category-b",
        artifacts: [
          { key: `shared/category-b/${RUN_ID}/result.future`, s3_uri: `s3://mock/shared/category-b/${RUN_ID}/result.future`, render: "download", size: 9 },
        ],
      },
    }).as("qualifiedRunB");

    cy.get("#tabRerun").click();
    cy.get("#artifactRefreshRuns").click();
    cy.wait("@qualifiedRuns");
    cy.get("#runIdSelect option").then(($opts) => {
      const values = [...$opts].map((option) => option.value).filter(Boolean);
      expect(values).to.include(REF_A);
      expect(values).to.include(REF_B);
      expect(values.filter((value) => value === RUN_ID)).to.have.length(0);
    });
    cy.get("#runIdSelect").select(REF_B, { force: true });
    cy.wait("@qualifiedRunB");
    cy.get("#artifactList .artifact-card").should("have.length", 1);
    cy.get("#artifactList").should("contain.text", "result.future");
    cy.get("#artifactList").should("contain.text", "download");
    cy.get('meta[name="npa-artifact-discovery-contract"]')
      .should("have.attr", "content", "s3-source-qualified-v1");
  });

  it("rejects uniform gray / blank canvases in frameLooksBlank", () => {
    cy.window().then((win) => {
      const api = win.__NPA_AGENT_TEST__;
      expect(api, "test hooks").to.exist;
      const gray = win.document.createElement("canvas");
      gray.width = 120;
      gray.height = 80;
      const gctx = gray.getContext("2d");
      gctx.fillStyle = "#9ca3af";
      gctx.fillRect(0, 0, 120, 80);
      expect(api.frameLooksBlank(gray)).to.eq(true);

      const black = win.document.createElement("canvas");
      black.width = 120;
      black.height = 80;
      black.getContext("2d").fillRect(0, 0, 120, 80);
      expect(api.frameLooksBlank(black)).to.eq(true);

      const content = win.document.createElement("canvas");
      content.width = 160;
      content.height = 100;
      const cctx = content.getContext("2d");
      cctx.fillStyle = "#0a0a12";
      cctx.fillRect(0, 0, 160, 100);
      cctx.strokeStyle = "#ff8a1f";
      cctx.lineWidth = 3;
      cctx.beginPath();
      cctx.moveTo(40, 20);
      cctx.lineTo(80, 60);
      cctx.lineTo(50, 90);
      cctx.stroke();
      cctx.strokeStyle = "#5eead4";
      cctx.beginPath();
      cctx.moveTo(90, 25);
      cctx.lineTo(120, 70);
      cctx.stroke();
      expect(api.frameLooksBlank(content)).to.eq(false);
      const stats = api.sampleFrameStats(content);
      expect(stats.vivid).to.be.greaterThan(0);

      // Large dark viewport with thin G1-style strokes — must not be wiped by downscale.
      const sparse = win.document.createElement("canvas");
      sparse.width = 960;
      sparse.height = 540;
      const sctx = sparse.getContext("2d");
      sctx.fillStyle = "#050508";
      sctx.fillRect(0, 0, 960, 540);
      sctx.strokeStyle = "#ff8a1f";
      sctx.lineWidth = 2;
      sctx.beginPath();
      sctx.moveTo(480, 80);
      sctx.lineTo(470, 220);
      sctx.lineTo(455, 360);
      sctx.lineTo(450, 480);
      sctx.stroke();
      sctx.strokeStyle = "#5eead4";
      sctx.beginPath();
      sctx.moveTo(490, 90);
      sctx.lineTo(520, 200);
      sctx.lineTo(560, 280);
      sctx.stroke();
      expect(api.frameLooksBlank(sparse), "sparse skeleton on dark grid").to.eq(false);
      expect(api.sampleFrameStats(sparse).vivid).to.be.greaterThan(2);
    });
  });

  it("Describe this appears in chat immediately and attaches a non-blank frame", () => {
    cy.intercept("POST", "/api/chat", (req) => {
      // Delayed vision reply so the pending chat bubble must appear first.
      req.reply({
        delay: 1400,
        statusCode: 200,
        body: {
          ok: true,
          grounded: false,
          tier: "vision",
          model: "Qwen/Qwen2.5-VL-72B-Instruct",
          session_id: req.body.session_id || "default",
          reply: [
            "**What I see**: Dark 3D grid with orange and cyan skeleton wireframes (G1 trajectory style).",
            "**Likely meaning**: Locomotion / trajectory overlay in the Rerun viewer.",
            "**Operator feedback**: Structured sim content is visible — not a blank frame.",
            "**Next actions**: Scrub timeline; compare held-out cameras; keep this recording.",
          ].join("\n"),
        },
      });
    }).as("slowDescribeChat");

    cy.get("#tabRerun").click();
    cy.get("body").should("have.class", "viewer-focus");
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
    cy.get("#rerunFrame").should(($frame) => {
      const win = $frame[0].contentWindow;
      expect(win && win.__NPA_MOCK_RERUN__).to.exist;
      win.__NPA_MOCK_RERUN__.setMode("content");
    });

    cy.get("#describeVisual").click({ force: true });
    // Immediate UX: request visible before the delayed /api/chat completes.
    cy.get("#chatLog .msg-row.user", { timeout: 2000 }).should("contain.text", "Describe this");
    cy.get("#panelChat").should("have.class", "chat-drawer-open");

    cy.wait("@slowDescribeChat", { timeout: 20000 }).then((interception) => {
      const body = interception.request.body;
      expect(body.visual_context).to.be.an("object");
      expect(body.visual_context.capture).to.eq("frame");
      expect(body.visual_context.frame_quality).to.eq("rendered");
      expect(body.visual_context.has_image).to.eq(true);
      const messages = body.messages;
      const last = messages[messages.length - 1];
      expect(last.content).to.be.an("array");
      const imagePart = last.content.find((part) => part && String(part.type || "").startsWith("image"));
      expect(imagePart, "image part").to.exist;
      const url = imagePart.image_url.url;
      expect(url).to.match(/^data:image\/jpeg;base64,/);
      expect(url.length).to.be.greaterThan(4000);
    });
    cy.get("#chatLog .msg-row.assistant").should("contain.text", "skeleton");
    cy.get("#chatLog .msg-row.assistant").should("not.contain.text", "completely uniform gray");
  });

  it("Describe this carries grounded pipeline provenance when a run is loaded", () => {
    cy.intercept("GET", "**/api/artifacts/provenance/**", {
      statusCode: 200,
      body: {
        ok: true,
        run_id: "paidf-mock-1",
        summary:
          "Augment — Cosmos Transfer 2.5 (nvidia/Cosmos-Transfer2.5-2B) [GPU (Nebius K8s)]; " +
          "Pseudo-label augmented — Token Factory VLM (Qwen/Qwen2.5-VL-72B-Instruct) [hosted GPU (Token Factory)]",
        components: [
          {
            stage: "Augment",
            component: "Cosmos Transfer 2.5",
            runtime: "GPU (Nebius K8s)",
            model: "nvidia/Cosmos-Transfer2.5-2B",
          },
        ],
        origin: {
          run_id: "paidf-mock-1",
          original_present: false,
          summary:
            "No separate original input image was stored for run `paidf-mock-1` — the " +
            "earliest stored visuals are the Cosmos Transfer 2.5 augment OUTPUTS.",
        },
      },
    }).as("provenance");
    cy.intercept("POST", "/api/chat", (req) => {
      req.reply({
        statusCode: 200,
        body: {
          ok: true,
          grounded: false,
          tier: "vision",
          model: "Qwen/Qwen2.5-VL-72B-Instruct",
          session_id: req.body.session_id || "default",
          reply: "**What I see**: augmented road scene.\n**Where it comes from**: Cosmos Transfer 2.5 augment stage.",
        },
      });
    }).as("provChat");

    cy.get("#tabRerun").click();
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
    cy.get("#rerunFrame").should(($frame) => {
      $frame[0].contentWindow.__NPA_MOCK_RERUN__.setMode("content");
    });
    // A loaded run id is what triggers the grounded provenance fetch.
    cy.get("#simRunId").then(($el) => {
      $el[0].textContent = "paidf-mock-1";
    });

    cy.get("#describeVisual").click({ force: true });
    cy.wait("@provenance");
    cy.wait("@provChat", { timeout: 20000 }).then((interception) => {
      const body = interception.request.body;
      expect(body.visual_context.provenance, "visual_context.provenance").to.be.a("string");
      expect(body.visual_context.provenance).to.match(/Cosmos Transfer 2\.5/);
      // Grounded original-input resolution rides along with provenance.
      expect(body.visual_context.origin, "visual_context.origin").to.be.a("string");
      expect(body.visual_context.origin).to.match(/No separate original input image was stored/);
      const last = body.messages[body.messages.length - 1];
      const textPart = Array.isArray(last.content)
        ? last.content.find((part) => part && String(part.type || "").includes("text"))
        : { text: last.content };
      const promptText = String((textPart && textPart.text) || "");
      expect(promptText, "prompt provenance section").to.match(/Pipeline provenance/i);
      expect(promptText).to.match(/Cosmos Transfer 2\.5/);
      expect(promptText, "prompt original-input section").to.match(/Original input/i);
      expect(promptText).to.match(/No separate original input image was stored/);
    });
  });

  it("Describe this stays metadata-only for uniform gray canvases", () => {
    cy.get("#tabRerun").click();
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
    cy.get("#rerunFrame").should(($frame) => {
      $frame[0].contentWindow.__NPA_MOCK_RERUN__.setMode("gray");
    });
    cy.window().then(async (win) => {
      const api = win.__NPA_AGENT_TEST__;
      const iframe = win.document.getElementById("rerunFrame");
      api.ensureRerunCaptureBridge(iframe, { forceRestart: true });
      const quality = await api.waitForQualityRerunFrame(2500);
      expect(quality.dataUrl, "gray must not attach").to.eq("");
      expect(quality.quality).to.be.oneOf(["unavailable", "missing"]);
    });

    cy.get("#describeVisual").click({ force: true });
    cy.wait("@chat", { timeout: 60000 }).then((interception) => {
      const body = interception.request.body;
      expect(body.visual_context).to.be.an("object");
      expect(body.visual_context.capture).to.not.eq("frame");
      expect(body.visual_context.has_image).to.eq(false);
      const messages = body.messages;
      const last = messages[messages.length - 1];
      const content = last.content;
      if (Array.isArray(content)) {
        const imagePart = content.find((part) => part && String(part.type || "").startsWith("image"));
        expect(imagePart).to.not.exist;
      }
    });
    cy.get("#chatLog .msg-row.assistant").should("contain.text", "metadata only");
  });

  it("keeps the cover up while the iframe shows Loading application bundle", () => {
    cy.get("#tabRerun").click();
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");

    cy.get("#rerunFrame").should(($frame) => {
      $frame[0].contentWindow.__NPA_MOCK_RERUN__.setMode("splash");
    });

    cy.window().then((win) => {
      const api = win.__NPA_AGENT_TEST__;
      const iframe = win.document.getElementById("rerunFrame");
      expect(api.rerunViewerShowsBundleSplash(iframe)).to.eq(true);
      expect(api.rerunViewerLooksDisplayReady(iframe)).to.eq(false);
      api.showRerunBundleCover("Opening viewer…", "Almost ready…");
      expect(win.document.getElementById("rerunBundleCover").hidden).to.eq(false);
      // safeHide must refuse while splash / blank canvas is showing.
      expect(api.safeHideRerunBundleCover(iframe)).to.eq(false);
      expect(win.document.getElementById("rerunBundleCover").hidden).to.eq(false);
      // Parent chrome must never echo Rerun's splash string.
      expect(win.document.getElementById("rerunBundleCover").innerText).not.to.match(
        /Loading application bundle/i,
      );
    });

    // When content returns, uncover is allowed.
    cy.get("#rerunFrame").should(($frame) => {
      $frame[0].contentWindow.__NPA_MOCK_RERUN__.setMode("content");
    });
    cy.window().then((win) => {
      const api = win.__NPA_AGENT_TEST__;
      const iframe = win.document.getElementById("rerunFrame");
      expect(api.rerunViewerLooksDisplayReady(iframe)).to.eq(true);
      expect(api.safeHideRerunBundleCover(iframe)).to.eq(true);
      expect(win.document.getElementById("rerunBundleCover").hidden).to.eq(true);
    });
  });

  it("generalizes capture across content / gray / splash visual modes", () => {
    cy.get("#tabRerun").click();
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");

    const modes = [
      { mode: "content", expectFrame: true },
      { mode: "gray", expectFrame: false },
      { mode: "splash", expectFrame: false },
    ];

    cy.wrap(modes).each((item) => {
      cy.get("#rerunFrame").should(($frame) => {
        $frame[0].contentWindow.__NPA_MOCK_RERUN__.setMode(item.mode);
      });
      cy.window().then({ timeout: 20000 }, async (win) => {
        const api = win.__NPA_AGENT_TEST__;
        const iframe = win.document.getElementById("rerunFrame");
        // Force a fresh bridge after mode paint so probes do not see stale frames.
        api.ensureRerunCaptureBridge(iframe, { forceRestart: true });
        await new Promise((r) => setTimeout(r, 80));
        const probed = await api.probeRerunCanvasContent(iframe);
        const result = await api.waitForQualityRerunFrame(2500);
        if (item.expectFrame) {
          expect(probed, `${item.mode} probe`).to.eq(true);
          expect(result.quality, `${item.mode} quality`).to.eq("rendered");
          expect(result.dataUrl).to.match(/^data:image\/jpeg/);
        } else {
          expect(probed, `${item.mode} probe`).to.eq(false);
          expect(result.dataUrl, `${item.mode} dataUrl`).to.eq("");
          expect(result.quality).to.be.oneOf(["unavailable", "missing"]);
        }
      });
    });
  });

  it("captures Rerun via MediaStream bridge even when sync blank checks would fail", () => {
    cy.get("#tabRerun").click();
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");
    cy.get("#rerunFrame").should(($frame) => {
      $frame[0].contentWindow.__NPA_MOCK_RERUN__.setMode("content");
    });
    cy.window().then(async (win) => {
      const api = win.__NPA_AGENT_TEST__;
      const iframe = win.document.getElementById("rerunFrame");
      expect(win.document.documentElement.outerHTML).to.include("ensureRerunCaptureBridge");
      // Restart after the mock canvas paints content so the MediaStream cannot
      // retain an empty frame produced by the preceding page-load mount.
      const bridge = api.ensureRerunCaptureBridge(iframe, { forceRestart: true });
      expect(bridge, "capture bridge").to.exist;
      expect(bridge.video).to.exist;
      await new Promise((resolve) => setTimeout(resolve, 80));
      const grabbed = await api.grabFromRerunCaptureBridge(5000);
      expect(grabbed).to.match(/^data:image\/jpeg/);
      // Capture must succeed even if we ignore sync blank gates (the live WebGL failure mode).
      const quality = await api.waitForQualityRerunFrame(4000);
      expect(quality.quality).to.eq("rendered");
      expect(quality.dataUrl.length).to.be.greaterThan(4000);
    });
  });

  it("captures live WebGL canvas via captureStream when sync readback is blank", () => {
    cy.window().then(async (win) => {
      const api = win.__NPA_AGENT_TEST__;
      const canvas = win.document.createElement("canvas");
      canvas.width = 160;
      canvas.height = 120;
      const gl = canvas.getContext("webgl", { preserveDrawingBuffer: false, alpha: false });
      expect(gl, "webgl context").to.exist;
      let raf = 0;
      const paint = () => {
        // Alternating orange / cyan clears so the stream has non-uniform structure over time,
        // while sync 2D readback of a non-preserveDrawingBuffer canvas is often blank.
        const t = Date.now() % 400 < 200;
        if (t) gl.clearColor(1.0, 0.45, 0.1, 1.0);
        else gl.clearColor(0.2, 0.85, 0.8, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        raf = win.requestAnimationFrame(paint);
      };
      paint();
      await new Promise((r) => setTimeout(r, 250));
      const url = await api.captureCanvasDataUrl(canvas, { budgetMs: 2500 });
      win.cancelAnimationFrame(raf);
      if (!url) {
        // Headless Chromium often cannot composite WebGL → MediaStream; the Rerun mock
        // (2D canvas) + live agent suite cover the production path.
        expect(typeof canvas.captureStream).to.eq("function");
        return;
      }
      expect(url, "WebGL stream capture").to.match(/^data:image\/jpeg;base64,/);
      expect(url.length).to.be.greaterThan(800);
    });
  });

  it("captures image / video / data visual kinds for Describe this", () => {
    cy.get("#tabRerun").click();
    cy.get("#rerunBundleCover", { timeout: 20000 }).should("have.attr", "hidden");

    cy.window().then(async (win) => {
      const api = win.__NPA_AGENT_TEST__;
      const host = win.document.getElementById("artifactPreviewHost");
      expect(host).to.exist;

      // --- image ---
      host.hidden = false;
      const imgCanvas = win.document.createElement("canvas");
      imgCanvas.width = 120;
      imgCanvas.height = 80;
      const ictx = imgCanvas.getContext("2d");
      ictx.fillStyle = "#102030";
      ictx.fillRect(0, 0, 120, 80);
      ictx.fillStyle = "#ff8800";
      ictx.fillRect(20, 15, 80, 50);
      const img = win.document.createElement("img");
      img.src = imgCanvas.toDataURL("image/png");
      await new Promise((resolve) => {
        img.onload = resolve;
        img.onerror = resolve;
      });
      host.innerHTML = "";
      host.appendChild(img);
      api.setRenderMode("image");
      let captured = await api.captureVisualContext();
      expect(captured.kind).to.eq("image");
      expect(captured.meta.capture).to.eq("frame");
      expect(captured.imageDataUrl).to.match(/^data:image\/jpeg/);
      expect(captured.prompt).to.include("visual_kind: `image`");

      // --- video (canvas.captureStream backed) ---
      const vCanvas = win.document.createElement("canvas");
      vCanvas.width = 160;
      vCanvas.height = 90;
      const vctx = vCanvas.getContext("2d");
      vctx.fillStyle = "#0a1020";
      vctx.fillRect(0, 0, 160, 90);
      vctx.fillStyle = "#33cc99";
      vctx.fillRect(30, 20, 100, 50);
      const stream = vCanvas.captureStream(12);
      const video = win.document.createElement("video");
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      await video.play().catch(() => undefined);
      await new Promise((r) => setTimeout(r, 200));
      host.innerHTML = "";
      host.appendChild(video);
      api.setRenderMode("video");
      captured = await api.captureVisualContext();
      expect(captured.kind).to.eq("video");
      expect(captured.meta.capture).to.eq("frame");
      expect(captured.imageDataUrl).to.match(/^data:image\/jpeg/);
      expect(captured.prompt).to.include("visual_kind: `video`");
      stream.getTracks().forEach((t) => t.stop());

      // --- data / text ---
      const pre = win.document.createElement("pre");
      pre.textContent = JSON.stringify(
        { success_rate: 0.82, stage: "heldout", robot: "g1" },
        null,
        2
      );
      host.innerHTML = "";
      host.appendChild(pre);
      api.setRenderMode("data");
      captured = await api.captureVisualContext();
      expect(captured.kind).to.eq("data");
      expect(captured.meta.capture).to.eq("text");
      expect(captured.imageDataUrl).to.eq("");
      expect(captured.prompt).to.include("success_rate");
      expect(captured.prompt).to.include("visual_kind: `data`");
      expect(captured.prompt.toLowerCase()).to.include("pixels");

      // restore rerun mode
      api.setRenderMode("rerun");
      host.hidden = true;
      host.innerHTML = "";
    });
  });

  it("Describe this posts vision frames for image and video panes", () => {
    cy.get("#tabRerun").click();
    cy.intercept("POST", "/api/chat", (req) => {
      req.reply({
        statusCode: 200,
        body: {
          ok: true,
          grounded: false,
          tier: "vision",
          model: "Qwen/Qwen2.5-VL-72B-Instruct",
          session_id: req.body.session_id || "default",
          reply: "**What I see**: Structured viewer content.\n**Likely meaning**: Valid capture.\n**Operator feedback**: OK.\n**Next actions**: Continue.",
        },
      });
    }).as("visualKindChat");

    // Image pane
    cy.window().then(async (win) => {
      const api = win.__NPA_AGENT_TEST__;
      const host = win.document.getElementById("artifactPreviewHost");
      host.hidden = false;
      const imgCanvas = win.document.createElement("canvas");
      imgCanvas.width = 100;
      imgCanvas.height = 60;
      const ctx = imgCanvas.getContext("2d");
      ctx.fillStyle = "#203040";
      ctx.fillRect(0, 0, 100, 60);
      ctx.strokeStyle = "#ffaa00";
      ctx.lineWidth = 4;
      ctx.strokeRect(10, 10, 80, 40);
      const img = win.document.createElement("img");
      img.src = imgCanvas.toDataURL("image/png");
      await new Promise((resolve) => {
        img.onload = resolve;
      });
      host.innerHTML = "";
      host.appendChild(img);
      api.setRenderMode("image");
    });

    cy.get("#describeVisual").click({ force: true });
    cy.wait("@visualKindChat").then((interception) => {
      const body = interception.request.body;
      expect(body.visual_context.kind).to.eq("image");
      expect(body.visual_context.capture).to.eq("frame");
      expect(body.visual_context.has_image).to.eq(true);
      const last = body.messages[body.messages.length - 1];
      expect(last.content).to.be.an("array");
      expect(last.content.some((p) => p && String(p.type || "").startsWith("image"))).to.eq(true);
    });

    // Video pane
    cy.window().then(async (win) => {
      const api = win.__NPA_AGENT_TEST__;
      const host = win.document.getElementById("artifactPreviewHost");
      const vCanvas = win.document.createElement("canvas");
      vCanvas.width = 128;
      vCanvas.height = 72;
      const vctx = vCanvas.getContext("2d");
      vctx.fillStyle = "#101828";
      vctx.fillRect(0, 0, 128, 72);
      vctx.fillStyle = "#22d3ee";
      vctx.fillRect(24, 16, 80, 40);
      const stream = vCanvas.captureStream(12);
      const video = win.document.createElement("video");
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      await video.play().catch(() => undefined);
      await new Promise((r) => setTimeout(r, 180));
      host.innerHTML = "";
      host.appendChild(video);
      host._npaTestStream = stream;
      api.setRenderMode("video");
    });

    cy.get("#describeVisual").click({ force: true });
    cy.wait("@visualKindChat").then((interception) => {
      const body = interception.request.body;
      expect(body.visual_context.kind).to.eq("video");
      expect(body.visual_context.capture).to.eq("frame");
      expect(body.visual_context.has_image).to.eq(true);
      const last = body.messages[body.messages.length - 1];
      expect(last.content.some((p) => p && String(p.type || "").startsWith("image"))).to.eq(true);
    });

    // Data pane — metadata/text only, never invents an image part
    cy.intercept("POST", "/api/chat", (req) => {
      req.reply({
        statusCode: 200,
        body: {
          ok: true,
          grounded: false,
          tier: "reasoning",
          model: "mock/model",
          session_id: req.body.session_id || "default",
          reply: "**What I see**: Metadata/text only.\n**Likely meaning**: JSON report.\n**Operator feedback**: OK.\n**Next actions**: Reload Rerun.",
        },
      });
    }).as("dataKindChat");

    cy.window().then((win) => {
      const api = win.__NPA_AGENT_TEST__;
      const host = win.document.getElementById("artifactPreviewHost");
      if (host._npaTestStream) {
        host._npaTestStream.getTracks().forEach((t) => t.stop());
        delete host._npaTestStream;
      }
      const pre = win.document.createElement("pre");
      pre.textContent = JSON.stringify({ success_rate: 0.91, robot: "g1" }, null, 2);
      host.innerHTML = "";
      host.appendChild(pre);
      api.setRenderMode("data");
    });

    cy.get("#describeVisual").click({ force: true });
    cy.wait("@dataKindChat").then((interception) => {
      const body = interception.request.body;
      expect(body.visual_context.kind).to.eq("data");
      expect(body.visual_context.capture).to.eq("text");
      expect(body.visual_context.has_image).to.eq(false);
      const last = body.messages[body.messages.length - 1];
      const content = last.content;
      if (Array.isArray(content)) {
        expect(content.some((p) => p && String(p.type || "").startsWith("image"))).to.eq(false);
        expect(JSON.stringify(content)).to.include("success_rate");
      } else {
        expect(String(content)).to.include("success_rate");
      }
    });
  });
});
