const defaultLeIsaacConfiguration = () => ({
  schema: "npa.leisaac.configuration.v1",
  robot: {
    id: "so101_follower",
    display_name: "Built-in SO-101 follower robot",
    source: "built-in-runtime",
  },
  scene: {
    id: "table_with_cube",
    display_name: "Built-in table and lift-cube scene",
    source: "built-in-runtime",
  },
  device: {
    id: "browser_keyboard_so101",
    display_name: "Browser keyboard SO-101 teleoperator (default test device)",
    source: "built-in-runtime",
  },
  task: {
    id: "LeIsaac-SO101-LiftCube-v0",
    display_name: "SO101 Lift Cube",
    source: "built-in-registry",
  },
  custom_bundle_count: 0,
});

const leisaacModeStatus = (viewMode = "single_fast", recordingMode = "primary_only") => ({
  view_mode_contract: {
    default: "single_fast",
    values: ["single_fast", "dual_slow"],
    labels: { single_fast: "Fast single", dual_slow: "Dual view — slower" },
  },
  recording_camera_contract: {
    default: "primary_only",
    values: ["primary_only", "primary_and_secondary"],
    labels: { primary_only: "Primary only", primary_and_secondary: "Primary + secondary" },
  },
  requested_view_mode: viewMode,
  applied_view_mode: viewMode,
  requested_recording_camera_mode: recordingMode,
  applied_recording_camera_mode: recordingMode,
  view_revision: 0,
  applied_view_revision: 0,
  recording_revision: 0,
  applied_recording_revision: 0,
});

const grantMockControllerLease = () => cy.window().then((win) =>
  win.__NPA_AGENT_TEST__.setLeIsaacControllerLeaseForTest(true),
);

describe("NPA agent LeIsaac capability tab", () => {
  beforeEach(() => {
    cy.intercept("POST", "/api/leisaac/ws-session*", {
      statusCode: 204,
    }).as("wsSession");
    cy.visitMockAgent();
    cy.wait("@session");
  });

  it("stays mounted with retry state when live capability is unavailable", () => {
    cy.intercept("GET", "/api/leisaac/status", {
      statusCode: 200,
      body: {
        available: false,
        episodes_available: false,
        run_id: "",
        reason: "No LeIsaac runtime is registered with this agent.",
      },
    }).as("readinessRetry");
    cy.get("#tabLeIsaac", { timeout: 10000 }).should("exist").click();
    cy.get("#panelLeIsaac").should("have.class", "is-active");
    cy.get("#leisaacReadiness")
      .should("be.visible")
      .and("have.attr", "data-contract", "LEISAAC_CONTROL_READINESS_CONTRACT");
    cy.get("#leisaacRetry").should("be.visible").and("not.be.disabled").focus();
    cy.focused().should("have.id", "leisaacRetry");
    cy.get("#leisaacConnect, #leisaacLiveGrid, #leisaacRecordStart, #leisaacResetDefaults")
      .should("not.exist");
    cy.get("#leisaacReadinessSummary").should("contain.text", "not displayed");
    cy.get("#leisaacPrerequisitesTitle").should("contain.text", "Operator readiness checklist");
    cy.get(".leisaac-launch-template")
      .should("contain.text", "npa workbench leisaac launch")
      .and("contain.text", "<project-alias>");
    cy.get("#panelLeIsaac")
      .should("contain.text", "single public ACCEPT_EULA input")
      .and("contain.text", "This page never changes that input or privacy/telemetry settings");
    cy.get("#leisaacRetry").click();
    cy.wait("@readinessRetry");
    cy.get("#leisaacAvailability").should("contain.text", "No LeIsaac runtime is registered");
  });

  it("keeps no-session readiness usable at a mobile viewport", () => {
    cy.viewport(390, 844);
    cy.get("#tabLeIsaac").click();
    cy.get("#leisaacReadiness").should("be.visible");
    cy.get("#leisaacRetry").should("be.visible").and("not.be.disabled");
    cy.get("#leisaacConnect, #leisaacLiveGrid, #leisaacRecordStart").should("not.exist");
    cy.window().then((win) => {
      expect(win.document.documentElement.scrollWidth).to.be.at.most(win.innerWidth);
    });
  });

  it("mounts before a slow capability request can block first paint", () => {
    cy.intercept("GET", "/api/leisaac/status*", {
      delay: 5000,
      statusCode: 200,
      body: {
        available: false,
        episodes_available: false,
        run_id: "",
        reason: "delayed unavailable capability",
      },
    }).as("slowLeIsaacStatus");
    cy.reload();
    cy.get("#tabLeIsaac", { timeout: 1000 }).should("be.visible").click();
    cy.get("#panelLeIsaac").should("have.class", "is-active");
    cy.get("#leisaacConnect").should("not.exist");
    cy.get("#leisaacRetry").should("be.visible").and("not.be.disabled");
    cy.get("#leisaacAvailability").should("contain.text", "No LeIsaac runtime is registered");
  });

  it("shows and preserves four real defaults for a completely fresh client", () => {
    const status = {
      available: true,
      episodes_available: true,
      run_id: "mock-run",
      task: "LeIsaac-SO101-LiftCube-v0",
      robot: "so101_follower",
      scene: "table_with_cube",
      device: "browser_keyboard_so101",
      configuration: defaultLeIsaacConfiguration(),
      task_registry: {
        default_task: "LeIsaac-SO101-LiftCube-v0",
        tasks: [
          { task: "LeIsaac-SO101-PickOrange-v0", display_name: "SO101 Pick Orange" },
          { task: "LeIsaac-SO101-LiftCube-v0", display_name: "SO101 Lift Cube" },
        ],
      },
      environment_id: "operator-0",
      environment_index: 0,
      seed: 42,
      dataset_uri: "s3://bucket/datasets/leisaac",
      ...leisaacModeStatus(),
      cameras: ["workspace"],
      stream_transport: "websocket-v1",
      control_ws_url: "/api/leisaac/transport/control?run_id=mock-run",
      video_ws_url: "/api/leisaac/transport/video?run_id=mock-run",
      bundle_reset_url: "/api/leisaac/bundles/reset?run_id=mock-run",
      recorder: { state: "idle", completed_episode_count: 0 },
      gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    };
    cy.intercept("GET", "/api/leisaac/status*", { statusCode: 200, body: status }).as("defaultStatus");
    cy.clearLocalStorage();
    cy.window().then((win) => win.sessionStorage.clear());
    cy.reload();
    cy.wait("@defaultStatus");
    cy.get("#tabLeIsaac", { timeout: 10000 }).click();
    cy.get("#leisaacRobotSelection").should("have.attr", "data-value", "so101_follower");
    cy.get("#leisaacSceneSelection").should("have.attr", "data-value", "table_with_cube");
    cy.get("#leisaacDeviceSelection").should("have.attr", "data-value", "browser_keyboard_so101");
    cy.get("#leisaacTaskSelection").should("have.attr", "data-value", "LeIsaac-SO101-LiftCube-v0");
    cy.get("#leisaacInputDevice")
      .should("have.value", "keyboard")
      .find("option:selected")
      .should("contain.text", "default test device");
    cy.get("#leisaacConnect").should("not.be.disabled");
    cy.get("#leisaacSendNeutralAction")
      .should("be.disabled")
      .and("have.attr", "aria-disabled", "true")
      .and("have.attr", "aria-describedby", "leisaacControllerPrerequisite");
    cy.get("#leisaacControllerPrerequisite").should("contain.text", "Connect teleoperation");
    cy.get("#leisaacCanvas").should("exist");
    cy.get("#leisaacViewMode").should("have.value", "single_fast");
    cy.get("#leisaacLiveGrid").should("have.class", "is-single");
    cy.get("#leisaacStreamHost").should("have.attr", "data-orbit-bound", "1");
    cy.get("#leisaacSecondaryHost").should("not.be.visible");
    cy.get("#leisaacSecondaryFrame").should("not.have.attr", "src");
    cy.get("#leisaacModeWarning").should("contain.text", "no secondary capture");
    cy.get("#panelLeIsaac").then(($panel) => {
      cy.window().then((win) => win.__NPA_AGENT_TEST__.refreshLeIsaacCapability("mock-run"));
      cy.wait("@defaultStatus");
      cy.get("#panelLeIsaac").then(($refreshed) => {
        expect($refreshed[0], "same single-viewport panel survives polling").to.equal($panel[0]);
      });
    });
  });

  it("reconciles mode options when an authoritative refresh follows a summary mount", () => {
    const status = {
      available: true,
      episodes_available: false,
      run_id: "mock-mode-refresh",
      task: "LeIsaac-SO101-LiftCube-v0",
      environment_id: "operator-0",
      environment_index: 0,
      seed: 42,
      configuration: defaultLeIsaacConfiguration(),
      ...leisaacModeStatus(),
      cameras: ["workspace"],
      stream_transport: "websocket-v1",
      recorder: { state: "idle", completed_episode_count: 0 },
      gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    };
    cy.intercept("GET", "/api/leisaac/status*", { statusCode: 200, body: status }).as("modeRefresh");
    cy.window().then((win) => win.__NPA_AGENT_TEST__.refreshLeIsaacCapability(status.run_id));
    cy.wait("@modeRefresh");
    cy.get("#tabLeIsaac").click();
    cy.get("#leisaacViewMode").then(($select) => {
      $select.empty().prop("disabled", true);
    });
    cy.get("#leisaacRecordingCameras").then(($select) => {
      $select.empty().prop("disabled", true);
    });
    cy.window().then((win) => win.__NPA_AGENT_TEST__.refreshLeIsaacCapability(status.run_id));
    cy.wait("@modeRefresh");
    cy.get("#leisaacViewMode").should("be.disabled");
    cy.get("#leisaacRecordingCameras").should("be.disabled");
    grantMockControllerLease();
    cy.get("#leisaacViewMode")
      .should("not.be.disabled")
      .and("have.value", "single_fast")
      .find("option")
      .should("have.length", 2);
    cy.get("#leisaacRecordingCameras")
      .should("not.be.disabled")
      .and("have.value", "primary_only")
      .find("option")
      .should("have.length", 2);
  });

  it("keeps a second browser locked and exposes controls only after its lease is active", () => {
    const status = {
      available: true,
      episodes_available: false,
      run_id: "mock-controller-lease",
      task: "LeIsaac-SO101-LiftCube-v0",
      configuration: defaultLeIsaacConfiguration(),
      ...leisaacModeStatus(),
      cameras: ["workspace"],
      stream_transport: "websocket-v1",
      recorder: { state: "idle", completed_episode_count: 0 },
    };
    cy.intercept("GET", "/api/leisaac/status*", { statusCode: 200, body: status }).as("leaseStatus");
    cy.window().then((win) => win.__NPA_AGENT_TEST__.refreshLeIsaacCapability(status.run_id));
    cy.wait("@leaseStatus");
    cy.get("#tabLeIsaac").click();
    cy.get("#leisaacConnect").should("be.visible").and("not.be.disabled");
    cy.window().then((win) => win.__NPA_AGENT_TEST__.setLeIsaacControllerLeaseForTest(
      false,
      "Another authenticated browser currently holds the controller lease. Controller actions remain locked here.",
    ));
    cy.get("#leisaacControllerPrerequisite").should("contain.text", "Another authenticated browser");
    cy.get("#leisaacViewMode, #leisaacRecordingCameras, #leisaacRecordStart, #leisaacSendNeutralAction, #leisaacResetDefaults")
      .each(($control) => {
        expect($control).to.be.disabled;
        expect($control).to.have.attr("aria-disabled", "true");
        expect($control.attr("aria-describedby")).to.contain("leisaacControllerPrerequisite");
      });
    cy.get("#leisaacStreamHost")
      .should("have.attr", "tabindex", "-1")
      .and("have.attr", "aria-disabled", "true");
    cy.get("#leisaacConnect")
      .should("not.be.disabled")
      .and("contain.text", "Retry controller lease");

    grantMockControllerLease();
    cy.get("#leisaacControllerPrerequisite").should("contain.text", "Controller lease active");
    cy.get("#leisaacViewMode, #leisaacRecordingCameras, #leisaacRecordStart, #leisaacSendNeutralAction, #leisaacResetDefaults")
      .each(($control) => {
        expect($control).not.to.be.disabled;
        expect($control).to.have.attr("aria-disabled", "false");
      });
    cy.get("#leisaacRecordSuccess, #leisaacRecordFailure, #leisaacRecordFinalize")
      .should("be.disabled");
    cy.get("#leisaacStreamHost")
      .should("have.attr", "tabindex", "0")
      .and("have.attr", "aria-disabled", "false");
  });

  it("shows task/environment metadata and enforces recorder transitions", () => {
    let recorderState = "idle";
    let pendingOutcome = "";
    let completed = 1;
    let lastOutcome = "failure";
    let lastCommandId = "";
    let lastCommand = "";
    const statusBody = () => ({
      available: true,
      run_id: "mock-recorder",
      task: "LeIsaac-SO101-LiftCube-v0",
      environment_id: "table-b",
      environment_index: 1,
      seed: 43,
      dataset_uri: "s3://bucket/datasets/leisaac",
      stream_transport: "jpeg-poll",
      frame_url: "/api/leisaac/frame.jpg?run_id=mock-recorder",
      input_url: "/api/leisaac/input?run_id=mock-recorder",
      recorder_url: "/api/leisaac/recorder?run_id=mock-recorder",
      recorder: {
        state: recorderState,
        active_episode: recorderState === "idle" ? null : "episode-uuid",
        frame_count: recorderState === "idle" ? 0 : 7,
        completed_episode_count: completed,
        pending_outcome: pendingOutcome,
        last_outcome: lastOutcome,
        last_upload_status: completed > 1 ? "uploaded" : "never",
        last_error: "",
        last_command_id: lastCommandId,
        last_command: lastCommand,
        command_revision: completed,
        dataset_version_uri:
          completed > 1 ? "s3://bucket/datasets/leisaac/versions/test" : "",
        last_episode_commit_uri:
          completed > 1
            ? "s3://bucket/datasets/leisaac/commits/latest.json"
            : "",
      },
      gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    });
    cy.intercept("GET", "/api/leisaac/status?run_id=mock-recorder", (req) =>
      req.reply(statusBody()),
    ).as("recorderStatus");
    cy.intercept("POST", "/api/leisaac/select", {
      statusCode: 200,
      body: { selected: true },
    });
    cy.intercept(
      "POST",
      "/api/leisaac/recorder?run_id=mock-recorder",
      (req) => {
        expect(req.headers["x-npa-leisaac-control"]).to.equal("1");
        expect(req.headers["x-npa-leisaac-client-id"]).to.match(/^[A-Za-z0-9._:-]+$/);
        expect(req.headers["x-npa-leisaac-lease-id"]).to.equal("a".repeat(64));
        expect(req.body.request_id).to.be.a("string").and.not.be.empty;
        if (req.body.command === "start" && recorderState === "idle")
          recorderState = "recording";
        else if (
          req.body.command === "mark-success" &&
          recorderState === "recording"
        ) {
          recorderState = "outcome-pending";
          pendingOutcome = "success";
        } else if (
          req.body.command === "mark-failure" &&
          recorderState === "recording"
        ) {
          recorderState = "outcome-pending";
          pendingOutcome = "failure";
        } else if (
          req.body.command === "finalize" &&
          recorderState === "outcome-pending"
        ) {
          recorderState = "idle";
          lastOutcome = pendingOutcome;
          pendingOutcome = "";
          completed += 1;
        } else {
          req.reply({
            statusCode: 409,
            body: { detail: "invalid transition" },
          });
          return;
        }
        lastCommandId = req.body.request_id;
        lastCommand = req.body.command;
        req.reply({ statusCode: 202, body: { accepted: true } });
      },
    ).as("recorderControl");

    cy.window().then((win) =>
      win.__NPA_AGENT_TEST__.refreshLeIsaacCapability("mock-recorder"),
    );
    cy.wait("@recorderStatus");
    cy.get("#tabLeIsaac").click();
    grantMockControllerLease();
    cy.get("#panelLeIsaac").should("contain.text", "LeIsaac-SO101-LiftCube-v0");
    cy.get("#panelLeIsaac").should("contain.text", "table-b [1]");
    cy.get("#panelLeIsaac").should(
      "contain.text",
      "s3://bucket/datasets/leisaac",
    );
    cy.get("#leisaacRecordStart").should("not.be.disabled");
    cy.get("#leisaacRecordSuccess")
      .should("be.disabled")
      .and("have.attr", "title")
      .and("contain", "Start an episode");
    cy.get("#leisaacRecordFinalize").should("be.disabled");
    cy.get("#leisaacRecorderGuidance").should(
      "contain.text",
      "Start an episode",
    );
    cy.get("#leisaacRecordStart").then(($button) => {
      $button[0].click();
      $button[0].click();
    });
    cy.wait("@recorderControl");
    cy.get("@recorderControl.all").should("have.length", 1);
    cy.get("#leisaacRecorderStatus").should("contain.text", "State: recording");
    cy.get("#leisaacRecordingCameras").should("be.disabled");
    cy.get("#leisaacRecordSuccess").should("not.be.disabled").click();
    cy.wait("@recorderControl");
    cy.get("#leisaacRecorderGuidance").should(
      "contain.text",
      "Outcome selected: success",
    );
    cy.get("#leisaacRecordFinalize").should("not.be.disabled").click();
    cy.wait("@recorderControl");
    cy.get("#leisaacRecorderStatus").should("contain.text", "completed: 2");
    cy.get("#leisaacRecordingCameras").should("not.be.disabled");
    cy.get("#leisaacRecorderArtifact").should(
      "contain.text",
      "Immutable dataset",
    );

    cy.get("#leisaacRecordStart").should("not.be.disabled").click();
    cy.wait("@recorderControl");
    cy.get("#leisaacRecordFailure").should("not.be.disabled").click();
    cy.wait("@recorderControl");
    cy.get("#leisaacRecorderGuidance").should(
      "contain.text",
      "Outcome selected: failure",
    );
    cy.get("#leisaacRecordFinalize").should("not.be.disabled").click();
    cy.wait("@recorderControl");
    cy.get("#leisaacRecorderStatus")
      .should("contain.text", "State: idle")
      .and("contain.text", "completed: 3")
      .and("contain.text", "failure/uploaded");
    cy.screenshot("leisaac-recorder-transition");
  });

  it("surfaces a failed upload and retries the same episode", () => {
    let recorderState = "outcome-pending";
    let attempts = 0;
    let lastCommandId = "";
    cy.intercept("GET", "/api/leisaac/status?run_id=mock-retry", (req) => {
      req.reply({
        available: true,
        run_id: "mock-retry",
        task: "LeIsaac-SO101-LiftCube-v0",
        environment_id: "counter-a",
        environment_index: 0,
        seed: 42,
        dataset_uri: "s3://bucket/datasets/retry",
        recorder_url: "/api/leisaac/recorder?run_id=mock-retry",
        recorder: {
          state: recorderState,
          active_episode: recorderState === "idle" ? null : "retry-episode",
          frame_count: recorderState === "idle" ? 0 : 8,
          completed_episode_count: recorderState === "idle" ? 1 : 0,
          pending_outcome: recorderState === "idle" ? "" : "failure",
          last_outcome: recorderState === "idle" ? "failure" : "",
          last_upload_status:
            recorderState === "upload-failed"
              ? "failed"
              : recorderState === "idle"
                ? "uploaded"
                : "recording",
          last_error:
            recorderState === "upload-failed"
              ? "temporary object-store failure"
              : "",
          last_command_id: lastCommandId,
          last_command: lastCommandId ? "finalize" : "",
          dataset_version_uri:
            recorderState === "idle"
              ? "s3://bucket/datasets/retry/versions/v1"
              : "",
        },
      });
    }).as("retryStatus");
    cy.intercept("POST", "/api/leisaac/select", {
      statusCode: 200,
      body: { selected: true },
    });
    cy.intercept("POST", "/api/leisaac/recorder?run_id=mock-retry", (req) => {
      attempts += 1;
      lastCommandId = req.body.request_id;
      if (attempts === 1) {
        recorderState = "upload-failed";
        req.reply({
          statusCode: 202,
          body: { accepted: true, request_id: lastCommandId },
        });
      } else {
        recorderState = "idle";
        req.reply({
          statusCode: 202,
          body: { accepted: true, request_id: lastCommandId },
        });
      }
    }).as("retryControl");

    cy.window().then((win) =>
      win.__NPA_AGENT_TEST__.refreshLeIsaacCapability("mock-retry"),
    );
    cy.wait("@retryStatus");
    cy.get("#tabLeIsaac").click();
    grantMockControllerLease();
    cy.get("#leisaacRecordFinalize").should("not.be.disabled").click();
    cy.wait("@retryControl");
    cy.get("#leisaacRecorderError").should(
      "contain.text",
      "temporary object-store failure",
    );
    cy.window().then((win) => win.__NPA_AGENT_TEST__.pollLeIsaacRecorder());
    cy.get("#leisaacRecorderStatus").should(
      "contain.text",
      "State: upload-failed",
    );
    cy.get("#leisaacRecordFinalize")
      .should("not.be.disabled")
      .and("have.attr", "title")
      .and("contain", "Retry");
    cy.get("#leisaacRecordFinalize").click();
    cy.wait("@retryControl");
    cy.get("#leisaacRecorderStatus")
      .should("contain.text", "State: idle")
      .and("contain.text", "completed: 1");
  });

  it("unlocks controls after a rejected recorder request and allows retry", () => {
    let recorderState = "idle";
    let lastCommandId = "";
    const status = () => ({
      available: true,
      run_id: "mock-network-retry",
      task: "LeIsaac-SO101-PickOrange-v0",
      environment_id: "counter-a",
      environment_index: 0,
      seed: 42,
      recorder_url: "/api/leisaac/recorder?run_id=mock-network-retry",
      recorder: {
        state: recorderState,
        active_episode: recorderState === "recording" ? "episode-retry" : null,
        frame_count: recorderState === "recording" ? 3 : 0,
        completed_episode_count: 0,
        pending_outcome: "",
        last_outcome: "",
        last_upload_status:
          recorderState === "recording" ? "recording" : "never",
        last_error: "",
        last_command_id: lastCommandId,
        last_command: lastCommandId ? "start" : "",
      },
    });
    cy.intercept("GET", "/api/leisaac/status*", (req) => {
      req.reply(status());
    }).as("networkRetryStatus");
    cy.intercept("POST", "/api/leisaac/select", {
      statusCode: 200,
      body: { selected: true },
    });
    cy.intercept(
      "POST",
      "/api/leisaac/recorder?run_id=mock-network-retry",
      (req) => {
        recorderState = "recording";
        lastCommandId = req.body.request_id;
        req.reply({
          statusCode: 202,
          body: { accepted: true, request_id: lastCommandId },
        });
      },
    ).as("networkRetryControl");

    cy.window().then((win) => {
      win.__NPA_AGENT_TEST__.selectActiveRunId("mock-network-retry");
      return win.__NPA_AGENT_TEST__.refreshLeIsaacCapability(
        "mock-network-retry",
      );
    });
    cy.wait("@networkRetryStatus");
    cy.get("#tabLeIsaac").click();
    grantMockControllerLease();
    cy.window().then((win) => {
      const originalFetch = win.fetch.bind(win);
      let rejectRecorderRequest = true;
      win.fetch = (url, options) => {
        if (
          rejectRecorderRequest &&
          String(url).includes("/api/leisaac/recorder?") &&
          options &&
          options.method === "POST"
        ) {
          rejectRecorderRequest = false;
          return Promise.reject(new TypeError("simulated network disconnect"));
        }
        return originalFetch(url, options);
      };
    });
    cy.get("#leisaacRecordStart").should("not.be.disabled").click();
    cy.get("#leisaacRecorderError")
      .should("contain.text", "Recorder command failed")
      .and("contain.text", "retry is available");
    cy.get("#leisaacRecordStart").should("not.be.disabled").click();
    cy.wait("@networkRetryControl");
    cy.get("#leisaacRecorderStatus").should("contain.text", "State: recording");
    cy.get("#leisaacRecordSuccess").should("not.be.disabled");
  });

  it("switches readiness into a live run and drives the upstream keyboard client", () => {
    cy.intercept("GET", "/api/leisaac/status?run_id=mock-run", {
      statusCode: 200,
      body: {
        available: true,
        run_id: "mock-run",
        transport: "agent-relay",
        stream_transport: "webrtc",
        control_ws_url: "/api/leisaac/transport/control?run_id=mock-run",
        requested_video_transport: "webrtc-kit-h264",
        active_video_transport: "webrtc-kit-h264",
        video_codec: "H264",
        hardware_acceleration: "runtime-nvenc",
        video_fallback_reason: "",
        task: "LeIsaac-SO101-PickOrange-v0",
        teleop_device: "keyboard",
        media_server: "203.0.113.50",
        media_port: 47998,
        ice_transport_policy: "relay",
        ice_servers: [
          {
            urls: [
              "turn:203.0.113.50:3478?transport=tcp",
            ],
            username: "mock-run",
            credential: "ephemeral-test-credential",
          },
        ],
        signaling_server: "same-origin",
        signaling_port: 443,
        signaling_path: "/api/leisaac/signal",
        control_ws_url: "/api/leisaac/transport/control?run_id=mock-run",
        client_module_url: "/api/leisaac/client/index.js?run_id=mock-run",
        source_version: "0.4.0",
        source_commit: "1651c321e9b0c1bb54233211fc7b3cd70d8373d5",
        isaac_sim_version: ["5", "1", "0", "0"].join("."),
        isaac_lab_version: "2.3.2.post1",
        image: "registry/npa-leisaac@sha256:test",
        gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition",
      },
    }).as("leisaacStatus");
    cy.intercept("GET", "/api/leisaac/client/index.js?run_id=mock-run", {
      statusCode: 200,
      headers: { "content-type": "text/javascript" },
      body: `window.OVWebStreamingLibrary = { AppStreamer: {
        connect: async function(props) {
          window.__LEISAAC_CONNECT_PROPS__ = props;
          window.__LEISAAC_SIGNAL_SOCKET__ = new window.WebSocket(
            "ws://" + window.location.hostname + ":443" + props.streamConfig.signalingPath,
          );
          new window.RTCPeerConnection({ iceServers: [{ urls: "stun:untrusted.invalid" }] });
          props.streamConfig.onStart({ status: "success" });
          return { status: "inProgress" };
        },
        terminate: async function() { window.__LEISAAC_TERMINATED__ = true; }
      }};`,
    }).as("leisaacClient");
    cy.intercept("POST", "/api/leisaac/select", {
      statusCode: 200,
      body: { selected: true, run_id: "mock-run", available: true },
    }).as("leisaacSelect");

    cy.window().then((win) =>
      win.__NPA_AGENT_TEST__.refreshLeIsaacCapability("mock-run"),
    );
    cy.wait("@leisaacStatus");
    cy.wait("@leisaacSelect");
    cy.get("#tabLeIsaac").should("exist").click();
    cy.get("#panelLeIsaac").should("have.class", "is-active");
    cy.intercept("GET", "/api/sim-viz/status?run_id=older-rerun-only-run", {
      statusCode: 200,
      delay: 200,
      body: {
        run_id: "older-rerun-only-run",
        active_run_id: "older-rerun-only-run",
        stage: "artifacts_available",
        available_runs: [],
      },
    }).as("selectedRunRefresh");
    cy.intercept("GET", "/api/leisaac/status?run_id=older-rerun-only-run", {
      statusCode: 200,
      body: { available: false, reason: "selected run is not LeIsaac" },
    }).as("unrelatedLeisaacStatus");
    cy.intercept("GET", /\/api\/leisaac\/status$/, {
      statusCode: 200,
      body: {
        available: true,
        run_id: "mock-run",
        transport: "agent-relay",
        stream_transport: "webrtc",
        requested_video_transport: "webrtc-kit-h264",
        active_video_transport: "webrtc-kit-h264",
        video_codec: "H264",
        hardware_acceleration: "runtime-nvenc",
        video_fallback_reason: "",
        task: "LeIsaac-SO101-PickOrange-v0",
        teleop_device: "keyboard",
        media_server: "203.0.113.50",
        media_port: 47998,
        ice_transport_policy: "relay",
        ice_servers: [
          {
            urls: [
              "turn:203.0.113.50:3478?transport=tcp",
            ],
            username: "mock-run",
            credential: "ephemeral-test-credential",
          },
        ],
        signaling_path: "/api/leisaac/signal",
        control_ws_url: "/api/leisaac/transport/control?run_id=mock-run",
        client_module_url: "/api/leisaac/client/index.js?run_id=mock-run",
        gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition",
      },
    }).as("rememberedLeisaacStatus");
    cy.window().then((win) => {
      win.__NPA_AGENT_TEST__.selectActiveRunId("older-rerun-only-run");
      const pendingRefresh = win.__NPA_AGENT_TEST__.refresh();
      win.__NPA_AGENT_TEST__.selectActiveRunId("mock-run");
      return pendingRefresh;
    });
    cy.wait("@selectedRunRefresh");
    cy.window().then((win) => {
      expect(win.__NPA_AGENT_TEST__.activeRunId()).to.equal("mock-run");
    });
    cy.get("#tabLeIsaac").should("exist");
    cy.window().then((win) => {
      win.__NPA_AGENT_TEST__.selectActiveRunId("older-rerun-only-run");
      expect(
        win.__NPA_AGENT_TEST__.leisaacPeriodicRefreshRunId(),
        "periodic refresh stays pinned to the mounted teleoperation run",
      ).to.equal("mock-run");
      win.__NPA_AGENT_TEST__.selectActiveRunId("mock-run");
    });
    cy.window().then((win) =>
      win.__NPA_AGENT_TEST__.refreshLeIsaacCapability("older-rerun-only-run"),
    );
    cy.wait("@unrelatedLeisaacStatus");
    cy.wait("@rememberedLeisaacStatus");
    cy.get("#tabLeIsaac").should("exist");
    cy.window().then((win) => {
      function CapturingPeerConnection(config) {
        win.__LEISAAC_PEER_CONFIG__ = config;
        this.connectionState = "new";
        this.iceConnectionState = "new";
        this.iceGatheringState = "new";
        this.signalingState = "stable";
        this.addEventListener = function addEventListener() {};
        this.getConfiguration = function getConfiguration() { return config; };
      }
      CapturingPeerConnection.prototype = {};
      win.__LEISAAC_NATIVE_PEER__ = CapturingPeerConnection;
      win.RTCPeerConnection = CapturingPeerConnection;
      function CapturingWebSocket(url) {
        this.url = String(url);
        this.readyState = CapturingWebSocket.CONNECTING;
        this.sent = [];
        if (this.url.includes("/transport/control")) {
          win.__LEISAAC_CONTROL_SOCKET__ = this;
        }
        win.setTimeout(() => {
          this.readyState = CapturingWebSocket.OPEN;
          if (this.onopen) this.onopen({ target: this });
        }, 0);
      }
      CapturingWebSocket.CONNECTING = 0;
      CapturingWebSocket.OPEN = 1;
      CapturingWebSocket.CLOSING = 2;
      CapturingWebSocket.CLOSED = 3;
      CapturingWebSocket.prototype.send = function send(raw) {
        this.sent.push(String(raw));
        const message = JSON.parse(String(raw));
        if (message.type !== "resume" && !this.url.includes("/transport/control")) return;
        const response = message.type === "resume"
          ? {
              v: 1,
              type: "resumed",
              run_id: message.run_id,
              client_id: message.client_id,
              lease_id: "a".repeat(64),
              lease_generation: 1,
              next_seq: 1,
              last_applied_seq: 0,
              keys_down: [],
            }
          : { ...message, type: "ack", phase: "applied" };
        win.setTimeout(() => {
          if (this.onmessage) this.onmessage({ data: JSON.stringify(response) });
        }, 0);
      };
      CapturingWebSocket.prototype.close = function close() {
        this.readyState = CapturingWebSocket.CLOSED;
        if (this.onclose) this.onclose({ target: this });
      };
      win.__LEISAAC_NATIVE_WEBSOCKET__ = CapturingWebSocket;
      win.WebSocket = CapturingWebSocket;
      win.__LEISAAC_FRAME_CALLBACKS__ = [];
      const video = win.document.getElementById("leisaacVideo");
      win.__LEISAAC_UNAUTHENTICATED_KEY_EVENTS__ = 0;
      win.document.getElementById("leisaacStreamHost").addEventListener(
        "keydown",
        () => { win.__LEISAAC_UNAUTHENTICATED_KEY_EVENTS__ += 1; },
      );
      video.requestVideoFrameCallback = (callback) => {
        win.__LEISAAC_FRAME_CALLBACKS__.push(callback);
        return win.__LEISAAC_FRAME_CALLBACKS__.length;
      };
    });
    cy.get("#leisaacConnect").click();
    cy.wait("@leisaacClient");
    cy.get("#leisaacStreamStatus").should(
      "contain.text",
      "keyboard teleoperation active",
    );
    cy.get("#leisaacTransportStatus")
      .should("contain.text", "native video")
      .and("contain.text", "active webrtc-kit-h264")
      .and("contain.text", "H264")
      .and("contain.text", "runtime-nvenc");
    cy.window()
      .its("__LEISAAC_CONNECT_PROPS__.streamConfig.signalingPath")
      .should("eq", "/api/leisaac/signal");
    cy.window()
      .its("__LEISAAC_SIGNAL_SOCKET__.url")
      .should("match", /^wss:\/\/[^/]+\/api\/leisaac\/signal$/);
    cy.window()
      .its("__LEISAAC_CONNECT_PROPS__.streamConfig")
      .should("not.have.property", "forceWSS");
    cy.window()
      .its("__LEISAAC_CONNECT_PROPS__.streamConfig.mediaPort")
      .should("eq", 47998);
    cy.window()
      .its("__LEISAAC_CONNECT_PROPS__.streamConfig.width")
      .should("eq", 1280);
    cy.window()
      .its("__LEISAAC_CONNECT_PROPS__.streamConfig.height")
      .should("eq", 720);
    cy.window()
      .its("__LEISAAC_CONNECT_PROPS__.streamConfig.fps")
      .should("eq", 30);
    cy.window()
      .its("__LEISAAC_CONNECT_PROPS__.streamConfig.nativeTouchEvents")
      .should("eq", false);
    cy.window()
      .its("__LEISAAC_CONNECT_PROPS__.streamConfig.localizeTextInput")
      .should("eq", false);
    cy.window()
      .its("__LEISAAC_PEER_CONFIG__.iceTransportPolicy")
      .should("eq", "relay");
    cy.window()
      .its("__LEISAAC_PEER_CONFIG__.iceServers.0.urls.0")
      .should("eq", "turn:203.0.113.50:3478?transport=tcp");
    cy.get("#leisaacStreamHost").trigger("keydown", { key: "W", code: "KeyW" });
    cy.get("#leisaacInputStatus")
      .should("contain.text", "Keyboard events sent: 1")
      .and("contain.text", "last W");
    cy.window().its("__LEISAAC_UNAUTHENTICATED_KEY_EVENTS__").should("eq", 0);
    cy.window().should((win) => {
      const socket = win.__LEISAAC_CONTROL_SOCKET__;
      expect(
        socket && socket.sent.map((raw) => JSON.parse(raw)).find(
          (item) => item.type === "control" && item.key === "W",
        ),
      ).to.be.an("object");
    });
    cy.window().then((win) => {
      const socket = win.__LEISAAC_CONTROL_SOCKET__;
      const control = socket.sent.map((raw) => JSON.parse(raw)).find(
        (item) => item.type === "control" && item.key === "W",
      );
      expect(control).to.be.an("object");
      win.__LEISAAC_CAUSAL_SEQUENCE__ = control.seq;
      if (!win.__NPA_AGENT_TEST__.leisaacTransportEvidenceLive().controls.some(
        (item) => item.seq === control.seq && item.phase === "applied",
      )) {
        socket.onmessage({
          data: JSON.stringify({ ...control, type: "ack", phase: "applied" }),
        });
      }
    });
    cy.window().should((win) => {
      expect(
        win.__NPA_AGENT_TEST__.leisaacTransportEvidenceLive().controls.some(
          (item) => item.seq === win.__LEISAAC_CAUSAL_SEQUENCE__ && item.phase === "applied" &&
            item.key === "W" && item.event === "press",
        ),
      ).to.equal(true);
    });
    cy.window().then((win) => {
      const first = win.__LEISAAC_FRAME_CALLBACKS__.shift();
      expect(first).to.be.a("function");
      first(win.performance.now(), {
        presentedFrames: 1,
        mediaTime: 0.033,
        captureTime: 0,
        expectedDisplayTime: win.performance.now(),
      });
      const frames = win.__NPA_AGENT_TEST__.leisaacTransportEvidenceLive().frames;
      expect(frames.at(-1).causal_action_sequence).to.equal(0);
      const second = win.__LEISAAC_FRAME_CALLBACKS__.shift();
      expect(second).to.be.a("function");
      second(win.performance.now() + 33, {
        presentedFrames: 2,
        mediaTime: 0.066,
        captureTime: win.performance.now() + 30,
        receiveTime: win.performance.now() + 31,
        rtpTimestamp: 9000,
        processingDuration: 0.004,
        expectedDisplayTime: win.performance.now() + 33,
      });
      expect(frames.at(-1).causal_action_sequence).to.equal(win.__LEISAAC_CAUSAL_SEQUENCE__);
      expect(frames.at(-1).rtp_timestamp).to.equal(9000);
      expect(frames.at(-1).decode_processing_ms).to.equal(4);
      expect(frames.at(-1).frame_age_ms).to.be.at.least(0);
      expect(frames.at(-1).view_revision).to.equal(0);
      expect(frames.at(-1).transport).to.equal("webrtc-kit-h264");
    });

    cy.intercept("GET", "/api/leisaac/status?run_id=mock-run", {
      statusCode: 200,
      body: { available: false, reason: "session expired" },
    }).as("leisaacGone");
    cy.intercept("GET", "/api/leisaac/status", {
      statusCode: 200,
      body: { available: false, reason: "session expired" },
    }).as("rememberedLeisaacGone");
    cy.window().then((win) =>
      win.__NPA_AGENT_TEST__.refreshLeIsaacCapability("mock-run"),
    );
    cy.wait("@leisaacGone");
    cy.wait("@rememberedLeisaacGone");
    cy.get("#tabLeIsaac").should("exist");
    cy.get("#panelLeIsaac").should("exist");
    cy.get("#leisaacAvailability").should("contain.text", "session expired");
    cy.get("#leisaacConnect, #leisaacLiveGrid, #leisaacRecordStart").should("not.exist");
    cy.get("#leisaacReadiness").should("be.visible");
    cy.get("#leisaacRetry").should("not.be.disabled");
    cy.window().then((win) =>
      {
        expect(win.RTCPeerConnection).to.equal(win.__LEISAAC_NATIVE_PEER__);
        expect(win.WebSocket).to.equal(win.__LEISAAC_NATIVE_WEBSOCKET__);
      },
    );
  });

  it("polls authenticated RTX frames and forwards keyboard controls", () => {
    cy.intercept("GET", "/api/leisaac/status?run_id=mock-jpeg", {
      statusCode: 200,
      body: {
        available: true,
        run_id: "mock-jpeg",
        task: "LeIsaac-SO101-PickOrange-v0",
        teleop_device: "keyboard",
        stream_transport: "jpeg-poll",
        control_ws_url: "/api/leisaac/transport/control?run_id=mock-jpeg",
        frame_url: "/api/leisaac/frame.jpg?run_id=mock-jpeg",
        input_url: "/api/leisaac/input?run_id=mock-jpeg",
        gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition",
      },
    }).as("jpegStatus");
    cy.intercept("GET", "/api/leisaac/frame.jpg?run_id=mock-jpeg&frame=*", {
      statusCode: 200,
      headers: { "content-type": "image/svg+xml", "cache-control": "no-store" },
      body: '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450"><rect width="800" height="450" fill="#2563eb"/></svg>',
    }).as("jpegFrame");
    let fallbackRequests = 0;
    cy.intercept("POST", "/api/leisaac/input?run_id=mock-jpeg", (req) => {
      fallbackRequests += 1;
      expect(req.headers["x-npa-leisaac-client-id"]).to.equal(req.body.client_id);
      expect(req.headers["x-npa-leisaac-lease-id"]).to.equal("a".repeat(64));
      req.reply({
        statusCode: 202,
        body: { accepted: true },
        delay: fallbackRequests === 1 ? 150 : 0,
      });
    }).as("jpegInput");
    cy.intercept("POST", "/api/leisaac/select", {
      statusCode: 200,
      body: { selected: true, run_id: "mock-jpeg", available: true },
    }).as("jpegSelect");

    cy.window().then((win) => {
      class FallbackControlWebSocket {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSING = 2;
        static CLOSED = 3;
        constructor(url, protocol) {
          this.url = String(url);
          this.protocol = String(protocol || "");
          this.readyState = FallbackControlWebSocket.CONNECTING;
          win.setTimeout(() => {
            this.readyState = FallbackControlWebSocket.OPEN;
            if (this.onopen) this.onopen({ target: this });
          }, 0);
        }
        send(raw) {
          const message = JSON.parse(String(raw));
          if (message.type !== "resume") return;
          win.setTimeout(() => {
            if (this.onmessage) this.onmessage({ data: JSON.stringify({
              v: 1,
              type: "resumed",
              run_id: message.run_id,
              client_id: message.client_id,
              lease_id: "a".repeat(64),
              lease_generation: 1,
              next_seq: 1,
              last_applied_seq: 0,
              keys_down: [],
            }) });
          }, 0);
        }
        close() {
          this.readyState = FallbackControlWebSocket.CLOSED;
          if (this.onclose) this.onclose({ target: this });
        }
      }
      win.WebSocket = FallbackControlWebSocket;
    });

    cy.window().then((win) =>
      win.__NPA_AGENT_TEST__.refreshLeIsaacCapability("mock-jpeg"),
    );
    cy.wait("@jpegStatus");
    cy.wait("@jpegSelect");
    cy.get("#tabLeIsaac").click();
    cy.get("#leisaacConnect").click();
    cy.wait("@jpegFrame");
    cy.get("#leisaacFrame")
      .should("be.visible")
      .and(($frame) => expect($frame[0].naturalWidth).to.equal(800));
    cy.get("#leisaacStreamStatus").should(
      "contain.text",
      "keyboard teleoperation active",
    );
    cy.get("#leisaacStreamHost")
      .click()
      .trigger("keydown", { key: "W", code: "KeyW" })
      .trigger("keyup", { key: "W", code: "KeyW" });
    cy.wait(["@jpegInput", "@jpegInput"]).then((interceptions) => {
      const requests = interceptions.map((item) => item.request);
      expect(requests[0].headers["x-npa-leisaac-control"]).to.equal("1");
      expect(requests.map((item) => [item.body.seq, item.body.event])).to.deep.equal([
        [1, "press"],
        [2, "release"],
      ]);
      expect(requests[0].body).to.include({
        v: 1,
        type: "control",
        run_id: "mock-jpeg",
        key: "W",
      });
      expect(requests[0].body.client_id).to.match(/^browser-/);
    });
    cy.get("#leisaacInputStatus").should(
      "contain.text", "Keyboard events sent: 1",
    );
    cy.get("#leisaacDisconnect").click();
  });

  it("negotiates authenticated reliable WebRTC control with bounded WebSocket video", () => {
    const status = {
      available: true,
      run_id: "mock-datachannel",
      task: "LeIsaac-SO101-LiftCube-v0",
      environment_id: "latency-test",
      environment_index: 2,
      seed: 47,
      ...leisaacModeStatus("dual_slow"),
      stream_transport: "websocket-v1",
      preferred_transport: "websocket-v1",
      preferred_control_transport: "webrtc-datachannel-v1",
      control_ws_url: "/api/leisaac/transport/control?run_id=mock-datachannel",
      control_datachannel_url: "/api/leisaac/transport/control-webrtc?run_id=mock-datachannel",
      video_ws_url: "/api/leisaac/transport/video?run_id=mock-datachannel",
      ice_transport_policy: "relay",
      ice_servers: [{
        urls: [
          "turn:203.0.113.50:3478?transport=tcp",
        ],
        username: "bounded-test-user",
        credential: "bounded-test-credential",
      }],
      cameras: ["workspace", "overview"],
      view_orbit: true,
      recorder: { state: "idle", completed_episode_count: 0 },
    };
    cy.intercept("GET", "/api/leisaac/status?run_id=mock-datachannel", status).as("dcStatus");
    cy.intercept("GET", "/api/leisaac/status?run_id=mock-run", status);
    cy.intercept("GET", "/api/leisaac/status", status);
    cy.intercept("POST", "/api/leisaac/select", {
      statusCode: 200,
      body: { selected: true, run_id: "mock-datachannel", available: true },
    });
    cy.intercept("POST", "/api/leisaac/transport/control-webrtc?run_id=mock-datachannel", (req) => {
      expect(req.headers["x-npa-leisaac-control"]).to.equal("1");
      expect(req.body).to.have.all.keys("v", "run_id", "type", "sdp");
      expect(req.body).to.include({ v: 1, run_id: "mock-datachannel", type: "offer" });
      expect(req.body.sdp).to.include("m=application");
      req.reply({
        statusCode: 200,
        headers: { "cache-control": "no-store" },
        body: { v: 1, type: "answer", sdp: "v=0\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n" },
      });
    }).as("dcOffer");

    cy.window().then((win) => {
      class FakeControlWebSocket {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSING = 2;
        static CLOSED = 3;
        constructor(url, protocol) {
          this.url = String(url);
          this.protocol = String(protocol || "");
          this.readyState = FakeControlWebSocket.CONNECTING;
          win.setTimeout(() => {
            this.readyState = FakeControlWebSocket.OPEN;
            if (this.onopen) this.onopen({ target: this });
          }, 0);
        }
        send(raw) {
          const message = JSON.parse(String(raw));
          if (message.type === "resume") {
            win.setTimeout(() => this.onmessage && this.onmessage({ data: JSON.stringify({
              v: 1,
              type: "resumed",
              run_id: message.run_id,
              client_id: message.client_id,
              lease_id: "a".repeat(64),
              lease_generation: 1,
              next_seq: 1,
              last_applied_seq: 0,
              keys_down: [],
            }) }), 0);
          }
        }
        close() { this.readyState = FakeControlWebSocket.CLOSED; }
      }
      class FakeDataChannel {
        constructor(label, options) {
          this.label = label;
          this.options = options;
          this.readyState = "connecting";
          this.binaryType = "blob";
          this.sent = [];
        }
        send(raw) {
          const message = JSON.parse(String(raw));
          this.sent.push(message);
          if (message.type === "resume") {
            win.setTimeout(() => this.onmessage && this.onmessage({ data: JSON.stringify({
              v: 1,
              type: "resumed",
              run_id: message.run_id,
              client_id: message.client_id,
              lease_id: "a".repeat(64),
              lease_generation: 1,
              next_seq: 1,
              last_applied_seq: 0,
              keys_down: [],
            }) }), 0);
          }
        }
        close() { this.readyState = "closed"; }
      }
      class FakePeerConnection {
        constructor(configuration) {
          this.configuration = configuration;
          this.iceGatheringState = "complete";
          this.connectionState = "new";
          this.sctp = { maxMessageSize: 65536 };
          win.__LEISAAC_DC_PEER__ = this;
        }
        createDataChannel(label, options) {
          this.channel = new FakeDataChannel(label, options);
          return this.channel;
        }
        async createOffer() {
          return {
            type: "offer",
            sdp: "v=0\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\na=candidate:1 1 udp 1 203.0.113.50 49152 typ relay\r\n",
          };
        }
        async setLocalDescription(description) { this.localDescription = description; }
        async setRemoteDescription(description) {
          this.remoteDescription = description;
          this.connectionState = "connected";
          this.channel.readyState = "open";
          win.setTimeout(() => this.channel.onopen && this.channel.onopen(), 0);
        }
        addEventListener() {}
        removeEventListener() {}
        close() { this.connectionState = "closed"; }
      }
      win.WebSocket = FakeControlWebSocket;
      win.RTCPeerConnection = FakePeerConnection;
    });

    cy.window().then((win) =>
      win.__NPA_AGENT_TEST__.refreshLeIsaacCapability("mock-datachannel"),
    );
    cy.wait("@dcStatus");
    cy.get("#tabLeIsaac").click();
    cy.get("#leisaacConnect").click();
    cy.wait("@dcOffer");
    cy.get("#leisaacTransportStatus", { timeout: 10000 })
      .should("contain.text", "WebRTC control + WebSocket video")
      .and("contain.text", "latest-frame-wins");
    cy.window().should((win) => {
      const peer = win.__LEISAAC_DC_PEER__;
      expect(peer.configuration.iceTransportPolicy).to.equal("relay");
      expect(peer.configuration.iceServers).to.deep.equal(status.ice_servers);
      expect(peer.channel.label).to.equal("npa-leisaac-control");
      expect(peer.channel.options).to.deep.include({ ordered: true });
      expect(peer.channel.sent.some((item) => item.type === "resume")).to.equal(true);
      const evidence = win.__NPA_AGENT_TEST__.leisaacTransportEvidence();
      expect(evidence.active).to.equal("websocket-v1");
      expect(evidence.control).to.equal("webrtc-datachannel-v1");
      expect(evidence.video).to.equal("websocket-v1");
    });
    cy.window().then((win) => {
      win.__NPA_AGENT_TEST__.queueLeIsaacControl("W", "press");
    });
    cy.window().should((win) => {
      expect(
        win.__LEISAAC_DC_PEER__.channel.sent.some(
          (item) => item.type === "control" && item.key === "W",
        ),
      ).to.equal(true);
    });
    cy.get("#leisaacDisconnect").click();
  });

  it("uses binary WebSocket video fallback while controls and recorder stay responsive", () => {
    let recorderState = "idle";
    const status = () => ({
      available: true,
      run_id: "mock-websocket",
      task: "LeIsaac-SO101-LiftCube-v0",
      environment_id: "latency-test",
      environment_index: 2,
      seed: 47,
      ...leisaacModeStatus("dual_slow"),
      stream_transport: "websocket-v1",
      preferred_transport: "websocket-v1",
      control_ws_url: "/api/leisaac/transport/control?run_id=mock-websocket",
      video_ws_url: "/api/leisaac/transport/video?run_id=mock-websocket",
      frame_url: "/api/leisaac/frame.jpg?run_id=mock-websocket",
      input_url: "/api/leisaac/input?run_id=mock-websocket",
      view_url: "/api/leisaac/view?run_id=mock-websocket",
      cameras: ["workspace", "overview"],
      view_orbit: true,
      recorder_url: "/api/leisaac/recorder?run_id=mock-websocket",
      gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition",
      recorder: {
        state: recorderState,
        active_episode: recorderState === "recording" ? "episode-live" : null,
        frame_count: recorderState === "recording" ? 12 : 0,
        completed_episode_count: 1,
        pending_outcome: "",
        last_outcome: "success",
        last_upload_status: "uploaded",
        last_error: "",
      },
    });
    cy.intercept("GET", "/api/leisaac/status?run_id=mock-websocket", (req) =>
      req.reply(status()),
    ).as("wsStatus");
    cy.intercept("GET", "/api/leisaac/status?run_id=mock-run", (req) =>
      req.reply(status()),
    );
    cy.intercept("GET", "/api/leisaac/status", (req) => req.reply(status()));
    cy.intercept("POST", "/api/leisaac/select", {
      statusCode: 200,
      body: { selected: true, run_id: "mock-websocket", available: true },
    });
    cy.intercept("POST", "/api/leisaac/view?run_id=mock-websocket", (req) => {
      expect(req.headers["x-npa-leisaac-control"]).to.equal("1");
      expect(req.body.camera).to.equal("workspace");
      expect(req.body.sequence).to.be.greaterThan(0);
      expect(Math.abs(req.body.distance_delta)).to.be.at.most(1);
      req.reply({ statusCode: 202, body: { accepted: true, sequence: req.body.sequence } });
    }).as("viewOrbit");
    cy.intercept(
      "POST",
      "/api/leisaac/recorder?run_id=mock-websocket",
      (req) => {
        recorderState = req.body.command === "start" ? "recording" : recorderState;
        req.reply({ statusCode: 202, body: { accepted: true } });
      },
    ).as("wsRecorder");

    cy.window().then((win) => {
      let nextExpected = 1;
      let frameSequence = 0;
      const sockets = [];
      const encodeFrame = () => {
        const jpeg = new win.Uint8Array([0xff, 0xd8, 1, 2, 3, 4, 0xff, 0xd9]);
        const payload = new win.ArrayBuffer(136 + jpeg.length);
        const view = new win.DataView(payload);
        [0x4e, 0x50, 0x41, 0x46].forEach((value, index) =>
          view.setUint8(index, value),
        );
        view.setUint8(4, 3);
        view.setUint16(6, 136, false);
        frameSequence += 1;
        view.setUint8(5, frameSequence % 2 === 0 ? 1 : 0);
        view.setBigUint64(8, BigInt(frameSequence), false);
        view.setBigUint64(16, BigInt(win.Date.now()) * 1000000n, false);
        view.setBigUint64(24, BigInt(Math.floor(win.performance.now() * 1000000)), false);
        view.setBigUint64(32, BigInt(win.Date.now()) * 1000000n, false);
        view.setBigUint64(40, BigInt(Math.floor(win.performance.now() * 1000000)), false);
        view.setBigUint64(48, 500n, false);
        view.setBigUint64(56, 600n, false);
        view.setBigUint64(64, 700n, false);
        view.setBigUint64(72, BigInt(frameSequence), false);
        view.setBigUint64(80, 650n, false);
        view.setBigUint64(88, BigInt(frameSequence), false);
        view.setUint32(96, jpeg.length, false);
        view.setUint32(100, Math.max(0, frameSequence - 1), false);
        new win.Uint8Array(payload, 136).set(jpeg);
        return payload;
      };
      class FakeWebSocket {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSING = 2;
        static CLOSED = 3;
        constructor(url, protocol) {
          this.url = String(url);
          this.protocol = protocol;
          this.readyState = FakeWebSocket.CONNECTING;
          this.binaryType = "blob";
          this.sent = [];
          this.timer = null;
          sockets.push(this);
          win.setTimeout(() => {
            this.readyState = FakeWebSocket.OPEN;
            if (this.onopen) this.onopen({ target: this });
            if (this.url.includes("/video")) {
              this.timer = win.setInterval(() => {
                if (this.onmessage) this.onmessage({ data: encodeFrame() });
              }, 12);
            }
          }, 5);
        }
        send(raw) {
          this.sent.push(String(raw));
          if (!this.url.includes("/control")) return;
          const message = JSON.parse(String(raw));
          let response = null;
          if (message.type === "resume") {
            if (win.__LEISAAC_REJECT_NEXT_RESUME__) {
              win.__LEISAAC_REJECT_NEXT_RESUME__ = false;
              response = {
                v: 1,
                type: "error",
                code: "controller_busy",
                detail: "another authenticated control transport owns this session",
              };
            } else {
              response = {
                v: 1,
                type: "resumed",
                run_id: message.run_id,
                client_id: message.client_id,
                lease_id: "a".repeat(64),
                lease_generation: 1,
                next_seq: nextExpected,
                last_applied_seq: nextExpected - 1,
                keys_down: [],
              };
            }
          } else if (message.type === "ping") {
            response = {
              ...message,
              type: "pong",
              runtime_wall_ns: String(BigInt(win.Date.now()) * 1000000n),
            };
          } else if (message.type === "control" || message.type === "action") {
            expect(message.seq).to.equal(nextExpected);
            nextExpected += 1;
            if (win.__LEISAAC_DROP_NEXT_CONTROL_ACKS__) {
              win.__LEISAAC_DROP_NEXT_CONTROL_ACKS__ = false;
              return;
            }
            response = { ...message, type: "ack", phase: "accepted" };
            win.setTimeout(() => {
              if (this.onmessage)
                this.onmessage({
                  data: JSON.stringify({
                    ...message,
                    type: "ack",
                    phase: "applied",
                    simulator_applied_mono_ns: "800",
                    simulator_step: message.seq,
                  }),
                });
            }, 2);
          } else if (message.type === "release-all") {
            response = {
              v: 1,
              type: "released",
              run_id: message.run_id,
              client_id: message.client_id,
              released_count: 0,
            };
          } else if (message.type === "view-mode" || message.type === "recording-cameras") {
            if (
              message.type === "view-mode" &&
              message.mode === "dual_slow" &&
              win.__LEISAAC_DROP_NEXT_MODE_REQUEST__
            ) {
              win.__LEISAAC_DROP_NEXT_MODE_REQUEST__ = false;
              return;
            }
            response = { ...message, type: "ack", request_type: message.type, phase: "accepted" };
            win.setTimeout(() => {
              if (this.onmessage) this.onmessage({ data: JSON.stringify({
                ...message,
                type: "ack",
                request_type: message.type,
                phase: "applied",
                mode_transition_ms: 2,
              }) });
            }, 2);
            if (message.type === "view-mode" && message.mode === "single_fast") {
              win.setTimeout(() => {
                if (this.onmessage) this.onmessage({ data: JSON.stringify({
                  ...message,
                  type: "ack",
                  request_type: message.type,
                  phase: "applied",
                  revision: message.revision - 1,
                  mode: "dual_slow",
                }) });
              }, 5);
            }
          }
          if (response) {
            const dispatch = () =>
              this.onmessage && this.onmessage({ data: JSON.stringify(response) });
            if (
              message.type === "resume" &&
              response.type === "error" &&
              win.__LEISAAC_SYNC_BUSY_RESUME__
            ) dispatch();
            else win.setTimeout(
              dispatch,
              response.type === "resumed"
                ? Number(win.__LEISAAC_RESUME_ACK_DELAY_MS__ || 0)
                : 0,
            );
          }
        }
        close() {
          if (this.timer) win.clearInterval(this.timer);
          this.readyState = FakeWebSocket.CLOSED;
        }
        fail() {
          if (this.timer) win.clearInterval(this.timer);
          this.readyState = FakeWebSocket.CLOSED;
          if (this.onclose) this.onclose({ code: 1012 });
        }
      }
      win.WebSocket = FakeWebSocket;
      // Deliver each camera every 24 ms while paint runs every 30 ms. A worker
      // that discards its in-flight decode whenever a newer frame arrives will
      // starve forever; the bounded worker must paint and then take the latest.
      win.requestAnimationFrame = (callback) => win.setTimeout(
        () => callback(win.performance.now()),
        30,
      );
      win.__LEISAAC_TEST_ERRORS__ = [];
      win.addEventListener("error", (event) => {
        win.__LEISAAC_TEST_ERRORS__.push(String(event.message || event.error || "error"));
      });
      win.addEventListener("unhandledrejection", (event) => {
        win.__LEISAAC_TEST_ERRORS__.push(String(event.reason || "rejection"));
      });
      let releaseFirstBitmap;
      let firstBitmap = true;
      let bitmapCloses = 0;
      win.createImageBitmap = async () => {
        if (firstBitmap) {
          firstBitmap = false;
          await new Promise((resolve) => {
            releaseFirstBitmap = resolve;
          });
        }
        return { width: 1280, height: 720, close() { bitmapCloses += 1; } };
      };
      win.__LEISAAC_BITMAP_CLOSES__ = () => bitmapCloses;
      win.__LEISAAC_RELEASE_FIRST_BITMAP__ = () => releaseFirstBitmap();
      class FallbackImage {
        constructor() {
          this.naturalWidth = 1280;
          this.naturalHeight = 720;
          this.onload = null;
          this.onerror = null;
        }
        set src(_value) {
          win.setTimeout(() => this.onload && this.onload(), 0);
        }
      }
      win.Image = FallbackImage;
      win.URL.createObjectURL = () => "blob:mock-leisaac-frame";
      win.URL.revokeObjectURL = () => {};
      const nativeGetContext = win.HTMLCanvasElement.prototype.getContext;
      let canvasFailures = 0;
      win.__LEISAAC_SET_CANVAS_FAILURES__ = (count) => {
        canvasFailures = Number(count);
      };
      win.HTMLCanvasElement.prototype.getContext = function getContext(kind, ...args) {
        if (["leisaacCanvas", "leisaacSecondaryCanvas"].includes(this.id) && kind === "2d") {
          if (canvasFailures === 2) {
            canvasFailures -= 1;
            return null;
          }
          if (canvasFailures === 1) {
            canvasFailures -= 1;
            return { drawImage() { throw new Error("synthetic draw failure"); } };
          }
          return { drawImage() {} };
        }
        return nativeGetContext.call(this, kind, ...args);
      };
      win.__LEISAAC_FAKE_SOCKETS__ = sockets;
    });

    cy.window().then((win) =>
      win.__NPA_AGENT_TEST__.refreshLeIsaacCapability("mock-websocket"),
    );
    cy.wait("@wsStatus");
    cy.get("#tabLeIsaac").click();
    cy.get("#leisaacConnect").click();
    cy.get("#leisaacTransportStatus", { timeout: 10000 })
      .should("contain.text", "WebSocket")
      .and("contain.text", "latest-frame-wins");
    cy.window().should((win) => {
      expect(win.__NPA_AGENT_TEST__.leisaacTransportEvidence().video)
        .to.equal("websocket-v1");
      const video = win.__LEISAAC_FAKE_SOCKETS__.find(
        (socket) => socket.url.includes("/video") && socket.readyState === 1,
      );
      const credits = video.sent
        .map((raw) => JSON.parse(raw))
        .filter((item) => item.type === "frame-ack");
      expect(credits, "receipt credits before a blocked decode").to.have.length
        .greaterThan(0);
      expect(
        win.__NPA_AGENT_TEST__.leisaacTransportEvidence().frames.some(
          (frame) => frame.camera === "overview",
        ),
        "overview paints independently while workspace decode is blocked",
      ).to.equal(true, JSON.stringify({
        errors: win.__LEISAAC_TEST_ERRORS__,
        status: win.document.getElementById("leisaacStreamStatus")?.textContent,
      }));
    });
    cy.wait(1100);
    cy.window().should((win) => {
      expect(
        win.__NPA_AGENT_TEST__.leisaacTransportEvidence().decode_fallbacks,
        "bounded image-element fallback after a stuck bitmap decode",
      ).to.be.greaterThan(0);
    });
    cy.window().then((win) => {
      win.__LEISAAC_SET_CANVAS_FAILURES__(2);
      win.__LEISAAC_RELEASE_FIRST_BITMAP__();
    });
    cy.get("#leisaacCanvas").should("be.visible");
    cy.get("#leisaacSecondaryCanvas", { timeout: 10000 }).should("be.visible");
    cy.window().should((win) => {
      expect(win.__LEISAAC_BITMAP_CLOSES__()).to.be.at.least(3);
    });
    cy.get("#leisaacStreamHost").trigger("wheel", { deltaY: 120 });
    cy.wait("@viewOrbit");
    cy.get("#leisaacStreamHost").should("have.attr", "data-orbit-bound", "1");
    cy.get("#panelLeIsaac").then(($panel) => {
      cy.get("#leisaacViewMode").select("single_fast");
      cy.get("#leisaacLiveGrid").should("have.class", "is-single");
      cy.get("#leisaacSecondaryHost").should("not.be.visible");
      cy.get("#leisaacViewMode").should("have.value", "single_fast");
      cy.get("#leisaacModeStatus", { timeout: 12000 })
        .should("contain.text", "Applied view")
        .and("contain.text", "Fast single");
      cy.get("#panelLeIsaac").then(($samePanel) => {
        expect($samePanel[0], "mode switch does not remount the tab").to.equal($panel[0]);
      });
      cy.get("#leisaacRecordingCameras").select("primary_and_secondary");
      cy.get("#leisaacModeWarning").should("contain.text", "reduces Fast single performance");
    });
    cy.get("#leisaacStreamStatus").should(
      "contain.text",
      "keyboard teleoperation active",
    );
    cy.window().then((win) => {
      const host = win.document.getElementById("leisaacStreamHost");
      host.focus();
      host.dispatchEvent(
        new win.KeyboardEvent("keydown", { key: "W", code: "KeyW", bubbles: true }),
      );
      host.dispatchEvent(
        new win.KeyboardEvent("keyup", { key: "W", code: "KeyW", bubbles: true }),
      );
    });
    cy.window().should((win) => {
      const controls = win.__NPA_AGENT_TEST__.leisaacTransportEvidence().controls;
      expect(controls.filter((item) => item.phase === "accepted")).to.have.length(2);
      expect(controls.filter((item) => item.phase === "applied")).to.have.length(2);
      const video = win.__LEISAAC_FAKE_SOCKETS__.find(
        (socket) => socket.url.includes("/video") && socket.readyState === 1,
      );
      expect(
        video.sent
          .map((raw) => JSON.parse(raw))
          .filter((item) => item.type === "frame-ack"),
      ).to.have.length.greaterThan(0);
      const evidence = win.__NPA_AGENT_TEST__.leisaacTransportEvidence();
      const lastByCamera = {};
      const expectedDrops = evidence.frames.reduce((total, frame) => {
        const prior = Number(lastByCamera[frame.camera] || 0);
        const gap = prior ? Math.max(0, Number(frame.sequence) - prior - 1) : 0;
        lastByCamera[frame.camera] = Number(frame.sequence);
        return total + Math.max(
          Number(frame.dropped_before || 0),
          gap,
          Number(frame.browser_dropped || 0),
        );
      }, 0);
      expect(evidence.dropped_frames).to.equal(expectedDrops);
      expect(evidence.frames.some((frame) => frame.causal_action_sequence > 0)).to.equal(true);
      expect(evidence.frames.some((frame) => frame.view_revision > 0)).to.equal(true);
      const recordedFrameCount = evidence.frames.length;
      evidence.frames.length = 0;
      expect(
        win.__NPA_AGENT_TEST__.leisaacTransportEvidence().frames,
        "diagnostic snapshots isolate their bounded arrays",
      ).to.have.length(recordedFrameCount);
      const liveEvidence = win.__NPA_AGENT_TEST__.leisaacTransportEvidenceLive();
      expect(liveEvidence.frames, "performance probes reuse the live bounded array")
        .to.equal(win.__NPA_AGENT_TEST__.leisaacTransportEvidenceLive().frames)
        .and.to.have.length(recordedFrameCount);
    });
    cy.window().then((win) => {
      const host = win.document.getElementById("leisaacStreamHost");
      host.focus();
      host.dispatchEvent(
        new win.KeyboardEvent("keydown", { key: "A", code: "KeyA", bubbles: true }),
      );
    });
    cy.window().should((win) => {
      expect(
        win.__NPA_AGENT_TEST__.leisaacTransportEvidence().controls.some(
          (item) => item.phase === "applied" && item.key === "A" && item.event === "press",
        ),
      ).to.equal(true);
    });
    cy.window().then((win) => {
      win.__LEISAAC_DROP_NEXT_CONTROL_ACKS__ = true;
      win.__LEISAAC_REJECT_NEXT_RESUME__ = true;
      // Reproduce the live open/resume gap: the contended lease rejects while
      // recovery is still pending, then the replacement socket opens well
      // before its authoritative resumed acknowledgement arrives.
      win.__LEISAAC_SYNC_BUSY_RESUME__ = true;
      win.__LEISAAC_RESUME_ACK_DELAY_MS__ = 400;
      const host = win.document.getElementById("leisaacStreamHost");
      host.dispatchEvent(
        new win.KeyboardEvent("keyup", { key: "A", code: "KeyA", bubbles: true }),
      );
      const video = win.__LEISAAC_FAKE_SOCKETS__.find(
        (socket) => socket.url.includes("/video") && socket.readyState === 1,
      );
      video.fail();
    });
    cy.get("#leisaacViewMode").should("be.disabled");
    cy.get("#leisaacControllerPrerequisite").should("contain.text", "Another authenticated browser");
    cy.window().then((win) => { win.__LEISAAC_DROP_NEXT_MODE_REQUEST__ = true; });
    cy.get("#leisaacViewMode", { timeout: 10000 })
      .should("not.be.disabled")
      .select("dual_slow");
    cy.get("#leisaacModeStatus", { timeout: 10000 })
      .should("contain.text", "Applied view")
      .and("contain.text", "Dual view");
    cy.window().should((win) => {
      const evidence = win.__NPA_AGENT_TEST__.leisaacTransportEvidence();
      expect(
        evidence.controls.some(
          (item) =>
            item.phase === "applied" &&
            item.key === "A" &&
            item.event === "release" &&
            item.recovered_on_resume === true,
        ),
      ).to.equal(true);
      expect(evidence.reconnects).to.be.greaterThan(0);
      const resumeEpochs = win.__LEISAAC_FAKE_SOCKETS__
        .filter((socket) => socket.url.includes("/control"))
        .flatMap((socket) => socket.sent.map((raw) => JSON.parse(raw)))
        .filter((item) => item.type === "resume")
        .map((item) => BigInt(item.client_wall_ns));
      expect(resumeEpochs).to.have.length.greaterThan(2);
      expect(resumeEpochs).to.have.length.at.most(4);
      expect(
        resumeEpochs.every((epoch, index) => index === 0 || epoch > resumeEpochs[index - 1]),
      ).to.equal(true);
      expect(new Set(resumeEpochs.map((epoch) => epoch / 1000000n)).size).to.equal(1);
    });
    cy.get("#leisaacInputDevice").select("custom-so101");
    cy.get("#leisaacSendNeutralAction").click();
    cy.window().should((win) => {
      const controlSocket = win.__LEISAAC_FAKE_SOCKETS__.find(
        (socket) => socket.url.includes("/control") && socket.readyState === 1,
      );
      const actions = controlSocket.sent.map((raw) => JSON.parse(raw)).filter((item) => item.type === "action");
      expect(actions).to.have.length.greaterThan(0);
      expect(actions[0].device).to.equal("custom-so101");
      expect(actions[0].action).to.deep.equal([0, 0, 0, 0, 0, 0, 0, 0]);
    });
    cy.get("#leisaacRecordStart").click();
    cy.wait("@wsRecorder");
    cy.get("#leisaacRecorderStatus").should("contain.text", "recording");

    cy.window().then((win) => {
      const video = win.__LEISAAC_FAKE_SOCKETS__.find(
        (socket) => socket.url.includes("/video") && socket.readyState === 1,
      );
      expect(video).to.exist;
      video.fail();
    });
    cy.get("#leisaacTransportStatus", { timeout: 10000 }).should(
      "contain.text",
      "WebSocket",
    );
    cy.window()
      .then((win) => win.__NPA_AGENT_TEST__.leisaacTransportEvidence())
      .its("reconnects")
      .should("be.gte", 0);
    cy.get("#leisaacDisconnect").click();
  });

  it("browses immutable episodes, opens the exact upload, and synchronizes playback", () => {
    const versionId = "v000001-" + "b".repeat(32);
    let episodeStatusRequests = 0;
    const episodeSummary = (index, outcome) => ({
      episode_index: index,
      episode_id: "episode-" + String(index),
      task: "LeIsaac-SO101-LiftCube-v0",
      environment_id: "table-b",
      outcome,
      recorded_at: "2026-08-06T01:00:0" + String(index) + "Z",
      frame_count: 2,
      robot: "custom-so101",
      scene: "custom-table",
      device: "spacemouse",
      bundle: "bundle-sha256",
    });
    cy.intercept("GET", "/api/leisaac/status*", (request) => {
      const uploadedIndex = episodeStatusRequests++ === 0 ? 0 : 1;
      request.reply({
        statusCode: 200,
        body: {
        available: false,
        episodes_available: true,
        reason: "live runtime is reconnecting",
        run_id: "mock-episodes",
        task: "LeIsaac-SO101-LiftCube-v0",
        environment_id: "table-b",
        dataset_uri: "s3://bucket/datasets/leisaac",
        recorder: {
          state: "idle",
          completed_episode_count: 2,
          last_outcome: "success",
          last_upload_status: "uploaded",
          last_episode_index: uploadedIndex,
          dataset_version_uri:
            "s3://bucket/datasets/leisaac/versions/" + versionId,
        },
        },
      });
    }).as("episodeStatus");
    cy.intercept("POST", "/api/leisaac/select", {
      statusCode: 200,
      body: { selected: true },
    });
    cy.intercept("GET", "/api/leisaac/episodes/versions?*", {
      delay: 800,
      statusCode: 200,
      body: {
        versions: [
          {
            version_id: versionId,
            episode_count: 2,
            created_at: "2026-08-06T01:00:10Z",
            lerobot_version: "0.5.1",
          },
        ],
        next_cursor: "",
        bounded: true,
      },
    }).as("episodeVersions");
    cy.intercept(
      { method: "GET", pathname: "/api/leisaac/episodes" },
      (req) => {
        req.alias = req.query.cursor
          ? "episodeListNext"
          : req.query.task
            ? "episodeListFiltered"
            : "episodeList";
        expect(req.query.limit).to.equal("20");
        if (req.query.task) {
          expect(req.query.task).to.equal("LeIsaac-SO101-LiftCube-v0");
          expect(req.query.version_id).to.equal(versionId);
        }
        req.reply({
          delay: 800,
          statusCode: 200,
          body: {
            episodes: req.query.cursor
              ? [episodeSummary(1, "failure")]
              : [episodeSummary(0, "success")],
            next_cursor: req.query.cursor ? "" : "bounded-next",
            bounded: true,
          },
        });
      },
    );
    cy.intercept(
      { method: "GET", pathname: "/api/leisaac/episodes/1" },
      (req) => {
        expect(req.query.version_id).to.equal(versionId);
        req.reply({
          // Keep the explicit detail open in flight while a status refresh
          // starts another automatic list discovery. The two requests must
          // not cancel each other.
          delay: 400,
          statusCode: 200,
          body: {
            ...episodeSummary(1, "failure"),
            run_id: "mock-episodes",
            dataset_version: versionId,
            start_timestamp: "2026-08-06T01:00:00Z",
            end_timestamp: "2026-08-06T01:00:01Z",
            duration_seconds: 1,
            commit_checksum: "c".repeat(64),
            source_uris: {
              commit: "s3://bucket/datasets/leisaac/commits/episode-000000.json",
            },
            markers: { success_frames: [1], reset_frames: [1] },
            cameras: [
              {
                id: "workspace",
                label: "Workspace",
                sha256: "d".repeat(64),
                media_url:
                  "/api/leisaac/episodes/1/media/workspace?run_id=mock-episodes&version_id=" +
                  versionId,
              },
              {
                id: "wrist",
                label: "Wrist",
                sha256: "e".repeat(64),
                media_url:
                  "/api/leisaac/episodes/1/media/wrist?run_id=mock-episodes&version_id=" +
                  versionId,
              },
            ],
            camera_mode: "synchronized-two-camera",
            artifacts: [
              {
                name: "records",
                kind: "timeline",
                bytes: 900,
                sha256: "a".repeat(64),
                download_url:
                  "/api/leisaac/episodes/1/download/records?run_id=mock-episodes&version_id=" +
                  versionId,
              },
              { name: "calibration", kind: "unknown", download_url: "" },
            ],
            export: {
              records_url:
                "/api/leisaac/episodes/1/download/records?run_id=mock-episodes&version_id=" +
                versionId,
              metadata_url:
                "/api/leisaac/episodes/1/download/metadata?run_id=mock-episodes&version_id=" +
                versionId,
            },
          },
        });
      },
    ).as("episodeDetail");
    cy.intercept(
      { method: "GET", pathname: "/api/leisaac/episodes/1/timeline" },
      {
        statusCode: 200,
        body: {
          rows: [
            { frame_index: 0, timestamp: 0, action: [0], reward: 0, success: false },
            {
              frame_index: 1,
              timestamp: 1,
              action: [1],
              observation_state: [2],
              reward: 1,
              success: true,
              done: true,
              reset_reason: "success",
            },
          ],
          checksum_state: "verified",
          sha256: "a".repeat(64),
        },
      },
    ).as("episodeTimeline");
    cy.intercept("GET", "/api/leisaac/episodes/1/media/*", {
      statusCode: 206,
      headers: {
        "content-type": "video/mp4",
        "accept-ranges": "bytes",
        "content-range": "bytes 0-3/4",
      },
      body: "mock",
    });

    cy.window().then((win) =>
      win.__NPA_AGENT_TEST__.refreshLeIsaacCapability("mock-episodes"),
    );
    cy.wait("@episodeStatus");
    cy.get("#tabLeIsaac").click();
    cy.get("#leisaacAvailability").should("contain.text", "reconnecting");
    cy.get("#leisaacEpisodesTitle").should("be.visible");
    cy.wait("@episodeList");
    cy.wait("@episodeVersions");
    cy.get("#leisaacEpisodeVersion").select(versionId);
    cy.get("#leisaacEpisodesNextPage").should("not.be.disabled").click();
    cy.wait("@episodeListNext");
    cy.contains("#leisaacEpisodeList button", "Open").click();
    cy.window().then((win) =>
      win.__NPA_AGENT_TEST__.refreshLeIsaacCapability("mock-episodes"),
    );
    cy.wait("@episodeDetail");
    cy.wait("@episodeTimeline");
    cy.get("#leisaacEpisodePlayer").should("be.visible");
    cy.get("#leisaacEpisodePlayerTitle").should("contain.text", "Episode 1 · failure");
    cy.get("#leisaacEpisodeSecondaryPane").should("be.visible");
    cy.get("#leisaacEpisodeSingleCamera").should("not.be.visible");
    cy.get("#leisaacEpisodeMetadata")
      .should("contain.text", "custom-so101/custom-table/spacemouse")
      .and("contain.text", versionId);
    cy.get("#leisaacEpisodeChecksum")
      .should("contain.text", "verified")
      .and("contain.text", "workspace=" + "d".repeat(64));
    cy.get("#leisaacEpisodeArtifacts")
      .should("contain.text", "records")
      .and("contain.text", "Unknown artifact is preserved");
    cy.get("#leisaacEpisodeTimeline").invoke("val", 1).trigger("input");
    cy.get("#leisaacEpisodeTimelineValues")
      .should("contain.text", '"reward": 1')
      .and("contain.text", '"reset_reason": "success"');
    cy.get("#leisaacEpisodeRate").select("2");
    cy.get("#leisaacEpisodePrimaryVideo").should(($video) => {
      expect($video[0].playbackRate).to.equal(2);
      expect($video[0].defaultPlaybackRate).to.equal(2);
    });
    cy.get("#leisaacEpisodeDescribe").click();
    cy.get("#panelLeIsaac").should("have.class", "is-active");
    cy.get("#chatLog .msg-row.user", { timeout: 10000 })
      .should("contain.text", "Describe this")
      .and("contain.text", '"viewer": "LeIsaac"')
      .and("contain.text", '"mode": "episode"');
    cy.window().then((win) => {
      win.document
        .getElementById("leisaacEpisodePrimaryVideo")
        .dispatchEvent(new win.Event("error"));
    });
    cy.get("#leisaacEpisodeError").should("contain.text", "exact immutable episode");
    cy.get("#leisaacEpisodesRetry").click();
    cy.wait("@episodeDetail");
    cy.wait("@episodeTimeline");
    cy.get("#leisaacEpisodeVersion").select(versionId);
    cy.get("#leisaacEpisodeTask").type("LeIsaac-SO101-LiftCube-v0");
    cy.get("#leisaacEpisodesApply").click();
    cy.wait("@episodeListFiltered");
    cy.get("#leisaacEpisodesNextPage").should("not.be.disabled").click();
    cy.wait("@episodeListNext");
    cy.get("#leisaacEpisodeList").should("contain.text", "Episode 1 · failure");
  });

  it("distinguishes a filtered empty S3 page from an empty collection", () => {
    cy.intercept("GET", "/api/leisaac/status*", {
      statusCode: 200,
      body: {
        available: false,
        episodes_available: true,
        reason: "runtime intentionally unavailable",
        run_id: "mock-filtered-page",
        task: "LeIsaac-SO101-LiftCube-v0",
        dataset_uri: "s3://bucket/datasets/leisaac",
        recorder: { state: "idle", completed_episode_count: 2 },
      },
    }).as("filteredPageStatus");
    cy.intercept("POST", "/api/leisaac/select", {
      statusCode: 200,
      body: { selected: true },
    });
    cy.intercept("GET", "/api/leisaac/episodes/versions?*", {
      statusCode: 200,
      body: { versions: [], next_cursor: "", bounded: true },
    });
    cy.intercept("GET", "/api/leisaac/episodes?*", {
      statusCode: 200,
      body: {
        episodes: [],
        next_cursor: "opaque-next-page",
        has_more_pages: true,
        source_count: 20,
        loaded_count: 20,
        filtered_count: 20,
        skipped_count: 0,
        bounded: true,
      },
    }).as("filteredEmptyPage");
    cy.window().then((win) =>
      win.__NPA_AGENT_TEST__.refreshLeIsaacCapability("mock-filtered-page"),
    );
    cy.wait("@filteredPageStatus");
    cy.get("#tabLeIsaac").click();
    cy.wait("@filteredEmptyPage");
    cy.get("#leisaacEpisodeStatus")
      .should("contain.text", "No filter matches on this bounded page")
      .and("contain.text", "Continue with Next");
    cy.get("#leisaacEpisodesNextPage").should("not.be.disabled");
  });

  it("keeps offline bundle storage useful but applies only with a live controller lease", () => {
    const digest = "d".repeat(64);
    const uploaded = [];
    let liveReady = false;
    let selectionPending = false;
    const status = () => ({
      available: liveReady,
      episodes_available: true,
      reason: liveReady ? "" : "registered runtime health is not ready",
      run_id: "mock-bundles",
      task: "LeIsaac-SO101-LiftCube-v0",
      dataset_uri: "s3://bucket/datasets/leisaac",
      bundles_url: "/api/leisaac/bundles?run_id=mock-bundles",
      bundle_select_url: "/api/leisaac/bundles/select?run_id=mock-bundles",
      bundle_reset_url: "/api/leisaac/bundles/reset?run_id=mock-bundles",
      bundle_selection_pending: selectionPending,
      bundle_selection_reason: selectionPending
        ? "Persisted checksum-verified bundles differ from this runtime. Connect teleoperation and obtain the controller lease to apply them."
        : "",
      configuration: defaultLeIsaacConfiguration(),
      ...leisaacModeStatus(),
      cameras: ["workspace"],
      recorder: { state: "idle", completed_episode_count: 0 },
    });
    cy.intercept("GET", "/api/leisaac/status?run_id=mock-bundles", (req) =>
      req.reply({ statusCode: 200, body: status() }),
    ).as("bundleStatus");
    cy.intercept("GET", "/api/leisaac/episodes/versions?*", {
      statusCode: 200,
      body: { versions: [], next_cursor: "", bounded: true },
    });
    cy.intercept("GET", "/api/leisaac/episodes?*", {
      statusCode: 200,
      body: { episodes: [], next_cursor: "", source_count: 0, loaded_count: 0, filtered_count: 0, skipped_count: 0 },
    });
    cy.intercept("GET", "/api/leisaac/bundles?run_id=mock-bundles", (req) =>
      req.reply({ statusCode: 200, body: { bundles: uploaded, bounded: true, truncated: false } }),
    ).as("bundleList");
    cy.intercept("POST", "/api/leisaac/bundles?run_id=mock-bundles", (req) => {
      expect(req.headers["x-npa-leisaac-control"]).to.equal("1");
      expect(req.body.files.every((file) => /^[a-f0-9]{64}$/.test(file.sha256))).to.equal(true);
      const index = {
        schema: "npa.leisaac.bundle.v1",
        kind: "robot",
        name: req.body.name,
        bundle_sha256: digest,
        entrypoint: req.body.entrypoint,
        bytes: 64,
      };
      uploaded.push(index);
      req.reply({ statusCode: 201, body: index });
    }).as("bundleUpload");
    cy.intercept("POST", "/api/leisaac/bundles/select?run_id=mock-bundles", (req) => {
      expect(req.headers["x-npa-leisaac-lease-id"]).to.equal("a".repeat(64));
      expect(req.body).to.deep.equal({ kind: "robot", bundle_sha256: digest });
      req.reply({ statusCode: 202, body: { selected: req.body, restarting: true } });
    }).as("bundleSelect");
    cy.intercept("POST", "/api/leisaac/bundles/reset?run_id=mock-bundles", (req) => {
      expect(req.headers["x-npa-leisaac-lease-id"]).to.equal("a".repeat(64));
      req.reply({ statusCode: 202, body: { reset: true, restarting: true } });
    }).as("bundleReset");

    cy.window().then((win) => win.__NPA_AGENT_TEST__.refreshLeIsaacCapability("mock-bundles"));
    cy.wait("@bundleStatus");
    cy.get("#tabLeIsaac").click();
    cy.wait("@bundleList");
    cy.get("#leisaacReadiness").should("be.visible");
    cy.get("#leisaacBundleUpload, #leisaacBundleRefresh").should("not.be.disabled");
    cy.get("#leisaacBundleSelect, #leisaacResetDefaults").should("not.exist");
    cy.get("#leisaacBundleName").type("custom-so101");
    cy.get("#leisaacBundleKind").select("robot");
    cy.get("#leisaacBundleFiles").selectFile({
      contents: Cypress.Buffer.from('#usda 1.0\ndef Xform "SO101" {}\n'),
      fileName: "robot.usda",
      mimeType: "application/octet-stream",
    });
    cy.get("#leisaacBundleUpload").click();
    cy.wait("@bundleUpload");
    cy.wait("@bundleList");
    cy.get("#leisaacBundleStatus").should("contain.text", "has not been applied");
    cy.get("#leisaacBundleLibrary").should("contain.text", "custom-so101");

    cy.then(() => {
      liveReady = true;
      selectionPending = true;
    });
    cy.window().then((win) => win.__NPA_AGENT_TEST__.refreshLeIsaacCapability("mock-bundles"));
    cy.wait("@bundleStatus");
    cy.get("#leisaacConfigurationStatus")
      .should("contain.text", "differ from this runtime")
      .and("contain.text", "controller lease");
    cy.get("#leisaacConnect").should("not.be.disabled");
    cy.get("#leisaacBundleSelection").select(digest);
    cy.get("#leisaacBundleSelect").should("be.disabled");
    grantMockControllerLease();
    cy.get("#leisaacBundleSelect")
      .should("not.be.disabled")
      .and("have.attr", "aria-disabled", "false")
      .click();
    cy.wait("@bundleSelect");
    cy.get("#leisaacBundleStatus")
      .should("contain.text", "Selected robot bundle")
      .and("contain.text", "restart accepted");
    cy.get("#leisaacResetDefaults").should("be.disabled");
    cy.get("#leisaacControllerPrerequisite").should("contain.text", "controller lease");
    grantMockControllerLease();
    cy.get("#leisaacResetDefaults").should("not.be.disabled").click();
    cy.wait("@bundleReset");
    cy.get("#leisaacBundleStatus").should("contain.text", "Built-in defaults selected");
  });

  it("falls back explicitly and self-heals after bounded preferred-transport retries", () => {
    const modeFields = {
      view_mode_contract: {
        default: "single_fast",
        values: ["single_fast", "dual_slow"],
        labels: { single_fast: "Fast single", dual_slow: "Slower dual" },
      },
      recording_camera_contract: {
        default: "primary_only",
        values: ["primary_only", "primary_and_secondary"],
        labels: {
          primary_only: "Primary only",
          primary_and_secondary: "Primary + secondary",
        },
      },
      requested_view_mode: "single_fast",
      applied_view_mode: "single_fast",
      view_revision: 0,
      requested_recording_camera_mode: "primary_only",
      applied_recording_camera_mode: "primary_only",
      recording_revision: 0,
    };
    let authoritativeRecordingRevision = 0;
    let authoritativeViewRevision = 0;
    const recordingRevisions = [];
    cy.intercept("GET", "/api/leisaac/status?run_id=mock-ws-fallback", (request) => {
      request.reply({
        statusCode: 200,
        body: {
        available: true,
        run_id: "mock-ws-fallback",
        task: "LeIsaac-SO101-PickOrange-v0",
        stream_transport: "websocket-v1",
        control_ws_url: "/api/leisaac/transport/control?run_id=mock-ws-fallback",
        video_ws_url: "/api/leisaac/transport/video?run_id=mock-ws-fallback",
        frame_url: "/api/leisaac/frame.jpg?run_id=mock-ws-fallback",
        input_url: "/api/leisaac/input?run_id=mock-ws-fallback",
        ...modeFields,
        requested_view_mode: authoritativeViewRevision ? "dual_slow" : "single_fast",
        applied_view_mode: authoritativeViewRevision ? "dual_slow" : "single_fast",
        view_revision: authoritativeViewRevision,
        applied_view_revision: authoritativeViewRevision,
        requested_recording_camera_mode: authoritativeRecordingRevision ? "primary_and_secondary" : "primary_only",
        applied_recording_camera_mode: authoritativeRecordingRevision ? "primary_and_secondary" : "primary_only",
        recording_revision: authoritativeRecordingRevision,
        applied_recording_revision: authoritativeRecordingRevision,
        },
      });
    }).as("wsFallbackStatus");
    cy.intercept("GET", "/api/leisaac/status?run_id=mock-run", {
      statusCode: 200,
      body: {
        available: true,
        run_id: "mock-ws-fallback",
        task: "LeIsaac-SO101-PickOrange-v0",
        stream_transport: "websocket-v1",
        control_ws_url: "/api/leisaac/transport/control?run_id=mock-ws-fallback",
        video_ws_url: "/api/leisaac/transport/video?run_id=mock-ws-fallback",
        frame_url: "/api/leisaac/frame.jpg?run_id=mock-ws-fallback",
        input_url: "/api/leisaac/input?run_id=mock-ws-fallback",
        ...modeFields,
      },
    });
    cy.intercept("GET", "/api/leisaac/status", (request) => {
      request.reply({
        statusCode: 200,
        body: {
        available: true,
        run_id: "mock-ws-fallback",
        task: "LeIsaac-SO101-PickOrange-v0",
        stream_transport: "websocket-v1",
        control_ws_url: "/api/leisaac/transport/control?run_id=mock-ws-fallback",
        video_ws_url: "/api/leisaac/transport/video?run_id=mock-ws-fallback",
        frame_url: "/api/leisaac/frame.jpg?run_id=mock-ws-fallback",
        input_url: "/api/leisaac/input?run_id=mock-ws-fallback",
        ...modeFields,
        requested_view_mode: authoritativeViewRevision ? "dual_slow" : "single_fast",
        applied_view_mode: authoritativeViewRevision ? "dual_slow" : "single_fast",
        view_revision: authoritativeViewRevision,
        applied_view_revision: authoritativeViewRevision,
        requested_recording_camera_mode: authoritativeRecordingRevision ? "primary_and_secondary" : "primary_only",
        applied_recording_camera_mode: authoritativeRecordingRevision ? "primary_and_secondary" : "primary_only",
        recording_revision: authoritativeRecordingRevision,
        applied_recording_revision: authoritativeRecordingRevision,
        },
      });
    });
    cy.intercept("POST", "/api/leisaac/select", {
      statusCode: 200,
      body: { selected: true },
    });
    cy.intercept("POST", "/api/leisaac/input?run_id=mock-ws-fallback", (request) => {
      expect(request.headers["x-npa-leisaac-control"]).to.equal("1");
      expect(request.body).to.include({
        v: 1,
        type: "recording-cameras",
        mode: "primary_and_secondary",
      });
      expect(request.body.revision).to.be.greaterThan(0);
      recordingRevisions.push(request.body.revision);
      authoritativeRecordingRevision = request.body.revision;
      request.reply({
        statusCode: 202,
        body: { v: 1, type: "ack", phase: "accepted" },
      });
    }).as("fallbackModeControl");
    cy.intercept("POST", "/api/leisaac/input?run_id=mock-ws-fallback", (request) => {
      if (request.body.type !== "view-mode") return;
      expect(request.headers["x-npa-leisaac-control"]).to.equal("1");
      if (request.body.mode !== "dual_slow") {
        request.reply({
          statusCode: 202,
          body: { v: 1, type: "ack", phase: "accepted" },
        });
        return;
      }
      expect(request.body).to.include({
        v: 1,
        type: "view-mode",
        mode: "dual_slow",
      });
      expect(request.body.revision).to.be.greaterThan(0);
      authoritativeViewRevision = request.body.revision;
      request.reply({
        statusCode: 202,
        body: { v: 1, type: "ack", phase: "accepted" },
      });
    }).as("preferredGapModeControl");
    cy.intercept("GET", "/api/leisaac/frame.jpg?run_id=mock-ws-fallback&frame=*", {
      statusCode: 200,
      headers: { "content-type": "image/svg+xml" },
      body: '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="#16a34a"/></svg>',
    }).as("wsFallbackFrame");
    cy.window().then((win) => {
      let failuresRemaining = 6;
      const preferredFrame = () => {
        const jpeg = new win.Uint8Array([0xff, 0xd8, 1, 2, 3, 4, 0xff, 0xd9]);
        const payload = new win.ArrayBuffer(128 + jpeg.length);
        const view = new win.DataView(payload);
        [0x4e, 0x50, 0x41, 0x46].forEach((value, index) => view.setUint8(index, value));
        view.setUint8(4, 2);
        view.setUint16(6, 128, false);
        view.setBigUint64(8, 1n, false);
        view.setBigUint64(16, BigInt(win.Date.now()) * 1000000n, false);
        view.setBigUint64(24, 1n, false);
        view.setBigUint64(32, BigInt(win.Date.now()) * 1000000n, false);
        view.setBigUint64(40, 2n, false);
        view.setBigUint64(48, 3n, false);
        view.setBigUint64(56, 4n, false);
        view.setBigUint64(64, 5n, false);
        view.setUint32(88, jpeg.length, false);
        new win.Uint8Array(payload, 128).set(jpeg);
        return payload;
      };
      class RecoveringWebSocket {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSED = 3;
        constructor(url) {
          this.url = String(url);
          this.readyState = RecoveringWebSocket.CONNECTING;
          if (failuresRemaining > 0) {
            failuresRemaining -= 1;
            win.setTimeout(() => this.onerror && this.onerror(new Error("blocked")), 0);
            return;
          }
          win.setTimeout(() => {
            this.readyState = RecoveringWebSocket.OPEN;
            if (this.url.includes("/control")) win.__mockLeIsaacRecoveredControl = this;
            if (this.onopen) this.onopen({ target: this });
            if (this.url.includes("/video") && this.onmessage) {
              this.onmessage({ data: preferredFrame() });
            }
          }, this.url.includes("/control") ? 50 : 0);
        }
        send(raw) {
          if (!this.url.includes("/control")) return;
          const message = JSON.parse(String(raw));
          if (message.type === "view-mode") {
            authoritativeViewRevision = message.revision;
            return;
          }
          if (message.type === "recording-cameras") {
            recordingRevisions.push(message.revision);
            authoritativeRecordingRevision = message.revision;
            return;
          }
          if (message.type !== "resume") return;
          win.setTimeout(() => {
            if (this.onmessage) this.onmessage({ data: JSON.stringify({
              v: 1,
              type: "resumed",
              run_id: message.run_id,
              client_id: message.client_id,
              lease_id: "a".repeat(64),
              lease_generation: 1,
              next_seq: 1,
              last_applied_seq: 0,
              keys_down: [],
            }) });
          }, 0);
        }
        close() { this.readyState = RecoveringWebSocket.CLOSED; }
      }
      win.WebSocket = RecoveringWebSocket;
      win.createImageBitmap = async () => ({ width: 1280, height: 720, close() {} });
      const nativeGetContext = win.HTMLCanvasElement.prototype.getContext;
      win.HTMLCanvasElement.prototype.getContext = function getContext(kind, ...args) {
        if (this.id === "leisaacCanvas" && kind === "2d") return { drawImage() {} };
        return nativeGetContext.call(this, kind, ...args);
      };
    });
    cy.window().then((win) =>
      win.__NPA_AGENT_TEST__.refreshLeIsaacCapability("mock-ws-fallback"),
    );
    cy.wait("@wsFallbackStatus");
    cy.get("#tabLeIsaac").click();
    cy.get("#leisaacRecordingCameras").should("be.disabled");
    cy.get("#leisaacConnect").click();
    cy.wait("@wsFallbackFrame");
    cy.get("#leisaacTransportStatus", { timeout: 10000 })
      .should("contain.text", "JPEG polling")
      .and("contain.text", "fallback");
    cy.get("#leisaacFrame").should("be.visible");
    cy.get("#leisaacTransportStatus", { timeout: 10000 })
      .should("contain.text", "WebSocket")
      .and("contain.text", "latest-frame-wins");
    cy.get("#leisaacFrame").should("not.be.visible");
    cy.get("#leisaacCanvas").should("be.visible");
    cy.get("#leisaacRecordingCameras")
      .should("not.be.disabled")
      .select("primary_and_secondary");
    cy.get("#leisaacModeStatus", { timeout: 5000 }).should(
      "contain.text",
      "recording: Primary + secondary",
    );
    cy.window().then((win) => {
      const control = win.__mockLeIsaacRecoveredControl;
      expect(control, "recovered preferred control socket").to.exist;
      control.readyState = win.WebSocket.CLOSED;
      if (control.onclose) control.onclose({ target: control });
    });
    cy.get("#leisaacViewMode", { timeout: 10000 })
      .should("not.be.disabled")
      .select("dual_slow");
    cy.window().should((win) => {
      const evidence = win.__NPA_AGENT_TEST__.leisaacTransportEvidence();
      expect(evidence.active).to.equal("websocket-v1");
      expect(evidence.video).to.equal("websocket-v1");
      expect(evidence.reconnects).to.be.greaterThan(0);
    });
    cy.then(() => {
      expect(recordingRevisions.length, "recording mode retransmissions").to.be.greaterThan(1);
      expect(new Set(recordingRevisions).size, "stable recording revision across reconnects").to.equal(1);
    });
    cy.window().then((win) => {
      // A bundle restarts only the simulator child. The authenticated socket
      // stays open, but runtime mode state returns to safe revision-zero
      // defaults and must be restored without waiting for another resume ack.
      authoritativeRecordingRevision = 0;
      authoritativeViewRevision = 0;
      return win.__NPA_AGENT_TEST__.refreshLeIsaacCapability("mock-ws-fallback");
    });
    cy.window().should(() => {
      expect(authoritativeRecordingRevision, "recording mode restored after child restart").to.be.greaterThan(0);
      expect(authoritativeViewRevision, "view mode restored after child restart").to.be.greaterThan(0);
    });
    cy.then(() => {
      expect(new Set(recordingRevisions).size, "stable recording revision after child restart").to.equal(1);
    });
    cy.get("#leisaacDisconnect").click();
  });
});
