const requiredLiveEnv = [
  "NPA_AGENT_BASE_URL",
  "NPA_AGENT_USER",
  "NPA_AGENT_PASSWORD",
  "NPA_AGENT_RUN_ID",
  "NPA_AGENT_TASK",
];

// Chromium may emit this observer-delivery diagnostic after Cypress restores a
// full-page screenshot viewport. It is not an application exception and does
// not invalidate the explicit viewport, media, or synchronization assertions.
Cypress.on("uncaught:exception", (error) => {
  const message = String(
    (error && (error.stack || error.message)) || error || "",
  );
  if (
    message.includes(
      "ResizeObserver loop completed with undelivered notifications.",
    )
  ) {
    return false;
  }
  return undefined;
});

function hasLiveEnv() {
  return requiredLiveEnv.every((name) => Boolean(Cypress.env(name)));
}

function runId() {
  return String(Cypress.env("NPA_AGENT_RUN_ID") || "").trim();
}

function expectedTask() {
  return String(
    Cypress.env("NPA_AGENT_TASK") || "LeIsaac-SO101-PickOrange-v0",
  ).trim();
}

function expectedEnvironment() {
  return String(Cypress.env("NPA_AGENT_ENVIRONMENT_ID") || "").trim();
}

function expectedCompletedEpisodes() {
  return Number(Cypress.env("NPA_AGENT_COMPLETED_EPISODES") || 0);
}

function loadRecorderStatus() {
  const selectedRun = runId();
  return cy.window().then(async (win) => {
    const response = await win.fetch(
      "/api/leisaac/status?run_id=" + encodeURIComponent(selectedRun),
      { credentials: "include", cache: "no-store" },
    );
    if (!response.ok) {
      throw new Error(
        "recorder status returned HTTP " + String(response.status),
      );
    }
    return response.json();
  });
}

function waitForRecorder(predicate) {
  return loadRecorderStatus().then((status) => {
    const recorder = status.recorder || {};
    if (predicate(recorder, status)) return status;
    if (recorder.last_error) {
      throw new Error(
        "recorder transition failed: " + String(recorder.last_error),
      );
    }
    return cy.wait(500).then(() => waitForRecorder(predicate));
  });
}

function waitForCapability(predicate) {
  return loadRecorderStatus().then((status) => {
    if (predicate(status)) return status;
    if (status.available === false && String(status.reason || "").includes("failed")) {
      throw new Error("LeIsaac capability failed: " + String(status.reason));
    }
    return cy.wait(1000).then(() => waitForCapability(predicate));
  });
}

function uploadAndApplyBundle(kind, name, fileName, contents) {
  cy.get("#leisaacBundleName")
    .clear()
    .type(name, { delay: 0 })
    .should("have.value", name);
  cy.get("#leisaacBundleKind").select(kind);
  cy.get("#leisaacBundleFiles").selectFile({
    contents: Cypress.Buffer.from(contents),
    fileName,
    mimeType: kind === "device" ? "application/json" : "application/octet-stream",
  });
  cy.get("#leisaacBundleUpload").click();
  cy.get("#leisaacBundleStatus", { timeout: 30000 }).should(
    "contain.text",
    "Validated immutable " + kind + " bundle",
  );
  cy.get("#leisaacBundleSelection").should("not.have.value", "");
  cy.get("#leisaacBundleSelect").click();
  cy.get("#leisaacBundleStatus", { timeout: 30000 })
    .should("contain.text", "Selected " + kind + " bundle")
    .and("contain.text", "restart accepted");
  return waitForCapability(
    (status) =>
      status.available === true &&
      Array.isArray(status.cameras) &&
      status.cameras.includes("workspace") &&
      status.selected_bundles &&
      status.selected_bundles[kind] &&
      status.selected_bundles[kind].name === name,
  );
}

function verifyExactUploadedEpisode(expectSecondary = true) {
  cy.get("#leisaacViewUploadedEpisode").should("be.visible").click();
  cy.get("#leisaacEpisodePlayer", { timeout: 30000 }).should("be.visible");
  cy.get("#leisaacEpisodePlayerTitle").invoke("text").should(
    "match",
    /Episode \d+ · (success|failure)/,
  );
  cy.get("#leisaacEpisodeMetadata")
    .should("contain.text", expectedTask())
    .and("contain.text", expectedEnvironment())
    .and("contain.text", "v000");
  cy.get("#leisaacEpisodeChecksum")
    .invoke("text")
    .should("contain", "verified")
    .and("match", /(?:primary|workspace)=/);
  if (expectSecondary) {
    cy.get("#leisaacEpisodeChecksum").invoke("text").should("match", /(?:secondary|overview)=/);
    cy.get("#leisaacEpisodeSecondaryPane").should("be.visible");
    cy.get("#leisaacEpisodeSingleCamera").should("not.be.visible");
  } else {
    cy.get("#leisaacEpisodeChecksum").invoke("text").should("not.match", /(?:secondary|overview)=/);
    cy.get("#leisaacEpisodeSecondaryPane").should("not.be.visible");
    cy.get("#leisaacEpisodeSingleCamera").should("be.visible");
  }
  cy.get("#leisaacEpisodePrimaryVideo", { timeout: 30000 }).should(($video) => {
    expect(Number($video[0].duration), "primary video duration").to.be.greaterThan(0);
  });
  if (expectSecondary) {
    cy.get("#leisaacEpisodeSecondaryVideo", { timeout: 30000 }).should(($video) => {
      expect(Number($video[0].duration), "secondary video duration").to.be.greaterThan(0);
    });
  }
  cy.get("#leisaacEpisodeTimeline").should(($timeline) => {
    expect(Number($timeline.attr("max")), "timeline rows").to.be.greaterThan(1);
  });
  cy.get("#leisaacEpisodeTimeline").then(($timeline) => {
    const target = Math.max(1, Math.floor(Number($timeline.attr("max")) / 2));
    cy.wrap($timeline).invoke("val", target).trigger("input");
  });
  cy.get("#leisaacEpisodeTimelineValues")
    .should("contain.text", '"action"')
    .and("contain.text", '"observation.state"')
    .and("contain.text", '"reward"');
  cy.get("#leisaacEpisodeRate").select("2");
  cy.get("#leisaacEpisodePrimaryVideo").should(($video) => {
    expect($video[0].playbackRate).to.equal(2);
  });
  if (expectSecondary) {
    cy.get("#leisaacEpisodeSecondaryVideo").should(($video) => {
      expect($video[0].playbackRate).to.equal(2);
    });
  }
  cy.get("#leisaacEpisodeRecordsExport")
    .should("have.attr", "href")
    .and("include", "/download/records");
  cy.get("#leisaacEpisodeMetadataExport")
    .should("have.attr", "href")
    .and("include", "/download/metadata");

  cy.window().then({ timeout: 60000 }, async (win) => {
    const primary = win.document.getElementById("leisaacEpisodePrimaryVideo");
    const secondary = win.document.getElementById("leisaacEpisodeSecondaryVideo");
    const mediaUrl = String(primary.getAttribute("src") || "");
    expect(mediaUrl, "exact immutable media URL").to.include("version_id=v000");
    for (const expected of [
      ["bytes=0-63", 206],
      ["bytes=-64", 206],
      ["bytes=64-", 206],
      ["bytes=999999999-", 416],
    ]) {
      const response = await win.fetch(mediaUrl, {
        credentials: "include",
        cache: "no-store",
        headers: { Range: expected[0] },
      });
      expect(response.status, expected[0]).to.equal(expected[1]);
      expect(response.headers.get("accept-ranges"), expected[0]).to.equal("bytes");
      expect(response.headers.get("content-range"), expected[0]).to.match(
        expected[1] === 206 ? /^bytes \d+-\d+\/\d+$/ : /^bytes \*\/\d+$/,
      );
      if (expected[1] === 206) {
        expect(response.headers.get("content-type"), expected[0]).to.contain(
          "video/mp4",
        );
      }
      await response.arrayBuffer();
    }
    const duration = Number(primary.duration);
    expect(duration, "primary duration").to.be.greaterThan(0);
    primary.currentTime = duration * 0.5;
    primary.dispatchEvent(new win.Event("seeking"));
    await new Promise((resolve) => win.setTimeout(resolve, 250));
    if (expectSecondary) {
      expect(
        Math.abs(Number(primary.currentTime) - Number(secondary.currentTime)),
        "synchronized camera seek",
      ).to.be.lessThan(0.12);
    }
  });
  cy.get("#leisaacEpisodePlayer").focus().trigger("keydown", { key: "ArrowRight" });
  cy.get("#leisaacEpisodeDescribe").click();
  cy.get("#chatLog .msg-row.user", { timeout: 5000 }).should(
    "contain.text",
    "Describe this",
  );
  cy.screenshot(expectSecondary
    ? "07-two-camera-episode-playback"
    : "06-primary-only-episode-playback", {
    capture: "fullPage",
  });
}

function dispatchTeleoperation(keys) {
  cy.get("#leisaacStreamHost")
    .click()
    .then(($host) => {
      for (const key of keys) {
        $host[0].dispatchEvent(
          new KeyboardEvent("keydown", { key, bubbles: true }),
        );
        $host[0].dispatchEvent(
          new KeyboardEvent("keyup", { key, bubbles: true }),
        );
      }
    });
}

function redactLiveEvidenceIdentifiers() {
  cy.document().then((doc) => {
    const style = doc.createElement("style");
    style.setAttribute("data-leisaac-evidence-redaction", "true");
    style.textContent = `
      #panelLeIsaac .leisaac-head > div:first-child > .hint:first-of-type strong,
      #panelLeIsaac .leisaac-head > div:first-child > .hint:nth-of-type(2) code,
      #leisaacRecorderArtifact,
      #leisaacEpisodeVersion,
      #leisaacEpisodeList,
      #leisaacEpisodeMetadata {
        visibility: hidden !important;
      }
    `;
    doc.head.appendChild(style);
  });
}

function waitForPaintedOrbitRevision() {
  let revision = 0;
  return cy.get("#leisaacStreamStatus")
    .should(($status) => {
      const match = String($status.text()).match(/orbit revision (\d+) accepted/);
      expect(match, "accepted orbit revision").not.to.equal(null);
      revision = Number(match[1]);
    })
    .then(() => {
      return cy.window().then(async (win) => {
        const deadline = win.performance.now() + 30000;
        while (win.performance.now() < deadline) {
          const frames = win.__NPA_AGENT_TEST__.leisaacTransportEvidence().frames;
          if (frames.some((frame) =>
            frame.camera === "workspace" && Number(frame.view_revision || 0) >= revision
          )) return revision;
          await new Promise((resolve) => win.setTimeout(resolve, 100));
        }
        throw new Error(`orbit revision ${revision} was not painted`);
      });
    });
}

function recordEpisode(outcome, episodeNumber, completedBefore) {
  cy.get("#leisaacRecordStart").should("not.be.disabled").click();
  return waitForRecorder(
    (recorder) =>
      recorder.state === "recording" &&
      Boolean(recorder.active_episode) &&
      Number(recorder.frame_count || 0) >= 2,
  )
    .then((recordingStatus) => {
      cy.get("#leisaacRecorderStatus")
        .should("contain.text", "State: recording")
        .and("not.contain.text", "active: none");
      cy.get("#leisaacRecordStart").should("be.disabled");
      cy.get("#leisaacRecordSuccess").should("not.be.disabled");
      cy.get("#leisaacRecordFailure").should("not.be.disabled");
      cy.get("#leisaacRecordFinalize")
        .should("be.disabled")
        .and("have.attr", "title")
        .and("contain", "Mark success or failure");
      cy.get("#leisaacRecorderGuidance").should(
        "contain.text",
        "Recording live simulator frames",
      );
      cy.screenshot(
        `0${4 + episodeNumber * 4}-episode-${episodeNumber}-${outcome}-recording`,
        { capture: "viewport" },
      );

      const inputBefore = Number(recordingStatus.input_events || 0);
      const framesBefore = Number(recordingStatus.recorder.frame_count || 0);
      dispatchTeleoperation(
        outcome === "success" ? ["W", "A", "U"] : ["S", "D", "O"],
      );
      return waitForRecorder(
        (recorder, status) =>
          recorder.state === "recording" &&
          Number(recorder.frame_count || 0) > framesBefore &&
          Number(status.input_events || 0) >= inputBefore + 3 &&
          Number(status.input_events || 0) ===
            Number(status.applied_inputs || 0),
      );
    })
    .then((appliedStatus) => {
      expect(appliedStatus.input_events, "accepted inputs").to.equal(
        appliedStatus.applied_inputs,
      );
      cy.get(
        outcome === "success"
          ? "#leisaacRecordSuccess"
          : "#leisaacRecordFailure",
      )
        .should("not.be.disabled")
        .click();
      return waitForRecorder(
        (recorder) =>
          recorder.state === "outcome-pending" &&
          recorder.pending_outcome === outcome,
      );
    })
    .then(() => {
      cy.get("#leisaacRecorderStatus").should(
        "contain.text",
        "State: outcome-pending",
      );
      cy.get("#leisaacRecorderGuidance").should(
        "contain.text",
        `Outcome selected: ${outcome}`,
      );
      cy.get("#leisaacRecordFinalize").should("not.be.disabled");
      cy.get("#leisaacRecordStart").should("be.disabled");
      cy.screenshot(
        `0${5 + episodeNumber * 4}-episode-${episodeNumber}-${outcome}-selected`,
        { capture: "viewport" },
      );
      cy.get("#leisaacRecordFinalize").click();
      return waitForRecorder(
        (recorder) =>
          recorder.state === "idle" &&
          recorder.last_upload_status === "uploaded" &&
          recorder.last_outcome === outcome &&
          Number(recorder.completed_episode_count || 0) ===
            completedBefore + 1 &&
          Boolean(recorder.dataset_version_uri) &&
          Boolean(recorder.last_episode_commit_uri),
      );
    })
    .then((completedStatus) => {
      cy.get("#leisaacRecorderStatus")
        .should("contain.text", "State: idle")
        .and("contain.text", `completed: ${completedBefore + 1}`)
        .and("contain.text", `${outcome}/uploaded`);
      cy.get("#leisaacRecorderArtifact")
        .should("contain.text", "Immutable dataset")
        .and(
          "contain.text",
          String(completedStatus.recorder.dataset_version_uri),
        );
      cy.get("#leisaacRecorderGuidance").should(
        "contain.text",
        "Upload complete",
      );
      cy.get("#leisaacRecordStart").should("not.be.disabled");
      cy.get("#leisaacRecordFinalize").should("be.disabled");
      cy.screenshot(
        `0${6 + episodeNumber * 4}-episode-${episodeNumber}-${outcome}-uploaded`,
        { capture: "viewport" },
      );
    });
}

(hasLiveEnv() ? describe : describe.skip)(
  "NPA agent live LeIsaac teleoperation",
  () => {
    beforeEach(() => {
      cy.viewport(1440, 1050);
      cy.visitLiveAgent();
    });

    it("discovers, renders, and controls the selected task through public authenticated HTTPS", () => {
      const selectedRun = runId();
      const selectedTask = expectedTask();
      const selectedEnvironment = expectedEnvironment();
      const completedEpisodes = expectedCompletedEpisodes();
      cy.get("#tabLeIsaac", { timeout: 30000 }).should("be.visible").click();
      cy.get("#panelLeIsaac")
        .should("have.class", "is-active")
        .and("have.attr", "data-run-id", selectedRun);
      cy.get("#panelLeIsaac .hint")
        .should("contain.text", selectedTask)
        .and("contain.text", selectedEnvironment)
        .and("contain.text", "RTX PRO 6000");
      redactLiveEvidenceIdentifiers();
      cy.get("#leisaacViewMode").should("have.value", "single_fast");
      cy.get("#leisaacRecordingCameras").should("have.value", "primary_only");
      cy.get("#leisaacLiveGrid").should("have.class", "is-single");
      cy.get("#leisaacRecorderStatus").should("contain.text", "State: idle");
      if (completedEpisodes > 0) {
        cy.get("#leisaacRecorderStatus")
          .should("contain.text", `completed: ${completedEpisodes}`)
          .and("contain.text", "uploaded");
      }
      cy.window().then(async (win) => {
        const response = await win.fetch(
          "/api/leisaac/status?run_id=" + encodeURIComponent(selectedRun),
          { credentials: "include", cache: "no-store" },
        );
        expect(response.ok, "authenticated capability status").to.equal(true);
        const status = await response.json();
        expect(status.available).to.equal(true);
        expect(status.task).to.equal(selectedTask);
        expect(status.environment_id).to.equal(selectedEnvironment);
        expect(status.recorder.state).to.equal("idle");
        if (completedEpisodes > 0) {
          expect(status.recorder.completed_episode_count).to.equal(
            completedEpisodes,
          );
        }
        expect(status.stream_transport).to.equal("websocket-v1");
        expect(status.control_ws_url).to.match(
          /^\/api\/leisaac\/transport\/control\?run_id=/,
        );
        expect(status.video_ws_url).to.match(
          /^\/api\/leisaac\/transport\/video\?run_id=/,
        );
        expect(status.frame_url).to.match(
          /^\/api\/leisaac\/frame\.jpg\?run_id=/,
        );
        expect(status.input_url).to.match(/^\/api\/leisaac\/input\?run_id=/);
        expect(status.gpu).to.contain("RTX PRO 6000");
        expect(status.frame_bytes, "substantive JPEG payload").to.be.greaterThan(
          1024,
        );
        win.__LEISAAC_INITIAL_STATUS__ = status;
      });

      cy.get("#leisaacConnect").click();
      cy.get("#leisaacStreamStatus", { timeout: 120000 }).should(
        "contain.text",
        "keyboard teleoperation active",
      );
      cy.get("#leisaacTransportStatus", { timeout: 120000 })
        .should("contain.text", "WebSocket")
        .and("contain.text", "preferred")
        .and("contain.text", "latest-frame-wins");
      cy.get("#leisaacLatencyStatus")
        .should("contain.text", "Latency: control")
        .and("contain.text", "FPS")
        .and("contain.text", "dropped/coalesced");
      cy.get("#leisaacCanvas", { timeout: 120000 })
        .should("be.visible")
        .and(($canvas) => {
          expect($canvas[0].width, "decoded frame width").to.be.greaterThan(640);
          expect($canvas[0].height, "decoded frame height").to.be.greaterThan(360);
          const context = $canvas[0].getContext("2d", {
            willReadFrequently: true,
          });
          const pixels = context.getImageData(
            0,
            0,
            $canvas[0].width,
            $canvas[0].height,
          ).data;
          let minimum = 255;
          let maximum = 0;
          for (let index = 0; index < pixels.length; index += 16) {
            minimum = Math.min(
              minimum,
              pixels[index],
              pixels[index + 1],
              pixels[index + 2],
            );
            maximum = Math.max(
              maximum,
              pixels[index],
              pixels[index + 1],
              pixels[index + 2],
            );
          }
          expect(
            maximum - minimum,
            "nonblank real rendered pixels",
          ).to.be.greaterThan(24);
        });
      cy.get("#leisaacSecondaryHost").should("not.be.visible");
      cy.window().then((win) => {
        win.__LEISAAC_CONTROLS_BEFORE_ORBIT__ =
          win.__NPA_AGENT_TEST__.leisaacTransportEvidence().controls.length;
      });
      cy.screenshot("01-fast-single-selector-one-large-viewport", { capture: "viewport" });
      cy.screenshot("02-fast-single-horizontal-orbit-before", { capture: "viewport" });
      cy.get("#leisaacStreamHost")
        .trigger("pointerdown", {
          pointerId: 41,
          pointerType: "mouse",
          clientX: 300,
          clientY: 240,
        })
        .trigger("pointermove", {
          pointerId: 41,
          pointerType: "mouse",
          clientX: 380,
          clientY: 240,
        })
        .trigger("pointerup", { pointerId: 41, pointerType: "mouse" });
      cy.get("#leisaacStreamStatus").should("contain.text", "orbit revision");
      waitForPaintedOrbitRevision();
      cy.screenshot("02-fast-single-horizontal-orbit-after", { capture: "viewport" });
      cy.screenshot("03-fast-single-touch-vertical-orbit-before", { capture: "viewport" });
      cy.get("#leisaacStreamHost")
        .trigger("pointerdown", {
          pointerId: 42,
          pointerType: "touch",
          clientX: 340,
          clientY: 280,
        })
        .trigger("pointermove", {
          pointerId: 42,
          pointerType: "touch",
          clientX: 340,
          clientY: 205,
        })
        .trigger("pointerup", { pointerId: 42, pointerType: "touch" });
      waitForPaintedOrbitRevision();
      cy.screenshot("03-fast-single-touch-vertical-orbit-after", { capture: "viewport" });
      cy.window().then((win) => {
        expect(
          win.__NPA_AGENT_TEST__.leisaacTransportEvidence().controls.length,
          "orbit gestures emit no robot controls",
        ).to.equal(win.__LEISAAC_CONTROLS_BEFORE_ORBIT__);
      });

      cy.get("#panelLeIsaac").then(($panel) => {
        cy.get("#leisaacViewMode").select("dual_slow");
        cy.get("#leisaacModeStatus", { timeout: 30000 })
          .should("contain.text", "Applied view")
          .and("contain.text", "Dual view");
        cy.get("#panelLeIsaac").then(($same) => {
          expect($same[0], "mode transition keeps the panel mounted").to.equal($panel[0]);
        });
      });
      cy.get("#leisaacSecondaryCanvas", { timeout: 120000 })
        .should("be.visible")
        .and(($canvas) => {
          expect($canvas[0].width, "overview frame width").to.be.greaterThan(640);
          expect($canvas[0].height, "overview frame height").to.be.greaterThan(360);
        });
      cy.screenshot("04-dual-slow-two-distinct-viewports", { capture: "viewport" });
      cy.window().then((win) => {
        const primary = win.document.getElementById("leisaacCanvas");
        const overview = win.document.getElementById("leisaacSecondaryCanvas");
        expect(primary.toDataURL("image/png"), "distinct real camera pixels").not.to.equal(
          overview.toDataURL("image/png"),
        );
      });

      cy.window().then(async (win) => {
        const status = win.__LEISAAC_INITIAL_STATUS__;
        const response = await win.fetch(status.frame_url + "&proof=1", {
          credentials: "include",
          cache: "no-store",
        });
        expect(response.ok, "authenticated frame route").to.equal(true);
        expect(response.headers.get("content-type")).to.contain("image/jpeg");
        const bytes = new Uint8Array(await response.arrayBuffer());
        expect(bytes.length, "real frame bytes").to.be.greaterThan(10000);
        expect(bytes[0]).to.equal(0xff);
        expect(bytes[1]).to.equal(0xd8);

        const bitmap = await win.createImageBitmap(
          new win.Blob([bytes], { type: "image/jpeg" }),
        );
        const canvas = win.document.createElement("canvas");
        canvas.width = 160;
        canvas.height = 90;
        const context = canvas.getContext("2d", { willReadFrequently: true });
        context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        bitmap.close();
        const pixels = context.getImageData(
          0,
          0,
          canvas.width,
          canvas.height,
        ).data;
        let minimum = 255;
        let maximum = 0;
        let total = 0;
        for (let index = 0; index < pixels.length; index += 4) {
          const luma = Math.round(
            pixels[index] * 0.2126 +
              pixels[index + 1] * 0.7152 +
              pixels[index + 2] * 0.0722,
          );
          minimum = Math.min(minimum, luma);
          maximum = Math.max(maximum, luma);
          total += luma;
        }
        expect(maximum - minimum, "rendered frame variance").to.be.greaterThan(
          12,
        );
        expect(
          Math.round(total / (pixels.length / 4)),
          "rendered frame mean luma",
        ).to.be.greaterThan(3);
      });
      cy.screenshot("04-dual-slow-live-detail", { capture: "viewport" });

      const controls = [
        "W",
        "S",
        "A",
        "D",
        "Q",
        "E",
        "U",
        "O",
        "J",
        "L",
        "I",
        "K",
        "L",
      ];
      cy.get("#leisaacStreamHost")
        .click()
        .then(($host) => {
          for (const key of controls) {
            $host[0].dispatchEvent(
              new KeyboardEvent("keydown", { key, bubbles: true }),
            );
            $host[0].dispatchEvent(
              new KeyboardEvent("keyup", { key, bubbles: true }),
            );
          }
        });
      cy.get("#leisaacInputStatus")
        .should("contain.text", "Keyboard events sent: 13")
        .and("contain.text", "last L");
      cy.window().then(async (win) => {
        const initial = win.__LEISAAC_INITIAL_STATUS__;
        while (true) {
          const response = await win.fetch(
            "/api/leisaac/status?run_id=" + encodeURIComponent(selectedRun),
            { credentials: "include", cache: "no-store" },
          );
          const status = await response.json();
          if (
            Number(status.input_events || 0) >=
              Number(initial.input_events || 0) + 13 &&
            Number(status.applied_inputs || 0) >=
              Number(initial.applied_inputs || 0) + 13 &&
            Number(status.input_events || 0) ===
              Number(status.applied_inputs || 0) &&
            String(status.frame_updated_at || "") >
              String(initial.frame_updated_at || "")
          ) {
            win.__LEISAAC_SERVER_INPUT_EVENTS__ = Number(status.input_events);
            win.__LEISAAC_APPLIED_INPUTS__ = Number(status.applied_inputs);
            return;
          }
          await new Promise((resolve) => win.setTimeout(resolve, 500));
        }
      });
      cy.window()
        .its("__LEISAAC_SERVER_INPUT_EVENTS__")
        .should("be.at.least", 13);
      cy.window().its("__LEISAAC_APPLIED_INPUTS__").should("be.at.least", 13);
      cy.window().then((win) => {
        expect(win.__LEISAAC_SERVER_INPUT_EVENTS__, "accepted inputs").to.equal(
          win.__LEISAAC_APPLIED_INPUTS__,
        );
      });
      cy.get("#leisaacCanvas").should(($canvas) => {
        expect($canvas[0].width, "post-input decoded frame").to.be.greaterThan(640);
        expect($canvas[0].height).to.be.greaterThan(360);
      });
      cy.screenshot("05-successful-manipulation", {
        capture: "viewport",
      });

      cy.get("#leisaacRecordStart").should("not.be.disabled");
      cy.get("#leisaacRecordSuccess")
        .should("be.disabled")
        .and("have.attr", "title")
        .and("contain", "Start an episode");
      cy.get("#leisaacRecordFinalize")
        .should("be.disabled")
        .and("have.attr", "title")
        .and("contain", "Mark success or failure");
      cy.get("#leisaacRecorderGuidance")
        .invoke("text")
        .should("match", /Start (?:an )?episode/i);
      cy.screenshot("04-recorder-idle-start-enabled", { capture: "viewport" });

      cy.get("#leisaacViewMode").select("single_fast");
      cy.get("#leisaacModeStatus", { timeout: 30000 }).should(
        "contain.text",
        "Applied view: Fast single",
      );
      cy.get("#leisaacRecordingCameras").select("primary_only");
      cy.get("#leisaacModeStatus", { timeout: 30000 })
        .should("contain.text", "Applied view: Fast single")
        .and("contain.text", "recording: Primary only");
      recordEpisode("success", 0, completedEpisodes)
        .then(() => verifyExactUploadedEpisode(false))
        .then(() => {
          cy.get("#leisaacRecordingCameras").select("primary_and_secondary");
          cy.get("#leisaacModeStatus", { timeout: 30000 }).should(
            "contain.text",
            "recording: Primary + secondary",
          );
          cy.get("#leisaacModeWarning").should(
            "contain.text",
            "Two-camera episode recording reduces Fast single performance",
          );
          cy.get("#leisaacViewMode").select("dual_slow");
          cy.get("#leisaacModeStatus", { timeout: 30000 })
            .should("contain.text", "Applied view: Dual view")
            .and("contain.text", "recording: Primary + secondary");
        })
        .then(() => recordEpisode("failure", 1, completedEpisodes + 1))
        .then(() => verifyExactUploadedEpisode(true));
    });

    it("applies checksum-verified SO-101, scene, and device bundles and records their provenance", () => {
      const selectedRun = runId();
      let completedBefore = 0;
      const robot = `#usda 1.0
(
    defaultPrim = "SO101"
)
def Xform "SO101" (
    prepend references = @/opt/leisaac-cache/assets/runtime/robots/so101_follower.usd@
) {}
`;
      const scene = `#usda 1.0
(
    defaultPrim = "Scene"
)
def Xform "Scene" (
    prepend references = @/opt/leisaac-cache/assets/runtime/scenes/table_with_cube/scene.usd@
) {}
`;
      const device = JSON.stringify({
        schema: "npa.leisaac.so101-device.v1",
        driver: "custom-so101",
        action_order: [
          "x",
          "y",
          "z",
          "roll",
          "pitch",
          "yaw",
          "shoulder_pan",
          "gripper",
        ],
        rate_hz: 60,
      });

      cy.get("#tabLeIsaac", { timeout: 30000 }).should("be.visible").click();
      redactLiveEvidenceIdentifiers();
      cy.window().then((win) =>
        win.__NPA_AGENT_TEST__.refreshLeIsaacCapability(selectedRun),
      );
      uploadAndApplyBundle("robot", "queue-custom-so101", "robot.usda", robot)
        .then(() =>
          uploadAndApplyBundle("scene", "queue-custom-table", "scene.usda", scene),
        )
        .then(() =>
          uploadAndApplyBundle(
            "device",
            "queue-custom-so101-device",
            "device.json",
            device,
          ),
        )
        .then(() => {
          cy.get("#leisaacConnect").then(($button) => {
            if (!$button.prop("disabled")) cy.wrap($button).click();
          });
          cy.get("#leisaacStreamStatus", { timeout: 180000 }).should(
            "contain.text",
            "keyboard teleoperation active",
          );
          cy.get("#leisaacRecordingCameras").select("primary_and_secondary");
          cy.get("#leisaacModeStatus", { timeout: 30000 }).should(
            "contain.text",
            "recording: Primary + secondary",
          );
          cy.get("#leisaacViewMode").select("dual_slow");
          cy.get("#leisaacModeStatus", { timeout: 30000 })
            .should("contain.text", "Applied view: Dual view")
            .and("contain.text", "recording: Primary + secondary");
          cy.get("#leisaacInputDevice").select("custom-so101");
          return loadRecorderStatus().then((before) => {
            completedBefore = Number(
              (before.recorder && before.recorder.completed_episode_count) || 0,
            );
            cy.get("#leisaacSendNeutralAction").click();
            return waitForCapability(
              (status) =>
                status.available === true &&
                Number(status.input_events || 0) >=
                  Number(before.input_events || 0) + 1 &&
                Number(status.input_events || 0) ===
                  Number(status.applied_inputs || 0),
            );
          });
        })
        .then(() => {
          cy.screenshot("12-custom-so101-scene-device-selected", {
            capture: "fullPage",
          });
          return recordEpisode("success", 2, completedBefore);
        })
        .then(() => recordEpisode("failure", 3, completedBefore + 1))
        .then(() => verifyExactUploadedEpisode())
        .then(() =>
          loadRecorderStatus().then((status) => {
            expect(status.selected_bundles.robot.name).to.equal(
              "queue-custom-so101",
            );
            expect(status.selected_bundles.scene.name).to.equal(
              "queue-custom-table",
            );
            expect(status.selected_bundles.device.name).to.equal(
              "queue-custom-so101-device",
            );
          }),
        );
    });
  },
);
