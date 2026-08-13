const requiredLiveEnv = [
  "NPA_AGENT_BASE_URL",
  "NPA_AGENT_USER",
  "NPA_AGENT_PASSWORD",
  "NPA_AGENT_RUN_ID",
  "NPA_LEISAAC_BENCHMARK_OUTPUT",
];

function hasLiveEnv() {
  return requiredLiveEnv.every((name) => Boolean(Cypress.env(name)));
}

function numberEnv(name, fallback) {
  const value = Number(Cypress.env(name));
  return Number.isFinite(value) && value >= 0 ? value : fallback;
}

function liveTransportEvidence(win) {
  const accessor = win.__NPA_AGENT_TEST__?.leisaacTransportEvidenceLive;
  if (typeof accessor !== "function") {
    throw new Error("zero-copy LeIsaac performance evidence accessor is unavailable");
  }
  return accessor();
}

function percentile(values, fraction) {
  if (!values.length) return null;
  const sorted = values.slice().sort((left, right) => left - right);
  const point = (sorted.length - 1) * fraction;
  const lower = Math.floor(point);
  const upper = Math.ceil(point);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (point - lower);
}

function distribution(values) {
  const finite = values.filter(Number.isFinite);
  const p50 = percentile(finite, 0.5);
  const absoluteDeviations = Number.isFinite(p50)
    ? finite.map((value) => Math.abs(value - p50))
    : [];
  return {
    n: finite.length,
    p50,
    p90: percentile(finite, 0.9),
    p95: percentile(finite, 0.95),
    p99: percentile(finite, 0.99),
    max: finite.length ? Math.max(...finite) : null,
    jitter_mad: percentile(absoluteDeviations, 0.5),
  };
}

async function waitUntil(win, predicate, timeoutMs, label, intervalMs = 5) {
  const deadline = win.performance.now() + timeoutMs;
  while (win.performance.now() < deadline) {
    const value = predicate();
    if (value) return value;
    await new Promise((resolve) => win.setTimeout(resolve, intervalMs));
  }
  throw new Error(`timed out waiting for ${label}`);
}

function sampleCanvas(win, id) {
  let source = win.document.getElementById(id);
  if (id === "leisaacCanvas" && (!source || source.hidden)) {
    source = win.document.getElementById("leisaacVideo");
  }
  const sourceWidth = Number(source && (source.videoWidth || source.width) || 0);
  const sourceHeight = Number(source && (source.videoHeight || source.height) || 0);
  if (!source || !sourceWidth || !sourceHeight || source.hidden) {
    throw new Error(`canvas ${id} is unavailable or blank`);
  }
  const width = 96;
  const height = 54;
  const probe = win.document.createElement("canvas");
  probe.width = width;
  probe.height = height;
  const context = probe.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("2D probe context is unavailable");
  context.drawImage(source, 0, 0, width, height);
  const rgba = context.getImageData(0, 0, width, height).data;
  const pixels = new Uint8Array(width * height * 3);
  let sum = 0;
  let sumSq = 0;
  let target = 0;
  for (let index = 0; index < rgba.length; index += 4) {
    const red = rgba[index];
    const green = rgba[index + 1];
    const blue = rgba[index + 2];
    pixels[target] = red;
    pixels[target + 1] = green;
    pixels[target + 2] = blue;
    target += 3;
    const luminance = (red + green + blue) / 3;
    sum += luminance;
    sumSq += luminance * luminance;
  }
  const count = width * height;
  const mean = sum / count;
  return {
    pixels,
    mean,
    variance: Math.max(0, sumSq / count - mean * mean),
  };
}

function pixelDifference(left, right) {
  if (!left || !right || left.pixels.length !== right.pixels.length) return null;
  let total = 0;
  for (let index = 0; index < left.pixels.length; index += 1) {
    total += Math.abs(left.pixels[index] - right.pixels[index]);
  }
  return total / left.pixels.length;
}

function frameStageSummary(frames) {
  const cameras = {};
  for (const camera of ["workspace", "overview"]) {
    const selected = frames.filter((frame) => frame.camera === camera);
    const first = selected[0];
    const last = selected[selected.length - 1];
    const seconds = first && last
      ? (last.painted_mono_ms - first.painted_mono_ms) / 1000
      : 0;
    cameras[camera] = {
      delivered_frames: selected.length,
      delivered_fps: seconds > 0 ? (selected.length - 1) / seconds : 0,
      bytes: distribution(selected.map((frame) => frame.bytes)),
      capture_to_encode_ms: distribution(selected.map((frame) => {
        const capture = Number(frame.capture_mono_ns || 0);
        const encoded = Number(frame.encoded_mono_ns || 0);
        return encoded >= capture && capture > 0 ? (encoded - capture) / 1000000 : null;
      })),
      encode_to_runtime_send_ms: distribution(selected.map((frame) => {
        const encoded = Number(frame.encoded_mono_ns || 0);
        const sent = Number(frame.runtime_send_mono_ns || 0);
        return sent >= encoded && encoded > 0 ? (sent - encoded) / 1000000 : null;
      })),
      agent_receive_to_send_ms: distribution(selected.map((frame) => {
        const received = Number(frame.agent_receive_mono_ns || 0);
        const sent = Number(frame.agent_send_mono_ns || 0);
        return sent >= received && received > 0 ? (sent - received) / 1000000 : null;
      })),
      browser_receive_to_paint_ms: distribution(selected.map(
        (frame) => frame.painted_mono_ms - frame.received_mono_ms,
      )),
      capture_to_browser_paint_ms: distribution(selected.map(
        (frame) => frame.frame_age_ms,
      )),
      clock_uncertainty_ms: distribution(selected.map(
        (frame) => frame.clock_uncertainty_ms,
      )),
      inter_frame_ms: distribution(selected.map((frame) => frame.inter_frame_ms)),
      stalls_250ms: selected.filter(
        (frame) => frame.stall === true || Number(frame.inter_frame_ms || 0) >= 250,
      ).length,
    };
  }
  return cameras;
}

(hasLiveEnv() ? describe : describe.skip)(
  "NPA agent live LeIsaac end-to-end performance",
  () => {
    it("measures one explicit view mode from input through causal frame paint", () => {
      Cypress.config("defaultCommandTimeout", 480000);
      const runId = String(
        Cypress.env("NPA_LEISAAC_RUN_ID") || Cypress.env("NPA_AGENT_RUN_ID"),
      );
      const output = String(Cypress.env("NPA_LEISAAC_BENCHMARK_OUTPUT"));
      const viewMode = String(
        Cypress.env("NPA_LEISAAC_VIEW_MODE") || "single_fast",
      );
      if (!["single_fast", "dual_slow"].includes(viewMode)) {
        throw new Error(`unsupported benchmark view mode ${viewMode}`);
      }
      const phase = viewMode;
      const trial = Math.floor(numberEnv("NPA_LEISAAC_BENCHMARK_TRIAL", 0));
      const warmupCount = Math.floor(numberEnv("NPA_LEISAAC_WARMUP_SAMPLES", 10));
      const measuredCount = Math.floor(numberEnv("NPA_LEISAAC_PRIMARY_SAMPLES", 80));
      const periodMs = numberEnv("NPA_LEISAAC_ACTION_PERIOD_MS", 2000);
      const releaseDelayMs = numberEnv("NPA_LEISAAC_RELEASE_DELAY_MS", 120);
      const pixelThreshold = numberEnv("NPA_LEISAAC_PIXEL_DIFF_THRESHOLD", 0);
      const idleFrameCount = Math.floor(numberEnv("NPA_LEISAAC_IDLE_FRAMES", 12));

      cy.viewport(1440, 1050);
      cy.visitLiveAgent();
      cy.window().then((win) => {
        // Capability refresh and the production periodic refresh share this
        // source of truth. Pin it before the direct refresh so a generic active
        // run cannot race the benchmark back to a non-LeIsaac selection.
        win.__NPA_AGENT_TEST__.selectActiveRunId(runId);
        return win.__NPA_AGENT_TEST__.refreshLeIsaacCapability(runId);
      });
      cy.get("#tabLeIsaac", { timeout: 30000 }).should("be.visible").click();

      cy.window().then(async (win) => {
        const benchmark = {
          schema: "npa.leisaac.e2e-performance.v1",
          phase,
          view_mode: viewMode,
          trial,
          source_commit: String(Cypress.env("NPA_LEISAAC_SOURCE_COMMIT") || "unknown"),
          started_at: new Date().toISOString(),
          controls: {
            warmup_samples: warmupCount,
            measured_samples: measuredCount,
            action_period_ms: periodMs,
            release_delay_ms: releaseDelayMs,
            pixel_difference_threshold: pixelThreshold,
            idle_frames: idleFrameCount,
            outlier_rule: "none; only the first fixed warmup_samples are excluded",
          },
          clock_method: {
            primary: "single-browser performance.now() event and painted timestamps",
            simulator: "runtime and simulator monotonic timestamps from one process",
            camera_to_browser:
              "runtime wall clock translated by the measured ping midpoint; uncertainty retained",
          },
          network: {
            user_agent: String(win.navigator.userAgent || ""),
            effective_type: String((win.navigator.connection || {}).effectiveType || ""),
            downlink_mbps: Number((win.navigator.connection || {}).downlink || 0),
            browser_rtt_ms: Number((win.navigator.connection || {}).rtt || 0),
          },
          idle_pixel_differences: [],
          action_samples: [],
          errors: [],
        };
        win.__NPA_LEISAAC_PERFORMANCE__ = benchmark;

        const wireSamples = new Map();
        const nativeSend = win.WebSocket.prototype.send;
        const nativeDataChannelSend = win.RTCDataChannel && win.RTCDataChannel.prototype.send;
        function recordWire(data, bufferedAmount) {
          try {
            const payload = typeof data === "string" ? JSON.parse(data) : null;
            if (payload && (payload.type === "control" || payload.type === "action")) {
              wireSamples.set(Number(payload.seq), {
                mono_ms: win.performance.now(),
                bytes: new win.TextEncoder().encode(data).byteLength,
                buffered_before_bytes: Number(bufferedAmount || 0),
              });
            }
          } catch (_error) {
            // Instrumentation is observational and must never alter the live path.
          }
        }
        win.WebSocket.prototype.send = function instrumentedSend(data) {
          recordWire(data, this.bufferedAmount);
          return nativeSend.call(this, data);
        };
        if (nativeDataChannelSend) {
          win.RTCDataChannel.prototype.send = function instrumentedDataChannelSend(data) {
            recordWire(data, this.bufferedAmount);
            return nativeDataChannelSend.call(this, data);
          };
        }

        const startingFrameCount = liveTransportEvidence(win).frames.length;
        win.document.getElementById("leisaacConnect").click();
        try {
          await waitUntil(
            win,
            () => {
              const evidence = liveTransportEvidence(win);
              return ["websocket-v1", "webrtc-datachannel-v1", "webrtc-native-h264"].includes(evidence.active) &&
                evidence.frames.slice(startingFrameCount)
                  .some((frame) => frame.camera === "workspace");
            },
            numberEnv("NPA_LEISAAC_CONNECT_TIMEOUT_MS", 120000),
            "preferred RTX primary camera",
          );
        } catch (error) {
          const evidence = liveTransportEvidence(win);
          throw new Error(`${error.message}; transport evidence=${JSON.stringify(evidence)}`);
        }
        const selector = win.document.getElementById("leisaacViewMode");
        selector.value = viewMode;
        selector.dispatchEvent(new win.Event("change", { bubbles: true }));
        await waitUntil(
          win,
          () => {
            const text = String(
              win.document.getElementById("leisaacModeStatus")?.textContent || "",
            );
            return text.includes("Applied view") &&
              (viewMode === "single_fast"
                ? text.includes("Fast single")
                : text.includes("Dual view"));
          },
          30000,
          `${viewMode} scheduler acknowledgement`,
        );
        const modeEvidenceStart = liveTransportEvidence(win).frames.length;
        await waitUntil(
          win,
          () => {
            const frames = liveTransportEvidence(win).frames.slice(modeEvidenceStart);
            return frames.filter((frame) => frame.camera === "workspace").length >= 2 &&
              (viewMode === "single_fast" ||
                frames.some((frame) => frame.camera === "overview"));
          },
          30000,
          `${viewMode} steady-state frames`,
        );
        if (viewMode === "single_fast") {
          await new Promise((resolve) => win.setTimeout(resolve, 1200));
          const frames = liveTransportEvidence(win).frames.slice(modeEvidenceStart);
          if (frames.some((frame) => frame.camera === "overview")) {
            throw new Error("Fast single decoded or painted an overview frame");
          }
        }

        const host = win.document.getElementById("leisaacStreamHost");
        if (!host) throw new Error("LeIsaac teleoperation host is missing");
        host.focus();

        let observedFrames = liveTransportEvidence(win).frames.length;
        let previousIdle = sampleCanvas(win, "leisaacCanvas");
        const workspaceIdleFrames = [];
        while (workspaceIdleFrames.length < idleFrameCount) {
          const next = await waitUntil(
            win,
            () => {
              const frames = liveTransportEvidence(win).frames;
              for (let index = observedFrames; index < frames.length; index += 1) {
                if (frames[index].camera === "workspace") return { frame: frames[index], end: frames.length };
              }
              observedFrames = frames.length;
              return null;
            },
            30000,
            `idle workspace frame ${workspaceIdleFrames.length + 1}`,
          );
          observedFrames = next.end;
          const pixels = sampleCanvas(win, "leisaacCanvas");
          benchmark.idle_pixel_differences.push(pixelDifference(previousIdle, pixels));
          workspaceIdleFrames.push(next.frame);
          previousIdle = pixels;
        }

        const keys = ["J", "L"];
        let nextActionAt = win.performance.now() + 500;
        let lastPrimarySequence = 0;
        for (let ordinal = 0; ordinal < warmupCount + measuredCount; ordinal += 1) {
          const remaining = nextActionAt - win.performance.now();
          if (remaining > 0) {
            await new Promise((resolve) => win.setTimeout(resolve, remaining));
          }
          const key = keys[ordinal % keys.length];
          const beforeControls = liveTransportEvidence(win).controls.length;
          const beforePixels = sampleCanvas(win, "leisaacCanvas");
          const eventMonoMs = win.performance.now();
          host.dispatchEvent(new win.KeyboardEvent("keydown", {
            key,
            code: `Key${key}`,
            bubbles: true,
          }));
          let releaseEventMonoMs = null;
          const releaseDispatched = new Promise((resolve) => {
            win.setTimeout(() => {
              releaseEventMonoMs = win.performance.now();
              host.dispatchEvent(new win.KeyboardEvent("keyup", {
                key,
                code: `Key${key}`,
                bubbles: true,
              }));
              resolve();
            }, releaseDelayMs);
          });

          const accepted = await waitUntil(
            win,
            () => liveTransportEvidence(win).controls
              .slice(beforeControls)
              .find((item) => item.phase === "accepted" && item.event === "press" && item.key === key),
            10000,
            `press accepted ${ordinal + 1}`,
          );
          const applied = await waitUntil(
            win,
            () => liveTransportEvidence(win).controls
              .slice(beforeControls)
              .find((item) => item.phase === "applied" && item.seq === accepted.seq),
            10000,
            `press applied ${ordinal + 1}`,
          );
          await releaseDispatched;
          const releaseAccepted = await waitUntil(
            win,
            () => liveTransportEvidence(win).controls
              .slice(beforeControls)
              .find((item) => item.phase === "accepted" && item.event === "release" && item.key === key),
            10000,
            `release accepted ${ordinal + 1}`,
          );
          const releaseApplied = await waitUntil(
            win,
            () => liveTransportEvidence(win).controls
              .slice(beforeControls)
              .find((item) => item.phase === "applied" && item.seq === releaseAccepted.seq),
            10000,
            `release applied ${ordinal + 1}`,
          );

          const appliedMonoNs = Number(applied.simulator_applied_mono_ns || 0);
          let cursor = observedFrames;
          const inspectedFrames = [];
          const causal = await waitUntil(
            win,
            () => {
              const frames = liveTransportEvidence(win).frames;
              for (let index = cursor; index < frames.length; index += 1) {
                const frame = frames[index];
                if (frame.camera !== "workspace") continue;
                if (frame.sequence <= lastPrimarySequence) continue;
                const captureMonoNs = Number(frame.capture_mono_ns || 0);
                const causalSequence = Number(frame.causal_action_sequence || 0);
                const causalAppliedMonoNs = Number(frame.causal_applied_mono_ns || 0);
                const causallyStamped = causalSequence > 0
                  ? causalSequence >= Number(accepted.seq) && causalAppliedMonoNs >= appliedMonoNs
                  : captureMonoNs >= appliedMonoNs;
                if (!(causallyStamped && frame.painted_mono_ms >= eventMonoMs)) continue;
                const pixels = sampleCanvas(win, "leisaacCanvas");
                const difference = pixelDifference(beforePixels, pixels);
                inspectedFrames.push({
                  sequence: frame.sequence,
                  pixel_difference: difference,
                  capture_after_apply_ms: (captureMonoNs - appliedMonoNs) / 1000000,
                });
                if (difference >= pixelThreshold) {
                  return { frame, difference, end: frames.length };
                }
              }
              cursor = frames.length;
              return null;
            },
            12000,
            `causal painted workspace frame ${ordinal + 1}`,
          );
          observedFrames = Math.max(observedFrames, causal.end);
          lastPrimarySequence = causal.frame.sequence;
          const wire = wireSamples.get(accepted.seq) || {};
          const releaseWire = wireSamples.get(releaseAccepted.seq) || {};
          const runtimeReceivedNs = Number(applied.runtime_received_mono_ns || 0);
          const actionSample = {
            ordinal: ordinal + 1,
            warmup: ordinal < warmupCount,
            key,
            press_seq: accepted.seq,
            release_seq: releaseAccepted.seq,
            event_mono_ms: eventMonoMs,
            wire_mono_ms: Number(wire.mono_ms),
            wire_bytes: Number(wire.bytes || 0),
            wire_buffered_before_bytes: Number(wire.buffered_before_bytes || 0),
            event_to_wire_ms: Number(wire.mono_ms) - eventMonoMs,
            event_to_accepted_ack_ms: accepted.event_to_ack_ms,
            event_to_applied_ack_ms: applied.event_to_ack_ms,
            runtime_receive_to_simulator_apply_ms:
              runtimeReceivedNs > 0 && appliedMonoNs >= runtimeReceivedNs
                ? (appliedMonoNs - runtimeReceivedNs) / 1000000
                : null,
            event_to_causal_frame_painted_ms: causal.frame.painted_mono_ms - eventMonoMs,
            applied_ack_to_causal_frame_painted_ms:
              causal.frame.painted_mono_ms - (eventMonoMs + applied.event_to_ack_ms),
            causal_frame_sequence: causal.frame.sequence,
            causal_frame_capture_mono_ns: causal.frame.capture_mono_ns,
            causal_action_sequence: Number(causal.frame.causal_action_sequence || 0),
            causal_applied_mono_ns: String(causal.frame.causal_applied_mono_ns || "0"),
            causal_frame_pixel_difference: causal.difference,
            causal_frame_age_ms: causal.frame.frame_age_ms,
            inspected_frames: inspectedFrames,
            release_event_mono_ms: releaseEventMonoMs,
            release_wire_mono_ms: Number(releaseWire.mono_ms),
            release_event_to_wire_ms: Number(releaseWire.mono_ms) - releaseEventMonoMs,
            release_event_to_applied_ack_ms: releaseApplied.event_to_ack_ms,
            release_applied: true,
          };
          benchmark.action_samples.push(actionSample);
          nextActionAt += periodMs;
          if (win.performance.now() > nextActionAt) nextActionAt = win.performance.now();
        }

        const finalEvidence = win.__NPA_AGENT_TEST__.leisaacTransportEvidence();
        const measured = benchmark.action_samples.filter((sample) => !sample.warmup);
        const workspace = sampleCanvas(win, "leisaacCanvas");
        const overview = viewMode === "dual_slow"
          ? sampleCanvas(win, "leisaacSecondaryCanvas")
          : null;
        benchmark.completed_at = new Date().toISOString();
        benchmark.transport = {
          control: String(finalEvidence.active || ""),
          video: String(finalEvidence.video || ""),
          failures: Array.isArray(finalEvidence.failures)
            ? finalEvidence.failures.slice(-16)
            : [],
          policy: finalEvidence.active === "webrtc-native-h264"
            ? "authenticated reliable control channel; browser-native Kit H.264/NVENC video; presentation via requestVideoFrameCallback"
            : finalEvidence.active === "webrtc-datachannel-v1"
            ? "TURN-only reliable ordered control RTCDataChannel; binary WebSocket latest-frame-wins video"
            : finalEvidence.video === "webrtc-datachannel-v1"
              ? "TURN-only unordered maxRetransmits=0 video; reliable ordered control WebSocket"
              : "binary same-origin WebSocket control/video; CUDA fixed-step simulation; latest-frame-wins video",
        };
        benchmark.quality = {
          workspace: {
            width: 1280,
            height: 720,
            codec: finalEvidence.active === "webrtc-native-h264" ? "H264" : "JPEG",
            jpeg_quality: finalEvidence.active === "webrtc-native-h264" ? null : 82,
            variance: workspace.variance,
          },
          overview: overview
            ? { width: 1280, height: 720, jpeg_quality: 82, variance: overview.variance }
            : null,
          viewport_pixel_difference: overview ? pixelDifference(workspace, overview) : null,
        };
        benchmark.summary = {
          primary_input_to_causal_frame_painted_ms: distribution(
            measured.map((sample) => sample.event_to_causal_frame_painted_ms),
          ),
          input_to_applied_ack_ms: distribution(
            measured.map((sample) => sample.event_to_applied_ack_ms),
          ),
          input_to_wire_ms: distribution(
            measured.map((sample) => sample.event_to_wire_ms),
          ),
          runtime_receive_to_simulator_apply_ms: distribution(
            measured.map((sample) => sample.runtime_receive_to_simulator_apply_ms),
          ),
          applied_ack_to_causal_frame_painted_ms: distribution(
            measured.map((sample) => sample.applied_ack_to_causal_frame_painted_ms),
          ),
          causal_frame_pixel_difference: distribution(
            measured.map((sample) => sample.causal_frame_pixel_difference),
          ),
          idle_frame_pixel_difference: distribution(benchmark.idle_pixel_differences),
          release_input_to_applied_ack_ms: distribution(
            measured.map((sample) => sample.release_event_to_applied_ack_ms),
          ),
          all_safety_releases_applied: measured.every((sample) => sample.release_applied),
          frame_stages: frameStageSummary(finalEvidence.frames),
          dropped_or_coalesced_frames: Number(finalEvidence.dropped_frames || 0),
          reconnects: Number(finalEvidence.reconnects || 0),
          action_samples: measured.length,
        };
        win.WebSocket.prototype.send = nativeSend;
        if (nativeDataChannelSend) win.RTCDataChannel.prototype.send = nativeDataChannelSend;

        if (measured.length !== measuredCount) {
          throw new Error(`expected ${measuredCount} measured actions, got ${measured.length}`);
        }
        if (!benchmark.summary.all_safety_releases_applied) {
          throw new Error("one or more safety releases lacked an applied acknowledgement");
        }
        if (workspace.variance < 25 || (overview && overview.variance < 25)) {
          throw new Error("one or more requested camera canvases are blank or uniform");
        }
        if (overview && benchmark.quality.viewport_pixel_difference < 1) {
          throw new Error("the two camera canvases are not visually distinct");
        }
        if (
          viewMode === "dual_slow" &&
          !(benchmark.summary.frame_stages.overview.delivered_fps >= 2 &&
            benchmark.summary.frame_stages.overview.delivered_fps <= 5)
        ) {
          throw new Error(
            `Dual slow overview FPS is outside 2–5: ${benchmark.summary.frame_stages.overview.delivered_fps}`,
          );
        }
      });

      cy.window()
        .its("__NPA_LEISAAC_PERFORMANCE__")
        .then((benchmark) => {
          return cy.writeFile(output, benchmark, { log: false }).then(() => benchmark);
        })
        .then((benchmark) => {
          const primary = benchmark.summary.primary_input_to_causal_frame_painted_ms;
          return cy.document().then((document) => {
            const previous = document.getElementById("leisaacBenchmarkProof");
            if (previous) previous.remove();
            const proof = document.createElement("section");
            proof.id = "leisaacBenchmarkProof";
            proof.setAttribute("aria-label", "LeIsaac real-browser latency benchmark proof");
            proof.style.cssText = [
              "position:fixed", "right:18px", "bottom:18px", "z-index:2147483647",
              "max-width:520px", "padding:14px 16px", "border:2px solid #3ddc97",
              "border-radius:10px", "background:rgba(7,18,27,.96)", "color:#f4fbff",
              "font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace",
              "box-shadow:0 10px 32px rgba(0,0,0,.45)",
            ].join(";");
            proof.textContent =
              `Real RTX browser benchmark · ${viewMode} · trial ${trial + 1} · n=${primary.n}\n` +
              `input→causal-frame-painted p50 ${primary.p50.toFixed(2)} ms · ` +
              `p95 ${primary.p95.toFixed(2)} ms\n` +
              `primary ${benchmark.summary.frame_stages.workspace.delivered_fps.toFixed(2)} FPS` +
              (viewMode === "dual_slow"
                ? ` · secondary ${benchmark.summary.frame_stages.overview.delivered_fps.toFixed(2)} FPS`
                : " · zero secondary decode/paint") +
              ` · ${benchmark.transport.control}/${benchmark.transport.video}`;
            document.body.appendChild(proof);
            return benchmark;
          });
        });
      // Keep evidence bounded to the two viewports and latency proof. The page
      // below the grid may contain immutable recorder URIs that are irrelevant
      // to this benchmark and must not be copied into screenshots.
      cy.get(".leisaac-live-grid")
        .scrollIntoView()
        .screenshot(`leisaac-performance-proof-${phase}-trial-${trial + 1}`);
    });
  },
);
