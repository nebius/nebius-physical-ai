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
  const source = win.document.getElementById(id);
  if (!source || !source.width || !source.height || source.hidden) {
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
    };
  }
  return cameras;
}

(hasLiveEnv() ? describe : describe.skip)(
  "NPA agent live LeIsaac end-to-end performance",
  () => {
    it("measures real input to causal frame paint and enforces the optimized gate", () => {
      Cypress.config("defaultCommandTimeout", 480000);
      const runId = String(
        Cypress.env("NPA_LEISAAC_RUN_ID") || Cypress.env("NPA_AGENT_RUN_ID"),
      );
      const output = String(Cypress.env("NPA_LEISAAC_BENCHMARK_OUTPUT"));
      const phase = String(Cypress.env("NPA_LEISAAC_BENCHMARK_PHASE") || "pilot");
      const trial = Math.floor(numberEnv("NPA_LEISAAC_BENCHMARK_TRIAL", 0));
      const warmupCount = Math.floor(numberEnv("NPA_LEISAAC_WARMUP_SAMPLES", 10));
      const measuredCount = Math.floor(numberEnv("NPA_LEISAAC_PRIMARY_SAMPLES", 80));
      const periodMs = numberEnv("NPA_LEISAAC_ACTION_PERIOD_MS", 2000);
      const releaseDelayMs = numberEnv("NPA_LEISAAC_RELEASE_DELAY_MS", 120);
      const pixelThreshold = numberEnv("NPA_LEISAAC_PIXEL_DIFF_THRESHOLD", 0);
      const idleFrameCount = Math.floor(numberEnv("NPA_LEISAAC_IDLE_FRAMES", 12));
      const baselineP50 = numberEnv("NPA_LEISAAC_BASELINE_P50_MS", 0);
      const baselineP95 = numberEnv("NPA_LEISAAC_BASELINE_P95_MS", 0);

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

        const startingEvidence = win.__NPA_AGENT_TEST__.leisaacTransportEvidence();
        win.document.getElementById("leisaacConnect").click();
        await waitUntil(
          win,
          () => {
            const evidence = win.__NPA_AGENT_TEST__.leisaacTransportEvidence();
            const cameras = new Set(evidence.frames.map((frame) => frame.camera));
            return ["websocket-v1", "webrtc-datachannel-v1"].includes(evidence.active) &&
              evidence.frames.length >= startingEvidence.frames.length + 2 &&
              cameras.has("workspace") && cameras.has("overview");
          },
          120000,
          "both preferred RTX cameras",
        );

        const host = win.document.getElementById("leisaacStreamHost");
        if (!host) throw new Error("LeIsaac teleoperation host is missing");
        host.focus();

        let observedFrames = win.__NPA_AGENT_TEST__.leisaacTransportEvidence().frames.length;
        let previousIdle = sampleCanvas(win, "leisaacCanvas");
        const workspaceIdleFrames = [];
        while (workspaceIdleFrames.length < idleFrameCount) {
          const next = await waitUntil(
            win,
            () => {
              const frames = win.__NPA_AGENT_TEST__.leisaacTransportEvidence().frames;
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
          const beforeControls = win.__NPA_AGENT_TEST__.leisaacTransportEvidence().controls.length;
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
            () => win.__NPA_AGENT_TEST__.leisaacTransportEvidence().controls
              .slice(beforeControls)
              .find((item) => item.phase === "accepted" && item.event === "press" && item.key === key),
            10000,
            `press accepted ${ordinal + 1}`,
          );
          const applied = await waitUntil(
            win,
            () => win.__NPA_AGENT_TEST__.leisaacTransportEvidence().controls
              .slice(beforeControls)
              .find((item) => item.phase === "applied" && item.seq === accepted.seq),
            10000,
            `press applied ${ordinal + 1}`,
          );
          await releaseDispatched;
          const releaseAccepted = await waitUntil(
            win,
            () => win.__NPA_AGENT_TEST__.leisaacTransportEvidence().controls
              .slice(beforeControls)
              .find((item) => item.phase === "accepted" && item.event === "release" && item.key === key),
            10000,
            `release accepted ${ordinal + 1}`,
          );
          const releaseApplied = await waitUntil(
            win,
            () => win.__NPA_AGENT_TEST__.leisaacTransportEvidence().controls
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
              const frames = win.__NPA_AGENT_TEST__.leisaacTransportEvidence().frames;
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
        const overview = sampleCanvas(win, "leisaacSecondaryCanvas");
        benchmark.completed_at = new Date().toISOString();
        benchmark.transport = {
          control: String(finalEvidence.active || ""),
          video: String(finalEvidence.video || ""),
          failures: Array.isArray(finalEvidence.failures)
            ? finalEvidence.failures.slice(-16)
            : [],
          policy: finalEvidence.active === "webrtc-datachannel-v1"
            ? "TURN-only reliable ordered control RTCDataChannel; binary WebSocket latest-frame-wins video"
            : finalEvidence.video === "webrtc-datachannel-v1"
              ? "TURN-only unordered maxRetransmits=0 video; reliable ordered control WebSocket"
              : "binary same-origin WebSocket control/video; CUDA fixed-step simulation; latest-frame-wins video",
        };
        benchmark.quality = {
          workspace: { width: 1280, height: 720, jpeg_quality: 82, variance: workspace.variance },
          overview: { width: 1280, height: 720, jpeg_quality: 82, variance: overview.variance },
          viewport_pixel_difference: pixelDifference(workspace, overview),
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
        if (workspace.variance < 25 || overview.variance < 25) {
          throw new Error("one or both camera canvases are blank or uniform");
        }
        if (benchmark.quality.viewport_pixel_difference < 1) {
          throw new Error("the two camera canvases are not visually distinct");
        }
      });

      cy.window()
        .its("__NPA_LEISAAC_PERFORMANCE__")
        .then((benchmark) => {
          if (phase === "optimized") {
            const primary = benchmark.summary.primary_input_to_causal_frame_painted_ms;
            benchmark.comparison = {
              baseline_p50_ms: baselineP50,
              baseline_p95_ms: baselineP95,
              optimized_p50_ms: primary.p50,
              optimized_p95_ms: primary.p95,
              p50_speedup: baselineP50 / primary.p50,
              p95_speedup: baselineP95 / primary.p95,
            };
          }
          return cy.writeFile(output, benchmark, { log: false }).then(() => {
            if (phase !== "optimized") return benchmark;
            if (benchmark.transport.video !== "websocket-v1") {
              throw new Error(
                `optimized benchmark requires the measured bounded WebSocket path, got ${benchmark.transport.video || "none"}`,
              );
            }
            if (benchmark.transport.control !== "websocket-v1") {
              throw new Error(
                `optimized benchmark requires the measured same-origin WebSocket control path, got ${benchmark.transport.control || "none"}`,
              );
            }
            if (!(baselineP50 > 0 && baselineP95 > 0)) {
              throw new Error("optimized gate requires aggregate baseline p50 and p95");
            }
            const primary = benchmark.summary.primary_input_to_causal_frame_painted_ms;
            if (!(primary.p50 <= baselineP50 / 2 && primary.p95 <= baselineP95 / 2)) {
              throw new Error(
                `2x gate failed: p50 ${primary.p50} > ${baselineP50 / 2} or ` +
                `p95 ${primary.p95} > ${baselineP95 / 2}`,
              );
            }
            return benchmark;
          });
        })
        .then((benchmark) => {
          const primary = benchmark.summary.primary_input_to_causal_frame_painted_ms;
          const comparison = benchmark.comparison || {};
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
            proof.textContent = phase === "optimized"
              ? `Real RTX browser benchmark · trial ${trial + 1} · n=${primary.n}\n` +
                `input→causal-frame-painted p50 ${primary.p50.toFixed(2)} ms ` +
                `(${comparison.p50_speedup.toFixed(2)}×; baseline ${baselineP50.toFixed(2)} ms)\n` +
                `p95 ${primary.p95.toFixed(2)} ms ` +
                `(${comparison.p95_speedup.toFixed(2)}×; baseline ${baselineP95.toFixed(2)} ms)\n` +
                `${benchmark.quality.workspace.width}×${benchmark.quality.workspace.height} q82 · ` +
                `two distinct viewports · ${benchmark.transport.control}/${benchmark.transport.video}`
              : `Real RTX browser profile · ${phase} · n=${primary.n}\n` +
                `input→causal-frame-painted p50 ${primary.p50.toFixed(2)} ms · ` +
                `p95 ${primary.p95.toFixed(2)} ms`;
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
