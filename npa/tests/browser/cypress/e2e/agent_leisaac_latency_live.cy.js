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
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : fallback;
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
  return {
    n: finite.length,
    p50: percentile(finite, 0.5),
    p95: percentile(finite, 0.95),
    p99: percentile(finite, 0.99),
    min: finite.length ? Math.min(...finite) : null,
    max: finite.length ? Math.max(...finite) : null,
  };
}

async function waitUntil(win, predicate, timeoutMs, label) {
  const deadline = win.performance.now() + timeoutMs;
  while (win.performance.now() < deadline) {
    const value = predicate();
    if (value) return value;
    await new Promise((resolve) => win.setTimeout(resolve, 5));
  }
  throw new Error(`timed out waiting for ${label}`);
}

async function fetchStatus(win, runId) {
  const startedMonoMs = win.performance.now();
  const startedWallMs = win.Date.now();
  const response = await win.fetch(
    `/api/leisaac/status?run_id=${encodeURIComponent(runId)}`,
    { credentials: "include", cache: "no-store" },
  );
  const receivedMonoMs = win.performance.now();
  const receivedWallMs = win.Date.now();
  if (!response.ok) {
    throw new Error(`LeIsaac status returned HTTP ${response.status}`);
  }
  const payload = await response.json();
  return {
    payload,
    startedMonoMs,
    startedWallMs,
    receivedMonoMs,
    receivedWallMs,
    rttMs: receivedMonoMs - startedMonoMs,
    serverDateMs: Date.parse(response.headers.get("date") || ""),
  };
}

(hasLiveEnv() ? describe : describe.skip)(
  "NPA agent live LeIsaac latency benchmark",
  () => {
    it("records browser-path control and frame latency without hiding clock uncertainty", () => {
      Cypress.config("defaultCommandTimeout", 240000);
      const runId = String(Cypress.env("NPA_AGENT_RUN_ID"));
      const output = String(Cypress.env("NPA_LEISAAC_BENCHMARK_OUTPUT"));
      const controlCount = numberEnv("NPA_LEISAAC_CONTROL_SAMPLES", 80);
      const frameCount = numberEnv("NPA_LEISAAC_FRAME_SAMPLES", 100);

      cy.viewport(1440, 1050);
      cy.visitLiveAgent();
      cy.window().then((win) =>
        win.__NPA_AGENT_TEST__.refreshLeIsaacCapability(runId),
      );
      cy.get("#tabLeIsaac", { timeout: 30000 }).should("be.visible").click();

      cy.window().then(async (win) => {
        const benchmark = {
          schema: "npa.leisaac.transport-benchmark.v1",
          phase: String(Cypress.env("NPA_LEISAAC_BENCHMARK_PHASE") || "baseline"),
          public_path: new URL(win.location.href).origin,
          run_id: runId,
          started_at: new Date().toISOString(),
          requested_samples: { controls: controlCount, frames: frameCount },
          clock_method: {
            browser: "performance.now() for intervals; Date.now() for UTC correlation",
            simulator_application:
              "aggregate status observation; each sample is an upper bound because the baseline has no sequence acknowledgement",
            frame_capture:
              "latest frame file mtime observed after browser decode; retained as a bounded observation, not false exact correlation",
            server_date:
              "HTTP Date midpoint estimate with one-second header quantization included in uncertainty",
          },
          control_samples: [],
          application_samples: [],
          frame_samples: [],
          polling_frame_samples: [],
          reconnect_samples: [],
          status_samples: [],
          errors: [],
        };
        win.__NPA_LEISAAC_BENCHMARK__ = benchmark;
        win.performance.setResourceTimingBufferSize(5000);

        // Estimate server-wall minus browser-wall using a minimum-uncertainty
        // NTP-style bracket. HTTP Date is only one-second resolution, so retain
        // +/-500 ms quantization rather than presenting false precision.
        for (let index = 0; index < 20; index += 1) {
          const sample = await fetchStatus(win, runId);
          benchmark.status_samples.push({
            started_mono_ms: sample.startedMonoMs,
            received_mono_ms: sample.receivedMonoMs,
            rtt_ms: sample.rttMs,
            server_date_ms: sample.serverDateMs,
            client_midpoint_wall_ms:
              (sample.startedWallMs + sample.receivedWallMs) / 2,
            server_minus_client_ms: Number.isFinite(sample.serverDateMs)
              ? sample.serverDateMs +
                500 -
                (sample.startedWallMs + sample.receivedWallMs) / 2
              : null,
            uncertainty_ms: sample.rttMs / 2 + 500,
          });
        }

        const capabilityProbe = await fetchStatus(win, runId);
        const preferred =
          String(capabilityProbe.payload.stream_transport || "") === "websocket-v1";
        benchmark.clock_method.simulator_application = preferred
          ? "sequenced simulator-applied acknowledgement; browser interval uses performance.now(), runtime/simulator stages use their shared monotonic clock"
          : "aggregate status observation; each sample is an upper bound because the baseline has no sequence acknowledgement";
        benchmark.clock_method.frame_capture = preferred
          ? "binary envelope carries runtime capture/encode monotonic+wall timestamps; application ping/pong records browser/runtime wall offset uncertainty"
          : "latest frame file mtime observed after browser decode; retained as a bounded observation, not false exact correlation";

        const nativeFetch = win.fetch.bind(win);
        const controlPromises = [];
        win.fetch = async (url, options) => {
          const address = String(url);
          if (!address.includes("/api/leisaac/input?")) {
            return nativeFetch(url, options);
          }
          const body = JSON.parse(String((options && options.body) || "{}"));
          const startedMonoMs = win.performance.now();
          const startedWallMs = win.Date.now();
          const pending = nativeFetch(url, options)
            .then((response) => {
              const receivedMonoMs = win.performance.now();
              benchmark.control_samples.push({
                key: String(body.key || ""),
                event: String(body.event || ""),
                started_mono_ms: startedMonoMs,
                started_wall_ms: startedWallMs,
                received_mono_ms: receivedMonoMs,
                rtt_ms: receivedMonoMs - startedMonoMs,
                status: response.status,
                ok: response.ok,
              });
              return response;
            })
            .catch((error) => {
              benchmark.errors.push({
                surface: "control",
                message: String(error && error.message ? error.message : error),
              });
              throw error;
            });
          controlPromises.push(pending);
          return pending;
        };

        const frame = win.document.getElementById("leisaacFrame");
        if (!frame) throw new Error("LeIsaac frame element is missing");
        const frameListener = () => {
          if (benchmark.frame_samples.length >= frameCount) return;
          const decodedMonoMs = win.performance.now();
          const decodedWallMs = win.Date.now();
          const currentSrc = String(frame.currentSrc || frame.src || "");
          const resources = win.performance.getEntriesByName(currentSrc);
          const resource = resources.length ? resources[resources.length - 1] : null;
          const item = {
            src_sequence: Number(new URL(currentSrc).searchParams.get("frame")),
            request_start_mono_ms: resource ? resource.startTime : null,
            response_end_mono_ms: resource ? resource.responseEnd : null,
            decoded_mono_ms: decodedMonoMs,
            decoded_wall_ms: decodedWallMs,
            network_ms: resource ? resource.responseEnd - resource.startTime : null,
            decode_after_response_ms: resource
              ? decodedMonoMs - resource.responseEnd
              : null,
            transfer_bytes: resource
              ? resource.transferSize || resource.encodedBodySize || 0
              : 0,
            painted_mono_ms: null,
            latest_capture_wall_ms: null,
            latest_capture_observation_age_ms: null,
            observation_rtt_ms: null,
          };
          benchmark.frame_samples.push(item);
          win.requestAnimationFrame(() =>
            win.requestAnimationFrame(() => {
              item.painted_mono_ms = win.performance.now();
            }),
          );
          fetchStatus(win, runId)
            .then((sample) => {
              const captured = Date.parse(
                String(sample.payload.frame_updated_at || ""),
              );
              item.latest_capture_wall_ms = Number.isFinite(captured)
                ? captured
                : null;
              item.latest_capture_observation_age_ms = Number.isFinite(captured)
                ? decodedWallMs - captured
                : null;
              item.observation_rtt_ms = sample.rttMs;
              item.observed_frame_bytes = Number(sample.payload.frame_bytes || 0);
            })
            .catch((error) =>
              benchmark.errors.push({
                surface: "frame-status",
                message: String(error && error.message ? error.message : error),
              }),
            );
        };
        if (!preferred) frame.addEventListener("load", frameListener);

        const startingEvidence = win.__NPA_AGENT_TEST__.leisaacTransportEvidence();
        const startingFrameEvidence = startingEvidence.frames.length;
        win.document.getElementById("leisaacConnect").click();
        if (preferred) {
          await waitUntil(
            win,
            () => {
              const evidence = win.__NPA_AGENT_TEST__.leisaacTransportEvidence();
              return evidence.active === "websocket-v1" && evidence.frames.length > startingFrameEvidence;
            },
            120000,
            "first preferred RTX frame",
          );
        } else {
          await waitUntil(
            win,
            () => frame.complete && frame.naturalWidth > 0,
            120000,
            "first decoded RTX frame",
          );
        }
        const host = win.document.getElementById("leisaacStreamHost");
        host.focus();
        const initial = await fetchStatus(win, runId);
        const startAccepted = Number(initial.payload.input_events || 0);
        const keys = ["W", "S", "A", "D", "Q", "E", "J", "L", "I", "K", "U", "O"];

        const dispatchPreferred = async (key, event, ordinal) => {
          const before = win.__NPA_AGENT_TEST__.leisaacTransportEvidence().controls.length;
          host.dispatchEvent(
            new win.KeyboardEvent(event === "press" ? "keydown" : "keyup", {
              key,
              code: `Key${key}`,
              bubbles: true,
            }),
          );
          const accepted = await waitUntil(
            win,
            () =>
              win
                .__NPA_AGENT_TEST__.leisaacTransportEvidence()
                .controls.slice(before)
                .find(
                  (item) =>
                    ["accepted", "applied"].includes(item.phase) &&
                    item.event === event,
                ),
            10000,
            `${event} accepted or terminal-applied acknowledgement ${ordinal}`,
          );
          const applied =
            accepted.phase === "applied"
              ? accepted
              : await waitUntil(
                  win,
                  () =>
                    win
                      .__NPA_AGENT_TEST__.leisaacTransportEvidence()
                      .controls.slice(before)
                      .find(
                        (item) =>
                          item.phase === "applied" && item.seq === accepted.seq,
                      ),
                  10000,
                  `${event} simulator-applied acknowledgement ${ordinal}`,
                );
          benchmark.control_samples.push({
            protocol: "websocket-v1",
            seq: accepted.seq,
            key,
            event,
            rtt_ms: accepted.event_to_ack_ms,
            status: 101,
            ok: true,
            terminal_phase: applied.phase,
            recovered_on_resume: Boolean(applied.recovered_on_resume),
            runtime_received_mono_ns: accepted.runtime_received_mono_ns,
            agent_received_mono_ns: accepted.agent_received_mono_ns,
            agent_send_mono_ns: accepted.agent_send_mono_ns,
          });
          return applied;
        };

        for (let index = 0; index < controlCount; index += 1) {
          const key = keys[index % keys.length];
          if (preferred) {
            const applied = await dispatchPreferred(key, "press", index);
            const runtimeReceived = Number(applied.runtime_received_mono_ns || 0);
            const simulatorApplied = Number(applied.simulator_applied_mono_ns || 0);
            benchmark.application_samples.push({
              key,
              ordinal: index + 1,
              seq: applied.seq,
              event_to_observed_applied_ms: applied.event_to_ack_ms,
              runtime_received_to_simulator_applied_ms:
                runtimeReceived > 0 && simulatorApplied >= runtimeReceived
                  ? (simulatorApplied - runtimeReceived) / 1000000
                  : null,
              simulator_step: applied.simulator_step,
              exact_sequence_ack: true,
            });
            await dispatchPreferred(key, "release", index);
            continue;
          }
          const pressSampleIndex = benchmark.control_samples.length;
          const eventMonoMs = win.performance.now();
          host.dispatchEvent(
            new win.KeyboardEvent("keydown", { key, code: `Key${key}`, bubbles: true }),
          );
          await waitUntil(
            win,
            () => benchmark.control_samples.length > pressSampleIndex,
            10000,
            `press acknowledgement ${index}`,
          );
          const expectedApplied = startAccepted + index + 1;
          let observed;
          while (true) {
            observed = await fetchStatus(win, runId);
            if (
              Number(observed.payload.input_events || 0) >= expectedApplied &&
              Number(observed.payload.applied_inputs || 0) >= expectedApplied
            ) {
              break;
            }
            await new Promise((resolve) => win.setTimeout(resolve, 5));
          }
          benchmark.application_samples.push({
            key,
            ordinal: index + 1,
            event_mono_ms: eventMonoMs,
            observed_mono_ms: observed.receivedMonoMs,
            event_to_observed_applied_ms: observed.receivedMonoMs - eventMonoMs,
            observation_rtt_ms: observed.rttMs,
            accepted_total: Number(observed.payload.input_events || 0),
            applied_total: Number(observed.payload.applied_inputs || 0),
          });
          const releaseSampleIndex = benchmark.control_samples.length;
          host.dispatchEvent(
            new win.KeyboardEvent("keyup", { key, code: `Key${key}`, bubbles: true }),
          );
          await waitUntil(
            win,
            () => benchmark.control_samples.length > releaseSampleIndex,
            10000,
            `release acknowledgement ${index}`,
          );
        }

        await Promise.allSettled(controlPromises);
        if (preferred) {
          await waitUntil(
            win,
            () =>
              win.__NPA_AGENT_TEST__.leisaacTransportEvidence().frames.length >=
              startingFrameEvidence + frameCount,
            120000,
            `${frameCount} preferred decoded frames`,
          );
          benchmark.frame_samples = win
            .__NPA_AGENT_TEST__.leisaacTransportEvidence()
            .frames.slice(startingFrameEvidence, startingFrameEvidence + frameCount)
            .map((item) => ({
              ...item,
              decoded_mono_ms: item.painted_mono_ms,
              transfer_bytes: item.bytes,
              network_ms: null,
              decode_after_response_ms: item.painted_mono_ms - item.received_mono_ms,
              response_end_mono_ms: item.received_mono_ms,
              latest_capture_observation_age_ms: item.frame_age_ms,
            }));
        } else {
          await waitUntil(
            win,
            () => benchmark.frame_samples.length >= frameCount,
            120000,
            `${frameCount} decoded frames`,
          );
        }
        await new Promise((resolve) => win.setTimeout(resolve, 1000));

        if (preferred) {
          const reconnectStarted = win.performance.now();
          const framesBeforeReconnect = win.__NPA_AGENT_TEST__.leisaacTransportEvidence().frames.length;
          await win.__NPA_AGENT_TEST__.disconnectLeIsaac();
          await win.__NPA_AGENT_TEST__.connectLeIsaac();
          await waitUntil(
            win,
            () => {
              const evidence = win.__NPA_AGENT_TEST__.leisaacTransportEvidence();
              return evidence.active === "websocket-v1" && evidence.frames.length > framesBeforeReconnect;
            },
            30000,
            "preferred transport reconnect frame",
          );
          benchmark.reconnect_samples.push({
            reconnect_ms: win.performance.now() - reconnectStarted,
            sequence_state_recovered: true,
          });
          // Measure the HTTP fallback as a fallback, not while its preferred
          // WebSocket replacement is concurrently consuming frame bandwidth.
          await win.__NPA_AGENT_TEST__.disconnectLeIsaac();
        }

        // Fetch/decode/paint the fallback route directly as well. The SHA-256
        // joins each browser sample to the pod-side frame monitor without
        // relying on cross-host clocks or a coincidental JPEG byte length.
        const paintCanvas = win.document.createElement("canvas");
        paintCanvas.width = 320;
        paintCanvas.height = 180;
        paintCanvas.style.position = "fixed";
        paintCanvas.style.left = "0";
        paintCanvas.style.bottom = "0";
        paintCanvas.style.width = "320px";
        paintCanvas.style.height = "180px";
        paintCanvas.style.zIndex = "99999";
        win.document.body.appendChild(paintCanvas);
        const paintContext = paintCanvas.getContext("2d");
        const frameUrl = String(initial.payload.frame_url || "");
        for (let index = 0; index < frameCount; index += 1) {
          const separator = frameUrl.includes("?") ? "&" : "?";
          const startedMonoMs = win.performance.now();
          const startedWallMs = win.Date.now();
          const response = await nativeFetch(
            `${frameUrl}${separator}benchmark=${index}`,
            { credentials: "include", cache: "no-store" },
          );
          const responseMonoMs = win.performance.now();
          const responseWallMs = win.Date.now();
          const bytes = await response.arrayBuffer();
          const receivedMonoMs = win.performance.now();
          const digestBytes = new Uint8Array(
            await win.crypto.subtle.digest("SHA-256", bytes),
          );
          const sha256 = Array.from(digestBytes, (value) =>
            value.toString(16).padStart(2, "0"),
          ).join("");
          const bitmap = await win.createImageBitmap(
            new win.Blob([bytes], { type: "image/jpeg" }),
          );
          const decodedMonoMs = win.performance.now();
          paintContext.drawImage(
            bitmap,
            0,
            0,
            paintCanvas.width,
            paintCanvas.height,
          );
          bitmap.close();
          await new Promise((resolve) =>
            win.requestAnimationFrame(() => win.requestAnimationFrame(resolve)),
          );
          benchmark.polling_frame_samples.push({
            ordinal: index + 1,
            started_mono_ms: startedMonoMs,
            started_wall_ms: startedWallMs,
            response_mono_ms: responseMonoMs,
            response_wall_ms: responseWallMs,
            received_mono_ms: receivedMonoMs,
            decoded_mono_ms: decodedMonoMs,
            painted_mono_ms: win.performance.now(),
            request_to_response_ms: responseMonoMs - startedMonoMs,
            body_read_ms: receivedMonoMs - responseMonoMs,
            response_to_decode_ms: decodedMonoMs - responseMonoMs,
            response_to_paint_ms: win.performance.now() - responseMonoMs,
            bytes: bytes.byteLength,
            sha256,
            status: response.status,
          });
        }
        paintCanvas.remove();
        if (!preferred) frame.removeEventListener("load", frameListener);
        win.fetch = nativeFetch;

        const final = await fetchStatus(win, runId);
        const frameSamples = benchmark.frame_samples.slice(0, frameCount);
        const firstFrame = frameSamples[0];
        const lastFrame = frameSamples[frameSamples.length - 1];
        const frameWindowSeconds =
          (lastFrame.decoded_mono_ms - firstFrame.decoded_mono_ms) / 1000;
        const painted = frameSamples.filter(
          (item) =>
            Number.isFinite(item.painted_mono_ms) &&
            Number.isFinite(item.response_end_mono_ms),
        );
        benchmark.completed_at = new Date().toISOString();
        benchmark.transport = String(initial.payload.stream_transport || "");
        benchmark.task = String(initial.payload.task || "");
        benchmark.environment_id = String(initial.payload.environment_id || "");
        benchmark.gpu = String(initial.payload.gpu || "");
        benchmark.start_totals = {
          accepted: startAccepted,
          applied: Number(initial.payload.applied_inputs || 0),
        };
        benchmark.final_totals = {
          accepted: Number(final.payload.input_events || 0),
          applied: Number(final.payload.applied_inputs || 0),
        };
        benchmark.summary = {
          control_ack_rtt_ms: distribution(
            benchmark.control_samples.map((item) => item.rtt_ms),
          ),
          event_to_observed_applied_ms: distribution(
            benchmark.application_samples.map(
              (item) => item.event_to_observed_applied_ms,
            ),
          ),
          runtime_received_to_simulator_applied_ms: distribution(
            benchmark.application_samples.map(
              (item) => item.runtime_received_to_simulator_applied_ms,
            ),
          ),
          frame_network_ms: distribution(
            frameSamples.map((item) => item.network_ms),
          ),
          frame_decode_after_response_ms: distribution(
            frameSamples.map((item) => item.decode_after_response_ms),
          ),
          latest_capture_observation_age_ms: distribution(
            frameSamples.map((item) => item.latest_capture_observation_age_ms),
          ),
          frame_capture_to_encode_ms: distribution(
            frameSamples.map((item) =>
              Number(item.encoded_mono_ns || 0) > Number(item.capture_mono_ns || 0)
                ? (Number(item.encoded_mono_ns) - Number(item.capture_mono_ns)) / 1000000
                : null,
            ),
          ),
          frame_encode_to_runtime_send_ms: distribution(
            frameSamples.map((item) =>
              Number(item.runtime_send_mono_ns || 0) > Number(item.encoded_mono_ns || 0)
                ? (Number(item.runtime_send_mono_ns) - Number(item.encoded_mono_ns)) / 1000000
                : null,
            ),
          ),
          frame_agent_receive_to_send_ms: distribution(
            frameSamples.map((item) =>
              Number(item.agent_send_mono_ns || 0) >= Number(item.agent_receive_mono_ns || 0)
                ? (Number(item.agent_send_mono_ns) - Number(item.agent_receive_mono_ns)) / 1000000
                : null,
            ),
          ),
          frame_response_to_paint_ms: distribution(
            painted.map(
              (item) => item.painted_mono_ms - item.response_end_mono_ms,
            ),
          ),
          delivered_fps:
            frameWindowSeconds > 0 ? (frameSamples.length - 1) / frameWindowSeconds : 0,
          frame_bytes: distribution(
            frameSamples.map(
              (item) => item.transfer_bytes || item.observed_frame_bytes,
            ),
          ),
          polling_frame_request_to_response_ms: distribution(
            benchmark.polling_frame_samples.map(
              (item) => item.request_to_response_ms,
            ),
          ),
          polling_frame_response_to_paint_ms: distribution(
            benchmark.polling_frame_samples.map(
              (item) => item.response_to_paint_ms,
            ),
          ),
          polling_frame_bytes: distribution(
            benchmark.polling_frame_samples.map((item) => item.bytes),
          ),
          control_error_rate:
            benchmark.control_samples.length > 0
              ? benchmark.control_samples.filter((item) => !item.ok).length /
                benchmark.control_samples.length
              : null,
          delivered_frames: frameSamples.length,
          painted_frames: painted.length,
          dropped_or_coalesced_frames: frameSamples.reduce(
            (total, item) => total + Number(item.dropped_before || 0),
            0,
          ),
          reconnect_ms: distribution(
            benchmark.reconnect_samples.map((item) => item.reconnect_ms),
          ),
          frame_clock_uncertainty_ms: distribution(
            frameSamples.map((item) => item.clock_uncertainty_ms),
          ),
        };
      });

      cy.window()
        .its("__NPA_LEISAAC_BENCHMARK__")
        .then((benchmark) => cy.writeFile(output, benchmark, { log: false }));
    });

    it("durably releases held controls when a browser control channel disappears", () => {
      Cypress.config("defaultCommandTimeout", 240000);
      const runId = String(Cypress.env("NPA_AGENT_RUN_ID"));
      const output = `${String(Cypress.env("NPA_LEISAAC_BENCHMARK_OUTPUT"))}.abrupt-disconnect.json`;

      cy.viewport(1440, 1050);
      cy.visitLiveAgent();
      cy.window().then(async (win) => {
        const authorize = async () => {
          const response = await win.fetch(
            `/api/leisaac/ws-session?run_id=${encodeURIComponent(runId)}`,
            {
              method: "POST",
              credentials: "include",
              cache: "no-store",
              headers: { "X-NPA-LeIsaac-Control": "1" },
            },
          );
          if (response.status !== 204) {
            throw new Error(
              `LeIsaac control transport authorization returned HTTP ${response.status}`,
            );
          }
        };
        const connect = async (status, received) => {
          const target = new URL(String(status.control_ws_url), win.location.href);
          target.protocol = win.location.protocol === "https:" ? "wss:" : "ws:";
          const socket = new win.WebSocket(
            target.toString(),
            "npa.leisaac.control.v1",
          );
          socket.addEventListener("message", (message) => {
            try {
              received.push(JSON.parse(String(message.data || "")));
            } catch (_error) {
              received.push({ type: "invalid-json" });
            }
          });
          await waitUntil(
            win,
            () => socket.readyState === win.WebSocket.OPEN,
            15000,
            "raw control WebSocket open",
          );
          return socket;
        };
        const envelope = (type, clientId) => ({
          v: 1,
          type,
          run_id: runId,
          client_id: clientId,
          client_mono_ns: String(Math.floor(win.performance.now() * 1000000)),
          client_wall_ns: String(BigInt(win.Date.now()) * 1000000n),
        });
        const waitForMessage = (messages, predicate, label) =>
          waitUntil(win, () => messages.find(predicate), 15000, label);

        const initial = await fetchStatus(win, runId);
        expect(String(initial.payload.stream_transport || "")).to.equal(
          "websocket-v1",
        );
        const clientId = `cypress-abrupt-${win.Date.now()}`;
        await authorize();
        const firstMessages = [];
        const first = await connect(initial.payload, firstMessages);
        first.send(
          JSON.stringify({
            ...envelope("resume", clientId),
            last_acked_seq: 0,
            keys_down: [],
          }),
        );
        const initialResume = await waitForMessage(
          firstMessages,
          (message) => message.type === "resumed",
          "initial control resume",
        );
        expect(Number(initialResume.next_seq)).to.equal(1);
        first.send(
          JSON.stringify({
            ...envelope("control", clientId),
            seq: 1,
            key: "W",
            event: "press",
          }),
        );
        await waitForMessage(
          firstMessages,
          (message) =>
            message.type === "ack" &&
            message.phase === "applied" &&
            Number(message.seq) === 1,
          "held control simulator-applied acknowledgement",
        );
        const beforeDisconnect = await fetchStatus(win, runId);

        // Deliberately omit release-all: the runtime must synthesize and durably
        // apply a release when this browser-owned channel disappears.
        first.close();
        await waitUntil(
          win,
          () => first.readyState === win.WebSocket.CLOSED,
          15000,
          "abrupt control WebSocket close",
        );
        let afterDisconnect = null;
        const deadline = win.performance.now() + 30000;
        while (win.performance.now() < deadline) {
          const observed = await fetchStatus(win, runId);
          if (
            Number(observed.payload.input_events || 0) >=
              Number(beforeDisconnect.payload.input_events || 0) + 1 &&
            Number(observed.payload.applied_inputs || 0) >=
              Number(beforeDisconnect.payload.applied_inputs || 0) + 1 &&
            Number(observed.payload.input_events || 0) ===
              Number(observed.payload.applied_inputs || 0)
          ) {
            afterDisconnect = observed;
            break;
          }
          await new Promise((resolve) => win.setTimeout(resolve, 10));
        }
        if (!afterDisconnect) {
          throw new Error("disconnect release was not durably applied within 30 seconds");
        }

        await authorize();
        const resumedMessages = [];
        const resumedSocket = await connect(initial.payload, resumedMessages);
        resumedSocket.send(
          JSON.stringify({
            ...envelope("resume", clientId),
            last_acked_seq: 1,
            keys_down: ["W"],
          }),
        );
        const resumed = await waitForMessage(
          resumedMessages,
          (message) => message.type === "resumed",
          "post-disconnect control resume",
        );
        expect(Number(resumed.next_seq)).to.equal(3);
        expect(Number(resumed.last_applied_seq)).to.be.at.least(2);
        expect(resumed.keys_down).to.deep.equal([]);
        resumedSocket.close();

        win.__NPA_LEISAAC_ABRUPT_DISCONNECT__ = {
          schema: "npa.leisaac.abrupt-disconnect-proof.v1",
          completed_at: new Date().toISOString(),
          held_sequence: 1,
          synthetic_release_sequence: 2,
          resumed_next_sequence: Number(resumed.next_seq),
          resumed_last_applied_sequence: Number(resumed.last_applied_seq),
          resumed_keys_down: resumed.keys_down,
          input_events_before_disconnect: Number(
            beforeDisconnect.payload.input_events || 0,
          ),
          input_events_after_disconnect: Number(
            afterDisconnect.payload.input_events || 0,
          ),
          applied_inputs_after_disconnect: Number(
            afterDisconnect.payload.applied_inputs || 0,
          ),
        };
      });

      cy.window()
        .its("__NPA_LEISAAC_ABRUPT_DISCONNECT__")
        .then((proof) => cy.writeFile(output, proof, { log: false }));
    });
  },
);
