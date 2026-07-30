/**
 * NPA Foxglove embed host glue.
 *
 * Thin, dependency-free ESM wrapper around the official Foxglove embedding SDK
 * (`@foxglove/embed`, https://docs.foxglove.dev/docs/embed/typescript-sdk).
 *
 * The same module is served by two hosts so the integration can never drift:
 *   * the NPA agent UI            -> /foxglove/app/npa-foxglove-host.js
 *   * the npa-foxglove-embed image -> /app/npa-foxglove-host.js
 * In both layouts the SDK sits next to it at `../sdk/index.js`.
 *
 * Everything here is pure browser JavaScript: no build step, no bundler, no
 * transpile. `dataSourceFromConfig` is deliberately side-effect free so it can be
 * unit-tested directly.
 */

import { FoxgloveViewer } from "../sdk/index.js";

/** Live data-source protocols supported by the Foxglove embedded viewer. */
export const FOXGLOVE_LIVE_PROTOCOLS = Object.freeze([
  "foxglove-websocket",
  "rosbridge-websocket",
]);

/** Default layout storage key used when the backend does not configure one. */
export const DEFAULT_LAYOUT_STORAGE_KEY = "npa-agent-foxglove";

/**
 * Resolve a possibly relative URL against an origin.
 *
 * Data-source URLs are handed to a cross-origin iframe, so they must be
 * absolute; the agent backend returns same-origin paths such as
 * `/foxglove/data/<token>.mcap`.
 */
export function absoluteUrl(url, origin) {
  const raw = String(url || "").trim();
  if (!raw) return "";
  if (/^[a-z][a-z0-9+.-]*:/i.test(raw)) return raw;
  const base = String(origin || "").replace(/\/+$/, "");
  if (!raw.startsWith("/")) return base ? `${base}/${raw}` : raw;
  return base ? `${base}${raw}` : raw;
}

/**
 * Build a Foxglove `DataSource` from an `/api/foxglove/config` (or `/status`)
 * payload. Returns `null` when the payload carries no usable source, so callers
 * can mount an empty viewer instead of inventing data.
 */
export function dataSourceFromConfig(config, options) {
  const cfg = config || {};
  const opts = options || {};
  const origin = opts.origin || (typeof location !== "undefined" ? location.origin : "");
  const source = cfg.data_source || null;

  if (source && typeof source === "object" && source.type) {
    if (source.type === "remote-file") {
      const urls = (source.urls || [])
        .map((entry) => absoluteUrl(entry, origin))
        .filter(Boolean);
      if (!urls.length) return null;
      const resolved = { type: "remote-file", urls };
      if (source.autoplay) resolved.autoplay = true;
      if (source.startTime !== undefined && source.startTime !== null) {
        resolved.startTime = source.startTime;
      }
      return resolved;
    }
    if (source.type === "live") {
      const url = String(source.url || "").trim();
      if (!url) return null;
      const protocol = FOXGLOVE_LIVE_PROTOCOLS.includes(source.protocol)
        ? source.protocol
        : FOXGLOVE_LIVE_PROTOCOLS[0];
      return { type: "live", protocol, url };
    }
    // Foxglove-hosted sources (recording / session / device / remote-access) are
    // passed through untouched — the backend already validated their shape.
    return source;
  }

  const liveUrl = String(cfg.live_url || "").trim();
  if (liveUrl) {
    return { type: "live", protocol: FOXGLOVE_LIVE_PROTOCOLS[0], url: liveUrl };
  }
  return null;
}

/** Normalize a Foxglove `error` event detail into a printable string. */
export function formatViewerError(detail) {
  if (!detail) return "Unknown Foxglove viewer error";
  if (typeof detail === "string") return detail;
  if (typeof detail === "object") {
    const message = detail.message || detail.error || detail.reason || "";
    if (message) return String(message);
    try {
      return JSON.stringify(detail);
    } catch (_err) {
      return String(detail);
    }
  }
  return String(detail);
}

/**
 * Mount a `FoxgloveViewer` into `parent` and return a small handle.
 *
 * @param {object} params
 * @param {HTMLElement} params.parent      container element for the SDK iframe
 * @param {object} params.config           /api/foxglove/config payload
 * @param {function} [params.onReady]      called once the embedded viewer is ready
 * @param {function} [params.onError]      called with a formatted error string
 * @param {function} [params.onDescribe]   Ctrl+Shift+S handler inside the viewer
 * @param {string} [params.origin]         origin used to absolutize data URLs
 * @returns {{viewer: FoxgloveViewer, setDataSource: function, selectLayout: function,
 *            seek: function, destroy: function, isReady: function}}
 */
export function mountFoxgloveViewer(params) {
  const { parent, config, onReady, onError, onDescribe } = params || {};
  if (!parent) throw new Error("mountFoxgloveViewer requires a parent element");
  const cfg = config || {};
  const origin = params.origin || (typeof location !== "undefined" ? location.origin : "");

  const keybindings = [];
  if (typeof onDescribe === "function") {
    // Keyboard parity with the UI's "Describe this" button while focus is inside
    // the Foxglove iframe (the SDK forwards the keypress back to us).
    keybindings.push({ key: "s", modifiers: ["Control", "Shift"], handler: onDescribe });
  }

  const options = {
    parent,
    orgSlug: String(cfg.org_slug || "").trim() || undefined,
    colorScheme: cfg.color_scheme === "light" || cfg.color_scheme === "auto"
      ? cfg.color_scheme
      : "dark",
    initialLayoutParams: {
      storageKey: String(cfg.layout_storage_key || "").trim() || DEFAULT_LAYOUT_STORAGE_KEY,
    },
  };
  const src = String(cfg.embed_src || "").trim();
  if (src) options.src = src;
  if (keybindings.length) options.keybindings = keybindings;

  const initialSource = dataSourceFromConfig(cfg, { origin });
  if (initialSource) options.initialDataSource = initialSource;

  const viewer = new FoxgloveViewer(options);

  if (typeof onReady === "function") {
    viewer.addEventListener("ready", () => onReady());
  }
  if (typeof onError === "function") {
    viewer.addEventListener("error", (event) => onError(formatViewerError(event && event.detail)));
  }

  return {
    viewer,
    isReady: () => viewer.isReady(),
    /** Apply a new data source from a config/status payload (or a raw DataSource). */
    setDataSource(next) {
      const source =
        next && typeof next === "object" && next.type
          ? dataSourceFromConfig({ data_source: next }, { origin })
          : dataSourceFromConfig(next || {}, { origin });
      if (!source) return null;
      viewer.setDataSource(source);
      return source;
    },
    selectLayout(storageKey, opaqueLayout, force) {
      const params2 = { storageKey: String(storageKey || DEFAULT_LAYOUT_STORAGE_KEY) };
      if (opaqueLayout !== undefined) params2.opaqueLayout = opaqueLayout;
      if (force) params2.force = true;
      viewer.selectLayout(params2);
    },
    seek(time) {
      viewer.seekPlayback(time);
    },
    destroy() {
      if (!viewer.isDestroyed()) viewer.destroy();
    },
  };
}
