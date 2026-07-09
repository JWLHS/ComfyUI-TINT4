/**
 * tint4_monitor.js — TINT4 LoRA Bypass Monitor v1.1
 * 
 * v1.1: graphToPrompt injection + upstream ModelLoader trace.
 *       Dual channel: (1) prompt JSON _tint4_force_reload field
 *                     (2) HTTP POST /custom/TINT4/signal backup
 *
 * Detects bypass/blocked state on TINT4 LoRA nodes.
 * On detection, injects _tint4_force_reload into the upstream
 * ModelLoader's prompt JSON entry AND sends a signal file via
 * HTTP endpoint. Python ModelLoader reads either signal and
 * forces _tint4_reset_all_loras on next execution.
 */
import { app } from "../../../scripts/app.js";

const TINT4_TYPES = ["TINT4LoRALoader", "TINT4LoRAStack"];

// ── Global state ──────────────────────────────────────────────
if (!window.__TINT4_LORA_STATE__) {
    window.__TINT4_LORA_STATE__ = { nodes: {}, version: "1.1" };
}
const STATE = window.__TINT4_LORA_STATE__;

// ── HTTP signal sender (backup channel) ───────────────────────
async function sendSignal(payload) {
    try {
        await fetch("/custom/TINT4/signal", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
    } catch (e) {
        console.debug("[TINT4 Monitor] Signal send failed (will retry):", e.message);
    }
}

// ── Trace upstream to find TINT4ModelLoader ───────────────────
function findUpstreamLoader(nodeId) {
    const node = app.graph._nodes_by_id ? app.graph._nodes_by_id[nodeId] : null;
    if (!node) return null;
    const visited = new Set();
    const queue = [node];
    while (queue.length > 0) {
        const current = queue.shift();
        if (visited.has(current.id)) continue;
        visited.add(current.id);
        if (current.type === "TINT4ModelLoader") return current;
        if (current.inputs && current.inputs[0] && current.inputs[0].link) {
            const link = app.graph.links[current.inputs[0].link];
            if (link) {
                const upstream = app.graph._nodes_by_id
                    ? app.graph._nodes_by_id[link.origin_id]
                    : app.graph.getNodeById(link.origin_id);
                if (upstream && !visited.has(upstream.id)) queue.push(upstream);
            }
        }
    }
    return null;
}

// ── Scan all TINT4 nodes for bypassed/blocked state ───────────
function scanAndCollect() {
    const needsClear = [];
    const nodes = app.graph._nodes || app.graph.nodes || [];
    for (const node of nodes) {
        if (!TINT4_TYPES.includes(node.type)) continue;
        const bypassed = node.mode === 4;

        // Update state
        STATE.nodes[node.id] = {
            type: node.type,
            bypassed,
            lora: node.widgets?.find(
                w => w.name === "lora_name" || (w.name && w.name.startsWith("lora_name"))
            )?.value || "",
            strength: node.widgets?.find(
                w => w.name === "strength" || (w.name && w.name.startsWith("strength"))
            )?.value ?? 1.0,
        };

        if (bypassed) {
            const loader = findUpstreamLoader(node.id);
            needsClear.push({
                lora_node_id: node.id,
                lora_name: STATE.nodes[node.id].lora,
                model_loader_id: loader ? loader.id : null,
            });
        }
    }
    return needsClear;
}

// ── Register extension ────────────────────────────────────────
app.registerExtension({
    name: "TINT4.LoRA.Monitor",

    async setup() {
        console.info("[TINT4 Monitor] v1.1 active — signal-file + graphToPrompt dual channel");

        // ── Hook: bypass toggled ──────────────────────────────
        const origOnBypass = LiteGraph.LGraph.prototype.onBypass || (() => {});
        LiteGraph.LGraph.prototype.onBypass = function (node) {
            if (node && TINT4_TYPES.includes(node.type)) {
                const bypassed = node.mode === 4;
                STATE.nodes[node.id] = {
                    ...(STATE.nodes[node.id] || {}),
                    bypassed,
                    type: node.type,
                };
                if (bypassed) {
                    console.warn(
                        `[TINT4] ⚠️ "${node.title || node.type}" bypassed!\n` +
                        `  LoRA state will be cleared on next Queue.`
                    );
                    // Immediate signal via HTTP
                    const needsClear = scanAndCollect();
                    if (needsClear.length > 0) {
                        sendSignal({ action: "clear", nodes: needsClear });
                    }
                }
            }
            return origOnBypass.call(this, node);
        };

        // ── Hook: graphToPrompt — inject into prompt JSON ─────
        const origGraphToPrompt = app.graphToPrompt ||
            (app.graph && app.graph.graphToPrompt) ||
            (() => ({}));
        app.graphToPrompt = async function (...args) {
            const result = await origGraphToPrompt.apply(this, args);
            if (!result || !result.output) return result;

            // Scan for bypassed/blocked LoRA nodes
            const needsClear = [];
            const nodes = app.graph._nodes || app.graph.nodes || [];
            for (const node of nodes) {
                if (!TINT4_TYPES.includes(node.type)) continue;
                const bypassed = node.mode === 4;
                STATE.nodes[node.id] = { ...(STATE.nodes[node.id] || {}), bypassed, type: node.type };
                if (bypassed) {
                    const loader = findUpstreamLoader(node.id);
                    needsClear.push({
                        lora_node_id: node.id,
                        model_loader_id: loader ? loader.id : null,
                    });
                }
            }

            if (needsClear.length > 0) {
                // Inject _tint4_force_reload into upstream ModelLoader entries
                for (const item of needsClear) {
                    if (item.model_loader_id != null &&
                        result.output[String(item.model_loader_id)]) {
                        result.output[String(item.model_loader_id)]._tint4_force_reload = true;
                    }
                }
                // Backup signal via HTTP
                await sendSignal({ action: "clear", nodes: needsClear });
            }
            return result;
        };

        console.info("[TINT4 Monitor] Hooks: bypass + graphToPrompt");
    },

    async teardown() {
        STATE.nodes = {};
        console.info("[TINT4 Monitor] Teardown");
    },
});

console.info("[TINT4 Monitor] API: window.__TINT4_LORA_STATE__");
