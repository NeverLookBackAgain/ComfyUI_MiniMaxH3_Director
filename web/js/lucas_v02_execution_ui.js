/** Opt-in v0.2 execution controls with backend preflight before submission. */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { cacheStatusPayload } from "./minimax_refine.js";

const KEY = "lucas_v02_ui";
const MODES = [
    ["first", "仅一采／重抽"],
    ["first_refine", "一采＋二采"],
    ["confirm", "先确认一采"],
    ["refine", "仅二采"],
    ["merge_first", "合成一采"],
    ["merge_refine", "合成二采"],
];
const FIRST = new Set(["first", "first_refine", "confirm"]);
const MERGE = new Set(["merge_first", "merge_refine"]);
const controllers = new WeakMap();
const widget = (n, key) => n?.widgets?.find((w) => w.name === key);
const kind = (n) => n?.comfyClass || n?.type;
const labelFor = (mode) => MODES.find(([key]) => key === mode)?.[1] || mode;
function drafts() {
    return (app.graph?._nodes || []).filter((n) => n.properties?.[KEY]);
}
function linked(n, name) {
    const input = n.inputs?.find((i) => i.name === name);
    const link = input?.link != null ? app.graph.links[input.link] : null;
    return link ? app.graph.getNodeById(link.origin_id) : null;
}
function modules(n) {
    return { refine: linked(n, "refine"), selflift: linked(n, "selflift"), face: linked(n, "face_refine") };
}
function element(tag, text, css = "") {
    const e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    if (css) e.style.cssText = css;
    return e;
}
function panel(title) {
    const root = element("div", undefined, "box-sizing:border-box;margin:6px 0;padding:10px;border:1px solid var(--border-color,#555);border-radius:6px;background:var(--comfy-menu-bg,#292929);color:var(--input-text,#ddd);font:13px/1.5 sans-serif;min-width:0;max-width:100%;overflow:hidden;overflow-wrap:anywhere;user-select:text;");
    root.append(element("strong", title));
    for (const event of ["pointerdown", "mousedown", "keydown", "wheel"]) root.addEventListener(event, (e) => e.stopPropagation());
    return root;
}
function select(root, title, choices, value, change) {
    const row = element("label", undefined, "display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.5fr);gap:8px;align-items:center;margin:8px 0;");
    const input = document.createElement("select");
    input.style.cssText = "width:100%;min-width:0;padding:5px;background:var(--comfy-input-bg,#222);color:var(--input-text,#ddd);border:1px solid var(--border-color,#555);border-radius:4px;font:inherit;";
    for (const [key, text] of choices) { const option = element("option", text); option.value = key; input.append(option); }
    input.value = value;
    input.addEventListener("change", () => change(input.value));
    row.append(element("span", title), input); root.append(row);
    return { row, input };
}
function toggle(root, title, value, change) {
    const row = element("label", undefined, "display:flex;gap:8px;align-items:center;margin:8px 0;");
    const input = document.createElement("input"); input.type = "checkbox"; input.checked = value;
    input.addEventListener("change", () => change(input.checked));
    row.append(input, element("span", title)); root.append(row);
    return { row, input };
}
function domWidget(n, name, root, minHeight, before) {
    root.style.width = "100%";
    root.style.minHeight = `${minHeight}px`;
    root.style.maxWidth = "100%";
    root.style.overflowY = "auto";
    const w = n.addDOMWidget(name, "lucas_design", root, {
        getValue: () => "", setValue: () => {}, getMinHeight: () => minHeight,
        hideOnZoom: false,
        onDraw() { w.width = n.size?.[0] || 340; root.style.height = "fit-content"; root.style.maxHeight = `${height()}px`; },
    });
    w.element = root;
    w.width = n.size?.[0] || 340;
    delete w.computeSize;
    const height = () => Math.max(minHeight, Math.min(520, 32 + Array.from(root.children).reduce((sum, child) => sum + (child.offsetHeight || 0) + (child.hidden ? 0 : 12), 0)));
    w.computeLayoutSize = () => ({ minHeight: height(), maxHeight: height(), minWidth: 220 });
    if (typeof ResizeObserver !== "undefined") {
        const observer = new ResizeObserver(() => { w.width = n.size?.[0] || 340; app.graph.setDirtyCanvas(true, true); });
        observer.observe(root);
        const removed = n.onRemoved;
        n.onRemoved = function (...args) { observer.disconnect(); return removed?.apply(this, args); };
    }
    w.serialize = false; w.options.serialize = false;
    const target = widget(n, before);
    if (target) {
        n.widgets.splice(n.widgets.indexOf(w), 1);
        n.widgets.splice(n.widgets.indexOf(target), 0, w);
    }
    return w;
}
function writeNative(w, value) {
    if (!w || w.value === value) return;
    w.value = value;
    w.callback?.(value);
}
function hook(w, callback) {
    if (!w) return;
    const old = w.callback;
    w.callback = function (...args) { const result = old?.apply(this, args); callback(args[0]); return result; };
}
function hideRow(control, hidden, display) {
    control.row.hidden = hidden;
    control.row.style.display = hidden ? "none" : display;
}
function stageText(s) {
    const first = s.first_source === "selflift" ? "SelfLift 一采" : "导演台原生一采";
    const face = s.face_enabled ? " → 脸修" : "";
    if (s.mode === "first") return first + face + " → 输出";
    if (s.mode === "first_refine") return first + " → 二采 × passes" + face + " → 输出";
    if (s.mode === "confirm") return first + " → 保存并暂停；确认后 → 二采" + face;
    if (s.mode === "refine") return "读取各段一采 → 执行或复用二采" + face + " → 输出";
    if (s.mode === "merge_first") return "读取一采及有效的一采脸修缓存 → 合成；不采样";
    return "读取二采及有效的二采脸修缓存 → 合成" + (s.export_first ? "，同时合成一采" : "");
}
function readTimeline(n) {
    try { return n._minimaxEditor?.timeline || JSON.parse(widget(n, "timeline_data")?.value || "{}"); } catch { return {}; }
}
function inspect(n) {
    const s = n.properties[KEY]; const m = modules(n); const t = readTimeline(n);
    const segments = t.segments || [];
    const issues = [];
    if (FIRST.has(s.mode) && s.first_source === "selflift" && (!m.selflift || m.selflift.mode === 4)) issues.push("SelfLift 未连接或仍被绕过");
    if (!MERGE.has(s.mode) && s.face_enabled && (!m.face || m.face.mode === 4)) issues.push("FaceRefine 未连接或仍被绕过");
    if (["first_refine", "confirm", "refine"].includes(s.mode) && (!m.refine || m.refine.mode === 4)) issues.push("二采节点未连接或被绕过");
    if (!segments.length) issues.push("没有素材分段");
    const selected = t.runSelectEnabled ? (t.runSelection || []).length : segments.length;
    if (!selected) issues.push("未选中运行片段");
    const lines = [
        `执行模式：${labelFor(s.mode)}；选中 ${selected}/${segments.length} 段`,
        `执行顺序：${stageText(s)}`,
        `一采方式：${FIRST.has(s.mode) ? (s.first_source === "selflift" ? "SelfLift" : "导演台原生") : "复用缓存，不重新一采"}`,
        `脸修：${MERGE.has(s.mode) ? "只读取已有有效结果" : s.face_enabled ? "开启" : "关闭"}`,
        `二采 seed：${s.seed_mode === "cached_first_pass" ? "跟随各段一采 seed" : s.seed_mode}`,
        `输出一采：${s.mode === "first" || s.mode === "merge_first" ? "主要输出" : s.export_first ? "开启" : "关闭"}`,
        `节点检查：${issues.length ? issues.join("；") : "通过"}`,
        "执行判断：请检查运行条件；Queue 前自动核对，执行时再次严格核验缓存。",
    ];
    // Reuse the existing report output node; replace its content, never append logs.
    const reportLink = Object.values(app.graph.links || {}).find((l) => l.origin_id === n.id &&
        n.outputs?.[l.origin_slot]?.name === "report" && kind(app.graph.getNodeById(l.target_id)) === "PreviewAny");
    const reportNode = reportLink ? app.graph.getNodeById(reportLink.target_id) : null;
    if (reportNode?.onExecuted) reportNode.onExecuted({ text: [lines.join("\n")] });
    app.graph.setDirtyCanvas(true, true);
    return lines.join("\n");
}
function showReport(n, text) {
    const link = Object.values(app.graph.links || {}).find((l) => l.origin_id === n.id && n.outputs?.[l.origin_slot]?.name === "report" && kind(app.graph.getNodeById(l.target_id)) === "PreviewAny");
    if (link) app.graph.getNodeById(link.target_id)?.onExecuted?.({ text: [text] });
}
async function checkExecution(n) {
    refresh(n);
    const response = await api.fetchApi("/minimax/director/first_pass_cache_status", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cacheStatusPayload(n, modules(n).refine)),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (!data.execution) throw new Error("后台尚未加载 v0.2 执行逻辑，请重启 ComfyUI 后重试");
    const result = data.execution;
    showReport(n, inspect(n) + "\n\n运行条件：" + (result.can_execute ? "通过" : result.errors.join("；")) + "\n" +
        result.segments.map((r) => `片段 ${r.index + 1}：一采 seed=${r.seed ?? "无缓存"}；${r.state}${r.legacy_model_identity ? "（旧缓存未记录模型身份）" : ""}`).join("\n"));
    return result;
}
function modelWitness(n) {
    const describe = (node, seen = new Set()) => {
        if (!node || seen.has(node.id)) return null;
        const next = new Set(seen); next.add(node.id);
        const values = {};
        for (const w of node.widgets || []) if (w.name && !w.name.startsWith("bd_") && w.serialize !== false && ["string", "number", "boolean"].includes(typeof w.value)) values[w.name] = w.value;
        return { type: kind(node), values, inputs: (node.inputs || []).filter((i) => i.link != null).map((i) => ({ name: i.name, node: describe(linked(node, i.name), next) })) };
    };
    const m = modules(n);
    return {
        first: ["model", "video_vae", "audio_vae", "clip", "semantic_bridge", ...(n.properties[KEY].first_source === "selflift" ? ["selflift"] : [])].map((key) => ({ key, node: describe(linked(n, key)) })),
        second: (m.refine?.inputs || []).filter((i) => i.link != null).map((i) => ({ key: i.name, node: describe(linked(m.refine, i.name)) })),
        face: (m.face?.inputs || []).filter((i) => i.link != null).map((i) => ({ key: i.name, node: describe(linked(m.face, i.name)) })),
    };
}
function refresh(n) {
    const s = n.properties[KEY], c = controllers.get(n); if (!c) return;
    const m = modules(n); c.syncing = true;
    c.mode.input.value = s.mode; c.source.input.value = s.first_source;
    hideRow(c.source, !FIRST.has(s.mode), "grid");
    c.face.input.checked = s.face_enabled; c.face.input.disabled = MERGE.has(s.mode);
    c.route.textContent = stageText(s);
    c.module.textContent = `SelfLift：${m.selflift?.mode === 4 ? "绕过" : m.selflift ? "启用" : "未连接"} · FaceRefine：${m.face?.mode === 4 ? "绕过" : m.face ? "启用" : "未连接"}`;
    if (c.skip) {
        c.skip.input.value = s.skip_completed ? "skip" : "rerun"; hideRow(c.skip, !["refine", "confirm", "first_refine"].includes(s.mode), "grid");
        c.output.input.value = s.export_first ? "yes" : "no"; hideRow(c.output, s.mode === "first" || s.mode === "merge_first", "grid");
    }
    if (["confirm", "first_refine"].includes(s.mode)) writeNative(widget(m.refine, "confirm_first_pass"), s.mode === "confirm");
    if (MERGE.has(s.mode)) {
        const t = readTimeline(n);
        if (!s.previous_export) s.previous_export = t.output?.exportMode || "all";
        if (t.output) t.output.exportMode = "all";
        const tw = widget(n, "timeline_data"); if (tw) tw.value = JSON.stringify(t);
    } else if (s.previous_export) {
        const t = readTimeline(n); if (t.output) t.output.exportMode = s.previous_export;
        const tw = widget(n, "timeline_data"); if (tw) tw.value = JSON.stringify(t);
        delete s.previous_export;
    }
    const timeline = readTimeline(n);
    timeline.lucasExecution = { ...s, enabled: true, model_witness: modelWitness(n) };
    delete timeline.lucasExecution.design_only;
    const tw = widget(n, "timeline_data"); if (tw) tw.value = JSON.stringify(timeline);
    c.syncing = false; app.graph.setDirtyCanvas(true, true);
}
function install(n) {
    if (controllers.has(n) || !n.properties?.[KEY] || !n.addDOMWidget) return;
    const s = n.properties[KEY], m = modules(n);
    const root = panel("Lucas v0.2 · 执行控制");
    const c = { syncing: false }; controllers.set(n, c);
    c.mode = select(root, "执行模式", MODES, s.mode, (value) => { s.mode = value; refresh(n); inspect(n); });
    c.source = select(root, "一采方式", [["director", "导演台原生一采"], ["selflift", "SelfLift 一采"]], s.first_source, (value) => {
        s.first_source = value; if (m.selflift) m.selflift.mode = value === "selflift" ? 0 : 4;
        refresh(n); inspect(n);
    });
    c.face = toggle(root, "执行脸修（对本轮最后采样结果）", s.face_enabled, (value) => {
        s.face_enabled = value; if (m.face) m.face.mode = value ? 0 : 4;
        refresh(n); inspect(n);
    });
    c.module = element("div", "", "opacity:.8;margin:6px 0;"); root.append(c.module);
    c.route = element("div", "", "padding:7px 0;border-top:1px solid var(--border-color,#555);word-break:break-word;"); root.append(c.route);
    const check = element("button", "检查运行条件", "padding:5px 10px;margin-top:6px;font:inherit;"); check.type = "button"; check.onclick = () => checkExecution(n).catch((e) => showReport(n, `执行检查失败：${e.message}`)); root.append(check);
    domWidget(n, "lucas_execution_controls", root, 250, "minimax_director_ui");
    if (m.refine?.addDOMWidget) {
        const r = panel("二采输出与缓存复用 · v0.2");
        c.skip = select(r, "二采缓存复用", [["skip", "跳过已完成且参数未变的二采"], ["rerun", "重新执行二采"]], s.skip_completed ? "skip" : "rerun", (value) => { s.skip_completed = value === "skip"; refresh(n); inspect(n); });
        c.output = select(r, "输出一采视频", [["yes", "输出"], ["no", "不输出"]], s.export_first ? "yes" : "no", (value) => { s.export_first = value === "yes"; refresh(n); inspect(n); });
        r.append(element("div", "cached_first_pass = 跟随各段一采 seed；passes 仍表示二采轮数。", "opacity:.8;"));
        domWidget(m.refine, "lucas_refine_controls", r, 155, "seed_mode");
        const cache = m.refine._mmxFirstPassCacheUI;
        if (cache) {
            Object.assign(cache.root.style, { width: "100%", maxWidth: "100%", minWidth: "0", margin: "4px 0", maxHeight: "460px", overflow: "auto" });
            Object.assign(cache.body.style, { maxWidth: "100%", minWidth: "0", overflowWrap: "anywhere", whiteSpace: "pre-wrap", maxHeight: "400px", overflowY: "auto" });
            const header = cache.root.firstElementChild;
            if (header) Object.assign(header.style, { flexWrap: "wrap", gap: "6px" });
            const cw = cache.widget;
            cw.width = m.refine.size?.[0] || 340;
            cw.options.onDraw = () => { cw.width = m.refine.size?.[0] || 340; };
            delete cw.computeSize;
            cw.computeLayoutSize = () => ({ minHeight: Math.max(188, Math.min(460, cache.root.scrollHeight || 188)), maxHeight: Math.max(188, Math.min(460, cache.root.scrollHeight || 188)), minWidth: 220 });
        }
        const seed = widget(m.refine, "seed_mode");
        if (seed) {
            const existing = seed.options.values;
            seed.options.values = typeof existing === "function" ? (...args) => [...new Set([...existing(...args), "cached_first_pass"])] : [...new Set([...(existing || []), "cached_first_pass"])];
            if (s.seed_mode === "cached_first_pass") seed.value = s.seed_mode;
            hook(seed, (value) => { if (!c.syncing) { s.seed_mode = value; refresh(n); inspect(n); } });
        }
        hook(widget(m.refine, "confirm_first_pass"), (value) => {
            if (c.syncing) return;
            s.mode = value ? "confirm" : "first_refine"; refresh(n); inspect(n);
        });
    }
    n._lucasFaceWitness = () => {
        const face = modules(n).face;
        if (!face || face.mode === 4) return null;
        const pack = { enabled: true, has_sigmas_tensor: face.inputs?.some((i) => i.name === "sigmas" && i.link != null) || false };
        for (const w of face.widgets || []) if (w.name && !w.name.startsWith("bd_")) pack[w.name] = w.value;
        return pack;
    };
    refresh(n); inspect(n);
}
app.registerExtension({
    name: "Lucas.Director.v02.ExecutionUIDesign",
    async setup() {
        api.addEventListener("execution_error", (event) => {
            const data = event.detail || {};
            const node = drafts().find((n) => String(n.id) === String(data.node_id));
            if (node) showReport(node, `执行失败：${data.exception_message || data.exception_type || "请查看后台日志"}`);
        });
        const original = app.queuePrompt;
        app.queuePrompt = async function (...args) {
            const marked = drafts();
            if (marked.length) {
                for (const n of marked) {
                    refresh(n);
                    try {
                        const check = await checkExecution(n);
                        if (!check.can_execute) {
                            const message = check.errors.join("；");
                            if (app.ui?.dialog?.show) app.ui.dialog.show(message); else window.alert(message);
                            return;
                        }
                    } catch (e) {
                        const message = `执行条件检查失败：${e.message}。未提交任务。`;
                        showReport(n, message);
                        if (app.ui?.dialog?.show) app.ui.dialog.show(message); else window.alert(message);
                        return;
                    }
                }
            }
            return original.apply(this, args);
        };
    },
    afterConfigureGraph() { setTimeout(() => drafts().forEach(install), 150); },
    loadedGraphNode(n) { if (n.properties?.[KEY]) setTimeout(() => install(n), 180); },
});
