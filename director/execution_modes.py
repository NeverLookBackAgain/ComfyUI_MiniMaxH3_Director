"""Opt-in Lucas execution controls. Legacy timelines retain their original path."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import folder_paths

MODES = {"first", "first_refine", "confirm", "refine", "merge_first", "merge_refine"}
MERGE = {"merge_first", "merge_refine"}
LABELS = {"first": "仅一采／重抽", "first_refine": "一采＋二采", "confirm": "先确认一采", "refine": "仅二采", "merge_first": "合成一采", "merge_refine": "合成二采"}
log = logging.getLogger("ComfyUI-MiniMaxH3-Director.execution")


def normalize_execution_timeline(timeline_data):
    """Keep the user selection intact for both sampling and cache-only merges."""
    return timeline_data


def configure_execution(plan):
    raw = (plan.raw or {}).get("lucasExecution")
    if not isinstance(raw, dict) or not raw.get("enabled"):
        plan.execution = None
        return plan
    mode = raw.get("mode", "first_refine")
    if mode not in MODES:
        raise ValueError(f"未知执行模式：{mode}")
    source = raw.get("first_source", "director")
    if source not in {"director", "selflift"}:
        raise ValueError(f"未知一采方式：{source}")
    plan.execution = {**raw, "mode": mode, "first_source": source}
    if source == "director":
        plan.selflift = None
    elif not plan.selflift or not plan.selflift.get("enabled"):
        raise ValueError("选择了 SelfLift 一采，但 SelfLift 节点未连接、未启用或仍被绕过。")
    if not raw.get("face_enabled"):
        plan.face_refine = None
    elif not plan.face_refine or not plan.face_refine.get("enabled"):
        raise ValueError("开启了脸修，但 FaceRefine 节点未连接、未启用或仍被绕过。")
    if mode in {"first", "merge_first"}:
        plan.refine = None
    else:
        if not plan.refine or not plan.refine.get("enabled"):
            raise ValueError("当前模式需要连接并启用二采节点。")
        plan.refine = dict(plan.refine)
        plan.refine["confirm_first_pass"] = mode == "confirm"
        seed_mode = raw.get("seed_mode", plan.refine.get("seed_mode", "inherit"))
        if seed_mode not in {"inherit", "offset", "independent", "cached_first_pass"}:
            raise ValueError(f"未知二采 seed 模式：{seed_mode}")
        plan.refine["seed_mode"] = seed_mode
    if mode in MERGE:
        plan.export_mode = "all"
    return plan


def model_witness_hash(plan, stage):
    import hashlib
    cfg = getattr(plan, "execution", None) or {}
    witness = cfg.get("model_witness", {}).get(stage)
    if witness is None:
        return None
    return hashlib.sha256(json.dumps(witness, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def first_fingerprint_matches(stored, expected):
    if not isinstance(stored, dict):
        return False
    # Old first caches did not record model wiring. Do not invent provenance:
    # accept their recorded keys, mark that limitation in the inspection report.
    if "first_model_witness" not in stored:
        expected = {k: v for k, v in expected.items() if k != "first_model_witness"}
    return stored == expected


def first_seed(plan, seg):
    return int(getattr(plan, "segment_seeds", {}).get(seg.index, plan.sample_seed))


def stamp_cached_seeds(node_id, plan):
    """Only substitute seed; every other first-pass fingerprint key stays strict."""
    plan.segment_seeds = {}
    cfg = getattr(plan, "execution", None)
    if not cfg:
        if not plan.refine or plan.refine.get("seed_mode") != "cached_first_pass":
            return
        cfg = {"mode": "confirm", "seed_mode": "cached_first_pass"}
    plan.execution_node_id = str(node_id or "")
    mode = cfg["mode"]
    if mode not in MERGE and not (mode in {"refine", "confirm"} and cfg.get("seed_mode") == "cached_first_pass"):
        return
    root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / str(node_id)
    for seg in plan.segments:
        try:
            stored = json.loads((root / f"seg_{seg.index:04d}.pre.meta.json").read_text(encoding="utf-8"))
            seed = stored["seed"]
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                continue
            plan.segment_seeds[seg.index] = seed
        except (OSError, ValueError, KeyError, TypeError):
            continue


def output_first(plan):
    cfg = getattr(plan, "execution", None)
    return not cfg or cfg["mode"] in {"first", "merge_first"} or (cfg["mode"] == "confirm" and getattr(plan, "execution_holding", False)) or bool(cfg.get("export_first", True))


def stage_report(plan, message):
    text = f"执行检查：{message}"
    log.info(text)
    return text


def preflight(plan, node_id, *, metadata_only=False):
    """Metadata-only validation, used before loading models or submitting a job."""
    from .segment_cache import first_pass_cache_fingerprint, segment_cache_fingerprint
    cfg = getattr(plan, "execution", None)
    if not cfg:
        return None
    stamp_cached_seeds(node_id, plan)
    root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / str(node_id)
    selected = plan.run_indices if plan.run_indices is not None else {s.index for s in plan.segments}
    rows = []
    for seg in plan.segments:
        pre_meta = root / f"seg_{seg.index:04d}.pre.meta.json"
        final_meta = root / f"seg_{seg.index:04d}.meta.json"
        def matches(path, expected, *, without_face=False):
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(stored, dict):
                    return False
                if path == pre_meta and "first_model_witness" not in stored:
                    expected = {k: v for k, v in expected.items() if k != "first_model_witness"}
                if without_face:
                    stored = {k: v for k, v in stored.items() if k != "face_refine" and not k.startswith("fr_")}
                    expected = {k: v for k, v in expected.items() if k != "face_refine" and not k.startswith("fr_")}
                if metadata_only:
                    from .segment_cache import _inspect_drop_keys
                    drop = _inspect_drop_keys(plan, first_pass=path == pre_meta)
                    if path != pre_meta and plan.face_refine and plan.face_refine.get("has_sigmas_tensor"):
                        drop.add("fr_sigmas")
                    if path != pre_meta and getattr(plan, "sample_sigmas_linked", False):
                        drop.add("lucas_first_pass")
                    return {k: v for k, v in stored.items() if k not in drop} == {k: v for k, v in expected.items() if k not in drop}
                return stored == expected
            except (OSError, ValueError):
                return False
        first_ok = matches(pre_meta, first_pass_cache_fingerprint(seg, plan)) and (root / f"seg_{seg.index:04d}.pre.av.pt").is_file()
        final_ok = matches(final_meta, segment_cache_fingerprint(seg, plan)) and any((root / f"seg_{seg.index:04d}{ext}").is_file() for ext in (".pt", ".frames.mkv"))
        refine_ok = bool(first_ok and plan.refine and matches(final_meta, segment_cache_fingerprint(seg, plan), without_face=True) and (root / f"seg_{seg.index:04d}.av.pt").is_file())
        legacy_model = False
        try:
            cached_meta = json.loads(pre_meta.read_text(encoding="utf-8"))
            cached_seed = cached_meta.get("seed")
            legacy_model = "first_model_witness" not in cached_meta
        except (OSError, ValueError, AttributeError):
            cached_seed = None
        rows.append({"index": seg.index, "seed": cached_seed, "next_seed": first_seed(plan, seg), "selected": seg.index in selected,
                     "first_valid": first_ok, "final_valid": final_ok, "refine_valid": refine_ok, "legacy_model_identity": legacy_model,
                     "state": "二采完成" if first_ok and final_ok and plan.refine else "二采完成／脸修需要更新" if refine_ok else "待二采" if first_ok else "需要更新"})
    mode = cfg["mode"]
    errors = []
    if not selected:
        errors.append("未选中运行片段")
    if mode in {"refine", "merge_first", "merge_refine"}:
        missing = [r["index"] + 1 for r in rows if r["selected"] and not r["first_valid"]]
        if missing:
            errors.append(f"一采缓存缺失或参数已变：片段 {missing}；请先仅一采重做")
    if mode == "merge_refine":
        missing = [r["index"] + 1 for r in rows if r["selected"] and not r["final_valid"]]
        if missing:
            errors.append(f"有效二采／脸修成片缺失：片段 {missing}；合成模式不会补采样")
    if mode == "merge_first" and plan.face_refine:
        missing = [r["index"] + 1 for r in rows if r["selected"] and not r["final_valid"]]
        if missing:
            errors.append(f"有效一采脸修结果缺失：片段 {missing}；请关闭脸修合成原始一采，或先完成脸修")
    return {"mode": mode, "label": LABELS[mode], "can_execute": not errors, "errors": errors, "segments": rows}
