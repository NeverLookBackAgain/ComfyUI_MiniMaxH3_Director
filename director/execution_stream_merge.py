"""Cache-only first/second-pass exports without an all-frames tensor.

The frame iterable is private to the native video writer. IMAGE sockets are
blocked after saving; downstream image nodes never receive a non-tensor object.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import torch
import folder_paths

from .audio_export import AUDIO_MODE_GENERATE, build_director_audio_outputs, resolve_audio_mode
from .segment_cache import (first_pass_cache_fingerprint, load_first_pass_frames_stale,
                            load_segment_cache, load_segment_audio)
from .execution_modes import first_fingerprint_matches
from .segment_continuity import prepare_continuity_pair
from ..lib.image_prep import pad_frames_to_canvas

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.stream_merge")


@dataclass
class StreamMergeResult:
    path: str
    saved: dict
    frame_count: int
    audio: dict
    additional_saved: list[dict] = field(default_factory=list)


class StreamingFrames:
    """Repeatable writer-only iterable, retaining at most two adjacent clips."""
    def __init__(self, load, counts, width, height, *, continuity=False, frame_limit=None, continuity_pairs=None):
        self.load = load
        self.counts = list(counts)
        self.continuity = continuity
        self.continuity_pairs = continuity_pairs
        # Match pad_or_trim_frames: trim excess, never fabricate missing frames.
        length = sum(counts) if frame_limit is None else min(int(frame_limit), sum(counts))
        self.shape = (length, height, width, 3)

    def __len__(self):
        return self.shape[0]

    def _canvas(self, frame):
        if tuple(frame.shape) == tuple(self.shape[1:]):
            return frame
        return pad_frames_to_canvas(frame.unsqueeze(0), self.shape[2], self.shape[1])[0]

    def __iter__(self):
        from comfy.model_management import throw_exception_if_processing_interrupted
        left = None
        emitted = 0
        for i, count in enumerate(self.counts):
            throw_exception_if_processing_interrupted()
            body = self.load(i)
            if int(body.shape[0]) != count:
                raise ValueError(f"片段 {i + 1} 缓存帧数在导出过程中发生变化，已停止。")
            if left is not None:
                if self.continuity and (self.continuity_pairs is None or self.continuity_pairs[i]):
                    left, body = prepare_continuity_pair(left, body)
                for frame in left:
                    if emitted >= self.shape[0]:
                        return
                    throw_exception_if_processing_interrupted()
                    # Isolate one frame so the encoder cannot retain its full clip.
                    yield self._canvas(frame).clone()
                    emitted += 1
                del frame, left
            left = body
            del body
            log.info("缓存流式导出：读取片段 %d/%d (%d 帧)", i + 1, len(self.counts), count)
        if left is not None:
            for frame in left:
                if emitted >= self.shape[0]:
                    return
                throw_exception_if_processing_interrupted()
                yield self._canvas(frame).clone()
                emitted += 1
            del frame, left


def merge_first_to_video(plan, *, node_id, vae, audio_vae):
    from .progress import report_director_finish
    result, report, counts = _export_cache_stage(plan, node_id=node_id, vae=vae,
        audio_vae=audio_vae, stage="first", first_face=True)
    report_director_finish(node_id, len(counts))
    return result, [], [], report, counts, result, [], False, None, []


def merge_refine_to_video(plan, *, node_id, vae, audio_vae):
    from .execution_modes import output_first
    from .progress import report_director_finish
    import gc
    result, report, counts = _export_cache_stage(plan, node_id=node_id, vae=vae,
        audio_vae=audio_vae, stage="refine")
    if output_first(plan):
        # Finish and release all second-pass frames before opening first-pass frames.
        gc.collect()
        try:
            first, first_report, _ = _export_cache_stage(plan, node_id=node_id,
                vae=vae, audio_vae=audio_vae, stage="first", first_face=False)
        except Exception as exc:
            log.exception("二采视频已保存，但同时导出一采失败：%s", result.path)
            raise RuntimeError(f"二采视频已保存：{result.path}；同时导出一采失败：{exc}") from exc
        result.additional_saved.append(first.saved)
        report += "\n\n同时导出原始一采（与二采顺序执行）：\n" + first_report
        del first
        gc.collect()
    report_director_finish(node_id, len(counts))
    return result, [], [], report, counts, result, [], False, None, []


def _export_cache_stage(plan, *, node_id, vae, audio_vae, stage, first_face=False):
    from .executor_core import _decode_av_latent, _trim_decoded_to_export, _unpack_node_output
    from comfy_api.latest import InputImpl, Types
    from comfy.model_management import throw_exception_if_processing_interrupted

    root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / str(node_id)
    segments = [seg for seg in plan.segments if plan.run_indices is None or seg.index in plan.run_indices]
    generated = resolve_audio_mode(plan) == AUDIO_MODE_GENERATE
    face = bool(plan.face_refine) and (stage == "refine" or first_face)
    final_cache = stage == "refine" or face
    label = "合成二采" if stage == "refine" else "合成一采"

    def latent(seg):
        stored = json.loads((root / f"seg_{seg.index:04d}.pre.meta.json").read_text(encoding="utf-8"))
        if not first_fingerprint_matches(stored, first_pass_cache_fingerprint(seg, plan)):
            raise ValueError(f"片段 {seg.index + 1} 一采参数已变，合成已停止。")
        payload = torch.load(root / f"seg_{seg.index:04d}.pre.av.pt", map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "samples" not in payload:
            raise ValueError(f"片段 {seg.index + 1} 一采 latent 损坏。")
        return payload

    def handoff(seg):
        path = root / f"seg_{seg.index:04d}.pre.handoff.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def load(i):
        seg = segments[i]
        if final_cache:
            frames = load_segment_cache(node_id, seg, plan)
        else:
            frames = load_first_pass_frames_stale(node_id, seg, plan)
            if frames is None:
                decoded, _ = _decode_av_latent(latent(seg), vae, audio_vae, decode_audio=False)
                h = handoff(seg)
                frames, _ = _trim_decoded_to_export(decoded, None,
                    trim_frames=int(h.get("trim_frames") or 0),
                    export_len=int(h.get("export_frames") or seg.frame_count), plan=plan)
        if frames is None or not frames.shape[0]:
            raise ValueError(f"片段 {seg.index + 1} 有效画面缓存无法读取，合成已停止。")
        return frames.cpu().float()

    counts, audios, sizes = [], [], []
    # Scan one clip at a time for true post-trim counts and audio alignment.
    for i, seg in enumerate(segments):
        throw_exception_if_processing_interrupted()
        frames = load(i)
        counts.append(int(frames.shape[0]))
        sizes.append((int(frames.shape[2]), int(frames.shape[1])))
        del frames
        audio = None
        if generated:
            if final_cache:
                audio = load_segment_audio(node_id, seg, plan)
            else:
                if audio_vae is None:
                    raise ValueError("生成音频模式需要连接音频 VAE。")
                try:
                    from comfy_extras.nodes_audio import VAEDecodeAudio
                except ImportError:
                    from comfy_extras.nodes_lt import VAEDecodeAudio
                audio = _unpack_node_output(VAEDecodeAudio.execute(audio_vae, latent(seg)))[0]
                h = handoff(seg)
                _, audio = _trim_decoded_to_export(torch.zeros((counts[-1] + int(h.get("trim_frames") or 0), 1, 1, 3)), audio,
                    trim_frames=int(h.get("trim_frames") or 0), export_len=counts[-1], plan=plan)
            if not audio or not torch.is_tensor(audio.get("waveform")) or not audio["waveform"].numel():
                raise ValueError(f"片段 {seg.index + 1} 音频缓存缺失或损坏，合成已停止。")
        audios.append(audio)
        log.info("%s流式检查：片段 %d/%d，%d 帧", label, i + 1, len(segments), counts[-1])
    if not counts:
        raise ValueError(f"{label}：没有可合成的片段。")
    width = (max(w for w, h in sizes) + 1) // 2 * 2
    height = (max(h for w, h in sizes) + 1) // 2 * 2
    keep_tail = bool(plan.continuity_enabled and getattr(plan, "continuity_keep_tail", True))
    total = sum(counts) if keep_tail else min(sum(counts), int(plan.total_frames))
    pairs = [False] + [segments[i].index == segments[i - 1].index + 1
        for i in range(1, len(segments))]
    frames = StreamingFrames(load, counts, width, height,
        continuity=bool(plan.continuity_enabled), frame_limit=total, continuity_pairs=pairs)
    if not generated and plan.run_indices is not None:
        # Extract each original timeline range, then join only selected audio.
        # Shape-only descriptors avoid reading the video clips again.
        from types import SimpleNamespace
        from .audio_export import _merge_generated_segment_audios
        clips = [SimpleNamespace(shape=(count, height, width, 3)) for count in counts]
        audio_out, fallback = build_director_audio_outputs(plan, clips, export_segments=True,
            segment_frame_counts=counts, audio_mode=resolve_audio_mode(plan))
        audio = _merge_generated_segment_audios(plan, audio_out, total_frames=total,
            fps=float(plan.frame_rate or 24), frame_counts=counts)
    else:
        audio_out, fallback = build_director_audio_outputs(plan, [frames], export_segments=False,
            output_frame_end=total, segment_audios=audios if generated else None,
            segment_frame_counts=counts if generated else None, audio_mode=resolve_audio_mode(plan))
        audio = audio_out[0] if audio_out else None
    output_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(
        f"video/MiniMaxH3_Director_{'second' if stage == 'refine' else 'first'}_stream",
        folder_paths.get_output_directory(), width, height)
    # UUID also prevents collisions across simultaneous workflows.
    filename = f"{name}_{counter:05}_{uuid.uuid4().hex[:8]}.mp4"
    path = Path(output_dir) / filename
    partial = path.with_suffix(".partial.mp4")
    writer_audio = audio if audio and torch.is_tensor(audio.get("waveform")) and audio["waveform"].numel() else None
    video = InputImpl.VideoFromComponents(Types.VideoComponents(images=frames,
        audio=writer_audio, frame_rate=Fraction(str(float(plan.frame_rate or 24)))))
    try:
        video.save_to(str(partial), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264,
            metadata={"director_stream_merge": {"node_id": str(node_id), "stage": stage,
                "face_refine": face, "selected_segments": [seg.index + 1 for seg in segments], "segment_frames": counts,
                "segment_seeds": getattr(plan, "segment_seeds", {}), "timeline": plan.raw}})
        throw_exception_if_processing_interrupted()
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    saved = {"filename": filename, "subfolder": subfolder, "type": "output"}
    result = StreamMergeResult(str(path), saved, total, audio)
    source = ("二采／脸修最终缓存" if face else "二采最终缓存") if stage == "refine" else ("一采脸修缓存" if face else "原始一采缓存")
    report = (f"{label}：流式导出完成，{len(counts)} 段、{total} 帧、{width}×{height}。\n"
              f"片段顺序：{[seg.index + 1 for seg in segments]}。\n"
              f"画面：{source}；音频模式：{resolve_audio_mode(plan)}。\n"
              f"文件：{path}\n"
              "完整视频已由导演台直接保存；画面输出口已阻断，下游无需再次创建或保存视频。")
    if fallback:
        report += f"\n源音频回退：{fallback}"
    return result, report, counts
