"""Assemble explicitly requested cache stages; never run conditioning or sampling."""
from __future__ import annotations

from .audio_export import resolve_audio_mode, AUDIO_MODE_GENERATE, empty_audio_dict
from .execution_modes import output_first, stage_report, LABELS
from .segment_cache import load_first_pass_cache, load_first_pass_frames_stale, load_segment_cache, load_segment_audio
from .segment_continuity import concat_continuous_chunks
from .progress import report_director_finish


def merge_cached_plan(plan, *, node_id, vae, audio_vae, progress_node_id=None):
    progress_node_id = node_id if progress_node_id is None else progress_node_id
    if plan.execution["mode"] == "merge_first":
        from .execution_stream_merge import merge_first_to_video
        return merge_first_to_video(plan, node_id=node_id, progress_node_id=progress_node_id, vae=vae, audio_vae=audio_vae)
    if plan.execution["mode"] == "merge_refine":
        from .execution_stream_merge import merge_refine_to_video
        return merge_refine_to_video(plan, node_id=node_id, progress_node_id=progress_node_id, vae=vae, audio_vae=audio_vae)
    from .executor_core import _decode_av_latent, _trim_decoded_to_export, _unpack_node_output
    chunks, pres, audios = [], [], []
    mode = plan.execution["mode"]
    report = [stage_report(plan, f"{LABELS[mode]}：读取全部 {len(plan.segments)} 段有效缓存，不执行采样")]
    for seg in plan.segments:
        need_first = mode == "merge_first" or output_first(plan)
        cache = load_first_pass_cache(node_id, seg, plan) if need_first else None
        if need_first and cache is None:
            raise ValueError(f"片段 {seg.timeline_index + 1} 一采缓存损坏或参数已变，合成已停止。")
        handoff = (cache or {}).get("handoff") or {}
        first = None
        first_audio = None
        if mode == "merge_first" or output_first(plan):
            first = load_first_pass_frames_stale(node_id, seg, plan)
            if first is None:
                decoded, first_audio = _decode_av_latent(cache["av_latent"], vae, audio_vae,
                    decode_audio=resolve_audio_mode(plan) == AUDIO_MODE_GENERATE)
                first, first_audio = _trim_decoded_to_export(decoded, first_audio,
                    trim_frames=int(handoff.get("trim_frames") or 0),
                    export_len=int(handoff.get("export_frames") or seg.frame_count), plan=plan)
        if mode == "merge_refine" or plan.face_refine:
            chunk = load_segment_cache(node_id, seg, plan)
            if chunk is None:
                raise ValueError(f"片段 {seg.timeline_index + 1} 有效成片无法读取，合成已停止。")
            audio = load_segment_audio(node_id, seg, plan) or empty_audio_dict()
        else:
            chunk = first
            if first_audio is None and resolve_audio_mode(plan) == AUDIO_MODE_GENERATE and audio_vae is not None:
                try:
                    from comfy_extras.nodes_audio import VAEDecodeAudio
                except ImportError:
                    from comfy_extras.nodes_lt import VAEDecodeAudio
                first_audio = _unpack_node_output(VAEDecodeAudio.execute(audio_vae, cache["av_latent"]))[0]
                _, first_audio = _trim_decoded_to_export(first.new_zeros((int(first.shape[0]) + int(handoff.get("trim_frames") or 0), 1, 1, 3)), first_audio,
                    trim_frames=int(handoff.get("trim_frames") or 0),
                    export_len=int(handoff.get("export_frames") or first.shape[0]), plan=plan)
            audio = first_audio or empty_audio_dict()
        chunks.append(chunk.float())
        if first is not None:
            pres.append(first.float())
        audios.append(audio)
        report.append(f"片段 {seg.timeline_index + 1}：读取缓存，一采 seed={plan.segment_seeds.get(seg.index, plan.sample_seed)}")
    combined = concat_continuous_chunks(chunks, plan.segments, plan)
    pre = concat_continuous_chunks(pres, plan.segments, plan) if pres and mode == "merge_refine" else combined
    report_director_finish(progress_node_id, len(chunks))
    return combined, chunks, audios, "\n".join(report), [int(c.shape[0]) for c in chunks], pre, pres, False, None, []
