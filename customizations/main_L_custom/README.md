# MiniMax H3 local customization snapshot

This package preserves the optimization deployed during the 2026-10-06 H3 Max local optimization study, plus the 2026-10-07 single-selected-segment decode-unload guard. It targets ComfyUI v0.38.2 on the measured Windows RTX 3080 20GB installation.

## Director changes

- After conditioning, multi-segment runs with segment cleanup enabled selectively release the MiniMaxH3TE text encoder instead of unloading every model.
- Before the final AV decode, unload models only when the number of segments selected for this run exceeds one. A twenty-segment timeline with only one segment selected skips this unload. Other cleanup settings retain their existing behavior.
- At the end of a segment, retain loaded models while clearing unused memory.

These changes are in the repository's `director/executor_core.py`. They do not implement the proposed six execution modes or mixed-seed whole-timeline refine flow.

## Runtime prerequisites preserved as a patch

`patches/comfyui-v0.38.2-local-runtime.patch` records the current installation's four tracked core modifications against the `core_base` commit in `manifest.json`:

- Extract H3 embedding/packing into `_embed_and_pack`.
- Make the H3 video VAE decoder layer count configurable and infer it from checkpoint keys.
- Cast VAE normalization weights through the existing offload-aware context.
- Load instance startup configuration before ComfyUI argument parsing.

These include runtime prerequisites inherited from earlier optimization work. The final 2026-10-06 deployment added TE-Speed startup whitelisting and optimized workflows on top of them. No core repository branch is changed by storing this patch here.

`runtime/` contains the startup script, configuration, and the exact-input singleton reference-image VAE cache. The reference cache does not share text conditioning between prompts. Review environment overrides before installing on another instance.

## Workflows and dependencies

`workflows/` contains four sanitized templates derived from the delivered GUI workflows: two with twenty empty segment slots and two with five empty segment slots, each in direct and confirm-first-pass variants. All generation prompts, input references, hidden workspace contents and preview data are removed. Segment timing, seeds, sampling settings, nodes, links and positions remain. Run selection is reset to all. Add your own inputs and prompts before generating; these empty templates cannot reproduce the historical benchmark without the private inputs.

The measured combination uses TE-Speed for the first pass, native SolAttn plus TE-Speed for refine, refine sparse window end 1.0, and no spatial tiling. T8 is not enabled. TE-Speed 3.5's third-party binary is not redistributed; its installed hash and the KJNodes commit are recorded in `manifest.json`. Use the matching installed dependencies.

## Applying on another matching installation

This is a reviewable source snapshot, not an automatic installer. First verify the target core revision, back up modified files and check the core patch with `git apply --check`. Apply it only to the matching clean base; do not reapply to this already patched installation. Copy runtime files to their matching core-relative locations and workflows to `user/default/workflows/`. Ensure the required plugins are installed and permitted by the instance launch whitelist. Restart is needed to load Python changes.

## Validation and limits

The source study measured three optimized repeats per five-segment generation mode: direct mean 1118.014 seconds versus baseline 1488.468; staged mean 1249.860 seconds versus baseline 1634.301. These historical results apply to that tested setup; they do not validate the new single-segment guard's performance. The study also completed a production-content segment-14 AV smoke test.

For this snapshot, four workflow JSONs and their link endpoints were checked, runtime Python parsed, and the updated Director syntax checked. No new GPU generation was submitted. The first-pass TE and R3 component quality had human acceptance; full-combination audio and visual equivalence remains subject to human review.

Upstream AIMixer `a8f57b8` was merged on 2026-10-07. The `director/plan.py` import conflict was resolved by retaining local `resolve_video_path` and upstream `normalize_lora_rows`. Local metadata-only cache inspection and all three memory scheduling changes remain. No new GPU generation or UI regression was run for this merge. Proposed publication target: `NeverLookBackAgain/ComfyUI_MiniMaxH3_Director`, branch `ComfyUI_MiniMaxH3_Director_Lucas_Custom`.
