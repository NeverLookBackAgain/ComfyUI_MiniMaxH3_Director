"""CPU regression: real cache files + executor, replacing only GPU sampling/decoding."""
import sys, types, unittest, tempfile
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT.parent.parent))
sys.argv.append('--cpu')
pkg = types.ModuleType('lucas_tests'); pkg.__path__ = [str(ROOT)]; sys.modules['lucas_tests'] = pkg
from lucas_tests.director import execution_modes as modes, executor_core as core, segment_cache as cache
from lucas_tests.director.plan import DirectorPlan, SegmentPlan
from lucas_tests.director.refine_pack import pack_refine, refine_seed_for
from lucas_tests.nodes.director_common import finalize_director_outputs
import folder_paths, torch
sys.argv.remove('--cpu')

class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dirpatch = patch.object(folder_paths, 'get_output_directory', return_value=self.tmp.name); self.dirpatch.start()
        self.events = []
        def sample(**kw):
            self.events.append(('first', kw['seed']))
            return {'samples': torch.full((1, 4, 1, 1), .2)}
        def refine(*args, **kw):
            self.events.append(('refine', kw['seed']))
            return {'samples': torch.full((1, 4, 1, 1), .6)}, 'refined'
        def selflift(**kw):
            self.events.append(('selflift', kw['seed']))
            return {'samples': torch.full((1, 4, 1, 1), .3)}, {'samples': torch.zeros(1, 4, 1, 1)}
        def decode(samples, *_args, **_kw):
            data=samples['samples']
            if not torch.is_tensor(data): data=data.unbind()[0]
            self.events.append(('decode', round(float(data.mean()), 1)))
            return torch.full((5, 32, 32, 3), float(data.mean())), {}
        from lucas_tests.director.face_refine import runtime
        def face(**kw):
            self.events.append(('face', kw['seed']))
            return kw['frames'], 'face test'
        self.patches = [
            patch.object(core, 'sample_single_stage', side_effect=sample),
            patch.object(core, 'sample_selflift_stage', side_effect=selflift),
            patch.object(core, 'apply_segment_refine', side_effect=refine),
            patch.object(core, 'run_minimax_conditioning', return_value=([], [], {'samples': torch.zeros(1,4,1,1)}, 'test')),
            patch.object(core, '_decode_av_latent', side_effect=decode),
            patch.object(core, 'cleanup_segment_vram'),
            patch.object(runtime, 'apply_segment_face_refine', side_effect=face),
            patch.object(core, 'report_director_progress'), patch.object(core, 'report_director_finish'),
        ]
        for p in self.patches: p.start()
    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.dirpatch.stop(); self.tmp.cleanup()
    def plan(self, mode, *, seed=99, face=False, source='director', export=True, skip=True):
        cfg={'enabled':True,'mode':mode,'first_source':source,'face_enabled':face,'export_first':export,'skip_completed':skip,'seed_mode':'cached_first_pass'}
        p=DirectorPlan(frame_rate=24,total_frames=10,width=32,height=32,ref_max_size=32,output_mode='fixed',source_width=32,source_height=32,global_task_type='t2v',global_task_key='t2v',global_prompt='',global_refs=[],segments=[SegmentPlan(i,i*5,i*5+5,f'prompt{i}','t2v','t2v',False) for i in range(2)],source_video=torch.full((1,32,32,3),.5),edit_mode='segment',raw={'output':{'audioMode':'mute'},'lucasExecution':cfg})
        p.refine=pack_refine(mode='refine',passes=1,seed_mode='cached_first_pass',skip_fl2v=False)
        p.selflift={'enabled':True,'native_scale':.5} if source=='selflift' else None
        p.face_refine={'enabled':True} if face else None
        p.sample_seed=seed;p.sample_cfg=1;p.sample_steps=25;p.sample_sampler='res_multistep';p.sample_scheduler='simple'
        return modes.configure_execution(p)
    def run_plan(self,p,seed=99):
        return core.execute_director_plan_core(p,node_id='12',model=None,vae=None,audio_vae=None,clip=None,seed=seed,clear_vram_between_segments=False)
    def cache_first(self,p,seeds=(10,20)):
        p.segment_seeds=dict(enumerate(seeds));p.execution_node_id='12'
        for seg in p.segments:
            cache.save_first_pass_cache('12',seg,p,av_latent={'samples':torch.full((1,4,1,1),.2)},frames=torch.full((5,32,32,3),.2),handoff={'trim_frames':0,'export_frames':5,'sample_frames':5})
    def test_mixed_seed_refine_and_reuse(self):
        p=self.plan('refine');self.cache_first(p)
        result=self.run_plan(p,seed=999)
        self.assertEqual([e for e in self.events if e[0]=='refine'], [('refine',10),('refine',20)])
        self.assertFalse(any(e[0]=='first' for e in self.events));self.assertFalse(result[7])
        self.events.clear();self.run_plan(p,seed=12345)
        self.assertFalse(any(e[0] in {'first','refine'} for e in self.events))
    def test_parameters_changed_reject_before_sampling(self):
        p=self.plan('refine');self.cache_first(p);p.segments[1].prompt='changed'
        with self.assertRaisesRegex(ValueError,'一采缓存'): self.run_plan(p)
        self.assertEqual(self.events,[])
    def test_first_always_rerolls_and_records_seed(self):
        p=self.plan('first');self.run_plan(p,seed=70);self.run_plan(p,seed=71)
        self.assertEqual([e[1] for e in self.events if e[0]=='first'],[70,70,71,71])
        q=self.plan('refine');check=modes.preflight(q,'12')
        self.assertTrue(check['can_execute']);self.assertEqual([r['seed'] for r in check['segments']],[71,71])
    def test_confirm_whole_selection_two_steps(self):
        p=self.plan('confirm');self.cache_first(p);p.segments[1].prompt='new prompt'
        out=self.run_plan(p)
        self.assertTrue(out[7]);self.assertFalse(any(e[0]=='refine' for e in self.events))
        self.events.clear();out=self.run_plan(p)
        self.assertFalse(out[7]);self.assertEqual(len([e for e in self.events if e[0]=='refine']),2)
        self.assertFalse(any(e[0]=='first' for e in self.events))
    def test_first_plus_refine_and_disabled_first_export(self):
        p=self.plan('first_refine',export=False);out=self.run_plan(p)
        self.assertEqual([e[0] for e in self.events if e[0] in {'first','refine'}],['first','refine','first','refine'])
        self.assertEqual(len([e for e in self.events if e[0]=='decode']),2)
        self.assertEqual(out[6],[])
        final=finalize_director_outputs(p,out[0],out[1],out[3],segment_audios=out[2],segment_frame_counts=out[4],pre_refine_combined=out[5],pre_refine_segments=out[6])
        from comfy_execution.graph import ExecutionBlocker
        self.assertIsInstance(final[6],ExecutionBlocker)
    def test_merge_first_selected_only_and_no_sampling(self):
        p=self.plan('merge_first');self.cache_first(p);p.run_indices=frozenset({0})
        # Reconfiguration must preserve the selected range.
        modes.configure_execution(p);out=self.run_plan(p,seed=999)
        self.assertEqual(p.run_indices,frozenset({0}))
        self.assertEqual(out[0].frame_count,5);self.assertEqual(self.events,[])
    def test_merge_refine_only_valid_final_then_first_reroll_expires(self):
        p=self.plan('refine');self.cache_first(p);self.run_plan(p)
        self.events.clear();q=self.plan('merge_refine');out=self.run_plan(q)
        self.assertEqual(out[0].frame_count,10);self.assertEqual(self.events,[])
        r=self.plan('first');r.run_indices=frozenset({0});self.run_plan(r,seed=88)
        self.events.clear()
        with self.assertRaisesRegex(ValueError,'有效二采'):self.run_plan(q)
        self.assertEqual(self.events,[])
    def test_passes_change_refine_again_without_first(self):
        p=self.plan('refine');self.cache_first(p);self.run_plan(p)
        p.refine['passes']=2;self.events.clear();self.run_plan(p)
        self.assertEqual(len([e for e in self.events if e[0]=='refine']),2)
        self.assertFalse(any(e[0]=='first' for e in self.events))
    def test_missing_merge_cache_is_explicit(self):
        p=self.plan('merge_first')
        with self.assertRaisesRegex(ValueError,'一采缓存'):self.run_plan(p)
        self.assertEqual(self.events,[])
    def test_selflift_and_face_order(self):
        p=self.plan('first_refine',source='selflift',face=True);self.run_plan(p)
        self.assertEqual([e[0] for e in self.events if e[0] in {'selflift','refine','face'}],['selflift','refine','face']*2)
    def test_face_change_reuses_second_latent(self):
        p=self.plan('refine',face=True);self.cache_first(p);self.run_plan(p)
        self.events.clear();p.face_refine['denoise']=.7;self.run_plan(p)
        self.assertFalse(any(e[0] in {'first','refine'} for e in self.events))
        self.assertEqual(len([e for e in self.events if e[0]=='face']),2)
    def test_skip_disabled_forces_refine_but_not_first(self):
        p=self.plan('refine');self.cache_first(p);self.run_plan(p)
        p.execution['skip_completed']=False;self.events.clear();self.run_plan(p)
        self.assertEqual(len([e for e in self.events if e[0]=='refine']),2)
        self.assertFalse(any(e[0]=='first' for e in self.events))
    def test_legacy_first_cache_supported_and_raw_lora_change_rejected(self):
        p=self.plan('refine');p.execution=None;self.cache_first(p)
        modes.configure_execution(p);check=modes.preflight(p,'12')
        self.assertTrue(check['can_execute'])
        p.segments[0].loras=[{'name':'changed.safetensors','strength':1.0,'active':True}]
        check=modes.preflight(p,'12');self.assertFalse(check['can_execute'])
    def test_no_previous_pixels_leak_when_first_decode_omitted(self):
        p=self.plan('first');self.run_plan(p,seed=99)
        q=self.plan('first_refine',export=False);self.run_plan(q,seed=100)
        modes.stamp_cached_seeds('12',q)
        self.assertIsNone(cache.load_first_pass_frames_stale('12',q.segments[0],q))
        self.events.clear();m=self.plan('merge_first');out=self.run_plan(m,seed=777)
        self.assertEqual(out[0].frame_count,10)
        self.assertFalse(any(e[0] in {'first','refine'} for e in self.events))
    def test_merge_empty_selection_preserved_and_rejected(self):
        import json
        raw={'runSelectEnabled':True,'runSelection':[],'lucasExecution':{'enabled':True,'mode':'merge_first'}}
        normalized=json.loads(modes.normalize_execution_timeline(json.dumps(raw)))
        self.assertTrue(normalized['runSelectEnabled'])
        self.assertEqual(normalized['runSelection'],[])
        p=self.plan('merge_first');p.run_indices=frozenset()
        with self.assertRaisesRegex(ValueError,'未选中'):self.run_plan(p)
        self.assertEqual(self.events,[])

    def test_merge_first_unselected_missing_cache_is_allowed(self):
        p=self.plan('merge_first');self.cache_first(p);p.run_indices=frozenset({1})
        for path in (Path(self.tmp.name)/'minimax_seg_cache'/'12').glob('seg_0000.*'):path.unlink()
        self.assertTrue(modes.preflight(p,'12')['can_execute'])
        out=self.run_plan(p)
        self.assertEqual(out[0].frame_count,5)
        self.assertIn('片段顺序：[2]',out[3]);self.assertEqual(self.events,[])

    def test_merge_refine_unselected_missing_or_changed_cache_is_allowed(self):
        p=self.plan('refine');self.cache_first(p);self.run_plan(p)
        q=self.plan('merge_refine',export=False);q.run_indices=frozenset({1})
        q.segments[0].prompt='unselected edit'
        for path in (Path(self.tmp.name)/'minimax_seg_cache'/'12').glob('seg_0000.*'):path.unlink()
        self.events.clear()
        self.assertTrue(modes.preflight(q,'12')['can_execute'])
        out=self.run_plan(q)
        self.assertEqual(out[0].frame_count,5)
        self.assertIn('片段顺序：[2]',out[3]);self.assertEqual(self.events,[])
        q.run_indices=None
        with self.assertRaisesRegex(ValueError,'一采缓存'):self.run_plan(q)
        self.assertEqual(self.events,[])

    def test_merge_selected_missing_final_rejected_before_export(self):
        p=self.plan('refine');self.cache_first(p);self.run_plan(p)
        q=self.plan('merge_refine',export=False);q.run_indices=frozenset({1})
        (Path(self.tmp.name)/'minimax_seg_cache'/'12'/'seg_0001.meta.json').unlink()
        self.events.clear()
        with self.assertRaisesRegex(ValueError,'有效二采'):self.run_plan(q)
        self.assertEqual(self.events,[])

    def test_nonadjacent_merge_video_contains_only_selected_clips_in_order(self):
        import av
        p=self.plan('merge_first')
        p.segments.append(SegmentPlan(2,10,15,'prompt2','t2v','t2v',False))
        p.total_frames=15;self.cache_first(p,seeds=(10,20,30))
        modes.stamp_cached_seeds('12',p)
        for seg in p.segments:
            cache.save_first_pass_cache('12',seg,p,
                av_latent={'samples':torch.full((1,4,1,1),.2)},
                frames=torch.full((5,32,32,3),(.1,.5,.9)[seg.index]),
                handoff={'trim_frames':0,'export_frames':5,'sample_frames':5})
        p.run_indices=frozenset({2,0});self.events.clear();out=self.run_plan(p)
        self.assertEqual(out[0].frame_count,10)
        self.assertIn('片段顺序：[1, 3]',out[3])
        with av.open(out[0].path) as video:
            means=[frame.to_ndarray(format='rgb24').mean()/255 for frame in video.decode(video=0)]
        self.assertEqual(len(means),10)
        self.assertLess(max(means[:5]),.15);self.assertGreater(min(means[5:]),.85)
        self.assertEqual(self.events,[])

    def test_selected_refine_generated_audio_matches_selected_clip(self):
        from lucas_tests.director import execution_stream_merge as stream
        p=self.plan('refine');self.cache_first(p);self.run_plan(p)
        q=self.plan('merge_refine',export=False);q.run_indices=frozenset({1})
        q.raw['output']['audioMode']='generate';self.events.clear()
        wave={'waveform':torch.ones(1,2,5000)*.25,'sample_rate':24000}
        with patch.object(stream,'load_segment_audio',return_value=wave) as load_audio:
            out=self.run_plan(q)
        self.assertEqual(load_audio.call_count,1)
        self.assertEqual(load_audio.call_args.args[1].index,1)
        self.assertEqual(out[0].frame_count,5)
        self.assertEqual(out[0].audio['waveform'].shape[-1],5000)
        self.assertEqual(self.events,[])

    def test_stream_nonadjacent_selection_has_no_seam_processing(self):
        from lucas_tests.director import execution_stream_merge as stream
        clips=[torch.full((2,32,32,3),v) for v in (.1,.3,.5)]
        frames=stream.StreamingFrames(lambda i:clips[i], [2,2,2],32,32,
            continuity=True,continuity_pairs=[False,False,True])
        with patch.object(stream,'prepare_continuity_pair',side_effect=lambda a,b:(a,b)) as seam:
            result=torch.stack(list(frames))
        self.assertEqual(seam.call_count,1)
        self.assertTrue(torch.equal(result,torch.cat(clips)))

    def test_partial_merge_source_audio_uses_selected_timeline_range(self):
        from lucas_tests.director import audio_export
        p=self.plan('merge_first');self.cache_first(p);p.run_indices=frozenset({1})
        p.raw['output']['audioMode']='source'
        p.global_task_key='r2v'
        wave={'waveform':torch.ones(1,2,10000)*.25,'sample_rate':24000}
        with patch.object(audio_export,'extract_timeline_audio',return_value=wave) as extract:
            out=self.run_plan(p)
        self.assertEqual(extract.call_args.args[1:3],(5,10))
        self.assertEqual(out[0].frame_count,5)
        self.assertEqual(out[0].audio['waveform'].shape[-1],5000)

    def test_model_wiring_changes_invalidate_correct_stage(self):
        p=self.plan('refine');p.execution['model_witness']={'first':['unetA'],'second':['refineA'],'face':[]}
        self.cache_first(p);self.run_plan(p)
        p.execution['model_witness']['second']=['refineB'];self.events.clear();self.run_plan(p)
        self.assertFalse(any(e[0]=='first' for e in self.events))
        self.assertEqual(len([e for e in self.events if e[0]=='refine']),2)
        p.execution['model_witness']['first']=['unetB'];self.events.clear()
        with self.assertRaisesRegex(ValueError,'一采缓存'):self.run_plan(p)
        self.assertEqual(self.events,[])
    def test_legacy_first_without_model_record_still_readable(self):
        p=self.plan('refine');self.cache_first(p)
        p.execution['model_witness']={'first':['currentUnet'],'second':[],'face':[]}
        check=modes.preflight(p,'12');self.assertTrue(check['can_execute'])
        self.assertTrue(all(r['legacy_model_identity'] for r in check['segments']))
        self.run_plan(p)
        self.assertFalse(any(e[0]=='first' for e in self.events))
    def test_native_cached_seed_choice_on_legacy_graph(self):
        p=self.plan('refine');self.cache_first(p);p.execution=None
        self.run_plan(p,seed=999)
        self.assertEqual([e[1] for e in self.events if e[0]=='refine'],[10,20])
        self.assertFalse(any(e[0]=='first' for e in self.events))

    def test_confirm_first_export_off_still_allows_initial_review(self):
        p=self.plan('confirm',export=False);out=self.run_plan(p)
        self.assertTrue(out[7]);self.assertTrue(modes.output_first(p));self.assertTrue(out[6])
        self.events.clear();out=self.run_plan(p)
        self.assertFalse(out[7]);self.assertFalse(modes.output_first(p));self.assertEqual(out[6],[])
    def test_merge_refine_export_off_does_not_read_first_latents(self):
        p=self.plan('refine');self.cache_first(p);self.run_plan(p)
        q=self.plan('merge_refine',export=False)
        from lucas_tests.director import execution_merge
        with patch.object(execution_merge,'load_first_pass_cache',side_effect=AssertionError('unneeded first latent read')):
            out=self.run_plan(q)
        self.assertEqual(out[0].frame_count,10)

    def test_first_merge_respects_successor_seam_trim(self):
        p=self.plan('merge_first');self.cache_first(p)
        cache.update_first_pass_export_handoff('12',p.segments[0],3)
        out=self.run_plan(p)
        self.assertEqual(out[0].frame_count,8)
        self.assertEqual(self.events,[])

    def test_legacy_no_mode_and_seed_rules(self):
        p=self.plan('first');p.raw.pop('lucasExecution');modes.configure_execution(p)
        self.assertIsNone(p.execution);self.assertTrue(modes.output_first(p))
        self.assertEqual(refine_seed_for({'seed_mode':'cached_first_pass'},10,2),10)
        self.assertEqual(refine_seed_for({'seed_mode':'offset'},10,2),13)
        self.assertEqual(refine_seed_for({'seed_mode':'independent','seed':100},10,2),102)

    def run_named(self, plan, name="project_A"):
        return core.execute_director_plan_core(plan, node_id="12", cache_name=name,
            model=None, vae=None, audio_vae=None, clip=None, seed=99,
            clear_vram_between_segments=False)

    def test_cache_name_sanitization_and_legacy_key(self):
        self.assertEqual(cache.resolve_segment_cache_key("", "12"), "12")
        self.assertEqual(cache.resolve_segment_cache_key("电影A", "12"), "电影A_12")
        self.assertNotEqual(cache.resolve_segment_cache_key("A", "12"), cache.resolve_segment_cache_key("A", "13"))
        self.assertEqual(cache.normalize_cache_name("CON"), "_CON")
        for name in ("../../outside", "..\\..\\outside", "a:b/c", "NUL", "x" * 100):
            key = cache.resolve_segment_cache_key(name, "12")
            root = cache._cache_dir(key).resolve()
            self.assertTrue(root.is_relative_to(Path(self.tmp.name).resolve() / "minimax_seg_cache"))
        self.assertEqual(len(cache.normalize_cache_name("x" * 100)), 64)
        for key in ("", "..", "../outside", "x/y", "x\\y"):
            with self.assertRaises(ValueError): cache._cache_dir(key)

    def test_named_cache_refine_and_selected_merges_keep_progress_node(self):
        from lucas_tests.director import progress
        p = self.plan("first")
        self.run_named(p)
        root = Path(self.tmp.name) / "minimax_seg_cache"
        self.assertTrue((root / "project_A_12/seg_0000.pre.av.pt").is_file())
        self.assertFalse((root / "12").exists())
        self.assertTrue(all(call.args[0] == "12" for call in core.report_director_progress.call_args_list))
        self.events.clear()
        p = self.plan("refine")
        self.run_named(p)
        self.assertFalse(any(e[0] == "first" for e in self.events))
        self.assertEqual(len([e for e in self.events if e[0] == "refine"]), 2)
        for mode in ("merge_first", "merge_refine"):
            q = self.plan(mode, export=False)
            q.run_indices = frozenset({1})
            self.events.clear()
            with patch.object(progress, "report_director_finish") as finish:
                out = self.run_named(q)
            self.assertEqual(out[0].frame_count, 5)
            self.assertEqual(self.events, [])
            finish.assert_called_once_with("12", 1)

    def test_named_clear_preserves_other_workflow_and_legacy_cache(self):
        p = self.plan("first")
        for name in ("project_A", "project_B", ""):
            self.run_named(self.plan("first"), name)
        for name in ("project_A", "project_B", ""):
            key = cache.resolve_segment_cache_key(name, "12")
            self.assertTrue(cache.first_pass_cache_disk_signature(key))
        self.assertGreater(cache.clear_segment_cache(cache.resolve_segment_cache_key("project_A", "12"), kind="all"), 0)
        self.assertFalse(cache.first_pass_cache_disk_signature("project_A_12"))
        self.assertTrue(cache.first_pass_cache_disk_signature("project_B_12"))
        self.assertTrue(cache.first_pass_cache_disk_signature("12"))

    def test_named_node_change_signature_watches_named_cache(self):
        import json
        from lucas_tests.nodes.director import MiniMaxH3Director
        timeline = json.dumps({"lucasExecution": {"enabled": True}})
        before = MiniMaxH3Director.IS_CHANGED(unique_id="12", cache_name="project_A", timeline_data=timeline)
        self.run_named(self.plan("first"))
        after = MiniMaxH3Director.IS_CHANGED(unique_id="12", cache_name="project_A", timeline_data=timeline)
        self.assertNotEqual(before, after)
        self.assertEqual(MiniMaxH3Director.IS_CHANGED(unique_id="12", cache_name="project_B", timeline_data=timeline), "")
        self.assertEqual(MiniMaxH3Director.IS_CHANGED(unique_id="12", cache_name="project_A"), cache.first_pass_cache_disk_signature("project_A_12"))

    def test_named_http_status_and_clear_resolve_same_key(self):
        import asyncio, json
        from lucas_tests.director import http_routes
        p = self.plan("first")
        self.run_named(p)
        class Request:
            async def json(self):
                return {"node_id": "12", "cache_name": "project_A", "timeline_data": p.raw, "kind": "all", "seed": 99, "sampler": "res_multistep", "scheduler": "simple"}
        with patch("lucas_tests.director.plan.build_director_plan", return_value=p):
            response = asyncio.run(http_routes.minimax_first_pass_cache_status(Request()))
        self.assertEqual(response.status, 200, response.text)
        payload = json.loads(response.text)
        self.assertTrue(all(row["first_valid"] for row in payload["execution"]["segments"]))
        response = asyncio.run(http_routes.minimax_clear_segment_cache(Request()))
        self.assertEqual(response.status, 200, response.text)
        self.assertGreater(json.loads(response.text)["removed"], 0)
        self.assertFalse(cache.first_pass_cache_disk_signature("project_A_12"))

if __name__=='__main__': unittest.main(verbosity=2)
