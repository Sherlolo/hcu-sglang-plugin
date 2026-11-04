import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from compare_saved_gsm8k import compare


class PairedMathComparisonTests(unittest.TestCase):
    def inputs(self):
        manifest={'dataset_sha256':'pinned','sample_indices':[5,6],'shot_indices':[0],'temperature':0,'max_tokens':1024,'parallel':8,'chat_template_kwargs':{'enable_thinking':False},'scorer_protocol':'fixed'}
        state={'status':'passed','cleanup_remaining_pids':[],'port_listening_after_cleanup':False,'config':{'args':['--model-path','same-checkpoint'],'recipe_commit':'same','physical_devices':list(range(8)),'env':{}}}
        def report(correct):return {'accuracy':sum(correct)/2,'invalid_responses':0,'responses':[{'sample_index':i,'correct':c,'predicted':str(i),'valid_response':True} for i,c in zip([5,6],correct)]}
        return report([True,False]),report([False,True]),manifest,copy.deepcopy(manifest),state,copy.deepcopy(state)

    def test_equal_aggregate_score_still_reports_opposite_paired_outcomes(self):
        report=compare(*self.inputs());self.assertEqual(report['status'],'comparable')
        self.assertEqual(report['outcomes'],{'both_correct':0,'baseline_only_correct':1,'candidate_only_correct':1,'both_incorrect':0})

    def test_load_aware_variant_is_not_silently_compared_to_plain_recipe(self):
        values=list(self.inputs());values[5]['config']['args']+=['--ep-dispatch-algorithm','load_aware']
        report=compare(*values);self.assertEqual(report['status'],'incomparable');self.assertIn('serving.args',report['mismatches']);self.assertNotIn('outcomes',report)

    def test_sampling_and_optimized_env_differences_are_rejected(self):
        values=list(self.inputs());values[3]['max_tokens']=2048;values[5]['config']['env']['SGLANG_USE_LIGHTOP']='0'
        report=compare(*values);self.assertIn('manifest.max_tokens',report['mismatches']);self.assertIn('serving.env.SGLANG_USE_LIGHTOP',report['mismatches'])

    def test_source_plugin_selector_and_task_output_paths_can_differ(self):
        values=list(self.inputs());values[4]['config']['env']={'SGLANG_PLUGINS':'none','SGLANG_TORCH_PROFILER_DIR':'baseline'};values[5]['config']['env']={'SGLANG_PLUGINS':'hcu','SGLANG_TORCH_PROFILER_DIR':'candidate'}
        self.assertEqual(compare(*values)['status'],'comparable')


if __name__=='__main__':unittest.main()
