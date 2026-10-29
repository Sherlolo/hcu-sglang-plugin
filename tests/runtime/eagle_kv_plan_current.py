from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.speculative import eagle_worker_common as S
from sglang.srt.plugins.hook_registry import HookRegistry,HookType
from hcu_sglang import speculative_compat as C

class EagleKVPlanTests(unittest.TestCase):
    def run_case(self,*,mhc,keyword=False):
        batch=SimpleNamespace(return_hidden_states_before_norm=False);plan=object();Prepared=namedtuple('Prepared','batch graph_flag kv_loc_plan');payload=Prepared(batch,True,plan)
        runner=SimpleNamespace(model_config=SimpleNamespace(hc_hidden_size=16384 if mhc else None));graph=SimpleNamespace(model_config=SimpleNamespace(hc_hidden_size=None if mhc else 123))
        def original(draft_input,req_to_token_pool,batch,cuda_graph_runner,draft_model_runner,topk,num_steps,*,target_translator=None,num_draft_tokens=None):
            self.assertIs(draft_model_runner,runner);self.assertIs(cuda_graph_runner,graph);return payload
        with patch.object(S,'prepare_for_draft',original):
            HookRegistry.reset();HookRegistry.register('sglang.srt.speculative.eagle_worker_common.prepare_for_draft',C.around_prepare_for_draft,HookType.AROUND);HookRegistry.apply_hooks()
            if keyword:result=S.prepare_for_draft(None,None,None,graph,draft_model_runner=runner,topk=1,num_steps=3)
            else:result=S.prepare_for_draft(None,None,None,graph,runner,1,3)
            actual,flag,kv_loc_plan=result
        HookRegistry.reset()
        self.assertIs(result,payload);self.assertIs(actual,batch);self.assertTrue(flag);self.assertIs(kv_loc_plan,plan)
        self.assertEqual(batch.return_hidden_states_before_norm,mhc)
    def test_mhc_positional_runner_and_plan_identity(self):self.run_case(mhc=True)
    def test_mhc_keyword_runner_and_plan_identity(self):self.run_case(mhc=True,keyword=True)
    def test_normal_model_keeps_original_hidden_rule(self):self.run_case(mhc=False)

if __name__=='__main__':unittest.main()
