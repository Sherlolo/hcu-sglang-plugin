from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import inspect,os,unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.mem_cache import memory_pool as N
from sglang.srt.arg_groups import speculative_hook as S
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.plugins.hook_registry import HookRegistry,HookType
from hcu_sglang.mem_cache import dsa as plugin,dsa_pool_methods as P,dsa_index_cache_mode as K
from hcu_sglang import speculative_compat as C

class DsaDsparkCurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_signature=inspect.signature(N.MLATokenToKVPool.__init__)
        HookRegistry.reset();plugin.register_hooks()
        HookRegistry.register('sglang.srt.arg_groups.speculative_hook._handle_dspark',C.around_handle_dspark,HookType.AROUND)
        HookRegistry.apply_hooks()

    def make_pool(self,*,fp8=True,compress=False):
        signature=self.base_signature
        def base(pool,*args,**kwargs):
            args=signature.bind(pool,*args,**kwargs);args.apply_defaults();d=args.arguments
            pool.size=d['size'];pool.page_size=d['page_size'];pool.dtype=d['dtype'];pool.device=d['device'];pool.layer_num=d['layer_num'];pool.start_layer=d['start_layer'];pool.end_layer=d['end_layer'];pool.custom_mem_pool=None;pool.layer_transfer_counter=None
            pool.memory_saver_adapter=SimpleNamespace(region=lambda *args:nullcontext())
            pool.kv_buffer=[torch.zeros(pool.size+pool.page_size,1,8) for _ in range(pool.layer_num)]
        with ExitStack() as stack:
            stack.enter_context(patch.object(N.MLATokenToKVPool,'__init__',base))
            stack.enter_context(patch.object(N.DSATokenToKVPool,'_finalize_allocation_log',lambda *args:None))
            stack.enter_context(patch.object(N,'get_parallel',return_value=SimpleNamespace(attn_dcp_size=2)))
            stack.enter_context(patch.object(plugin,'is_dcu',return_value=True))
            stack.enter_context(patch.object(P,'_is_hcu',True))
            stack.enter_context(patch.object(K,'is_hcu',return_value=True))
            stack.enter_context(patch.object(K,'is_hcu_gfx936',return_value=False))
            stack.enter_context(patch.object(K,'is_hcu_native_fp8_supported',return_value=fp8))
            stack.enter_context(patch.dict(os.environ,{'SGLANG_DSA_HCU_INT8_INDEX_K_CACHE':'0'}))
            pool=N.DSATokenToKVPool(size=128,page_size=64,kv_lora_rank=4,dtype=torch.float8_e4m3fn,qk_rope_head_dim=4,layer_num=3,device='cpu',index_head_dim=128,enable_memory_saver=False,kv_cache_dim=8,start_layer=4,end_layer=7,index_buf_size=128,index_kpool=2 if compress else 1,index_kpool_compress=compress,tail_extra_slots=1,max_running_requests=2,indexer_layer_ids=[6,4],dcp_replicated=True)
        return pool

    def test_current_constructor_owns_compact_layers_pages_and_compress_tails(self):
        pool=self.make_pool(compress=True)
        self.assertEqual(pool.indexer_layer_ids,(4,6));self.assertEqual(pool.indexer_layer_num,2)
        self.assertEqual(pool.index_page_size,64);self.assertEqual(pool.index_kpool,2)
        self.assertEqual(pool.get_state_layer_ids(),[4,6])
        self.assertEqual(pool.get_kv_layer_ids(),[4,5,6])
        self.assertEqual(len(pool.index_key_cache.buffer),2)
        self.assertEqual(pool._compress_tail_k[0].shape,(3,3,128));self.assertEqual(pool._compress_tail_k[1].shape[0],0)
        self.assertIsNone(pool.index_k_dequant_workspace)

    def test_scaled_cache_maps_global_layers_and_moves_key_and_scale(self):
        pool=self.make_pool();buf=pool.index_key_cache.buffer[1];page=buf[0];key=page[:64*128].view(64,128);scale=page[64*128:].view(torch.float32)
        key[3].fill_(17);scale[3]=.125
        with patch.object(plugin,'is_dcu',return_value=True),patch.object(N.MLATokenToKVPool,'move_kv_cache',lambda *args:None):
            self.assertEqual(pool.get_index_k_with_scale_buffer(6).data_ptr(),buf.data_ptr())
            pool.move_kv_cache(torch.tensor([9]),torch.tensor([3]))
        self.assertTrue(torch.equal(key[9],key[3]));self.assertEqual(scale[9].item(),.125)
        self.assertEqual(pool.index_key_cache.buffer[0].sum().item(),0)
        with self.assertRaises(ValueError):pool.index_key_cache.get_local_buffer(5)

    def test_bf16_cache_producer_consumer_use_same_compact_layer(self):
        pool=self.make_pool(fp8=False);value=torch.arange(256,dtype=torch.float32).view(2,1,128)
        pool.set_index_k_buffer(6,torch.tensor([1,65]),value)
        out=pool.get_index_k_buffer(6)
        self.assertEqual(out.dtype,torch.bfloat16);self.assertTrue(torch.equal(out[0,1],value[0].bfloat16()))
        self.assertTrue(torch.equal(out[1,1],value[1].bfloat16()));self.assertEqual(pool.index_k_buffer[0].sum().item(),0)

    def raw_dspark(self):
        return SimpleNamespace(device=None,model_path='/target',speculative_draft_model_path='/draft',attn_cp_size=1,tp_size=4,dp_size=4,disaggregation_mode='null',disaggregation_transfer_backend='mooncake',enable_prefill_cp=False,cp_strategy=None,enable_dp_lm_head=True,moe_a2a_backend='deepep',moe_runner_backend='deep_gemm',speculative_moe_a2a_backend='deepep',speculative_moe_runner_backend='deep_gemm',pp_size=1,speculative_num_steps=None,speculative_eagle_topk=None,speculative_dspark_block_size=3,speculative_num_draft_tokens=None,max_running_requests=None,speculative_dspark_align_verify_tokens_to_graph_tier=False,speculative_dspark_sps_table_path=None,_resolved_overrides=[('platform',{'device':'cuda','attn_dp_size':4})])

    def test_dspark_uses_live_resolution_and_keeps_raw_inputs(self):
        raw=self.raw_dspark()
        with patch.object(C,'is_dcu',return_value=True),patch('hcu_sglang.utils.detection.is_dcu',return_value=True),patch.object(S,'model_config_of',return_value=SimpleNamespace(hf_config=SimpleNamespace(architectures=['DeepseekV4ForCausalLM']))),patch('sglang.srt.speculative.dspark_components.dspark_config.read_draft_checkpoint_config',return_value=None),patch.dict(os.environ,{'SGLANG_RAGGED_VERIFY_MODE':'static'}):
            S._handle_dspark(raw)
        self.assertIsNone(raw.device);self.assertIsNone(raw.speculative_num_draft_tokens)
        view=resolving_view(raw);self.assertEqual(view.device,'cuda');self.assertEqual(view.speculative_num_draft_tokens,4);self.assertEqual(view.speculative_num_steps,1);self.assertEqual(view.max_running_requests,48)
        self.assertEqual(view.moe_a2a_backend,'deepep')

    def test_non_hcu_pool_arguments_are_forwarded_without_loss(self):
        args=object();seen=[]
        with patch.object(plugin,'is_dcu',return_value=False):
            plugin.around_init(lambda *a,**kw:seen.append((a,kw)),args,index_kpool=2,index_kpool_compress=True,tail_extra_slots=3,max_running_requests=7,dcp_replicated=True)
        self.assertIs(seen[0][0][0],args);self.assertEqual(seen[0][1]['tail_extra_slots'],3);self.assertTrue(seen[0][1]['dcp_replicated'])

if __name__=='__main__':unittest.main()
