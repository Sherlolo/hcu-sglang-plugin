from types import SimpleNamespace
from unittest.mock import patch
import ast,os,unittest
from pathlib import Path
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.layers.attention.current_dsa import backend as B, flashmla_backend as F
from sglang.srt.model_executor.forward_batch_info import ForwardMode

class FlashMlaCurrentTests(unittest.TestCase):
    def test_scheduler_object_from_live_metadata_reaches_attention(self):
        backend=object.__new__(B.DeepseekSparseAttnBackend)
        backend.flashmla_kv_num_q_heads=64;backend.dsa_index_topk=2;backend.physical_page_size=64;backend.kv_cache_dim=4;backend.dsa_kv_cache_store_fp8=True;backend._hcu_flashmla_decode_h16_enabled=False
        made=[];calls=[]
        def metadata(**kwargs):
            obj=SimpleNamespace(num_splits=torch.tensor([len(made)]));made.append((obj,kwargs));return obj,None
        def attention(**kwargs):
            calls.append(kwargs);q=kwargs['q'];return torch.ones((*q.shape[:-1],2)),None
        runtime=SimpleNamespace(get_mla_metadata=metadata,flash_mla_with_kvcache=attention)
        cache_seqlens=torch.tensor([3,0],dtype=torch.int32)
        with patch.object(B,'_is_hcu',True),patch.object(F,'import_module',return_value=runtime) as import_runtime:
            initial=backend._compute_flashmla_metadata(cache_seqlens,1)
            batch=SimpleNamespace(forward_mode=ForwardMode.DECODE,spec_info=SimpleNamespace(num_tokens_per_req=1),_original_batch_size=1)
            md=SimpleNamespace(dsa_cache_seqlens_int32=cache_seqlens,flashmla_metadata=initial)
            out=backend._forward_flashmla_kv(torch.zeros(2,2,4),torch.zeros(64,1,4),2,.25,SimpleNamespace(tp_q_head_num=2,head_dim=4),md,torch.zeros(2,2,dtype=torch.int32),batch)
        self.assertEqual(import_runtime.call_args.args[0],'flash_mla.flash_mla_interface')
        self.assertEqual(len(made),2)
        self.assertIs(calls[0]['tile_scheduler_metadata'],made[1][0])
        self.assertIs(calls[0]['num_splits'],made[1][0].num_splits)
        self.assertEqual(calls[0]['q'].shape[0],1);self.assertEqual(out.shape,(2,1,2,2))
        self.assertTrue(torch.equal(out[0],torch.ones_like(out[0])));self.assertEqual(out[1].sum().item(),0)
        self.assertIn('attn_sink',calls[0]);self.assertNotIn('kv_format',calls[0])

    def test_refresh_keeps_hcu_source_object_and_cuda_tensor_addresses(self):
        dest=F.DSAFlashMLAMetadata(torch.zeros(2),torch.zeros(2,dtype=torch.int32))
        src=F.DSAFlashMLAMetadata(torch.ones(1),torch.ones(1,dtype=torch.int32))
        ptr=dest.flashmla_metadata.data_ptr()
        actual=F.refresh_flashmla_metadata(dest,src,slice(0,1),is_hcu=False)
        self.assertEqual(actual.flashmla_metadata.data_ptr(),ptr)
        self.assertIs(F.refresh_flashmla_metadata(dest,src,slice(0,1),is_hcu=True),src)
        self.assertTrue(F.can_fuse_flashmla_metadata(dest,src))
        self.assertFalse(F.can_fuse_flashmla_metadata(F.DSAFlashMLAMetadata(object(),None)))


class DsaTopkCurrentTests(unittest.TestCase):
    def test_hcu_paged_v2_opt_in_reaches_graph_metadata_producer(self):
        from hcu_sglang.environ import register_envs
        register_envs()
        from hcu_sglang.layers.attention.current_dsa import dsa_topk_backend as T
        backend=SimpleNamespace(dsa_topk_backend=T.DSATopKBackend.SGL_KERNEL)
        with patch.object(T,'_is_hcu',True),patch.dict(os.environ,{'SGLANG_OPT_USE_TOPK_V2':'1','SGLANG_HCU_TOPK_V2_PAGED':'0'}):
            self.assertTrue(backend.dsa_topk_backend.should_use_topk_v2())
            self.assertFalse(backend.dsa_topk_backend.should_use_topk_v2_paged())
            self.assertIsNone(B.DeepseekSparseAttnBackend._build_topk_v2_plan(backend,torch.tensor([1],dtype=torch.int32)))
        with patch.object(T,'_is_hcu',True),patch.dict(os.environ,{'SGLANG_OPT_USE_TOPK_V2':'1','SGLANG_HCU_TOPK_V2_PAGED':'1'}):
            self.assertTrue(backend.dsa_topk_backend.should_use_topk_v2_paged())

    def test_ragged_v2_layout_preserves_vector_load_constraints(self):
        from hcu_sglang.layers.attention.current_dsa import dsa_topk_backend as T
        self.assertTrue(T._supports_topk_v2_ragged_layout(torch.zeros(2,8)))
        self.assertFalse(T._supports_topk_v2_ragged_layout(torch.zeros(2,7)))
        self.assertFalse(T._supports_topk_v2_ragged_layout(torch.zeros(2,8).half()))
        self.assertFalse(T._supports_topk_v2_ragged_layout(torch.zeros(8,2).T))

if __name__=='__main__':unittest.main()
