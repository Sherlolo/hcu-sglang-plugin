from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.layers.attention.current_dsa import indexer as I, gate_scale as G, hcu_sparse_mqa as H
from hcu_sglang.mem_cache.dsa_index_cache_mode import IndexKCacheMode

class IndexerCurrentTests(unittest.TestCase):
    def test_cache_read_uses_broadcast_storage_and_hcu_page64_abi(self):
        owner=torch.ones(3,64*132,dtype=torch.uint8);read=torch.full_like(owner,2)
        pool=SimpleNamespace(page_size=64,index_k_cache_mode=IndexKCacheMode.FP8_SCALED,get_index_k_with_scale_buffer=lambda **kw:owner,get_broadcastable_index_k_with_scale_buffer=lambda layer:read)
        indexer=object.__new__(I.Indexer);torch.nn.Module.__init__(indexer);indexer.head_dim=128
        with patch.object(I,'_is_hcu',True):
            result,bf16=indexer._get_hcu_paged_index_k_cache(pool,2,torch.zeros(1,1,dtype=torch.int32),torch.ones(1,dtype=torch.int32))
        self.assertEqual(result.shape,(3,64,1,132));self.assertEqual(result.data_ptr(),read.data_ptr());self.assertFalse(bf16)
        self.assertEqual(owner.sum().item(),owner.numel())

    def test_dense_logits_producer_keeps_clean_output_and_argument_identity(self):
        q=torch.zeros(2,32,128);cache=torch.zeros(3,64,1,132,dtype=torch.uint8);weights=torch.ones(2,32);lens=torch.tensor([1,1],dtype=torch.int32);blocks=torch.zeros(2,1,dtype=torch.int32);schedule=object();marker=object()
        with patch.object(I.lightop_attention,'paged_mqa_logits',return_value=marker) as kernel:
            result=I._hcu_paged_mqa_logits(q,cache,weights,lens,blocks,schedule,64)
        self.assertIs(result,marker);self.assertEqual(kernel.call_args.args[0].shape,(2,1,32,128))
        self.assertIs(kernel.call_args.args[1],cache);self.assertIs(kernel.call_args.args[5],schedule);self.assertTrue(kernel.call_args.kwargs['clean_logits'])

    def test_sparse_producer_requires_paired_consumer_and_valid_layout(self):
        args=dict(enabled=True,api_available=True,is_hcu=True,arch_name='gfx938',num_cus=64,page_size=64,topk=2048,fuse_topk=True,force_unfused_topk=False,topk_transform_method_name='PAGED',q=torch.empty(2,1,32,128,dtype=torch.float8_e4m3fn),fused_kv_cache=torch.empty(3,64,1,132,dtype=torch.uint8),weights=torch.ones(2,32),context_lens=torch.ones(2,dtype=torch.int32),block_table=torch.zeros(2,2,dtype=torch.int32),max_context_len=128,batch_size=2,is_target_verify=False,is_draft_extend_v2=False,mtp_group_size=1,grouping_lens_cpu=[])
        self.assertEqual(H.select_lightop_sparse_mqa_route(**args).group_size,1)
        for change in [dict(api_available=False),dict(force_unfused_topk=True),dict(page_size=1),dict(q=args['q'].float())]:
            self.assertIsNone(H.select_lightop_sparse_mqa_route(**dict(args,**change)))

    @unittest.skipUnless(torch.cuda.is_available(),'GPU unavailable')
    def test_real_hcu_head_gate_scale_kernel(self):
        weights=torch.arange(64,dtype=torch.float32,device='cuda').reshape(2,32)/32
        scales=torch.linspace(.1,1,64,device='cuda').reshape(2,32,1)
        out=G.fused_get_logits_head_gate_triton(weights,scales,32,.25)
        expected=weights.unsqueeze(-1)*scales*(32**-.5)*.25
        torch.testing.assert_close(out,expected,rtol=1e-5,atol=1e-6)

class PagedMqaDispatchTests(unittest.TestCase):
    def test_hcu_auto_and_explicit_backend_use_lightop(self):
        from hcu_sglang.layers.attention.current_dsa import paged_mqa_logits_backend as P
        self.assertIs(I.DSAPagedMQALogitsBackend,P.DSAPagedMQALogitsBackend)
        with patch.object(P,'is_hcu',return_value=True),patch.object(P,'is_hip',return_value=True):
            self.assertTrue(P.DSAPagedMQALogitsBackend.resolve('auto').is_lightop())
            self.assertTrue(P.DSAPagedMQALogitsBackend.resolve('lightop').is_lightop())
            with self.assertRaises(ValueError):P.DSAPagedMQALogitsBackend.resolve('aiter')
        with patch.object(P,'is_hcu',return_value=False),patch.object(P,'is_hip',return_value=True):
            self.assertTrue(P.DSAPagedMQALogitsBackend.resolve('auto').is_aiter())

if __name__=='__main__':unittest.main()
