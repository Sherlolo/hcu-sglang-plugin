from __future__ import annotations
import unittest
import os
from types import SimpleNamespace
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import fp8_moe as C, fp8_quant as Q
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod,Fp8Config
from sglang.srt.plugins.hook_registry import HookRegistry


class Fp8MoeLayoutGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hcu_sglang import register
        HookRegistry.reset();register();HookRegistry.apply_hooks()

    def make(self,fp4=False):
        cfg=Fp8Config(is_checkpoint_fp8_serialized=True,activation_scheme='dynamic',weight_block_size=[128,128],is_fp4_experts=fp4)
        m=Fp8MoEMethod(cfg);l=torch.nn.Module();l.moe_tp_size=1;l.num_local_experts=2;l.num_experts=2;l.moe_runner_config=SimpleNamespace(is_gated=True)
        with torch.device('cuda'):m.create_weights(l,2,256,256,torch.bfloat16)
        return m,l

    def test_real_asm_provider_preserves_all_expert_bytes_and_parameter_storage(self):
        m,l=self.make();torch.manual_seed(4901)
        for w in [l.w13_weight,l.w2_weight]:w.data.copy_((torch.randn(w.shape,device='cuda')*.25).to(w.dtype))
        originals=[l.w13_weight.view(torch.uint8).clone(),l.w2_weight.view(torch.uint8).clone()]
        params=[l.w13_weight,l.w2_weight];pointers=[w.data_ptr() for w in params]
        with patch.object(C.P,'_use_aiter',True):m.process_weights_after_loading(l)
        torch.cuda.synchronize()
        for stage,(w,old) in enumerate(zip(params,originals),1):
            self.assertEqual(w.data_ptr(),pointers[stage-1])
            e,n,k=w.shape;bn,bk=128,(256 if stage==1 else 128)
            packed=w.view(torch.uint8).reshape(e,n//bn,k//bk,bn//64,bk//64,4,4,16,16)
            inverse=packed.permute(0,1,3,5,7,2,4,6,8).contiguous().reshape(e,n,k)
            torch.testing.assert_close(inverse,old,rtol=0,atol=0)
        self.assertIs(l.w13_weight,params[0]);self.assertIs(l.w2_weight,params[1])

    def fp4_reference(self,packed,scale):
        raw=packed.view(torch.uint8).cpu().long()
        table=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
        values=torch.stack([table[raw%16],table[raw//16]],dim=-1).flatten(-2)
        return values*scale.cpu().repeat_interleave(32,-1)

    def test_native_nonserialized_fp8_quantization_uses_owned_producer(self):
        cfg=Fp8Config(is_checkpoint_fp8_serialized=False,activation_scheme='dynamic')
        m=Fp8MoEMethod(cfg);l=torch.nn.Module();l.moe_tp_size=1;l.num_local_experts=2;l.num_experts=2;l.moe_runner_config=SimpleNamespace(is_gated=True)
        with torch.device('cuda'):m.create_weights(l,2,256,256,torch.bfloat16)
        torch.manual_seed(4902)
        originals=[]
        for w in [l.w13_weight,l.w2_weight]:
            w.data.copy_(torch.randn(w.shape,device='cuda',dtype=torch.bfloat16)*.25);originals.append(w.data.clone())
        with patch.object(Q,'scaled_fp8_quant',wraps=Q.scaled_fp8_quant) as producer:
            m.process_weights_after_loading(l)
        torch.cuda.synchronize();self.assertEqual(producer.call_count,4)
        for w,s,original in zip([l.w13_weight,l.w2_weight],[l.w13_weight_scale,l.w2_weight_scale],originals):
            self.assertEqual(w.dtype,torch.float8_e4m3fn);self.assertEqual(s.dtype,torch.float32)
            decoded=w.float()*s.reshape(-1,1,1)
            torch.testing.assert_close(decoded,original.float(),rtol=.075,atol=.005)

    def test_native_routed_fp8_forward_after_public_weight_loading(self):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.runtime_context import publish, get_parallel
        args=ServerArgs(model_path=os.environ['HCU_TEST_MODEL_PATH'],trust_remote_code=True,moe_runner_backend='triton')
        publish(args,role='scheduler')
        m,l=self.make();l.moe_tp_rank=0;l.moe_ep_size=1;l.moe_ep_rank=0
        torch.manual_seed(4903)
        for w in [l.w13_weight,l.w2_weight]:w.data.copy_((torch.randn(w.shape,device='cuda')*.25).to(w.dtype))
        m.process_weights_after_loading(l)
        config=MoeRunnerConfig(num_experts=2,num_local_experts=2,hidden_size=256,intermediate_size_per_partition=256,layer_id=0,top_k=1,num_fused_shared_experts=0,params_dtype=torch.bfloat16,inplace=False,gate_up_interleaved=False,layer=l)
        m.create_moe_runner(l,config)
        self.assertTrue(m.runner.runner_backend.is_triton())
        x=torch.randn(16,256,device='cuda',dtype=torch.bfloat16)*.1
        ids=(torch.arange(16,device='cuda')%2).to(torch.int32).reshape(-1,1)
        weights=torch.ones(16,1,device='cuda')
        dispatch=StandardDispatchOutput(x,None,StandardTopKOutput(weights,ids,None))
        from hcu_sglang.layers.moe import alignment as A
        from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe as F
        stages=[];invoke=F.invoke_fused_moe_kernel
        def capture(*args,**kwargs):
            result=invoke(*args,**kwargs)
            stages.append((args[0].clone(),args[3].clone(),None if args[4] is None else args[4].clone()))
            return result
        with get_parallel().override(tp_group=None),patch.object(A,'moe_align_block_size',wraps=A.moe_align_block_size) as align,patch.object(F,'invoke_fused_moe_kernel',side_effect=capture):
            out=m.apply(l,dispatch).hidden_states
        self.assertEqual(align.call_count,1);self.assertEqual(len(stages),2)
        torch.cuda.synchronize()
        def quantized_pair(value):
            shape=value.shape
            groups=value.float().reshape(shape[0],-1,128)
            scale=groups.abs().amax(-1).clamp_min(1e-10)/448
            q=(groups/scale.unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn)
            return q.reshape(shape),scale
        def block_gemm(q,scale,w,weight_scale):
            result=torch.zeros(q.shape[0],w.shape[0],device=q.device)
            for group in range(q.shape[1]//128):
                block=q[:,group*128:(group+1)*128].float()@w[:,group*128:(group+1)*128].float().T
                result+=block*scale[:,group:group+1]*weight_scale[:,group].repeat_interleave(128).reshape(1,-1)
            return result
        quantized_x,x_scale=quantized_pair(x)
        expected=torch.empty_like(x)
        for e in range(2):
            mask=ids[:,0]==e
            gu=block_gemm(quantized_x[mask],x_scale[mask],l.w13_weight[e],l.w13_weight_scale_inv[e]).to(x.dtype)
            actual_gu=stages[0][1].reshape(16,512)[mask]
            torch.testing.assert_close(actual_gu,gu,rtol=.025,atol=.025)
            hidden=(torch.nn.functional.silu(actual_gu.float()[:,:256])*actual_gu.float()[:,256:]).to(x.dtype)
            actual_hidden=stages[1][0].reshape(16,256)[mask]
            torch.testing.assert_close(actual_hidden,hidden,rtol=.025,atol=.025)
            qhidden,hidden_scale=quantized_pair(actual_hidden)
            expected[mask]=block_gemm(qhidden,hidden_scale,l.w2_weight[e],l.w2_weight_scale_inv[e]).to(x.dtype)
        torch.testing.assert_close(out,expected,rtol=.08,atol=.03)

    def test_actual_native_fp4_to_fp8_conversion_preserves_expert_values_and_aliases(self):
        m,l=self.make(fp4=True);m.dequant_fp4_to_fp8=True
        self.assertFalse(m.is_deepgemm_moe_runner_backend_enabled())
        before=[];weights=[l.w13_weight,l.w2_weight];scales=[l.w13_weight_scale_inv,l.w2_weight_scale_inv]
        for w,s in zip(weights,scales):
            w.data.copy_(torch.arange(w.numel(),device='cuda').remainder(256).reshape(w.shape).to(torch.uint8).view(torch.int8))
            pattern=torch.tensor([1.,2.,4.,.5],device='cuda').repeat(s.shape[-1]//4)
            s.data.copy_(pattern.expand_as(s));before.append(self.fp4_reference(w,s))
        m.process_weights_after_loading(l);torch.cuda.synchronize()
        self.assertFalse(m.is_fp4_expert)
        for w,s,expected in zip(weights,scales,before):
            self.assertEqual(w.dtype,torch.float8_e4m3fn);self.assertEqual(s.dtype,torch.float32);self.assertFalse(s.format_ue8m0)
            decoded=w.float().cpu()*s.cpu().repeat_interleave(128,-2).repeat_interleave(128,-1)
            torch.testing.assert_close(decoded,expected,rtol=0,atol=0)
        self.assertIs(l.w13_weight,weights[0]);self.assertIs(l.w2_weight,weights[1])


if __name__=='__main__':unittest.main()
