import os,json
os.environ['SGLANG_ROCM_USE_AITER_TILELANG_MHC']='1'
import torch
from unittest.mock import patch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.kernels import mhc
from types import SimpleNamespace
x=torch.randn(8,4,7168,device='cuda',dtype=torch.bfloat16)
f=torch.randn(24,28672,device='cuda',dtype=torch.float32)*.01
scale=torch.ones(3,device='cuda');base=torch.zeros(24,device='cuda')
args=(x,f,scale,base,1e-6,1e-6,1e-6,2.,20)
with patch.object(mhc,'get_parallel',return_value=SimpleNamespace(tp_group=None)),patch.object(mhc,'is_allocation_symmetric',return_value=False):
 out=mhc.mhc_pre(*args)
 ref=mhc._mhc_pre_torch(*args)
 torch.cuda.synchronize()
 for a,b in zip(out,ref):
  print(a.shape,'max_abs',(a.float()-b.float()).abs().max().item(),flush=True)
  torch.testing.assert_close(a,b,rtol=.03,atol=.03)
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):out2=mhc.mhc_pre(*args)
 g.replay();torch.cuda.synchronize()
 print('current DAS mHC HCU graph capture/replay passed',flush=True)
