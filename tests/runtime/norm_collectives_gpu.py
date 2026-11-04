from __future__ import annotations

from contextlib import ExitStack
from datetime import timedelta
import json
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers import norm_collectives as C
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce


def main():
    rank = int(os.environ['RANK']); local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    assert dist.get_world_size() == 8
    owned = []; coordinators = {}; references = {}; members = {}
    try:
        for kind, partitions in [('tp', [list(range(8))]), ('attn_tp', [list(range(4)), list(range(4, 8))]), ('moe_tp', [list(range(i, i + 2)) for i in range(0, 8, 2)])]:
            for ranks in partitions:
                cpu_group = dist.new_group(ranks=ranks, backend='gloo', timeout=timedelta(seconds=180))
                gpu_group = dist.new_group(ranks=ranks, backend='nccl', timeout=timedelta(seconds=180))
                if rank in ranks:
                    communicator = CustomAllreduce(group=cpu_group, device=torch.device('cuda', local_rank), enable_register_for_capturing=False, max_size=4 * 1024 * 1024)
                    assert not communicator.disabled, (rank, kind, 'AITER communicator disabled')
                    owned.append(communicator)
                    coordinator = object.__new__(C.P.GroupCoordinator)
                    coordinator.ca_comm = communicator; coordinator.world_size = len(ranks)
                    coordinators[kind] = coordinator; references[kind] = gpu_group; members[kind] = ranks
        with ExitStack() as scope:
            for kind, coordinator in coordinators.items():
                scope.enter_context(patch.object(C.P, 'get_' + kind + '_group', return_value=coordinator))
            for kind in ('tp', 'attn_tp', 'moe_tp'):
                axis = torch.arange(7168, device='cuda', dtype=torch.float32)
                x = ((rank + 1) * .1 + axis.reshape(1, -1) * .0001).repeat(8, 1).to(torch.bfloat16)
                residual = torch.full_like(x, .125)
                weight = (1 + axis * .00001).to(torch.bfloat16)
                reduced = x.clone(); dist.all_reduce(reduced, group=references[kind]); torch.cuda.synchronize()
                expected_sum = reduced.float() + residual.float()
                expected_norm = (expected_sum * torch.rsqrt(expected_sum.square().mean(-1, keepdim=True) + 1e-6) * weight.float()).to(x.dtype)
                result = C.tensor_model_parallel_fused_allreduce_rmsnorm(x, residual, weight, 1e-6, group=kind)
                assert result is not None, (rank, kind, 'AITER fusion declined')
                output, updated = result; torch.cuda.synchronize()
                torch.testing.assert_close(updated, expected_sum.to(x.dtype), rtol=.02, atol=.02)
                torch.testing.assert_close(output, expected_norm, rtol=.02, atol=.02)
                result_quant = C.tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group(x, residual, weight, 1e-6, group=kind)
                assert result_quant is None, 'gfx938 must preserve the gfx95-only fused-quant decline'
                if rank == 0:
                    print(json.dumps({'group':kind,'rank0_members':members[kind],'native_GroupCoordinator':True,'real_AITER_IPC_fused_norm_matches_RCCL_reference':True,'gfx95_only_quant_declines_on_gfx938':True}),flush=True)
                dist.barrier()
    finally:
        torch.cuda.synchronize()
        for communicator in reversed(owned):
            if hasattr(communicator, 'close'): communicator.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
