"""HCU group all-to-all including the variable-split RLC contract."""
from __future__ import annotations
import sglang.srt.distributed.parallel_state as upstream
globals().update({name:value for name,value in vars(upstream).items() if not name.startswith('__')})

def _all_to_all_single(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    pynccl_comm = self.pynccl_comm
    if pynccl_comm is not None and (not pynccl_comm.disabled):
        with pynccl_comm.change_state(enable=True):
            pynccl_comm.all_to_all_single(output, input)
    else:
        torch.distributed.all_to_all_single(output, input, group=self.device_group)
    return output

def all_to_all_single(self, output: torch.Tensor, input: torch.Tensor, output_split_sizes: Optional[List[int]]=None, input_split_sizes: Optional[List[int]]=None) -> torch.Tensor:
    """All-to-all over dim 0.

        Even splits use the registered op (CUDA-graph friendly). Variable
        per-rank splits (DSV4 compressor RLC repartition) go through
        ``torch.distributed.all_to_all_single`` on the device group.
        """
    if self.world_size == 1:
        output.copy_(input)
        return output
    if output_split_sizes is None and input_split_sizes is None:
        reg_all_to_all_single(output, input, group_name=self.unique_name)
        return output
    torch.distributed.all_to_all_single(output, input, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes, group=self.device_group)
    return output

