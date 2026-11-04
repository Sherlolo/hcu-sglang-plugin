from __future__ import annotations
from types import SimpleNamespace as S
import os,unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import encoder_swa_replay as O
from sglang.srt.model_executor.forward_batch_info import ForwardMode


@unittest.skipUnless(os.environ.get('HCU_PLUGIN_RUN_GPU_TESTS')=='1','explicit GPU opt-in required')
class EncoderSwaGpuTests(unittest.TestCase):
    def test_fold_metadata_dense_and_tail_drop_across_changed_input_graph_replay(self):
        batch=S(forward_mode=ForwardMode.EXTEND,reqs=[S(full_untruncated_fill_ids=list(range(i*100,i*100+200))) for i in range(3)],encoder_swa_reset=[True,False,True],prefix_lens=[4,0,130],extend_lens=[2,1,2],extend_num_tokens=5,input_ids=torch.tensor([900,901,902,903,904],device='cuda'),prefill_input_ids_cpu=None,out_cache_loc=torch.arange(500,505,device='cuda'),req_pool_indices=torch.arange(3,device='cuda'),extend_logprob_start_lens=[1,0,1],engram_history=None)
        worker=S(model_runner=S(device='cuda',token_to_kv_pool=S(request_window=S(reset=lambda slots:None)),req_to_token_pool=S(req_to_token=torch.arange(600,device='cuda').reshape(3,200))))
        folded=O.fold_encoder_swa_replay(worker,batch)
        out=torch.empty((137,1),dtype=torch.float32,device='cuda')
        def kernel():out[:,0].copy_(folded.batch.input_ids.float()+folded.row_floor.float())
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):kernel()
        torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):kernel()
        for shift in [0,17]:
            if shift:
                batch.input_ids.add_(shift);new=O.fold_encoder_swa_replay(worker,batch);folded.batch.input_ids.copy_(new.batch.input_ids);folded.row_floor.copy_(new.row_floor)
            graph.replay();torch.cuda.synchronize()
            dense=S(hidden_states=out,hidden_states_token_indices=None);O.drop_folded_rows(logits_output=dense,folded=folded)
            expected=batch.input_ids.float()+torch.tensor([0,0,0,2,2],device='cuda')
            torch.testing.assert_close(dense.hidden_states[:,0],expected,rtol=0,atol=0)
            indices=torch.tensor([3,4,5,6,133,134,135,136],device='cuda');tail=S(hidden_states=out[indices],hidden_states_token_indices=indices)
            O.drop_folded_rows(logits_output=tail,folded=folded,tail_lens=[3,1,4])
            torch.testing.assert_close(tail.hidden_states[:,0],expected,rtol=0,atol=0);self.assertEqual(tail.hidden_states_token_indices.cpu().tolist(),[0,1,2,3,4])


if __name__=='__main__':unittest.main()
