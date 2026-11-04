from __future__ import annotations
import unittest
import torch
from hcu_sglang.layers.moe.ep_balance import make_ep_balanced_expert_ids as balanced


class EpBalanceTests(unittest.TestCase):
    def oracle(self,m,k,e,size,rank,layer):
        local=e//size;rows=[]
        for row in range(m):
            ids=[]
            for slot in range(k):
                assignment=row*k+slot;dest=(assignment+rank)%size
                if k<=size:position=assignment
                else:
                    d=slot%size
                    position=row*k+d*(k//size)+min(d,k%size)+slot//size
                ids.append(dest*local+(position+(0 if layer is None else layer%local))%local)
            rows.append(ids)
        return rows

    def test_exact_ids_across_ep_rank_topk_and_layer_rotation(self):
        for size in [1,2,4,8]:
            for k in [1,3,8,17,31,128]:
                for rank in range(size):
                    for layer in [None,-1,0,7]:
                        actual=balanced(5,k,128,ep_size=size,ep_rank=rank,device='cpu',dtype=torch.int32,layer_id=layer)
                        self.assertEqual(actual.tolist(),self.oracle(5,k,128,size,rank,layer))
                        for row in actual.tolist():self.assertEqual(len(set(row)),k)

    def test_equal_source_rows_have_exact_global_destination_balance(self):
        for k in [1,3,7,8,17,31]:
            counts=torch.zeros(8,dtype=torch.int64)
            for rank in range(8):
                ids=balanced(13,k,256,ep_size=8,ep_rank=rank,device='cpu',dtype=torch.int64)
                counts+=torch.bincount((ids//32).flatten(),minlength=8)
            torch.testing.assert_close(counts,torch.full_like(counts,13*k),rtol=0,atol=0)

    def test_layer_rotation_preserves_rank_fanout_and_destination_pattern(self):
        for k in [3,8,17]:
            a=balanced(13,k,256,ep_size=8,ep_rank=2,device='cpu',dtype=torch.int64,layer_id=0)
            b=balanced(13,k,256,ep_size=8,ep_rank=2,device='cpu',dtype=torch.int64,layer_id=9)
            torch.testing.assert_close(a//32,b//32,rtol=0,atol=0)
            for row in (a//32).tolist():self.assertEqual(len(set(row)),min(k,8))

    def test_empty_and_zero_topk_preserve_shapes_and_integer_dtype(self):
        for dtype in [torch.int32,torch.int64]:
            for m,k in [(0,8),(3,0),(0,0)]:
                ids=balanced(m,k,256,ep_size=8,ep_rank=0,device='cpu',dtype=dtype)
                self.assertEqual(tuple(ids.shape),(m,k));self.assertEqual(ids.dtype,dtype)

    def test_invalid_counts_ranks_and_expert_placement_are_rejected(self):
        base={'num_tokens':3,'topk':8,'num_experts':256,'ep_size':8,'ep_rank':0,'device':'cpu','dtype':torch.int32}
        for field,value in [('num_tokens',-1),('topk',-1),('topk',257),('ep_size',0),('ep_rank',-1),('ep_rank',8),('num_experts',255),('num_experts',0)]:
            kwargs=dict(base);kwargs[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):balanced(**kwargs)


if __name__=='__main__':unittest.main()
