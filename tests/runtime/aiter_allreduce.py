from contextlib import ExitStack, contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import os
import importlib
import sys
import unittest

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat

bootstrap_aiter_compat()

from sglang.srt.arg_groups import model_override_base
from sglang.srt import runtime_context
from sglang.srt.distributed.device_communicators import custom_all_reduce as S
from hcu_sglang import distributed as hooks


class FakeAiter:
    def __init__(self, *args, **kwargs):
        self.options = kwargs
        self._IS_CAPTURING = True
        self.disabled = False
        self.calls = []

    def should_custom_ar(self, value):
        return value.numel() > 0

    def all_reduce(self, value, **kwargs):
        self.calls.append(('all_reduce', kwargs))
        return value * 4

    def custom_all_reduce(self, value, **kwargs):
        self.calls.append(('base_all_reduce', kwargs))
        return torch.zeros_like(value)

    def fused_ar_rms(self, value, residual, **kwargs):
        self.calls.append(('fused_ar_rms', value, residual, kwargs))
        return value + residual

    def custom_fused_ar_rms(self, value, residual, weight, eps, use_1stage=False):
        self.calls.append(('base_fused_ar_rms',))
        return residual


class AiterAlignmentTests(unittest.TestCase):
    @contextmanager
    def environment(self, **overrides):
        fake = ModuleType('aiter.dist.device_communicators.custom_all_reduce')
        fake.CustomAllreduce = FakeAiter
        values = {'AITER_AR_TRANSPORT':'ipc','SGLANG_MEMORY_SAVER_CUDA_GRAPH':'0',
                  'SGLANG_USE_1STAGE_ALLREDUCE':'0','SGLANG_ENABLE_DETERMINISTIC_INFERENCE':'0',
                  'SGLANG_AITER_AR_REAL_GRAPH_WARMUP':'1'}
        values.update(overrides)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, values))
            os.environ.pop('AITER_AR_MAX_SIZE_MB', None)
            stack.enter_context(patch.dict(sys.modules, {fake.__name__:fake}))
            stack.enter_context(patch.object(hooks, 'is_dcu', return_value=True))
            stack.enter_context(patch.object(S, '_is_cuda', False))
            stack.enter_context(patch.object(S, '_is_hip', True))
            stack.enter_context(patch.object(S, '_is_musa', False))
            stack.enter_context(patch.object(S, '_use_amd_deterministic_impl', return_value=False))
            stack.enter_context(patch.object(S.torch.cuda, 'is_current_stream_capturing', return_value=False))
            stack.enter_context(patch.object(S, 'is_in_tc_piecewise_cuda_graph', return_value=False))
            stack.enter_context(patch.object(runtime_context, 'get_server_args', return_value=SimpleNamespace()))
            stack.enter_context(patch.object(model_override_base, 'model_config_of', return_value=SimpleNamespace(hf_config=SimpleNamespace(model_type='qwen3_moe'))))
            try:
                runtime = importlib.import_module('hcu_sglang.distributed_aiter')
            except ModuleNotFoundError as exc:
                if exc.name != 'hcu_sglang.distributed_aiter':
                    raise
            else:
                stack.enter_context(patch.object(runtime, 'is_dcu', return_value=True))
            yield

    def factory(self):
        return hooks.around_dispatch_custom_allreduce(lambda *a: S.CustomAllreduce, None, None)

    def test_auto_selects_aiter_with_default_and_legacy_zero(self):
        with self.environment():
            os.environ.pop('SGLANG_USE_AITER_AR', None)
            factory = self.factory()
            self.assertTrue(hasattr(factory, 'func'), 'HCU auto must select the AITER factory')
            self.assertIsInstance(factory(), FakeAiter)
            os.environ['SGLANG_USE_AITER_AR'] = '0'
            self.assertIsInstance(self.factory()(), FakeAiter)

    def test_graph_warmup_runs_collective_and_capture_delegates(self):
        with self.environment():
            comm = self.factory()()
            value = torch.arange(4, dtype=torch.float32)
            self.assertTrue(torch.equal(comm.custom_all_reduce(value), value * 4))
            self.assertFalse(comm.calls[-1][1]['registered_input'])
            with patch.object(S.torch.cuda, 'is_current_stream_capturing', return_value=True):
                self.assertTrue(torch.equal(comm.custom_all_reduce(value), torch.zeros_like(value)))
            self.assertEqual(comm.calls[-1][0], 'base_all_reduce')
            with patch.object(S, 'is_in_tc_piecewise_cuda_graph', return_value=True):
                comm.custom_all_reduce(value)
            self.assertEqual(comm.calls[-1][0], 'base_all_reduce')
            comm.disabled = True
            self.assertIsNone(comm.custom_all_reduce(value))

    def test_fused_warmup_keeps_live_residual_and_copy_in(self):
        with self.environment():
            comm = self.factory()()
            value, residual, weight = torch.ones(4), torch.arange(4), torch.ones(4)
            self.assertTrue(torch.equal(comm.custom_fused_ar_rms(value, residual, weight, 1e-6), value + residual))
            call = comm.calls[-1]
            self.assertIs(call[1], value)
            self.assertIs(call[2], residual)
            self.assertIs(call[3]['w'], weight)
            self.assertFalse(call[3]['registered'])
            self.assertEqual(call[3]['eps'], 1e-6)

    def test_transport_model_and_workspace_options(self):
        with self.environment(SGLANG_MEMORY_SAVER_CUDA_GRAPH='1'):
            self.assertTrue(self.factory().keywords['enable_register_for_capturing'])
            with patch.object(model_override_base, 'model_config_of', return_value=SimpleNamespace(hf_config=SimpleNamespace(model_type='deepseek_v3'))):
                self.assertFalse(self.factory().keywords['enable_register_for_capturing'])
            for transport in ['fabric','auto','unknown']:
                os.environ['AITER_AR_TRANSPORT'] = transport
                self.assertFalse(self.factory().keywords['enable_register_for_capturing'])
            os.environ['AITER_AR_MAX_SIZE_MB'] = '32'
            self.assertEqual(self.factory().keywords['max_size'], 32 * 1024 * 1024)
            for invalid in ['0','-1','bad']:
                os.environ['AITER_AR_MAX_SIZE_MB'] = invalid
                with self.assertRaises(ValueError):
                    self.factory()

    def test_deterministic_and_explicit_backend_contract(self):
        with self.environment():
            from hcu_sglang import distributed_aiter as P
            with patch.object(S, '_use_amd_deterministic_impl', return_value=True):
                self.assertIs(self.factory(), S.CustomAllreduce)
                with self.assertRaises(RuntimeError):
                    P.dispatch_custom_allreduce(None, None, backend='aiter')
            self.assertIs(P.dispatch_custom_allreduce(None, None, backend='native'), S.CustomAllreduce)
            self.assertIsNone(P.dispatch_custom_allreduce(None, None, backend='off'))

    def test_non_hcu_preserves_effective_original(self):
        calls = []
        group, device, sentinel = object(), object(), object()
        with patch.object(hooks, 'is_dcu', return_value=False):
            result = hooks.around_dispatch_custom_allreduce(lambda *a: calls.append(a) or sentinel, group, device)
        self.assertIs(result, sentinel)
        self.assertEqual(calls, [(group, device)])


if __name__ == '__main__':
    unittest.main()
