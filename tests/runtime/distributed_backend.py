from __future__ import annotations

import argparse
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch
import os
import subprocess
import sys
import unittest

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang import config, distributed as H, distributed_backend as C, server_args_hooks as A
from sglang.srt import runtime_context as R
from sglang.srt import server_args as S
from sglang.srt.arg_groups.arg_utils import record_fields
from sglang.srt.distributed import bootstrap as B
from sglang.srt.distributed.device_communicators import quick_all_reduce as Q, pcie_ipc_ar as I
from sglang.srt.plugins.hook_registry import HookRegistry


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset(); A.register(); H.register_hooks(); HookRegistry.apply_hooks()

    def setUp(self):
        self.scope = ExitStack(); self.addCleanup(self.scope.close)
        self.scope.enter_context(patch.dict(os.environ, {config.CUSTOM_ALL_REDUCE_BACKEND_ENV:'auto', 'SGLANG_USE_AITER_AR':'0', 'AITER_AR_TRANSPORT':'ipc'}))
        C.set_custom_all_reduce_backend('auto')

    def fixture(self, *, construction=None, enabled=True, transport='ipc'):
        self.scope.enter_context(patch.object(C.P, 'is_cuda_alike', return_value=False))
        self.scope.enter_context(patch.object(C.P, 'is_hip', return_value=True))
        self.scope.enter_context(patch.object(C.P, 'get_torch_distributed_pg_options', return_value=None))
        self.scope.enter_context(patch.object(C.P.torch.distributed, 'get_rank', return_value=0))
        self.scope.enter_context(patch.object(C.P.torch.distributed, 'new_group', side_effect=lambda *a,**k:object()))
        self.scope.enter_context(patch.object(C.P.GroupCoordinator, '_deterministic_collectives_enabled', return_value=False))
        self.scope.enter_context(patch.object(I, 'eligible_group', return_value=False))
        self.scope.enter_context(patch.object(C, 'use_quick_custom_allreduce', True))
        self.scope.enter_context(patch.object(Q, 'qr_rocm_arch_available', return_value=True))
        qr = self.scope.enter_context(patch.object(Q, 'QuickAllReduce'))
        instance = SimpleNamespace(disabled=not enabled, transport=transport)
        ctor = Mock(return_value=instance, side_effect=construction)
        dispatch = self.scope.enter_context(patch('hcu_sglang.distributed_aiter.dispatch_custom_allreduce', return_value=ctor))
        return instance, ctor, dispatch, qr

    def create_group(self, backend=None, use_custom=True):
        kwargs = {} if backend is None else {'custom_all_reduce_backend':backend}
        return C.P.init_model_parallel_group([[0,1]], 0, 'nccl', use_pynccl=False, use_mscclpp=False, use_custom_allreduce=use_custom, use_torch_symm_mem_allreduce=False, group_name='backend-test', **kwargs)

    def test_actual_cli_choices_do_not_mutate_raw_record_schema(self):
        field_names = {f.name for f in record_fields(S.ServerArgs)}
        self.assertNotIn('custom_all_reduce_backend', field_names)
        for choice in config.CUSTOM_ALL_REDUCE_BACKENDS:
            parser = argparse.ArgumentParser(); S.ServerArgs.add_cli_args(parser)
            args = parser.parse_args(['--model-path','dummy','--custom-all-reduce-backend',choice])
            raw = S.ServerArgs.from_cli_args(args)
            self.assertEqual(config.custom_all_reduce_backend(), choice)
            self.assertFalse(hasattr(raw, 'custom_all_reduce_backend'))
        parser = argparse.ArgumentParser(); S.ServerArgs.add_cli_args(parser)
        with self.assertRaises(SystemExit):parser.parse_args(['--model-path','dummy','--custom-all-reduce-backend','bogus'])
        args = parser.parse_args(['--model-path','dummy']); S.ServerArgs.from_cli_args(args)
        self.assertEqual(config.custom_all_reduce_backend(), 'auto')

    def test_cookbook_nsa_aliases_preserve_canonical_fields_and_defaults(self):
        parser=argparse.ArgumentParser();S.ServerArgs.add_cli_args(parser)
        args=parser.parse_args(['--model-path','dummy','--nsa-prefill-backend','flashmla_auto','--nsa-decode-backend','flashmla_kv','--custom-all-reduce-backend','aiter'])
        raw=S.ServerArgs.from_cli_args(args)
        self.assertEqual(raw.dsa_prefill_backend,'flashmla_auto');self.assertEqual(raw.dsa_decode_backend,'flashmla_kv')
        args=parser.parse_args(['--model-path','dummy','--dsa-prefill-backend','fa3'])
        self.assertEqual(args.dsa_prefill_backend,'fa3');self.assertIsNone(args.dsa_decode_backend)
        A.after_add_cli_args(None,parser)
        self.assertEqual(sum('--nsa-prefill-backend' in action.option_strings for action in parser._actions),1)

    def test_child_process_inherits_backend_without_importing_torch(self):
        config.configure_custom_all_reduce_backend('aiter')
        result = subprocess.check_output([sys.executable,'-c','from hcu_sglang.config import custom_all_reduce_backend;import sys;print(custom_all_reduce_backend(),"torch" in sys.modules)'],text=True)
        self.assertEqual(result.strip(), 'aiter False')

    def test_normalization_disable_wins_and_legacy_auto_promotes(self):
        config.configure_custom_all_reduce_backend('aiter')
        self.assertEqual(config.resolved_custom_all_reduce_backend(True), 'off')
        config.configure_custom_all_reduce_backend('auto')
        with patch.dict(os.environ, {'SGLANG_USE_AITER_AR':'1'}):self.assertEqual(config.resolved_custom_all_reduce_backend(), 'aiter')
        config.configure_custom_all_reduce_backend('native')
        with patch.dict(os.environ, {'SGLANG_USE_AITER_AR':'1'}):self.assertEqual(config.resolved_custom_all_reduce_backend(), 'native')
        with self.assertRaises(ValueError):config.configure_custom_all_reduce_backend('bogus')

    def test_public_bootstrap_publishes_resolved_backend(self):
        comm = SimpleNamespace(disable_custom_all_reduce=False,enable_mscclpp=False,enable_torch_symm_mem=False,flashinfer_allreduce_fusion_backend=None)
        from sglang.srt.layers import flashinfer_comm_fusion as F
        config.configure_custom_all_reduce_backend('aiter')
        with patch.object(B,'get_exec',return_value=SimpleNamespace(comm=comm)),patch.object(R,'get_exec',return_value=SimpleNamespace(comm=comm)),patch.object(F,'uses_cutedsl_ar_fusion',return_value=False):
            B._set_all_reduce_flags();self.assertEqual(C.P.get_custom_all_reduce_backend(),'aiter')
            comm.disable_custom_all_reduce=True
            B._set_all_reduce_flags();self.assertEqual(C.P.get_custom_all_reduce_backend(),'off')

    def test_public_group_constructor_receives_selected_backend(self):
        for backend in ('auto','native','aiter'):
            with self.subTest(backend=backend):
                instance, ctor, dispatch, qr = self.fixture()
                C.set_custom_all_reduce_backend(backend)
                group = self.create_group()
                self.assertIsInstance(group, C.P.GroupCoordinator)
                self.assertIs(group.ca_comm, instance)
                self.assertEqual(group.custom_all_reduce_backend, backend)
                self.assertEqual(dispatch.call_args.kwargs['backend'], backend)
                ctor.assert_called_once_with(group=group.cpu_group,device=group.device)
                self.assertIsNone(group.pcie_ipc_comm)
                self.assertEqual(group.ranks,[0,1])

    def test_explicit_off_never_constructs_custom_or_quick_communicator(self):
        instance, ctor, dispatch, qr = self.fixture()
        group = self.create_group('off')
        self.assertIsNone(group.ca_comm);self.assertIsNone(group.qr_comm)
        dispatch.assert_not_called();ctor.assert_not_called();qr.assert_not_called()

    def test_strict_fabric_rethrows_original_construction_error(self):
        error = RuntimeError('fabric topology admission failure')
        instance, ctor, dispatch, qr = self.fixture(construction=error)
        with patch.dict(os.environ, {'AITER_AR_TRANSPORT':'fabric'}):
            with self.assertRaises(RuntimeError) as caught:self.create_group('aiter')
        self.assertIs(caught.exception,error);qr.assert_not_called()

    def test_non_strict_constructor_error_keeps_source_rccl_fallback(self):
        instance, ctor, dispatch, qr = self.fixture(construction=RuntimeError('optional native init failure'))
        group = self.create_group('native')
        self.assertIsNone(group.ca_comm)

    def test_strict_fabric_disabled_is_rejected_and_qr_is_suppressed(self):
        instance, ctor, dispatch, qr = self.fixture(enabled=False,transport='fabric')
        with patch.dict(os.environ, {'AITER_AR_TRANSPORT':'fabric'}):
            with self.assertRaisesRegex(RuntimeError,'Strict Fabric requested'):self.create_group('aiter')
        qr.assert_not_called()

    def test_fabric_and_auto_transport_do_not_construct_quick_allreduce(self):
        instance, ctor, dispatch, qr = self.fixture(transport='fabric')
        with patch.dict(os.environ, {'AITER_AR_TRANSPORT':'fabric'}):
            group=self.create_group('aiter');self.assertIs(group.ca_comm,instance)
        qr.assert_not_called()
        with patch.dict(os.environ, {'AITER_AR_TRANSPORT':'auto'}):self.create_group('auto')
        qr.assert_not_called()

    def test_world_group_uses_off_and_explicit_group_choice_beats_default(self):
        instance, ctor, dispatch, qr = self.fixture();C.set_custom_all_reduce_backend('aiter')
        world=C.P.init_world_group([0,1],0,'gloo')
        self.assertEqual(world.custom_all_reduce_backend,'off');self.assertIsNone(world.ca_comm)
        self.create_group('native');self.assertEqual(dispatch.call_args.kwargs['backend'],'native')

    def test_non_hcu_preserves_original_group_callable(self):
        marker=object();original=Mock(return_value=marker)
        with patch.object(C,'is_hcu',return_value=False):
            for wrapper in [C.around_init_model_parallel_group,C.around_init_world_group]:
                self.assertIs(wrapper(original,setting='value'),marker)
            self.assertIs(C.around_coordinator_init(original,marker,setting='value'),marker)


if __name__ == '__main__':
    unittest.main()
