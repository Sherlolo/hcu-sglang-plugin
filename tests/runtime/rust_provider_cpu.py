from __future__ import annotations

from pathlib import Path
import shlex
import sys
import tempfile
from types import ModuleType, SimpleNamespace as S
import unittest
from unittest.mock import patch

from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.mem_cache.rust_tree_core import provider as P


class RustProviderCpuTests(unittest.TestCase):
    def test_native_options_preserved_and_build_inputs_owned(self):
        env = {'CXXFLAGS': '-O3 -DNAME="two words" -include "/old path/torch_compat.h"',
               'RUSTFLAGS': '-C link-arg=-Wl,-rpath,$ORIGIN/../../../../torch/lib', 'OTHER': 'kept'}
        fp = {'caller': 'kept'}
        with patch.object(P, 'is_hcu', return_value=True):
            module, kw = P.around_load(lambda module, **kw:(module, kw), P._NATIVE_MODULE,
                mode='never', cache_dir=Path('/cache'), additional_features=('inspection',),
                build_environment=env, build_fingerprint=fp)
        self.assertEqual(module, P._NATIVE_MODULE)
        self.assertEqual(kw['mode'], 'never')
        self.assertEqual(kw['additional_features'], ('inspection',))
        self.assertEqual(kw['workspace'], P._CRATE)
        self.assertEqual(kw['extension_module'], P._OWNED_MODULE)
        tokens = shlex.split(kw['build_environment']['CXXFLAGS'])
        self.assertIn('-DNAME=two words', tokens)
        self.assertEqual(tokens[-1], str(P._CRATE / 'torch_compat.h'))
        self.assertIn('$ORIGIN/../../../torch/lib', kw['build_environment']['RUSTFLAGS'])
        self.assertEqual(kw['build_environment']['OTHER'], 'kept')
        self.assertEqual(kw['build_fingerprint']['caller'], 'kept')
        self.assertEqual(fp, {'caller': 'kept'})
        self.assertIn('/old path', env['CXXFLAGS'])

    def test_inspection_name_maps_to_separate_owned_module(self):
        with patch.object(P, 'is_hcu', return_value=True):
            _, kw = P.around_load(lambda m, **kw:(m, kw), P._NATIVE_MODULE,
                extension_module='sglang.srt.mem_cache.rust_tree_core.mem_cache_inspection',
                build_environment={'CXXFLAGS':'-O2'}, mode='force')
        self.assertEqual(kw['extension_module'], P._OWNED_INSPECTION)
        self.assertEqual(kw['mode'], 'force')
        self.assertEqual(shlex.split(kw['build_environment']['CXXFLAGS'])[-2:], ['-include', str(P._CRATE / 'torch_compat.h')])

    def test_custom_workspace_extension_other_modules_and_non_hcu_delegate(self):
        for enabled, module, kwargs in [(False, P._NATIVE_MODULE, {}),
                (True, 'another.extension', {}), (True, P._NATIVE_MODULE, {'workspace':Path('/custom')}),
                (True, P._NATIVE_MODULE, {'extension_module':'custom.extension'})]:
            calls = []
            with patch.object(P, 'is_hcu', return_value=enabled):
                result = P.around_load(lambda m, **kw:calls.append((m, kw)) or 17, module, **kwargs)
            self.assertEqual(result, 17)
            self.assertEqual(calls, [(module, kwargs)])

    def test_fresh_environment_uses_real_torch_build_configuration(self):
        with patch.object(P, 'is_hcu', return_value=True):
            _, kw = P.around_load(lambda m, **kw:(m, kw), P._NATIVE_MODULE)
        self.assertEqual(shlex.split(kw['build_environment']['CXXFLAGS'])[-1], str(P._CRATE / 'torch_compat.h'))
        self.assertIn('torch_version', kw['build_fingerprint'])
        self.assertIsNotNone(kw['build_fingerprint']['compat_header_sha256'])

    def test_spaced_owned_header_survives_shell_flag_parsing(self):
        with tempfile.TemporaryDirectory(prefix='hcu rust space ') as root:
            crate = Path(root)
            (crate/'torch_compat.h').write_text('')
            with patch.object(P, '_CRATE', crate), patch.object(P, 'is_hcu', return_value=True):
                _, kw = P.around_load(lambda m, **kw:(m, kw), P._NATIVE_MODULE,
                    build_environment={'CXXFLAGS':'-O2'})
            self.assertEqual(shlex.split(kw['build_environment']['CXXFLAGS'])[-1], str(crate/'torch_compat.h'))

    def test_attach_complete_source_recorder_to_owned_binding(self):
        from hcu_sglang.mem_cache.kv_events import KvEventLoraNames
        from hcu_sglang.mem_cache.rust_tree_core.events import _RustKVCacheEventRecorder
        module = ModuleType(P._OWNED_MODULE)
        names = KvEventLoraNames()
        names.register(S(lora_id='id', extra_key='ns'), 'adapter')
        raw = [('block_stored', [1], None, [7], 1, 'GPU', None, None, 'ns'),
               ('block_removed', [1], 'GPU', 'ns')]
        class Queue:
            def take_events(self):
                result = raw[:]
                raw.clear()
                return result
        core = S(_bindings=module, _binding=Queue(), kv_events=S(lora_names=names))
        with patch.dict(sys.modules, {P._OWNED_MODULE:module}):
            P.attach_event_recorder(core, S(enable_kv_cache_events=True, dynamo_kv_event_format=False))
        self.assertIs(type(core.kv_events), _RustKVCacheEventRecorder)
        events = core.kv_events.take()
        self.assertEqual(events[0].lora_name, 'adapter')
        self.assertEqual(events[0].block_hashes, events[1].block_hashes)
        self.assertNotEqual(events[0].block_hashes, [1])

    def test_preloaded_foreign_binding_fails_before_mixing_event_abi(self):
        core = S(_bindings=ModuleType('foreign'))
        with self.assertRaisesRegex(RuntimeError, 'loaded before'):
            P.attach_event_recorder(core, S())

    def test_preflight_preserves_capability_and_custom_reasons(self):
        for reason in [None, 'session-aware caching requires the Python TreeCore', 'custom policy']:
            with patch.object(P, 'is_hcu', return_value=True):
                self.assertEqual(P.around_fallback_reason(lambda params:reason, S()), reason)

    def test_preflight_uses_owned_workspace_when_native_sources_missing(self):
        reason = 'this installation contains neither the Rust TreeCore extension nor its sources'
        with patch.object(P, 'is_hcu', return_value=True), patch.object(P.subprocess, 'run') as run:
            self.assertIsNone(P.around_fallback_reason(lambda params:reason, S()))
            self.assertEqual(run.call_count, 2)
            self.assertTrue(all(call.kwargs['cwd'] == P._CRATE for call in run.call_args_list))
        with patch.object(P, 'is_hcu', return_value=True), patch.object(P.subprocess, 'run', side_effect=OSError('toolchain')):
            self.assertEqual(P.around_fallback_reason(lambda params:reason, S()), reason)

    def test_optional_native_rust_module_not_eagerly_imported(self):
        self.assertNotIn('sglang.srt.mem_cache.rust_tree_core.extension', sys.modules)
        self.assertNotIn(P._OWNED_MODULE, sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
