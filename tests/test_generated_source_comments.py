import ast
import io
import tempfile
import tokenize
import unittest
from pathlib import Path

from tools import reconstruct_sglang_das as generator
from tools import sync_deepseek_methods as sync


def comments(source):
    return [token.string for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT]


class GeneratedSourceCommentsTests(unittest.TestCase):
    def test_method_rendering_preserves_source_without_adding_explanations(self):
        with tempfile.TemporaryDirectory() as task_dir:
            source = Path(task_dir) / "model.py"
            source.write_text('class Owner:\n    def forward(self, value):\n'
                              '        """Original method documentation."""\n'
                              '        # Original source comment.\n'
                              '        return value + 1  # Original inline comment.\n')
            for header in (sync.V2_HEADER, sync.V4_HEADER, sync.COMMON_HEADER, sync.NEXTN_HEADER):
                with self.subTest(header=header):
                    result = sync.render_methods(header, [(source, "Owner", "forward")], {})
                    self.assertEqual(comments(result), comments(source.read_text()))
                    tree = ast.parse(result)
                    self.assertIsNone(ast.get_docstring(tree))
                    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
                    self.assertEqual(ast.get_docstring(method), "Original method documentation.")
                    namespace = {}
                    exec(compile(ast.Module(body=[method], type_ignores=[]), "method", "exec"), namespace)
                    self.assertEqual(namespace["forward"](None, 41), 42)

    def test_injection_preserves_comments_and_refuses_duplicate_without_markers(self):
        cases = [
            (generator.LOADER_ANCHOR, generator.LOADER_BLOCK, generator.LOADER_BEGIN,
             generator.LOADER_END, "register_embedded_dcu", False,
             'def load_plugins():\n    plugins = {}\n    # Original loader comment.\n'
             '    for name, (func, dist_name) in plugins.items():\n        func()\n'),
            (generator.BOOTSTRAP_ANCHOR, generator.BOOTSTRAP_BLOCK, generator.BOOTSTRAP_BEGIN,
             generator.BOOTSTRAP_END, "_bootstrap_dcu_aiter_compat", True,
             '"""Original API documentation."""\n# Original API comment.\n'
             'def _redirect_third_party_caches():\n    pass\n_redirect_third_party_caches()\n'),
        ]
        with tempfile.TemporaryDirectory() as task_dir:
            path = Path(task_dir) / "module.py"
            for anchor, block, begin, end, symbol, after, source in cases:
                with self.subTest(symbol=symbol):
                    path.write_text(source)
                    generator._inject_once(path, anchor, block, begin, end, after=after, symbol=symbol)
                    output = path.read_text()
                    self.assertEqual(comments(output), comments(source))
                    self.assertEqual(ast.get_docstring(ast.parse(output)), ast.get_docstring(ast.parse(source)))
                    with self.assertRaisesRegex(generator.ReconstructionError, "already exists"):
                        generator._inject_once(path, anchor, block, begin, end, after=after, symbol=symbol)
                    self.assertEqual(path.read_text(), output)
                    for legacy in (begin, end):
                        path.write_text(source + "\n" + legacy + "\n")
                        with self.assertRaisesRegex(generator.ReconstructionError, "already exists"):
                            generator._inject_once(path, anchor, block, begin, end, after=after, symbol=symbol)


if __name__ == "__main__":
    unittest.main()
