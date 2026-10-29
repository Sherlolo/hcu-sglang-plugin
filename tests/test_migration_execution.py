import unittest
from tools.build_migration_execution import build_execution


class MigrationExecutionTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'baselines': {}, 'files': [
            {'path': 'same.py', 'relation': 'same', 'group': 'runtime'},
            {'path': 'new.py', 'relation': 'fork_only', 'group': 'runtime', 'has_mapping_clue': True},
            {'path': 'old.py', 'relation': 'official_only', 'group': 'runtime'}]}

    def test_all_differences_default_pending_even_with_mapping_clues(self):
        report = build_execution(self.plan, {})
        self.assertEqual(report['summary'], {'pending': 2})
        self.assertEqual({row['path'] for row in report['files']}, {'new.py', 'old.py'})

    def test_decisions_require_known_path_implementation_and_validation_evidence(self):
        for decisions in ({'missing.py': {}}, {'new.py': {'status': 'validated'}},
                          {'new.py': {'status': 'validated', 'rationale': 'reviewed', 'implementation': ['plugin.py']}}):
            with self.assertRaises(ValueError):
                build_execution(self.plan, decisions)
        report = build_execution(self.plan, {'new.py': {'status': 'implemented', 'rationale': 'reviewed', 'implementation': ['plugin.py']}})
        self.assertEqual(report['summary'], {'implemented': 1, 'pending': 1})
