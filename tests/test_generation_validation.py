"""Saved responses must distinguish correct answers from incidental tokens."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.validate_generation_response import validate_response

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools/validate_generation_response.py'


def completion(text='2', finish='stop', **message):
    return {'model': 'model-under-test', 'choices': [{
        'message': {'role': 'assistant', 'content': text, **message},
        'finish_reason': finish,
    }]}


class GenerationValidationTests(unittest.TestCase):
    def test_plain_expected_answer_passes(self):
        self.assertEqual(validate_response(completion(' 2\n'), ['2'])['status'], 'passed')

    def test_incidental_digit_in_corrupted_answer_cannot_pass(self):
        # The old service checker accepted this independent 2 inside 2.0.2.
        result = validate_response(completion('Add firactetw /libs/ 2.0.2 ...'), ['2'])
        self.assertEqual(result['status'], 'failed')

    def test_correct_text_with_truncated_finish_cannot_pass(self):
        self.assertEqual(validate_response(completion('2', 'length'), ['2'])['status'], 'failed')

    def test_control_characters_reject_even_when_final_answer_matches(self):
        for text in ['<think>bad\x00f</think>2', '<think>bad\x7f</think>2']:
            with self.subTest(text=text):
                self.assertEqual(validate_response(completion(text), ['2'],
                    allow_think_prefix=True)['status'], 'failed')

    def test_inline_reasoning_requires_explicit_protocol_choice(self):
        data = completion('<think>1+1=2.</think>\n2')
        self.assertEqual(validate_response(data, ['2'])['status'], 'failed')
        result = validate_response(data, ['2'], allow_think_prefix=True)
        self.assertEqual((result['status'], result['final_text']), ('passed', '2'))

    def test_unclosed_nested_or_empty_final_reasoning_cannot_pass(self):
        for text in ['<think>answer 2', '<think><think>x</think>2', '<think>x</think>']:
            with self.subTest(text=text):
                self.assertEqual(validate_response(completion(text), ['2'],
                    allow_think_prefix=True)['status'], 'failed')

    def test_separate_reasoning_is_preserved_but_must_be_text(self):
        good = validate_response(completion(reasoning_content='1+1=2'), ['2'])
        self.assertEqual((good['status'], good['separate_reasoning']), ('passed', True))
        for reasoning in ['bad\x00f', {'answer': '2'}]:
            with self.subTest(reasoning=reasoning):
                self.assertEqual(validate_response(completion(reasoning_content=reasoning),
                    ['2'])['status'], 'failed')

    def test_wrong_model_cannot_satisfy_expected_model(self):
        self.assertEqual(validate_response(completion(), ['2'], expected_model='another-model')['status'], 'failed')

    def test_empty_multiple_or_malformed_choices_are_not_success(self):
        for choices in [[], [completion()['choices'][0]] * 2, [{}], None]:
            with self.subTest(choices=choices):
                data = {'model': 'model-under-test', 'choices': choices}
                self.assertEqual(validate_response(data, ['2'])['status'], 'failed')

    def test_refusal_tool_calls_and_non_text_are_not_plain_answers(self):
        for data in [completion(refusal='no'), completion(tool_calls=[{'id': 'call'}]),
                     completion(None), completion(['2'])]:
            self.assertEqual(validate_response(data, ['2'])['status'], 'failed')

    def test_only_explicit_answer_alternatives_are_accepted(self):
        self.assertEqual(validate_response(completion('Paris.'), ['Paris'])['status'], 'failed')
        self.assertEqual(validate_response(completion('Paris.'), ['Paris', 'Paris.'])['status'], 'passed')

    def test_invalid_or_empty_oracle_is_configuration_error(self):
        for expected in [[], [''], ['\x00'], '2']:
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError): validate_response(completion(), expected)

    def test_cli_preserves_raw_response_and_reports_its_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / 'response.json'
            out = Path(folder) / 'report.json'
            raw.write_text(json.dumps(completion()), encoding='utf-8')
            original = raw.read_bytes()
            proc = subprocess.run([sys.executable, str(TOOL), '--response', str(raw),
                '--expected-final-text', '2', '--expected-model', 'model-under-test',
                '--output', str(out)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(out.read_text())
            self.assertEqual(report['response_sha256'], hashlib.sha256(original).hexdigest())
            self.assertEqual(raw.read_bytes(), original)

    def test_cli_cannot_overwrite_evidence_even_via_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / 'response.json'
            raw.write_text(json.dumps(completion()), encoding='utf-8')
            original = raw.read_bytes()
            alias = Path(folder) / 'alias.json'
            alias.symlink_to(raw)
            proc = subprocess.run([sys.executable, str(TOOL), '--response', str(raw),
                '--expected-final-text', '2', '--output', str(alias)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(raw.read_bytes(), original)

    def test_cli_cannot_overwrite_evidence_via_hardlink(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / 'response.json'
            raw.write_text(json.dumps(completion()), encoding='utf-8')
            original = raw.read_bytes()
            alias = Path(folder) / 'alias.json'
            os.link(raw, alias)
            proc = subprocess.run([sys.executable, str(TOOL), '--response', str(raw),
                '--expected-final-text', '2', '--output', str(alias)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(raw.read_bytes(), original)

    def test_cli_invalid_json_is_a_failed_result(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / 'response.json'
            out = Path(folder) / 'report.json'
            raw.write_text('{not-json', encoding='utf-8')
            proc = subprocess.run([sys.executable, str(TOOL), '--response', str(raw),
                '--expected-final-text', '2', '--output', str(out)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            self.assertEqual(json.loads(out.read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
