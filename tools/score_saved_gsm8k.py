#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from validate_generation_response import validate_response


NUMBER = re.compile(r'-?\d+\.?\d*')


def number(text: str) -> Decimal | None:
    matches = NUMBER.findall(text.replace(',', ''))
    if not matches:
        return None
    try:
        return Decimal(matches[-1])
    except InvalidOperation:
        return None


def selection(manifest: dict, size: int) -> list[int]:
    samples = manifest.get('sample_indices')
    shots = manifest.get('shot_indices', [])
    for label, values, nonempty in [('sample_indices', samples, True), ('shot_indices', shots, False)]:
        if not isinstance(values, list) or (nonempty and not values):
            raise ValueError(f'{label} must be a list' + (' with samples' if nonempty else ''))
        if any(type(i) is not int or i < 0 or i >= size for i in values) or len(set(values)) != len(values):
            raise ValueError(f'{label} has duplicate or out-of-range indices')
    if set(samples) & set(shots):
        raise ValueError('scored examples overlap few-shot prefix examples')
    if not isinstance(manifest.get('expected_model'), str) or not manifest['expected_model'].strip():
        raise ValueError('manifest must pin expected_model')
    return samples


def score_case(response: Any, http_status: int | None, expected: Decimal, model: str) -> dict:
    result = {'correct': False, 'expected': str(expected), 'predicted': None, 'valid_response': False, 'errors': []}
    if http_status != 200:
        result['errors'].append('http_status')
        return result
    text = ''
    if isinstance(response, dict):
        choices = response.get('choices')
        if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
            message = choices[0].get('message')
            if isinstance(message, dict) and isinstance(message.get('content'), str):
                text = message['content']
    envelope = validate_response(response, [text] if text.strip() else ['<missing-final-answer>'], expected_model=model)
    if envelope['status'] != 'passed':
        result['errors'] = [item['field'] + ':' + item['reason'] for item in envelope['errors']]
        return result
    predicted = number(envelope['final_text'])
    if predicted is None:
        result['errors'].append('unparseable_answer')
        return result
    result.update(valid_response=True, predicted=str(predicted), correct=predicted == expected)
    return result


def evaluate(dataset_path: Path, manifest_path: Path, run_dir: Path) -> dict:
    raw = dataset_path.read_bytes()
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(raw).hexdigest()
    if manifest.get('dataset_sha256') != digest:
        raise ValueError('dataset digest does not match pinned manifest')
    dataset = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    indices = selection(manifest, len(dataset))
    records = []
    response_hashes = []
    root = run_dir.resolve()
    for index in indices:
        answer = dataset[index].get('answer')
        if not isinstance(answer, str) or '####' not in answer:
            raise ValueError(f'dataset row {index} has no final-answer marker')
        expected = number(answer.rsplit('####', 1)[1])
        if expected is None:
            raise ValueError(f'dataset row {index} has no numeric oracle')
        path = root / f'{index}.response.json'
        metadata_path = root / f'{index}.transport.json'
        if path.is_symlink() or metadata_path.is_symlink():
            raise ValueError('response inputs must not be symlinks')
        response = None
        status = None
        try:
            saved = path.read_bytes()
            response_hashes.append({'sample_index': index, 'sha256': hashlib.sha256(saved).hexdigest()})
            response = json.loads(saved)
            status = json.loads(metadata_path.read_text()).get('http_status')
        except (OSError, UnicodeError, ValueError):
            pass
        row = score_case(response, status, expected, manifest['expected_model'])
        row['sample_index'] = index
        records.append(row)
    correct = sum(row['correct'] for row in records)
    invalid = sum(not row['valid_response'] for row in records)
    return {'schema_version': 1, 'dataset_sha256': digest, 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'expected_model': manifest['expected_model'], 'protocol': 'SGLang GSM8KEval signed last-number extraction; require HTTP200, exact model and normal stop first',
            'sample_indices': indices, 'shot_indices': manifest.get('shot_indices', []), 'num_samples': len(indices),
            'correct': correct, 'invalid_responses': invalid, 'accuracy': correct / len(indices),
            'error_counts': dict(Counter(error for row in records for error in row['errors'])), 'responses': records, 'response_hashes': response_hashes,
            'limits': 'Offline scoring of this pinned subset; no listener ownership, latency, full-benchmark or baseline-regression claim. Missing/error/truncated responses remain in the denominator.'}


def main() -> int:
    parser = argparse.ArgumentParser(description='Score pinned saved GSM8K responses without querying or retrying a model.')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    same_input = args.output.exists() and any(path.exists() and args.output.samefile(path) for path in [args.dataset, args.manifest])
    if same_input or output in {args.dataset.resolve(), args.manifest.resolve()} or output.is_relative_to(args.responses.resolve()):
        parser.error('output must be separate from dataset, manifest and raw response directory')
    try:
        report = evaluate(args.dataset, args.manifest, args.responses)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['num_samples', 'correct', 'invalid_responses', 'accuracy']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
