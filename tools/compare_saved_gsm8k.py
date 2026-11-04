#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from score_saved_gsm8k import evaluate


MANIFEST_FIELDS = ('dataset_sha256', 'sample_indices', 'shot_indices', 'temperature', 'max_tokens',
                   'parallel', 'chat_template_kwargs', 'scorer_protocol')
IGNORED_ENV = {'SGLANG_PLUGINS', 'SGLANG_TORCH_PROFILER_DIR', 'SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR'}


def compare(baseline: dict, candidate: dict, baseline_manifest: dict, candidate_manifest: dict,
            baseline_state: dict, candidate_state: dict) -> dict:
    mismatches = []
    for field in MANIFEST_FIELDS:
        if field not in baseline_manifest or field not in candidate_manifest or baseline_manifest[field] != candidate_manifest[field]:
            mismatches.append('manifest.' + field)
    for label, state in [('baseline', baseline_state), ('candidate', candidate_state)]:
        if state.get('status') != 'passed' or state.get('cleanup_remaining_pids') != [] or state.get('port_listening_after_cleanup') is not False:
            mismatches.append(label + '.service_completion')
    left = baseline_state.get('config', {})
    right = candidate_state.get('config', {})
    for field in ('args', 'recipe_commit', 'physical_devices'):
        if field not in left or field not in right or left[field] != right[field]:
            mismatches.append('serving.' + field)
    env_left, env_right = left.get('env', {}), right.get('env', {})
    for key in sorted(set(env_left) | set(env_right)):
        if key not in IGNORED_ENV and env_left.get(key) != env_right.get(key):
            mismatches.append('serving.env.' + key)
    result = {'status': 'incomparable' if mismatches else 'comparable', 'configuration_scope': 'declared_serving_args_and_env_only', 'live_resolved_config_verified': False, 'mismatches': mismatches,
              'baseline_accuracy': baseline['accuracy'], 'candidate_accuracy': candidate['accuracy'],
              'baseline_invalid': baseline['invalid_responses'], 'candidate_invalid': candidate['invalid_responses'],
              'limits': 'Paired outcomes for this pinned subset and declared matching serving arguments; no weight-file hash, full-benchmark accuracy, statistical significance or acceptance threshold is established.'}
    if mismatches:
        return result
    by_index = {row['sample_index']: row for row in baseline['responses']}
    other = {row['sample_index']: row for row in candidate['responses']}
    if set(by_index) != set(other) or set(by_index) != set(baseline_manifest['sample_indices']):
        raise ValueError('scored response indices do not match the pinned subset')
    paired = []
    counts = {'both_correct': 0, 'baseline_only_correct': 0, 'candidate_only_correct': 0, 'both_incorrect': 0}
    for index in baseline_manifest['sample_indices']:
        a, b = by_index[index], other[index]
        bucket = ('both_correct' if a['correct'] and b['correct'] else
                  'baseline_only_correct' if a['correct'] else 'candidate_only_correct' if b['correct'] else 'both_incorrect')
        counts[bucket] += 1
        paired.append({'sample_index': index, 'outcome': bucket, 'baseline_prediction': a['predicted'],
                       'candidate_prediction': b['predicted'], 'baseline_valid': a['valid_response'],
                       'candidate_valid': b['valid_response']})
    result.update(outcomes=counts, paired=paired, accuracy_delta=candidate['accuracy'] - baseline['accuracy'])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description='Rescore paired saved GSM8K runs and refuse configuration-mismatched comparisons.')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--candidate-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.output.exists() and args.output.stat().st_nlink > 1:
        parser.error('output must not be an existing hard-linked file')
    if output == args.dataset.resolve() or any(output.is_relative_to(run.resolve()) for run in [args.baseline_run, args.candidate_run]):
        parser.error('output must be separate from dataset and both saved run directories')
    try:
        results, manifests, states = [], [], []
        for run in [args.baseline_run, args.candidate_run]:
            manifest = run / 'gsm8k/manifest.json'
            results.append(evaluate(args.dataset, manifest, run / 'gsm8k/responses'))
            manifests.append(json.loads(manifest.read_text()))
            states.append(json.loads((run / 'state.json').read_text()))
        report = compare(*results, *manifests, *states)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'paired'}))
    return 0 if report['status'] == 'comparable' else 1


if __name__ == '__main__':
    raise SystemExit(main())
