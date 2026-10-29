from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_execution(plan, decisions):
    files = {item['path']: item for item in plan['files'] if item['relation'] != 'same'}
    unknown = set(decisions) - files.keys()
    if unknown:
        raise ValueError(f'decisions refer to paths outside the review set: {sorted(unknown)}')
    rows = []
    for path, source in sorted(files.items()):
        decision = decisions.get(path, {})
        status = decision.get('status', 'pending')
        if status not in {'pending', 'in_progress', 'implemented', 'validated', 'upstream', 'separate_component'}:
            raise ValueError(f'{path}: invalid status {status}')
        if status != 'pending' and not decision.get('rationale'):
            raise ValueError(f'{path}: disposition needs a rationale')
        if status in {'implemented', 'validated'} and not decision.get('implementation'):
            raise ValueError(f'{path}: implementation paths are required')
        if status == 'validated' and not decision.get('evidence'):
            raise ValueError(f'{path}: validation evidence is required')
        rows.append({'path': path, 'group': source['group'], 'relation': source['relation'],
                     'fork_oid': source.get('fork_oid'), 'official_oid': source.get('official_oid'),
                     'status': status, **{k: v for k, v in decision.items() if k != 'status'}})
    return {'schema_version': 1, 'inventory_baselines': plan['baselines'],
            'semantics': 'File dispositions are review records, not a runtime equivalence or completion percentage. Unlisted decisions remain pending. Historical mapping clues never imply validation.',
            'summary': dict(Counter(row['status'] for row in rows)), 'files': rows}


def main():
    parser = argparse.ArgumentParser(description='Build a complete, explicit DAS difference execution ledger.')
    parser.add_argument('--plan', type=Path, default=ROOT / 'plans/sglang-das-migration.json')
    parser.add_argument('--decisions', type=Path, default=ROOT / 'plans/sglang-das-decisions.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'plans/sglang-das-execution.json')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = build_execution(json.loads(args.plan.read_text()), json.loads(args.decisions.read_text()))
    content = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.check:
        return int(not args.output.is_file() or args.output.read_text() != content)
    args.output.write_text(content)
    print(json.dumps(report['summary']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
