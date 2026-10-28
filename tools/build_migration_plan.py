#!/usr/bin/env python3
"""Collect pinned Git trees and render a standalone DAS-to-plugin migration plan."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NATIVE = {'.cu', '.cuh', '.cpp', '.hpp', '.h', '.cc', '.c', '.hip'}
CODE = NATIVE | {'.py', '.json'}
CONTRACT = 'tests/contracts/targets_v0_5_22.json'


@dataclass
class Repository:
    head: str
    status: str
    files: dict[str, dict]
    blobs: dict[str, bytes]


def git(root: Path, *args: str, input_data: bytes | None = None) -> bytes:
    return subprocess.check_output(['git', '-C', str(root), *args], input=input_data,
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'))


def collect(root: Path) -> Repository:
    head = git(root, 'rev-parse', 'HEAD').decode().strip()
    files = {}
    for entry in git(root, 'ls-tree', '-rlz', '--full-tree', head).split(b'\0'):
        if not entry:
            continue
        meta, raw_path = entry.split(b'\t', 1)
        mode, kind, oid, size = meta.decode().split()
        name = raw_path.decode('utf-8')
        files[name] = {'oid': oid, 'mode': mode, 'kind': kind,
                       'size': int(size) if size != '-' else 0}
    requests = {item['oid'] for name, item in files.items()
                if item['kind'] == 'blob' and item['mode'] != '120000'
                and Path(name).suffix in CODE}
    raw = git(root, 'cat-file', '--batch', input_data=('\n'.join(sorted(requests)) + '\n').encode()) if requests else b''
    blobs = {}
    offset = 0
    while offset < len(raw):
        end = raw.index(b'\n', offset)
        oid, kind, size = raw[offset:end].decode().split()
        if kind != 'blob':
            raise ValueError('Unexpected object type')
        offset = end + 1
        blobs[oid] = raw[offset:offset + int(size)]
        offset += int(size) + 1
    if set(blobs) != requests:
        raise ValueError('Incomplete Git object read')
    if git(root, 'rev-parse', 'HEAD').decode().strip() != head:
        raise ValueError('HEAD moved during collection; retry without resetting the source')
    return Repository(head, git(root, 'status', '--porcelain=v1').decode(), files, blobs)


def content(repo: Repository, path: str) -> bytes | None:
    item = repo.files.get(path)
    return repo.blobs.get(item['oid']) if item else None


def python_info(body: bytes) -> dict:
    result = {'lines': len(body.splitlines()), 'symbols': [], 'classes': 0, 'functions': 0}
    try:
        tree = ast.parse(body)
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = 'class' if isinstance(node, ast.ClassDef) else 'async function' if isinstance(node, ast.AsyncFunctionDef) else 'function'
                result['symbols'].append([node.name, node.lineno, kind])
                result['classes' if kind == 'class' else 'functions'] += 1
    except (SyntaxError, ValueError, UnicodeError) as error:
        result['parse_error'] = '{}: {}'.format(type(error).__name__, str(error).splitlines()[0])
    return result


def group_for(path: str, groups: list[dict]) -> str:
    candidates = [(len(prefix), group['id']) for group in groups for prefix in group['prefixes']
                  if path.startswith(prefix)]
    return max(candidates)[1]


def target_path(target: str, files: set[str], packages: dict[str, set[str]] | None = None) -> str | None:
    parts = target.split('.')
    while parts:
        for suffix in ('.py',):
            path = 'python/' + '/'.join(parts) + suffix
            if path in files:
                return path
        package = 'python/' + '/'.join(parts) + '/__init__.py'
        tail = target.split('.')[len(parts):]
        if tail and package in files and tail[0] in (packages or {}).get(package, set()):
            return package
        parts.pop()
    return None


def extract_provenance(fork: Repository, plugin: Repository) -> tuple[dict, list]:
    mappings = defaultdict(list)
    diagnostics = []
    source_names = set(fork.files)
    source_heads = {'fork_commit', 'fork_head'}
    for path in sorted(plugin.files):
        if not path.startswith('sglang_dcu_plugin/') or 'provenance' not in path or not path.endswith('.json'):
            continue
        try:
            document = json.loads(content(plugin, path) or b'null')
        except (ValueError, UnicodeError):
            diagnostics.append({'file': path, 'issue': 'invalid JSON'})
            continue
        if isinstance(document, dict):
            for key in source_heads:
                if document.get(key) and document[key] != fork.head:
                    diagnostics.append({'file': path, 'issue': 'declared fork differs from scanned HEAD'})
        def normalize(value, prefix=''):
            if not isinstance(value, str):
                return None
            candidates = [value, prefix.rstrip('/') + '/' + value]
            if 'python/sglang/' in value:
                candidates.append('python/sglang/' + value.split('python/sglang/', 1)[1])
            for candidate in candidates:
                if candidate in source_names:
                    return candidate
            return None
        def walk(node, prefix=''):
            if isinstance(node, dict):
                prefix = node.get('source_root', prefix)
                if not isinstance(prefix, str):
                    prefix = ''
                name = None
                for key in ('source', 'source_file', 'source_path', 'file', 'path'):
                    name = normalize(node.get(key), prefix)
                    if name:
                        break
                if name:
                    expected = node.get('source_sha256') or node.get('sha256')
                    original = content(fork, name)
                    status = 'declared_reference'
                    if expected and original is not None:
                        status = 'source_hash_match' if hashlib.sha256(original).hexdigest() == expected else 'source_hash_mismatch'
                    target = node.get('plugin') or node.get('destination') or node.get('target')
                    target = target if isinstance(target, str) and target in plugin.files else None
                    record = {'manifest': path, 'status': status}
                    if target:
                        record['plugin'] = target
                        expected_target = node.get('plugin_sha256')
                        actual = content(plugin, target)
                        if expected_target and actual is not None:
                            record['plugin_hash_match'] = hashlib.sha256(actual).hexdigest() == expected_target
                    if record not in mappings[name]:
                        mappings[name].append(record)
                for child in node.values():
                    walk(child, prefix)
            elif isinstance(node, list):
                for child in node:
                    walk(child, prefix)
        walk(document)
    return mappings, diagnostics


def validate_plan(plan: dict) -> None:
    ids = {group['id'] for group in plan['groups']}
    phases = {phase['id'] for phase in plan['phases']}
    if len(phases) != len(plan['phases']):
        raise ValueError('Duplicate phase ID')
    for phase in plan['phases']:
        if set(phase['depends_on']) - phases:
            raise ValueError('Invalid phase dependency')
    if len(ids) != len(plan['groups']):
        raise ValueError('Duplicate module group ID')
    for group in plan['groups']:
        if group['phase'] not in phases or set(group['depends_on']) - ids:
            raise ValueError('Invalid dependency or phase: ' + group['id'])
    visiting, visited = set(), set()
    by_id = {group['id']: group for group in plan['groups']}
    def visit(key):
        if key in visiting:
            raise ValueError('Cyclic module dependency: ' + key)
        if key in visited:
            return
        visiting.add(key)
        for dep in by_id[key]['depends_on']:
            visit(dep)
        visiting.remove(key)
        visited.add(key)
    for key in ids:
        visit(key)
    visiting.clear()
    visited.clear()
    by_id = {phase['id']: phase for phase in plan['phases']}
    for key in phases:
        visit(key)


def inventory(official: Repository, fork: Repository, plugin: Repository, plan: dict) -> dict:
    validate_plan(plan)
    if CONTRACT not in plugin.files:
        raise ValueError('Pinned DCU target contract is missing: ' + CONTRACT)
    contract = json.loads(content(plugin, CONTRACT) or b'[]')
    if not isinstance(contract, list) or not contract or any(not isinstance(value, str) for value in contract):
        raise ValueError('Expected a string-list target contract')
    targets = sorted(set(contract))
    target_set = set(targets)
    literal_sites = defaultdict(list)
    for path in sorted(plugin.files):
        if not path.startswith('sglang_dcu_plugin/') or not path.endswith('.py'):
            continue
        body = content(plugin, path)
        if body is None:
            continue
        try:
            tree = ast.parse(body)
        except (SyntaxError, ValueError, UnicodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in target_set:
                literal_sites[node.value].append([path, node.lineno])
    by_file = defaultdict(list)
    unmatched = []
    all_paths = set(fork.files) | set(official.files)
    packages = {}
    for name in all_paths:
        if name.endswith('/__init__.py'):
            body = content(fork if name in fork.files else official, name)
            if body is not None:
                packages[name] = {symbol[0] for symbol in python_info(body)['symbols']}
    for target in targets:
        path = target_path(target, all_paths, packages)
        record = {'target': target, 'literal_sites': literal_sites[target]}
        if path:
            by_file[path].append(record)
        else:
            unmatched.append(record)
    provenance, provenance_diagnostics = extract_provenance(fork, plugin)
    native_by_oid = defaultdict(list)
    for path, item in plugin.files.items():
        if path.startswith('sglang_dcu_plugin/') and Path(path).suffix in NATIVE and item['size'] >= 64:
            native_by_oid[item['oid']].append(path)
    rows = []
    semantics = {}
    for path in sorted(all_paths):
        left, right = fork.files.get(path), official.files.get(path)
        relation = 'same' if left and right and left['oid'] == right['oid'] and left['mode'] == right['mode'] else 'changed' if left and right else 'fork_only' if left else 'official_only'
        item = left or right
        row = {'path': path, 'group': group_for(path, plan['groups']), 'relation': relation,
               'kind': item['kind'], 'mode': item['mode'], 'size': item['size']}
        if left:
            row['fork_oid'] = left['oid']
        if right:
            row['official_oid'] = right['oid']
        repo = fork if left else official
        if path.endswith('.py') and item['kind'] == 'blob' and item['mode'] != '120000':
            body = content(repo, path)
            if body is not None:
                if item['oid'] not in semantics:
                    semantics[item['oid']] = python_info(body)
                row.update(semantics[item['oid']])
                row['symbol_source'] = 'fork' if left else 'official'
        if path in by_file:
            row['targets'] = by_file[path]
        if path in provenance:
            row['provenance'] = provenance[path]
        if left and Path(path).suffix in NATIVE and left['size'] >= 64 and left['oid'] in native_by_oid:
            row['identical_native'] = native_by_oid[left['oid']]
        row['has_mapping_clue'] = bool(row.get('targets') or row.get('provenance') or row.get('identical_native'))
        rows.append(row)
    summaries = []
    for group in plan['groups']:
        members = [row for row in rows if row['group'] == group['id']]
        counts = dict(Counter(row['relation'] for row in members))
        summary = dict(group)
        summary['counts'] = counts
        summary['fork_files'] = sum('fork_oid' in row for row in members)
        summary['python_files'] = sum(row['path'].endswith('.py') and 'fork_oid' in row for row in members)
        summary['target_count'] = sum(len(row.get('targets', [])) for row in members)
        summary['mapped_files'] = sum(row['has_mapping_clue'] and 'fork_oid' in row for row in members)
        summary['review_files'] = sum(row['relation'] in {'changed', 'fork_only', 'official_only'} for row in members)
        summary['example_files'] = [row['path'] for row in sorted(members, key=lambda row: (not row['has_mapping_clue'], row['relation'] == 'same', -row['size'], row['path']))[:6]]
        summary['tasks'] = [{'id': '{}-{:02d}'.format(group['id'], number), 'title': title,
                             'status': 'proposed', 'assignee': None} for number, title in enumerate(group['tasks'], 1)]
        summaries.append(summary)
    runtime = [row for row in rows if row['path'].startswith('python/sglang/')]
    counts = dict(Counter(row['relation'] for row in rows))
    top_level = defaultdict(Counter)
    for row in rows:
        top_level[row['path'].split('/')[0]][row['relation']] += 1
    signature = hashlib.sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'schema_version': 1, 'plan_signature': signature, 'title': plan['title'],
            'collected_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'plan_reviewed_on': plan['reviewed_on'],
            'baselines': {key: {'head': repo.head, 'dirty': bool(repo.status), 'worktree_analyzed': False, 'tracked_paths': len(repo.files)}
                          for key, repo in [('fork', fork), ('official', official), ('plugin', plugin)]},
            'summary': {'relations': counts, 'fork_runtime_files': sum('fork_oid' in row for row in runtime),
                        'fork_python_modules': sum(row['path'].endswith('.py') and 'fork_oid' in row for row in runtime),
                        'runtime_review_files': sum(row['relation'] != 'same' for row in runtime),
                        'declared_targets': len(targets), 'unmatched_targets': len(unmatched),
                        'mapping_clue_files': sum(row['has_mapping_clue'] and 'fork_oid' in row for row in rows),
                        'parse_errors': sum('parse_error' in row for row in rows),
                        'proposed_tasks': sum(len(group['tasks']) for group in summaries)},
            'top_level': [{'name': name, 'counts': dict(value)} for name, value in sorted(top_level.items())],
            'phases': plan['phases'], 'groups': summaries, 'files': rows,
            'unmatched_targets': unmatched, 'provenance_diagnostics': provenance_diagnostics,
            'target_contract': CONTRACT, 'historical_evidence': plan['historical_evidence'],
            'semantics': {'comparison': 'Pinned HEAD trees only; dirty working-tree edits are recorded but not analyzed. Current same-path blob/mode comparison, not merge-base attribution or behavioral equivalence.',
                          'symbols': 'Top-level Python class/function/async-function names and source line numbers; non-Python symbols are not parsed.',
                          'mapping': 'Declared target contracts, literal source locations, provenance claims and identical native blobs are static clues, not runtime coverage.',
                          'tasks': 'Proposed work packages awaiting review and assignment, not committed deadlines or all confirmed implementation gaps.',
                          'submodules': 'Gitlink objects are recorded; nested submodule files are not read.',
                          'runtime': 'No model, GPU, registration apply or service test was executed by this collector.'}}


def render(snapshot: dict, template: str) -> str:
    payload = json.dumps(snapshot, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c').replace('&', '\\u0026')
    if template.count('__MIGRATION_DATA__') != 1:
        raise ValueError('Expected one data placeholder')
    return template.replace('__MIGRATION_DATA__', payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official', type=Path)
    parser.add_argument('--fork', type=Path)
    parser.add_argument('--plugin', type=Path)
    parser.add_argument('--plan', type=Path, default=ROOT / 'templates/sglang-das-migration-plan.json')
    parser.add_argument('--snapshot', type=Path, default=ROOT / 'plans/sglang-das-migration.json')
    parser.add_argument('--template', type=Path, default=ROOT / 'templates/migration-plan.html')
    parser.add_argument('--output', type=Path, default=ROOT / 'migration-plan.html')
    parser.add_argument('--render-only', action='store_true', help='render an existing snapshot without source repositories')
    parser.add_argument('--check', action='store_true', help='read-only freshness check against the saved snapshot')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if args.check or args.render_only:
        snapshot = json.loads(args.snapshot.read_text(encoding='utf-8'))
        signature = hashlib.sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if snapshot.get('plan_signature') != signature:
            print('Plan definitions changed; collect a new snapshot with the three input repositories')
            return 1
    else:
        if not all((args.official, args.fork, args.plugin)):
            parser.error('--official, --fork and --plugin are required for collection')
        for source in (args.official, args.fork, args.plugin):
            for output in (args.snapshot, args.output):
                if output.resolve().is_relative_to(source.resolve()):
                    parser.error('Outputs must be outside input repositories')
        snapshot = inventory(collect(args.official), collect(args.fork), collect(args.plugin),
                             plan)
    rendered = render(snapshot, args.template.read_text(encoding='utf-8'))
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding='utf-8') != rendered:
            print('Migration plan is stale; render the saved snapshot again')
            return 1
        print('Migration plan matches the saved snapshot; this does not check for newer source commits')
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.render_only:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    args.output.write_text(rendered, encoding='utf-8')
    print(json.dumps(snapshot['summary'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
