from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tokenize

ENV = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0', 'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_NO_LAZY_FETCH': '1'}


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], env=ENV)


def snapshot(repo):
    tree = {}
    for record in git(repo, 'ls-tree', '-r', '-z', 'HEAD').split(b'\0'):
        if not record:
            continue
        metadata, path = record.split(b'\t', 1)
        mode, kind, oid = metadata.decode().split()
        tree[path.decode()] = (mode, kind, oid)
    index = Path(git(repo, 'rev-parse', '--git-path', 'index').decode().strip())
    if not index.is_absolute():
        index = repo / index
    return {'head': git(repo, 'rev-parse', 'HEAD').decode().strip(),
            'dirty': bool(git(repo, 'status', '--porcelain=v1', '--untracked-files=all').strip()),
            'index_sha256': hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None,
            'tree': tree}


class Blobs:
    def __init__(self, repo):
        self.proc = subprocess.Popen(['git', '-C', str(repo), 'cat-file', '--batch'],
                                     env=ENV, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def __enter__(self):
        return self

    def read(self, oid):
        if not re.fullmatch('[0-9a-f]{40,64}', oid):
            raise ValueError('Invalid object ID')
        self.proc.stdin.write(oid.encode() + b'\n'); self.proc.stdin.flush()
        header = self.proc.stdout.readline().decode().split()
        if len(header) != 3 or header[1] != 'blob':
            raise ValueError('Expected committed blob: ' + oid)
        data = self.proc.stdout.read(int(header[2]))
        if self.proc.stdout.read(1) != b'\n':
            raise ValueError('Invalid cat-file frame')
        return data

    def __exit__(self, *exc):
        self.proc.stdin.close(); self.proc.stdout.close(); self.proc.wait(timeout=10)


def ast_hash(data, path):
    encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    tree = ast.parse(data.decode(encoding), filename=path, type_comments=True)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def review(plan, snapshots, readers):
    rows = []
    for item in sorted(plan['files'], key=lambda row: row['path']):
        path = item['path']
        if item['relation'] != 'changed' or not path.endswith('.py'):
            continue
        row = {'path': path, 'group': item.get('group'), 'status': 'different'}
        pairs = [snapshots[key]['tree'].get(path) for key in ('official', 'fork')]
        for key, entry in zip(('official', 'fork'), pairs):
            if entry is None or entry[2] != item.get(key + '_oid'):
                raise ValueError('Inventory/object drift at ' + path)
        if any(entry[0] not in {'100644', '100755'} or entry[1] != 'blob' for entry in pairs) or pairs[0][0] != pairs[1][0]:
            row['status'] = 'mode_or_kind_drift'; rows.append(row); continue
        a, b = [readers[key].read(entry[2]) for key, entry in zip(('official', 'fork'), pairs)]
        row.update(official_oid=pairs[0][2], fork_oid=pairs[1][2],
                   official_sha256=hashlib.sha256(a).hexdigest(), fork_sha256=hashlib.sha256(b).hexdigest())
        first = [data.splitlines()[0] if data.splitlines() else b'' for data in (a, b)]
        if any(line.startswith(b'#!') for line in first) and first[0] != first[1]:
            row['status'] = 'invocation_header_drift'; rows.append(row); continue
        try:
            hashes = [ast_hash(data, path) for data in (a, b)]
            if hashes[0] == hashes[1]:
                row.update(status='equivalent', ast_sha256=hashes[0])
        except (SyntaxError, UnicodeError, LookupError, ValueError) as error:
            row.update(status='unparsed', error=type(error).__name__)
        rows.append(row)
    return {'schema_version': 1, 'python_version': sys.version.split()[0],
            'semantics': 'Committed regular Python files only. Full AST includes imports, guards, decorators, docstrings and type comments; location attributes are omitted. Equivalent means no per-file Python code delta, not dependency/whole-program/runtime validation. Existing decisions are never overwritten.',
            'baselines': {key: {k: value for k, value in state.items() if k != 'tree'} for key, state in snapshots.items()},
            'summary': dict(Counter(row['status'] for row in rows)), 'files': rows}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    descriptor, temporary = tempfile.mkstemp(prefix='.equivalence-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Audit complete committed Python AST equality without importing source code.')
    parser.add_argument('--official-repo', type=Path, required=True)
    parser.add_argument('--fork-repo', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--apply-decisions', type=Path)
    args = parser.parse_args(argv)
    try:
        repos = {'official': args.official_repo.resolve(), 'fork': args.fork_repo.resolve()}
        outputs = [args.report.resolve()] + ([args.apply_decisions.resolve()] if args.apply_decisions else [])
        if len(set(outputs)) != len(outputs) or args.plan.resolve() in outputs:
            raise ValueError('Output paths must be distinct from the plan and each other')
        if any(out == repo or repo in out.parents for out in outputs for repo in repos.values()):
            raise ValueError('Outputs must be outside both source repositories')
        plan = json.loads(args.plan.read_text())
        previous = args.apply_decisions.read_bytes() if args.apply_decisions and args.apply_decisions.exists() else None
        decisions = json.loads(previous) if previous is not None else {}
        states = {key: snapshot(repo) for key, repo in repos.items()}
        if any(state['head'] != plan['baselines'][key]['head'] for key, state in states.items()):
            raise ValueError('Inventory HEAD drift; refresh and review the plan first')
        with ExitStack() as scope:
            readers = {key: scope.enter_context(Blobs(repo)) for key, repo in repos.items()}
            report = review(plan, states, readers)
        after = {key: snapshot(repo) for key, repo in repos.items()}
        stable = all(after[key]['head'] == state['head'] and after[key]['index_sha256'] == state['index_sha256'] for key, state in states.items())
        report['source_heads_and_indexes_unchanged'] = stable
        write_json(args.report, report)
        if not stable:
            raise ValueError('Source HEAD/index changed during audit; no decisions written')
        added = 0
        if args.apply_decisions:
            if any(state['dirty'] or after[key]['dirty'] for key, state in states.items()):
                raise ValueError('Dirty source worktree: report is commit-only; no decisions written')
            current = args.apply_decisions.read_bytes() if args.apply_decisions.exists() else None
            if current != previous:
                raise ValueError('Concurrent decisions edit; refusing to overwrite it')
            for row in report['files']:
                if row['status'] != 'equivalent' or row['path'] in decisions:
                    continue
                decisions[row['path']] = {'status': 'upstream',
                    'rationale': 'Whole committed Python AST matches the pinned official file, including imports, guards, decorators, values, docstrings and type comments. Reuse official code unchanged; dependencies/other paths remain separate review scopes.',
                    'evidence': {'official_oid': row['official_oid'], 'fork_oid': row['fork_oid'], 'ast_sha256': row['ast_sha256']},
                    'validation_scope': 'Per-file committed Python code equality only; not runtime, dependency equivalence or whole-DAS completion.'}
                added += 1
            if added:
                write_json(args.apply_decisions, decisions)
        print(json.dumps({'summary': report['summary'], 'new_upstream_decisions': added, 'source_heads_and_indexes_unchanged': stable}))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print('error: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
