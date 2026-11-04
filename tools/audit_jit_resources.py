from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

ENV = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0', 'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_NO_LAZY_FETCH': '1'}
PREFIX = 'python/sglang/kernels/jit/'


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], env=ENV)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blob_oid(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def dependency_key(entries):
    value = hashlib.sha256()
    for row in entries:
        value.update(repr(tuple(row)).encode())
        value.update(b'\0')
    return value.hexdigest()[:16]


def read_compile_evidence(cache_root, plugin_root, install_base, package):
    evidence = {}
    rejected = Counter()
    if cache_root is None:
        return evidence, dict(rejected)
    marker = '/' + package + '/jit_overlay/'
    for manifest in sorted(cache_root.rglob('sgl_deps.json')):
        try:
            entries = json.loads(manifest.read_text())
            if not isinstance(entries, list) or not entries or any(
                not isinstance(row, list) or len(row) != 3 or not all(isinstance(x, str) for x in row)
                for row in entries
            ):
                raise ValueError('invalid dependency rows')
            if manifest.parent.name != 'deps-' + dependency_key(entries):
                rejected['manifest_key_mismatch'] += 1
                continue
            libraries = sorted(p for p in manifest.parent.glob('*.so') if p.is_file() and not p.is_symlink())
            if not libraries:
                rejected['missing_library'] += 1
                continue
            for anchor, value, expected in entries:
                if anchor != 'abs' or marker not in value:
                    continue
                historical = Path(value)
                if not historical.is_absolute():
                    rejected['nonabsolute_dependency'] += 1
                    continue
                allowed = [plugin_root]
                if install_base is not None:
                    allowed.append(install_base)
                if not any(historical.resolve().is_relative_to(root.resolve()) for root in allowed):
                    rejected['outside_install_scope'] += 1
                    continue
                relative = package + '/jit_overlay/' + value.split(marker, 1)[1]
                current = plugin_root / relative
                if not current.is_file() or current.is_symlink() or not current.resolve().is_relative_to(plugin_root.resolve()):
                    rejected['missing_or_external_owner'] += 1
                    continue
                if not historical.is_file() or digest(historical) != expected:
                    rejected['historical_dependency_drift'] += 1
                    continue
                if digest(current) != expected:
                    rejected['current_dependency_drift'] += 1
                    continue
                evidence.setdefault(relative, []).append({
                    'manifest': manifest.relative_to(cache_root).as_posix(),
                    'manifest_sha256': digest(manifest),
                    'source_sha256': expected,
                    'install_label': historical.relative_to(install_base).parts[0]
                    if install_base is not None and historical.is_relative_to(install_base) else 'plugin-worktree',
                    'library_names': [p.name for p in libraries],
                    'executed_in_current_run': False,
                })
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            rejected['unreadable_or_invalid_manifest'] += 1
    return evidence, dict(rejected)


def review(plan, fork_repo, plugin_root, wheel=None, cache_root=None, install_base=None, package='hcu_sglang'):
    head = git(fork_repo, 'rev-parse', 'HEAD').decode().strip()
    expected_head = plan['inventory_baselines']['fork']['head']
    if head != expected_head:
        raise ValueError('Fork HEAD differs from inventory baseline')
    tree = {}
    for record in git(fork_repo, 'ls-tree', '-r', '-z', 'HEAD').split(b'\0'):
        if record:
            info, name = record.split(b'\t', 1)
            mode, kind, oid = info.decode().split()
            tree[name.decode()] = (mode, kind, oid)
    compiled, rejected = read_compile_evidence(cache_root, plugin_root, install_base, package)
    wheel_data = {}
    if wheel is not None:
        with zipfile.ZipFile(wheel) as archive:
            wheel_data = {n: hashlib.sha256(archive.read(n)).hexdigest()
                          for n in archive.namelist() if n.startswith(package + '/jit_overlay/') and not n.endswith('/')}
    rows = []
    for item in sorted(plan['files'], key=lambda row: row['path']):
        source = item['path']
        if not source.startswith(PREFIX) or not source[len(PREFIX):].startswith(('csrc/', 'include/')):
            continue
        entry = tree.get(source)
        if entry is None and item.get('fork_oid') is None:
            owner = package + '/jit_overlay/' + source[len(PREFIX):]
            rows.append({'source_path': source, 'fork_oid': None, 'owner': owner,
                         'disposition_before': item['status'], 'status': 'removed_in_fork',
                         'retained_owner_exists': (plugin_root / owner).is_file()})
            continue
        if entry is None or entry[2] != item.get('fork_oid'):
            raise ValueError('Inventory/object drift: ' + source)
        owner = package + '/jit_overlay/' + source[len(PREFIX):]
        path = plugin_root / owner
        row = {'source_path': source, 'fork_oid': entry[2], 'source_mode': entry[0],
               'owner': owner, 'disposition_before': item['status'], 'status': 'missing_owner'}
        if entry[1] != 'blob' or entry[0] not in {'100644', '100755'}:
            row['status'] = 'nonregular_source'
        elif path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(plugin_root.resolve()):
            data = path.read_bytes()
            row['owner_sha256'] = hashlib.sha256(data).hexdigest()
            row['owner_mode_matches'] = bool(path.stat().st_mode & 0o111) == (entry[0] == '100755')
            row['source_bytes_match'] = blob_oid(data) == entry[2]
            row['wheel_bytes_match'] = wheel_data.get(owner) == row['owner_sha256'] if wheel else None
            row['compile_evidence'] = compiled.get(owner, [])
            if not row['source_bytes_match']:
                row['status'] = 'owner_differs'
            elif not row['owner_mode_matches']:
                row['status'] = 'owner_mode_drift'
            elif wheel and not row['wheel_bytes_match']:
                row['status'] = 'not_packaged_identically'
            elif row['compile_evidence']:
                row['status'] = 'recorded_compile_dependency'
            else:
                row['status'] = 'identical_owner_only'
        rows.append(row)
    if git(fork_repo, 'rev-parse', 'HEAD').decode().strip() != head:
        raise ValueError('Fork HEAD moved during review')
    return {
        'schema_version': 1,
        'semantics': 'Read-only committed-source to owned WORKTREE resource review. Compiler dependency records are historical build evidence, not current execution, loaded-symbol or numerical proof. No migration dispositions are changed.',
        'fork_head': head,
        'fork_worktree_dirty': bool(git(fork_repo, 'status', '--porcelain').strip()),
        'owned_worktree_analyzed': True,
        'wheel_sha256': digest(wheel) if wheel else None,
        'rejected_compile_records': rejected,
        'summary': dict(Counter(row['status'] for row in rows)),
        'files': rows,
    }


def main():
    parser = argparse.ArgumentParser(description='Review JIT resource bytes, wheel packaging and historical compiler dependency closure without executing source or kernels.')
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--fork', type=Path, required=True)
    parser.add_argument('--plugin-root', type=Path, required=True)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--install-base', type=Path)
    parser.add_argument('--package', default='hcu_sglang')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.fork.resolve()):
        parser.error('Output must not overlap the read-only source repository')
    report = review(json.loads(args.plan.read_text()), args.fork, args.plugin_root,
                    args.wheel, args.cache_root, args.install_base, args.package)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], sort_keys=True))


if __name__ == '__main__':
    main()
