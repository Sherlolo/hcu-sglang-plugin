from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / 'tools/audit_python_equivalence.py'


class PythonEquivalenceTests(unittest.TestCase):
    def git(self, repo, *args):
        return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.DEVNULL).decode().strip()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.repos={key:self.root/key for key in ['official','fork']}
        for key,repo in self.repos.items():
            repo.mkdir();self.git(repo,'init','-q');self.git(repo,'config','user.name','Test');self.git(repo,'config','user.email','test@example.com')
            values={'equal.py':'VALUE = 1\n','second.py':'ITEMS = [1,2]\n','different.py':'VALUE = 1\n','docs.py':'"""Original"""\n','invalid.py':'def broken(\n','exec.py':'#!/usr/bin/env python3\nVALUE = 1\n','mode.py':'VALUE = 1\n'}
            if key=='fork':
                values.update({'equal.py':'# Copyright source\nVALUE = 1\n','second.py':'# Source header\nITEMS = [1, 2]\n','different.py':'VALUE = True\n','docs.py':'"""Changed"""\n','invalid.py':'# Source header\ndef broken(\n','exec.py':'#!/usr/bin/env python2\nVALUE = 1\n'})
            for name,text in values.items():(repo/name).write_text(text)
            (repo/'exec.py').chmod(0o755)
            if key=='fork':(repo/'mode.py').chmod(0o755)
            self.git(repo,'add','.');self.git(repo,'commit','-qm','fixture')
        self.plan=self.root/'plan.json';self.report=self.root/'report.json';self.decisions=self.root/'decisions.json'
        self.decisions.write_text(json.dumps({'equal.py':{'status':'pending','rationale':'manual review retained'}}))
        files=[]
        for name in values:
            files.append({'path':name,'relation':'changed','group':'models',**{key+'_oid':self.git(repo,'rev-parse','HEAD:'+name) for key,repo in self.repos.items()}})
        self.plan.write_text(json.dumps({'baselines':{key:{'head':self.git(repo,'rev-parse','HEAD')} for key,repo in self.repos.items()},'files':files}))

    def run_tool(self, apply=False, report=None):
        command=[sys.executable,str(TOOL),'--official-repo',str(self.repos['official']),'--fork-repo',str(self.repos['fork']),'--plan',str(self.plan),'--report',str(report or self.report)]
        if apply:command+=['--apply-decisions',str(self.decisions)]
        return subprocess.run(command,capture_output=True,text=True)

    def test_only_complete_ast_matches_are_added_and_manual_records_survive(self):
        result=self.run_tool(apply=True);self.assertEqual(result.returncode,0,result.stderr)
        decisions=json.loads(self.decisions.read_text())
        self.assertEqual(decisions['equal.py']['status'],'pending')
        self.assertEqual(decisions['second.py']['status'],'upstream')
        self.assertEqual(set(decisions),{'equal.py','second.py'})
        self.assertTrue(json.loads(self.report.read_text())['source_heads_and_indexes_unchanged'])

    def test_boolean_docstring_invalid_code_and_invocation_drift_stay_open(self):
        result=self.run_tool();self.assertEqual(result.returncode,0,result.stderr)
        rows={row['path']:row for row in json.loads(self.report.read_text())['files']}
        self.assertEqual(rows['equal.py']['status'],'equivalent')
        self.assertEqual(rows['different.py']['status'],'different')
        self.assertEqual(rows['docs.py']['status'],'different')
        self.assertEqual(rows['invalid.py']['status'],'unparsed')
        self.assertEqual(rows['exec.py']['status'],'invocation_header_drift')
        self.assertEqual(rows['mode.py']['status'],'mode_or_kind_drift')

    def test_stale_stat_caches_do_not_rewrite_source_indexes(self):
        before={key:(repo/'.git/index').read_bytes() for key,repo in self.repos.items()}
        for repo in self.repos.values():
            path=repo/'equal.py';stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+5_000_000_000))
        result=self.run_tool(apply=True);self.assertEqual(result.returncode,0,result.stderr)
        for key,repo in self.repos.items():self.assertEqual((repo/'.git/index').read_bytes(),before[key])

    def test_dirty_worktree_allows_commit_report_but_not_decision_application(self):
        (self.repos['fork']/'different.py').write_text('VALUE = 123\n')
        before=self.decisions.read_bytes();result=self.run_tool(apply=True)
        self.assertEqual(result.returncode,2);self.assertIn('Dirty source',result.stderr)
        self.assertEqual(self.decisions.read_bytes(),before)
        rows={row['path']:row for row in json.loads(self.report.read_text())['files']}
        self.assertEqual(rows['different.py']['status'],'different')

    def test_stale_inventory_and_output_collision_do_not_mutate_inputs(self):
        data=json.loads(self.plan.read_text());data['files'][-1]['official_oid']='a'*40;self.plan.write_text(json.dumps(data))
        before=self.decisions.read_bytes();result=self.run_tool(apply=True)
        self.assertEqual(result.returncode,2);self.assertIn('Inventory/object drift',result.stderr)
        self.assertEqual(self.decisions.read_bytes(),before)
        plan_before=self.plan.read_bytes();result=self.run_tool(report=self.plan)
        self.assertEqual(result.returncode,2);self.assertEqual(self.plan.read_bytes(),plan_before)
        result=self.run_tool(report=self.repos['fork']/'report.json')
        self.assertEqual(result.returncode,2);self.assertFalse((self.repos['fork']/'report.json').exists())


if __name__=='__main__':unittest.main()
