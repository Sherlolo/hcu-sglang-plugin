from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile
from tools import audit_jit_resources as A


class JitResourceAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.fork=self.root/'fork';self.fork.mkdir()
        self.source='python/sglang/kernels/jit/csrc/demo.cuh'
        p=self.fork/self.source;p.parent.mkdir(parents=True);p.write_bytes(b'int resource = 7;\n')
        def git(*args):return subprocess.check_output(['git','-C',str(self.fork),*args],stderr=subprocess.DEVNULL).decode().strip()
        self.git=git;git('init');git('config','user.name','Audit Test');git('config','user.email','audit@example.invalid');git('add','.');git('commit','-m','source')
        self.head=git('rev-parse','HEAD');self.oid=git('rev-parse','HEAD:'+self.source)
        self.plan={'inventory_baselines':{'fork':{'head':self.head}},'files':[{'path':self.source,'fork_oid':self.oid,'status':'pending'}]}
        self.plugin=self.root/'plugin';self.owner='hcu_sglang/jit_overlay/csrc/demo.cuh';self.current=self.plugin/self.owner
        self.current.parent.mkdir(parents=True);self.current.write_bytes(p.read_bytes())
        self.wheel=self.root/'plugin.whl'
        with zipfile.ZipFile(self.wheel,'w') as archive:archive.writestr(self.owner,self.current.read_bytes())
        self.installs=self.root/'installs';self.historical=self.installs/'site-dev1'/self.owner
        self.historical.parent.mkdir(parents=True);self.historical.write_bytes(p.read_bytes())
        self.entries=[['abs',str(self.historical),hashlib.sha256(self.historical.read_bytes()).hexdigest()]]
        key=hashlib.sha256((repr(tuple(self.entries[0]))+'\0').encode()).hexdigest()[:16]
        self.cache=self.root/'cache';self.leaf=self.cache/'gfx938/module/build-example'/('deps-'+key)
        self.leaf.mkdir(parents=True);(self.leaf/'sgl_deps.json').write_text(json.dumps(self.entries));(self.leaf/'module.so').write_bytes(b'presence fixture, never executable')

    def review(self):return A.review(self.plan,self.fork,self.plugin,self.wheel,self.cache,self.installs)

    def test_exact_source_wheel_and_historical_record_do_not_claim_execution_or_change_disposition(self):
        before=self.git('status','--porcelain');index=(self.fork/'.git/index').read_bytes();original=json.dumps(self.plan,sort_keys=True)
        report=self.review();row=report['files'][0]
        self.assertEqual(row['status'],'recorded_compile_dependency');self.assertTrue(row['wheel_bytes_match'])
        self.assertFalse(row['compile_evidence'][0]['executed_in_current_run']);self.assertEqual(row['disposition_before'],'pending')
        self.assertEqual(json.dumps(self.plan,sort_keys=True),original);self.assertEqual(self.git('status','--porcelain'),before)
        self.assertEqual((self.fork/'.git/index').read_bytes(),index)

    def test_tampered_manifest_key_and_missing_library_cannot_establish_build_evidence(self):
        (self.leaf/'sgl_deps.json').write_text(json.dumps(self.entries+[['sys','fake.h','bad']]))
        report=self.review();self.assertEqual(report['files'][0]['status'],'identical_owner_only');self.assertEqual(report['rejected_compile_records']['manifest_key_mismatch'],1)
        (self.leaf/'sgl_deps.json').write_text(json.dumps(self.entries));(self.leaf/'module.so').unlink()
        self.assertEqual(self.review()['rejected_compile_records']['missing_library'],1)

    def test_historical_and_current_file_drift_are_detected_independently(self):
        self.historical.write_bytes(b'changed historical source')
        report=self.review();self.assertEqual(report['files'][0]['status'],'identical_owner_only');self.assertEqual(report['rejected_compile_records']['historical_dependency_drift'],1)
        self.historical.write_bytes(self.current.read_bytes());self.current.write_bytes(b'changed current source')
        report=self.review();self.assertEqual(report['files'][0]['status'],'owner_differs');self.assertEqual(report['rejected_compile_records']['current_dependency_drift'],1)

    def test_wheel_omission_and_external_install_path_are_not_accepted(self):
        with zipfile.ZipFile(self.wheel,'w') as archive:archive.writestr('unrelated.txt','not packaged')
        self.assertEqual(self.review()['files'][0]['status'],'not_packaged_identically')
        evidence,rejected=A.read_compile_evidence(self.cache,self.plugin,self.root/'other-installs','hcu_sglang')
        self.assertEqual(evidence,{});self.assertEqual(rejected['outside_install_scope'],1)

    def test_removed_source_is_recorded_without_claiming_a_missing_port(self):
        self.plan['files'].append({'path':'python/sglang/kernels/jit/csrc/removed.cuh','fork_oid':None,'status':'pending'})
        report=self.review();removed=next(row for row in report['files'] if row['fork_oid'] is None)
        self.assertEqual(removed['status'],'removed_in_fork');self.assertEqual(removed['disposition_before'],'pending')
        self.assertFalse(removed['retained_owner_exists'])

    def test_inventory_head_and_object_drift_fail_closed(self):
        self.plan['inventory_baselines']['fork']['head']='0'*40
        with self.assertRaisesRegex(ValueError,'HEAD'):self.review()
        self.plan['inventory_baselines']['fork']['head']=self.head;self.plan['files'][0]['fork_oid']='0'*40
        with self.assertRaisesRegex(ValueError,'object drift'):self.review()


if __name__=='__main__':unittest.main()
