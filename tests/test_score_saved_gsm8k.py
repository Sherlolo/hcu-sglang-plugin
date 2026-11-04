from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from score_saved_gsm8k import evaluate, score_case, selection


def response(text='Answer: -1,234.0', finish='stop', model='owned'):
    return {'model': model, 'choices': [{'finish_reason': finish, 'message': {'role': 'assistant', 'content': text}}]}


class SavedMathScoreTests(unittest.TestCase):
    def test_signed_commas_decimals_and_math_mistakes_are_distinct_from_transport(self):
        self.assertTrue(score_case(response(), 200, Decimal('-1234'), 'owned')['correct'])
        wrong = score_case(response('Answer: 3'), 200, Decimal('4'), 'owned')
        self.assertTrue(wrong['valid_response']);self.assertFalse(wrong['correct'])

    def test_truncation_wrong_model_and_non200_cannot_count_as_correct(self):
        for item, status in [(response('4','length'),200),(response('4',model='other'),200),(response('4'),500)]:
            row=score_case(item,status,Decimal('4'),'owned')
            self.assertFalse(row['correct']);self.assertFalse(row['valid_response'])

    def test_prefix_leak_duplicates_and_bool_indices_are_rejected(self):
        for indices,shots in [([1],[1]),([1,1],[]),([True],[]),([4],[])]:
            with self.assertRaises(ValueError):selection({'sample_indices':indices,'shot_indices':shots,'expected_model':'owned'},4)


    def test_generation_prompt_uses_only_prefix_oracles_and_current_question(self):
        from run_saved_gsm8k import prompt
        rows=[{'question':'prefix','answer':'hidden-prefix #### 42'},{'question':'scored-question','answer':'never-send-this-oracle #### 987654'}]
        value=prompt(rows,[0],1)
        self.assertIn('hidden-prefix #### 42',value)
        self.assertIn('scored-question',value)
        self.assertNotIn('never-send-this-oracle',value)
        self.assertNotIn('987654',value)


    def test_runner_preserves_each_transport_error_and_scores_without_retry(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from threading import Thread
        from types import SimpleNamespace
        from run_saved_gsm8k import run
        seen=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                self.send_response(200);self.end_headers();self.wfile.write(b'{"data":[{"id":"owned"}]}')
            def do_POST(self):
                data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append(data)
                failed='second-scored' in data['messages'][0]['content']
                self.send_response(500 if failed else 200);self.end_headers()
                self.wfile.write(json.dumps({'error':'fixture'} if failed else response('4')).encode())
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);worker=Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);dataset=root/'data.jsonl';manifest=root/'manifest.json'
                rows=[{'question':q,'answer':'#### 4'} for q in ['prefix','first-scored','second-scored']]
                dataset.write_text('\n'.join(map(json.dumps,rows))+'\n')
                manifest.write_text(json.dumps({'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),'sample_indices':[1,2],'shot_indices':[0]}))
                args=SimpleNamespace(dataset=dataset,manifest=manifest,model='owned',parallel=2,max_tokens=16,timeout=5,chat_template_kwargs='{}',base_url='http://127.0.0.1:'+str(server.server_port),output=root/'run')
                result=run(args)
                self.assertEqual((result['correct'],result['invalid_responses'],result['accuracy']),(1,1,.5))
                self.assertEqual(len(seen),2)
                self.assertEqual(json.loads((root/'run/responses/2.transport.json').read_text())['http_status'],500)
                self.assertTrue((root/'run/responses/1.request.json').is_file())
        finally:server.shutdown();worker.join();server.server_close()

    def test_missing_and_invalid_samples_stay_in_accuracy_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset=root/'data.jsonl';manifest=root/'manifest.json';saved=root/'responses';saved.mkdir()
            dataset.write_text('\n'.join(json.dumps({'question':str(i),'answer':'reason #### 4'}) for i in range(3))+'\n')
            manifest.write_text(json.dumps({'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),'sample_indices':[1,2],'shot_indices':[0],'expected_model':'owned'}))
            (saved/'1.response.json').write_text(json.dumps(response('4')));(saved/'1.transport.json').write_text('{"http_status":200}')
            result=evaluate(dataset,manifest,saved)
            self.assertEqual((result['correct'],result['num_samples'],result['invalid_responses'],result['accuracy']),(1,2,1,.5))
            dataset.write_text(dataset.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'digest'):evaluate(dataset,manifest,saved)


if __name__=='__main__':unittest.main()
