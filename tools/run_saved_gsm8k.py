#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from score_saved_gsm8k import evaluate, selection


def prompt(dataset: list[dict], shots: list[int], index: int) -> str:
    pieces = []
    for shot in shots:
        pieces.append('Question: ' + dataset[shot]['question'] + '\nAnswer: ' + dataset[shot]['answer'] + '\n\n')
    pieces.append('Question: ' + dataset[index]['question'] + '\nAnswer:')
    return ''.join(pieces)


def run(args) -> dict:
    raw = args.dataset.read_bytes()
    manifest = json.loads(args.manifest.read_text())
    digest = hashlib.sha256(raw).hexdigest()
    if manifest.get('dataset_sha256') != digest:
        raise ValueError('dataset digest does not match manifest')
    dataset = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    manifest['expected_model'] = args.model
    indices = selection(manifest, len(dataset))
    if args.parallel < 1 or args.max_tokens < 1 or args.timeout <= 0:
        raise ValueError('parallel, max-tokens and timeout must be positive')
    kwargs = json.loads(args.chat_template_kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError('chat-template-kwargs must be a JSON object')
    base = args.base_url.rstrip('/')
    address = urlparse(base)
    if address.scheme not in ('http', 'https') or not address.netloc or address.query or address.fragment:
        raise ValueError('base-url must be an HTTP(S) server root without query or fragment')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + '/v1/models', timeout=args.timeout) as source:
        identity = source.read()
    if args.model not in [row.get('id') for row in json.loads(identity).get('data', [])]:
        raise ValueError('expected model is absent from server identity response')
    args.output.mkdir(parents=True, exist_ok=False)
    responses = args.output / 'responses'
    responses.mkdir()
    manifest.update(base_url=base, dataset_path=str(args.dataset.resolve()), parallel=args.parallel,
                    temperature=0, max_tokens=args.max_tokens, chat_template_kwargs=kwargs,
                    scorer_protocol='source_last_number_normal_stop', server_models_sha256=hashlib.sha256(identity).hexdigest())
    saved_manifest = args.output / 'manifest.json'
    saved_manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'models.response.json').write_bytes(identity)

    def one(index):
        request = {'model': args.model, 'messages': [{'role': 'user', 'content': prompt(dataset, manifest.get('shot_indices', []), index)}],
                   'temperature': 0, 'max_tokens': args.max_tokens, 'stream': False, 'chat_template_kwargs': kwargs}
        body = json.dumps(request).encode()
        (responses / f'{index}.request.json').write_bytes(body)
        code = None
        error = None
        data = b''
        started = time.monotonic()
        try:
            call = urllib.request.Request(base + '/v1/chat/completions', data=body, headers={'Content-Type': 'application/json'})
            with opener.open(call, timeout=args.timeout) as source:
                code, data = source.status, source.read()
        except urllib.error.HTTPError as exc:
            code, data = exc.code, exc.read()
        except (OSError, urllib.error.URLError) as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        (responses / f'{index}.response.json').write_bytes(data)
        (responses / f'{index}.transport.json').write_text(json.dumps({'http_status': code, 'error': error,
            'seconds': time.monotonic() - started, 'request_sha256': hashlib.sha256(body).hexdigest(),
            'response_sha256': hashlib.sha256(data).hexdigest()}, indent=2) + '\n')

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(one, indices))
    report = evaluate(args.dataset, saved_manifest, responses)
    report['wall_seconds'] = time.monotonic() - started
    (args.output / 'score.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description='Run a pinned GSM8K subset once and preserve every request, response and transport result.')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parallel', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--chat-template-kwargs', default='{}')
    args = parser.parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: result[key] for key in ['num_samples', 'correct', 'invalid_responses', 'accuracy']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
