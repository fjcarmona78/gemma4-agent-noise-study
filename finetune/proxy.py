"""Pass-through proxy that records every chat request/response between the harness and vLLM.

Captures the exact messages, tools, and model output the agent exchanged, so fine-tuning data
matches what the model actually sees. Forwards every request to vLLM unchanged.

    python finetune/proxy.py --log finetune/requests/<run>.jsonl [--port 8001] [--upstream http://127.0.0.1:8000]

Point the harness at it with MODEL_PROXY_URL=http://127.0.0.1:8001/v1 (read by swegemma's
model registry). Each JSONL line: {"ts", "path", "status", "seconds", "request", "response"}.
"""

import argparse
import json
import time

import aiohttp
from aiohttp import web

HOP_BY_HOP = {'connection', 'keep-alive', 'transfer-encoding', 'content-length', 'host', 'content-encoding'}


def make_app(upstream: str, log_path: str) -> web.Application:
    log = open(log_path, 'a', buffering=1)

    async def forward(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        start = time.time()
        async with request.app['session'].request(
                request.method, upstream + request.path_qs, data=body, headers=headers) as resp:
            payload = await resp.read()
            out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP_BY_HOP}
            if request.path.endswith('/chat/completions'):
                try:
                    req_json = json.loads(body)
                except ValueError:
                    req_json = body.decode(errors='replace')
                try:
                    resp_json = json.loads(payload)
                except ValueError:
                    resp_json = payload.decode(errors='replace')
                log.write(json.dumps({'ts': start, 'path': request.path, 'status': resp.status,
                                      'seconds': round(time.time() - start, 3),
                                      'request': req_json, 'response': resp_json}) + '\n')
            return web.Response(status=resp.status, body=payload, headers=out_headers)

    async def on_startup(app: web.Application) -> None:
        app['session'] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))

    async def on_cleanup(app: web.Application) -> None:
        await app['session'].close()
        log.close()

    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_route('*', '/{tail:.*}', forward)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--upstream', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    web.run_app(make_app(args.upstream.rstrip('/'), args.log), host='127.0.0.1', port=args.port,
                access_log=None, print=None)


if __name__ == '__main__':
    main()
