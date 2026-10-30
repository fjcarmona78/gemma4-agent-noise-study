"""Fetch Hugging Face dataset cards/metadata when WSL's DNS resolver is down (same DoH fallback as
kaggle_doh.py: only this process's name lookups change; TLS still verifies real hostnames).

    python hf_doh.py OUT_DIR REPO_ID [REPO_ID ...]
"""
import json
import socket
import sys
import urllib.request
from pathlib import Path

_system_getaddrinfo = socket.getaddrinfo
_cache = {}


def _doh(host):
    if host not in _cache:
        req = urllib.request.Request(f'https://1.1.1.1/dns-query?name={host}&type=A', headers={'accept': 'application/dns-json'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            _cache[host] = [a['data'] for a in json.load(resp).get('Answer', []) if a.get('type') == 1]
    return _cache[host]


def _getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return _system_getaddrinfo(host, port, family, type, proto, flags)
    except socket.gaierror:
        ips = _doh(host) if isinstance(host, str) else []
        if not ips:
            raise
        return [(socket.AF_INET, type or socket.SOCK_STREAM, proto or 6, '', (ip, port)) for ip in ips]


socket.getaddrinfo = _getaddrinfo


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    out = Path(sys.argv[1])
    api = HfApi()
    for repo in sys.argv[2:]:
        d = out / repo.replace('/', '__')
        d.mkdir(parents=True, exist_ok=True)
        try:
            info = api.dataset_info(repo, files_metadata=True)
            files = [(s.rfilename, s.size) for s in (info.siblings or [])]
            meta = {'id': info.id, 'license': (info.card_data or {}).get('license') if info.card_data else None,
                    'tags': info.tags, 'downloads': info.downloads, 'n_files': len(files),
                    'total_gb': round(sum(s or 0 for _, s in files) / 1e9, 2), 'files_head': files[:15]}
            (d / 'meta.json').write_text(json.dumps(meta, indent=1, default=str))
            print(repo, json.dumps({k: meta[k] for k in ('license', 'n_files', 'total_gb')}))
            hf_hub_download(repo, 'README.md', repo_type='dataset', local_dir=d)
        except Exception as error:  # report and continue with the next repo
            print(repo, 'ERROR', type(error).__name__, str(error)[:200])


if __name__ == '__main__':
    main()
