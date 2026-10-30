"""Run the kaggle CLI when WSL's DNS resolver is down but IP connectivity works.

Only this process's name lookups change: when the system resolver fails, hostnames are resolved
through Cloudflare's DNS-over-HTTPS endpoint (https://1.1.1.1/dns-query), reached by IP. TLS still
verifies each server's certificate against its real hostname. Nothing on the system is modified.

    python kaggle_doh.py competitions submissions gemma-4-developer-agent
"""
import json
import socket
import sys
import urllib.request

_system_getaddrinfo = socket.getaddrinfo
_cache: dict[str, list[str]] = {}


def _doh(host: str) -> list[str]:
    if host not in _cache:
        req = urllib.request.Request(f'https://1.1.1.1/dns-query?name={host}&type=A',
                                     headers={'accept': 'application/dns-json'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            answers = json.load(resp).get('Answer', [])
        _cache[host] = [a['data'] for a in answers if a.get('type') == 1]
    return _cache[host]


def _getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return _system_getaddrinfo(host, port, family, type, proto, flags)
    except socket.gaierror:
        ips = _doh(host) if isinstance(host, str) else []
        if not ips:
            raise
        sock_type = type or socket.SOCK_STREAM
        return [(socket.AF_INET, sock_type, proto or 6, '', (ip, port)) for ip in ips]


socket.getaddrinfo = _getaddrinfo

from kaggle.cli import main  # noqa: E402

sys.argv = ['kaggle'] + sys.argv[1:]
sys.exit(main())
