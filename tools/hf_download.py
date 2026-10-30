"""Download a Hugging Face dataset (default) or model repo with the DoH DNS fallback (see hf_doh.py).

    python hf_download.py REPO_ID LOCAL_DIR [dataset|model]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv, args = sys.argv[:1], sys.argv[1:]
import hf_doh  # noqa: E402,F401  (installs the getaddrinfo fallback on import)
from huggingface_hub import snapshot_download  # noqa: E402

path = snapshot_download(args[0], repo_type=args[2] if len(args) > 2 else 'dataset', local_dir=args[1], max_workers=4)
print('downloaded to', path)
