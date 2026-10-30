"""Model metadata via the Hugging Face API with the DoH DNS fallback: license, architecture, size."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_doh  # noqa: E402,F401
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

api = HfApi()
if sys.argv[1] == 'search':
    for m in api.list_models(search=sys.argv[2], sort='downloads', limit=int(sys.argv[3]) if len(sys.argv) > 3 else 15):
        print(m.id, '| downloads', m.downloads, '| tags', [t for t in (m.tags or []) if t.startswith(('license', 'awq', 'gptq', 'fp8', 'compressed'))][:4])
else:
    for repo in sys.argv[1:]:
        try:
            info = api.model_info(repo, files_metadata=True)
            size = sum((s.size or 0) for s in info.siblings or [] if s.rfilename.endswith('.safetensors')) / 1e9
            cfg = json.loads(Path(hf_hub_download(repo, 'config.json')).read_text())
            print(repo, '| license', (info.card_data or {}).get('license') if info.card_data else None, '| arch', cfg.get('architectures'),
                  '| model_type', cfg.get('model_type'), '| quant', (cfg.get('quantization_config') or {}).get('quant_method'), f'| {size:.1f} GB')
        except Exception as error:
            print(repo, 'ERROR', type(error).__name__, str(error)[:150])
