import argparse
import sys
from pathlib import Path

P = Path.home() / 'projects' / 'gemma-agent'
sys.path.insert(0, str(P / 'finetune'))
import build_dataset as bd  # noqa: E402

bd.REQUESTS = Path(sys.argv[0]).with_name('hintcheck') / 'requests'
bd.build(argparse.Namespace(max_tokens=20480, max_per_task=1, out=str(Path(sys.argv[0]).with_name('hintcheck') / 'v2_test.jsonl')))
