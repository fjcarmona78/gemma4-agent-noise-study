import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[0]).parent))
import strip_v2 as sv  # noqa: E402
bd = sv.bd
bd.REQUESTS = Path(sys.argv[0]).with_name('hintcheck')
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(str(bd.MODEL_DIR))
tools = bd.load_tools()
for task, chains in bd.recorded_trajectories('dev91_hinted1'):
    stripped = sv.strip_hints(chains, tok, tools)
    messages = stripped[0][0]
    for i, m in enumerate(messages):
        blob = json.dumps([m.get('reasoning'), m.get('content')])
        for h in re.finditer(r'.{0,80}\b(?:hint\w*|task authors).{0,80}', blob, re.I):
            print(task, i, m['role'], repr(h.group(0)))
