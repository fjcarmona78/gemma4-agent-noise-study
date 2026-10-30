"""Validate a submission directory the way the Kaggle scorer does. Usage: validate_submission.py DIR"""

import os
import sys
from pathlib import Path

from adk_submission import validate_directory
from swegemma.config import build_submission_limits
from swegemma.models.discovery import validate_single_declared_model

# 12B allowed since 2026-10-11 (host post: the scorer serves it with the Gemma4ForCausalLM override).
# The four variants the hosts list in forum topic 748287 (2026-10-11); one variant per submission.
ALLOWED_MODELS = {'gemma-4-31b-it-qat-w4a16-ct', 'gemma-4-12b-it-qat-w4a16-ct', 'gemma-4-e4b-it-qat-w4a16-ct',
                  'gemma-4-e2b-it-qat-w4a16-ct'}
# Local-only teacher runs (distillation data, never submitted) may name extra models through
# LOCAL_TEACHER_MODELS=name[,name]. Unset when packaging, so a submission still gets the strict check.
TEACHER_MODELS = {m for m in os.environ.get('LOCAL_TEACHER_MODELS', '').split(',') if m}

submission = Path(sys.argv[1])
limits, _generation_constraints = build_submission_limits()
validate_directory(submission, limits)
model = validate_single_declared_model(submission)
if model in TEACHER_MODELS:
    print(f'{submission}: valid for LOCAL teacher runs only ({model}); never submit this directory')
    sys.exit(0)
if model not in ALLOWED_MODELS:
    sys.exit(f'{submission}: declares {model}, but only {sorted(ALLOWED_MODELS)} are allowed')
print(f'{submission}: valid ({model})')
