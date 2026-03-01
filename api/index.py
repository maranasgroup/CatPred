from __future__ import annotations

from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if os.environ.get("VERCEL"):
    os.environ.setdefault("CATPRED_API_RUNTIME_ROOT", "/tmp/catpred")
    os.environ.setdefault("CATPRED_MODAL_FALLBACK_TO_LOCAL", "0")
    if os.environ.get("CATPRED_MODAL_ENDPOINT"):
        os.environ.setdefault("CATPRED_DEFAULT_BACKEND", "modal")

from catpred.web.app import app


__all__ = ["app"]
