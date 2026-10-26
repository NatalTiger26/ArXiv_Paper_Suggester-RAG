"""
Streamlit web UI for ArXiv Paper Suggester.

Delegates directly to the unified project web application (`src/app.py`),
providing seamless backwards-compatibility for existing launchers and documentation.

Run with:
  uv run python -m streamlit run src/app.py
  (or: streamlit run src/rag/app.py)
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if __name__ == "__main__" or "streamlit" in sys.modules:
    target_app = _SRC / "app.py"
    runpy.run_path(str(target_app), run_name="__main__")
