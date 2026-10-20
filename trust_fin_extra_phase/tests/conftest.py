"""pytest configuration: put the trust_fin_extra_phase workspace on sys.path."""
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
if str(WS) not in sys.path:
    sys.path.insert(0, str(WS))
