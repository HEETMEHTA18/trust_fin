#!/usr/bin/env python3
"""paths.py — machine-independent path discovery for the TrustFin repository.

Phase-5A hardening (docs/next_phases_roadmap.md §5, §11): runtime code must not
hard-code /home/<user>/... paths. Instead:

  repo root   : discovered by walking up from this file until the marker
                ``trust_fin_extra_phase/PLAN.md`` is found
                (override with ``$TRUSTFIN_REPO_ROOT``).
  workspace   : <repo root>/trust_fin_extra_phase
  interpreter : ``$TRUSTFIN_PYTHON`` if set, else the project venv at
                ``~/trustfin-slm/.venv/bin/python[3]`` if it exists, else the
                interpreter currently running this script (documented default).

Frozen manifests (Phase-2.1/3/4) may still *record* absolute paths — those are
immutable contract files and are never edited. Use :func:`resolve_recorded` to
map a recorded path back onto the current machine.

CLI
  python3 tools/paths.py --python       print the project interpreter
  python3 tools/paths.py --repo-root    print the repository root
  python3 tools/paths.py --workspace    print the trust_fin_extra_phase dir
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

#: A directory containing this relative path is the repository root.
MARKER = "trust_fin_extra_phase/PLAN.md"

_used_fallback_python = False


def find_repo_root() -> Path:
    """Return the repository root ('Research Papers') for this checkout."""
    env = os.environ.get("TRUSTFIN_REPO_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / MARKER).exists():
            return p
        raise FileNotFoundError(
            f"TRUSTFIN_REPO_ROOT={p} does not look like the TrustFin repo root "
            f"(missing {MARKER}).")
    here = Path(__file__).resolve()
    for cand in (here.parent, *here.parents):
        if (cand / MARKER).exists():
            return cand
    raise FileNotFoundError(
        f"TrustFin repo root not found from {here.parent} "
        f"(no ancestor contains {MARKER}); set TRUSTFIN_REPO_ROOT.")


def workspace() -> Path:
    """Return <repo root>/trust_fin_extra_phase (the Phase-4/5 workspace)."""
    return find_repo_root() / "trust_fin_extra_phase"


def find_python() -> Path:
    """Return the interpreter for the project environment (see module docstring)."""
    global _used_fallback_python
    env = os.environ.get("TRUSTFIN_PYTHON")
    if env:
        return Path(env).expanduser()
    venv_dir = Path.home() / "trustfin-slm" / ".venv" / "bin"
    for name in ("python3", "python"):
        cand = venv_dir / name
        if cand.exists():
            return cand
    _used_fallback_python = True
    return Path(sys.executable)


def using_fallback_python() -> bool:
    """True when find_python() had to fall back to the current interpreter."""
    return _used_fallback_python


def resolve_recorded(recorded: str | os.PathLike) -> Path:
    """Map a path recorded in a frozen manifest onto this machine.

    Frozen manifests store paths from the machine that produced them — often
    absolute (``/home/.../Research Papers/artifacts/...``), sometimes relative
    to a different working directory. Candidates tried, in order:

      1. the path as written (exists / CWD-relative), then repo-root + as-written
         for repo-relative paths;
      2. for every well-known anchor component (``trust_fin_extra_phase``,
         ``artifacts``, ``data``) in recorded order: repo-root + suffix.

    Returns the first candidate that exists, else the original path.
    """
    p = Path(recorded).expanduser()
    candidates: list[Path] = [p]
    if not p.is_absolute():
        candidates.append(find_repo_root() / p)
    parts = p.parts
    for i, part in enumerate(parts):
        if part in ("trust_fin_extra_phase", "artifacts", "data"):
            candidates.append(find_repo_root().joinpath(*parts[i:]))
    for c in candidates:
        if c.exists():
            return c
    return p  # not resolvable; caller decides what to do


def main() -> None:
    if "--python" in sys.argv:
        print(find_python())
    elif "--repo-root" in sys.argv:
        print(find_repo_root())
    elif "--workspace" in sys.argv:
        print(workspace())
    else:
        sys.exit("usage: paths.py --python | --repo-root | --workspace")


if __name__ == "__main__":
    main()
