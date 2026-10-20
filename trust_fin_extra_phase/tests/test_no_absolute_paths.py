"""Roadmap §11 step 2: no runtime dependence on absolute machine paths.

Scans the runnable Phase-5 code (tools, runtime, tests, build files) for
user-specific absolute paths. Frozen manifests, notebooks and docs are exempt:
they are immutable records of the machine that produced them.
"""
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
REPO = WS.parent

FORBIDDEN = "/home/heet18"

SCAN = [
    *sorted((WS / "tools").glob("*.py")),
    *sorted((WS / "runtime").rglob("*.py")),
    *sorted((WS / "runtime").rglob("*.json")),
    *sorted((WS / "tests").glob("*.py")),
    *(p for p in (WS / "scripts").glob("*.py")),
    REPO / "Makefile",
    REPO / "Dockerfile",
    REPO / ".dockerignore",
    WS / "requirements.txt",
]


def test_runtime_code_has_no_absolute_machine_paths():
    offenders = []
    self_path = Path(__file__).resolve()
    for p in SCAN:
        if not p.exists():
            continue
        if p.resolve() == self_path:
            continue  # this test file names the forbidden literal on purpose
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if FORBIDDEN in line:
                offenders.append(f"{p.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offenders, ("absolute machine paths leaked into runtime code:\n  "
                           + "\n  ".join(offenders))


def test_paths_module_discovers_workspace():
    from tools.paths import find_repo_root, find_python, workspace
    root = find_repo_root()
    assert (root / "trust_fin_extra_phase" / "PLAN.md").exists()
    assert workspace() == root / "trust_fin_extra_phase"
    assert Path(find_python()).exists()
