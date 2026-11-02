"""The EXO hook installer, and a guard that the package ships no internal addresses, hosts or paths as defaults."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pytest

from kvhandoff.exo import hook

SRC = Path(__file__).resolve().parent.parent / "src"

RUNNER = """class Runner:
    def start(self):
        def resolve(request: PrefillRequest, wfile: BinaryIO) -> bool:
            req = PrefillTask(request=request)
            return True
"""


def fake_checkout(tmp_path: Path, text: str = RUNNER) -> Path:
    p = tmp_path / "exo" / hook.RUNNER
    p.parent.mkdir(parents=True)
    p.write_text(text)
    return tmp_path / "exo"


def test_install_is_idempotent_and_remove_restores_bytes(tmp_path: Path) -> None:
    root = fake_checkout(tmp_path)
    runner = root / hook.RUNNER
    assert hook.status(str(root)) == "absent"
    assert hook.install(str(root)) == "installed"
    assert hook.install(str(root)) == "already installed"
    text = runner.read_text()
    assert text.count(hook.MARK) == 1 and "KVHANDOFF_BRIDGE_CONFIG" in text
    compile(text, "runner.py", "exec")  # still valid Python
    assert hook.remove(str(root)).startswith("hook removed (matches the backup")
    assert runner.read_text() == RUNNER and not os.path.exists(str(runner) + hook.BACKUP_SUFFIX)
    assert hook.remove(str(root)) == "not installed"


def test_remove_after_exo_was_updated_keeps_the_update(tmp_path: Path) -> None:
    root = fake_checkout(tmp_path)
    runner = root / hook.RUNNER
    hook.install(str(root))
    runner.write_text(runner.read_text() + "# a later EXO change\n")
    assert "kept at" in hook.remove(str(root))
    assert runner.read_text() == RUNNER + "# a later EXO change\n"


def test_missing_anchor_is_an_error(tmp_path: Path) -> None:
    root = fake_checkout(tmp_path, "def other():\n    pass\n")
    with pytest.raises(hook.HookError, match="anchor found 0 times"):
        hook.install(str(root))
    with pytest.raises(hook.HookError, match="not found"):
        hook.install(str(tmp_path / "nowhere"))
    assert hook.main(["status", "--exo-root", str(tmp_path / "nowhere")]) == 1


def test_install_into_a_real_exo_runner(tmp_path: Path) -> None:
    src = os.environ.get("EXO_SRC")
    if not src or not (Path(src) / "exo" / "worker" / "runner" / "runner.py").exists():
        pytest.skip("EXO_SRC not set")
    root = tmp_path / "exo"
    (root / "src" / "exo" / "worker" / "runner").mkdir(parents=True)
    shutil.copy(Path(src) / "exo" / "worker" / "runner" / "runner.py", root / hook.RUNNER)
    original = (root / hook.RUNNER).read_text()
    assert hook.main(["install", "--exo-root", str(root)]) == 0
    compile((root / hook.RUNNER).read_text(), "runner.py", "exec")
    assert hook.main(["remove", "--exo-root", str(root)]) == 0
    assert (root / hook.RUNNER).read_text() == original


# ----------------------------------------------------------------------------------------------- hygiene
FORBIDDEN = [
    (re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "private (RFC1918) address"),
    (re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b"), "private (RFC1918) address"),
    (re.compile(r"\b172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"), "private (RFC1918) address"),
    (re.compile(r"/(?:home|Users)/[A-Za-z]"), "user home directory path"),
    (re.compile(r"\"/dev/shm[^\"]*\"\s*[,)]|=\s*\"/dev/shm"), "/dev/shm used as a default"),
]
# Site-specific names (host names, accounts) to keep out of src/, supplied at test time rather than published here:
# KVHANDOFF_FORBIDDEN_NAMES="name1,name2" pytest
_names = [n.strip() for n in os.environ.get("KVHANDOFF_FORBIDDEN_NAMES", "").split(",") if n.strip()]
if _names:
    FORBIDDEN.append((re.compile("|".join(re.escape(n) for n in _names), re.IGNORECASE), "site-specific name"))


def test_no_internal_addresses_hosts_or_default_paths_in_src() -> None:
    hits: list[str] = []
    for path in SRC.rglob("*.py"):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            for pat, what in FORBIDDEN:
                if pat.search(line) and "noqa: S108" not in line:
                    hits.append(f"{path.relative_to(SRC)}:{i}: {what}: {line.strip()}")
    assert not hits, "\n".join(hits)
