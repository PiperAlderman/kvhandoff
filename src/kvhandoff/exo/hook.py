"""Install or remove the kvhandoff hook in an EXO checkout (replaces the prototype's patch script).

The hook is four lines at the top of the runner's prefill ``resolve`` and is inert unless KVHANDOFF_BRIDGE_CONFIG is
set in EXO's environment. kvhandoff must be importable from EXO's interpreter (``uv pip install --no-deps`` it into
EXO's venv). install keeps a byte-exact backup and is idempotent; remove restores the backup (no git needed), or strips
the hook if the backup is missing. The checkout path is an argument, never a default.

  kvhandoff-exo-hook install --exo-root <checkout>
  kvhandoff-exo-hook remove  --exo-root <checkout>
  kvhandoff-exo-hook status  --exo-root <checkout>
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

RUNNER = os.path.join("src", "exo", "worker", "runner", "runner.py")
ANCHOR = "        def resolve(request: PrefillRequest, wfile: BinaryIO) -> bool:\n"
MARK = "# kvhandoff hook"
HOOK = (
    f"            import os as _kvh_os  {MARK}\n"
    '            if _kvh_os.environ.get("KVHANDOFF_BRIDGE_CONFIG"):\n'
    "                from kvhandoff.exo.bridge import resolve as _kvh_resolve\n"
    "                return _kvh_resolve(request, wfile)\n"
)
BACKUP_SUFFIX = ".kvhandoff-orig"


class HookError(RuntimeError):
    pass


def runner_path(exo_root: str) -> str:
    p = os.path.join(exo_root, RUNNER)
    if not os.path.isfile(p):
        raise HookError(f"{p} not found (is {exo_root} an EXO checkout?)")
    return p


def status(exo_root: str) -> str:
    with open(runner_path(exo_root), encoding="utf-8") as f:
        return "installed" if MARK in f.read() else "absent"


def install(exo_root: str) -> str:
    p = runner_path(exo_root)
    with open(p, encoding="utf-8") as f:
        src = f.read()
    if MARK in src:
        return "already installed"
    if src.count(ANCHOR) != 1:
        raise HookError(f"anchor found {src.count(ANCHOR)} times in {p}; this EXO version needs a new anchor")
    if not os.path.exists(p + BACKUP_SUFFIX):
        shutil.copy2(p, p + BACKUP_SUFFIX)
    tmp = p + ".kvhandoff-tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(src.replace(ANCHOR, ANCHOR + HOOK))
    shutil.copymode(p, tmp)
    os.replace(tmp, p)
    return "installed"


def remove(exo_root: str) -> str:
    """Strip the hook (never restore an old backup over a newer runner.py); drop the backup when it matches."""
    p = runner_path(exo_root)
    backup = p + BACKUP_SUFFIX
    with open(p, encoding="utf-8") as f:
        src = f.read()
    if MARK not in src:
        return "not installed"
    if HOOK not in src:
        raise HookError(f"{p} carries a modified kvhandoff hook; remove it by hand (backup: {backup})")
    stripped = src.replace(HOOK, "")
    tmp = p + ".kvhandoff-tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(stripped)
    shutil.copymode(p, tmp)
    os.replace(tmp, p)
    if os.path.exists(backup):
        with open(backup, encoding="utf-8") as f:
            if f.read() == stripped:
                os.unlink(backup)
                return "hook removed (matches the backup, backup deleted)"
        return f"hook removed (runner.py differs from the backup, kept at {backup})"
    return "hook removed"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kvhandoff-exo-hook", description="Install the kvhandoff hook into EXO")
    ap.add_argument("action", choices=("install", "remove", "status"))
    ap.add_argument("--exo-root", required=True)
    a = ap.parse_args(argv)
    try:
        fn = {"install": install, "remove": remove, "status": status}[a.action]
        print(fn(a.exo_root))
    except HookError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
