"""Shared-token authentication. Tokens are read from files (never argv, never environment values), and a token file
that other users can read is refused, so a misconfigured deployment fails at start instead of leaking."""

from __future__ import annotations

import hmac
import os
import stat
from pathlib import Path

MIN_TOKEN_CHARS = 32


class TokenFileError(RuntimeError):
    pass


def load_token(path: str | os.PathLike[str], *, allow_group_read: bool = False) -> str:
    """Read a bearer token from ``path``.

    Refused when: the file is missing, not a regular file, writable by the group, readable by others, readable by
    the group (unless
    ``allow_group_read``, for a token shared by a service group), shorter than 32 characters, or has inner whitespace.
    The error never contains the token.
    """
    p = Path(path)
    try:
        st = p.stat()
    except OSError as exc:
        raise TokenFileError(f"token file {p}: {exc.strerror}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise TokenFileError(f"token file {p} is not a regular file")
    if st.st_mode & stat.S_IWGRP:
        raise TokenFileError(f"token file {p} is group-writable (mode {stat.S_IMODE(st.st_mode):o}); chmod 640 or 600")
    if st.st_mode & stat.S_IROTH or st.st_mode & stat.S_IWOTH:
        raise TokenFileError(f"token file {p} is accessible by other users (mode {stat.S_IMODE(st.st_mode):o}); chmod 600")
    if st.st_mode & stat.S_IRGRP and not allow_group_read:
        raise TokenFileError(f"token file {p} is group-readable (mode {stat.S_IMODE(st.st_mode):o}); chmod 600 or allow it")
    token = p.read_text(encoding="utf-8").strip()
    if len(token) < MIN_TOKEN_CHARS or any(c.isspace() for c in token):
        raise TokenFileError(f"token file {p} must hold one token of at least {MIN_TOKEN_CHARS} characters")
    return token


def bearer_ok(header_value: str | None, token: str) -> bool:
    """Constant-time check of an ``Authorization: Bearer <token>`` header."""
    if not header_value:
        return False
    scheme, _, presented = header_value.partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(presented.strip().encode(), token.encode())


def bearer_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
