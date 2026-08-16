from __future__ import annotations

import os
import secrets
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _windows_identity() -> str:
    if os.name != "nt":
        return ""
    try:
        completed = subprocess.run(
            ["whoami.exe"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("could not resolve the current Windows security principal") from exc
    identity = completed.stdout.strip()
    if not identity or any(character in identity for character in "\r\n"):
        raise RuntimeError("could not resolve the current Windows security principal")
    return identity


def restrict_owner_access(path: str | Path) -> Path:
    """Restrict a local secret/state file to the current OS user."""

    target = Path(path).expanduser().resolve()
    if os.name != "nt":
        target.chmod(0o600)
        return target
    try:
        completed = subprocess.run(
            [
                "icacls.exe",
                str(target),
                "/inheritance:r",
                "/grant:r",
                f"{_windows_identity()}:(F)",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("could not restrict a local secret file to the current user") from exc
    if completed.returncode != 0:
        raise RuntimeError("could not restrict a local secret file to the current user")
    return target


def restrict_owner_directory(path: str | Path) -> Path:
    """Create a private local data directory and restrict traversal to its owner."""

    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        target.chmod(0o700)
        return target
    try:
        completed = subprocess.run(
            [
                "icacls.exe",
                str(target),
                "/inheritance:r",
                "/grant:r",
                f"{_windows_identity()}:(OI)(CI)F",
                "/Q",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "could not restrict a local TMCRA data directory to the current user"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            "could not restrict a local TMCRA data directory to the current user"
        )
    return target


def token_path(config_root: str | Path) -> Path:
    return Path(config_root).expanduser().resolve() / "runtime" / "secrets" / "local-api.token"


def ensure_local_token(config_root: str | Path) -> tuple[Path, str, bool]:
    path = token_path(config_root)
    restrict_owner_directory(path.parent)
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if len(value) < 32:
            raise RuntimeError(f"local API token file is invalid: {path}")
        restrict_owner_access(path)
        return path, value, False
    value = secrets.token_urlsafe(48)
    fd, temporary_name = tempfile.mkstemp(prefix=".local-api.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            temporary.chmod(0o600)
        os.replace(temporary, path)
        try:
            restrict_owner_access(path)
        except Exception:
            path.unlink(missing_ok=True)
            raise
    finally:
        temporary.unlink(missing_ok=True)
    return path, value, True


def write_secret_file(path: str | Path, value: str) -> Path:
    target = Path(path).expanduser().resolve()
    secret = str(value or "").strip()
    if not secret:
        raise ValueError("secret cannot be empty")
    restrict_owner_directory(target.parent)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(secret + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            temporary.chmod(0o600)
        os.replace(temporary, target)
        try:
            restrict_owner_access(target)
        except Exception:
            target.unlink(missing_ok=True)
            raise
    finally:
        temporary.unlink(missing_ok=True)
    return target


__all__ = [
    "ensure_local_token",
    "restrict_owner_access",
    "restrict_owner_directory",
    "token_path",
    "write_secret_file",
]
