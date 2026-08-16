#!/usr/bin/env python3
"""Fail closed when the public source tree crosses TMCRA's release boundary.

The report prints categories and file locations only. It deliberately never
prints a matching secret value or the matching source line.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "PUBLIC_RELEASE_MANIFEST.json"
SELF_PATH = Path(__file__).resolve()
IPV4_PATTERN = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")


@dataclass(frozen=True, order=True)
class Finding:
    category: str
    path: str
    line: int = 0


def _git(*arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _release_files() -> list[Path]:
    payload = _git(
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    )
    names = sorted({name for name in payload.decode("utf-8").split("\0") if name})
    paths = [(ROOT / PurePosixPath(name)).resolve() for name in names]
    return [path for path in paths if path.is_file()]


def _text_lines(path: Path) -> list[str] | None:
    if path == SELF_PATH or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        return None
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return None


def _patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    # Build high-confidence credential prefixes in pieces so this scanner does
    # not flag its own source when reviewed outside _text_lines().
    return (
        (
            "private-key-material",
            re.compile("BEGIN " + r"(?:RSA |OPENSSH |EC )?PRIVATE KEY"),
        ),
        ("openai-style-secret", re.compile("s" + r"k-[A-Za-z0-9_-]{20,}")),
        ("huggingface-secret", re.compile("h" + r"f_[A-Za-z0-9]{20,}")),
        (
            "github-secret",
            re.compile(r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
        ),
        ("google-api-secret", re.compile("AI" + r"za[0-9A-Za-z_-]{20,}")),
        ("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
        ("tencent-secret-id", re.compile(r"AKID[A-Za-z0-9]{13,}")),
        ("slack-secret", re.compile("xo" + r"x[baprs]-[A-Za-z0-9-]{10,}")),
        (
            "jwt-token",
            re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
        ),
        (
            "literal-bearer-secret",
            re.compile(r"Bearer\s+[A-Za-z0-9._~-]{24,}", re.IGNORECASE),
        ),
        (
            "credential-in-url",
            re.compile(r"https?://[^\s/:@]+:[^\s/@]+@", re.IGNORECASE),
        ),
        (
            "generic-credential-literal",
            re.compile(
                r'''\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|'''
                r'''client[_ -]?secret|password|passwd|secret)\b["']?\s*'''
                r'''(?:=|:)\s*["'](?![$<{])[^"'\\\s]{8,}["']''',
                re.IGNORECASE,
            ),
        ),
        (
            "database-credential-in-url",
            re.compile(
                r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://"
                r"[^\s/:@]+:[^\s/@]+@",
                re.IGNORECASE,
            ),
        ),
        (
            "production-service-reference",
            re.compile(
                r"(?:api\." + r"tmcra\.com|gpu" + r"home\.cc|sc01-" + r"ssh|119\." + r"28\.)",
                re.IGNORECASE,
            ),
        ),
        (
            "developer-machine-path",
            re.compile(
                r"(?:[A-Za-z]:[\\/](?:Users|Documents|Desktop)[\\/]|"
                r"[A-Za-z]:[\\/]新建文件夹|/root/|/home/[A-Za-z0-9._-]+/)",
                re.IGNORECASE,
            ),
        ),
        (
            "internal-deployment-claim",
            re.compile(
                r"(?:tmcra-production-serving-verified|"
                r"TMCRA production deployment manifest|"
                r"TMCRA production currently uses|"
                r"生产同款|TMCRA 生产服务当前)",
                re.IGNORECASE,
            ),
        ),
    )


def _path_findings(files: Iterable[Path], manifest: dict) -> list[Finding]:
    allowed_dirs = set(manifest["allowed_top_level_directories"])
    allowed_files = set(manifest["allowed_root_files"])
    forbidden_suffixes = {
        ".db",
        ".key",
        ".log",
        ".p12",
        ".pem",
        ".pfx",
        ".sqlite",
        ".sqlite3",
    }
    forbidden_parts = {
        ".deploy-out",
        ".tmcra",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
    findings: list[Finding] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        pure = PurePosixPath(relative)
        if len(pure.parts) == 1:
            if relative not in allowed_files:
                findings.append(Finding("unapproved-root-file", relative))
        elif pure.parts[0] not in allowed_dirs:
            findings.append(Finding("unapproved-top-level-directory", relative))
        if forbidden_parts.intersection(part.lower() for part in pure.parts):
            findings.append(Finding("generated-or-private-path", relative))
        lower_name = pure.name.lower()
        if PurePosixPath(lower_name).suffix in forbidden_suffixes:
            findings.append(Finding("credential-or-state-file", relative))
        if lower_name in {".env", "credentials.json", "secrets.json"}:
            findings.append(Finding("credential-or-state-file", relative))
        if ".egg-info" in relative.lower():
            findings.append(Finding("generated-package-metadata", relative))
        if pure.parts[0] == "models" and (
            "checkpoints" in (part.lower() for part in pure.parts)
            or lower_name.startswith(("train_", "training_", "nohup"))
            or lower_name == "launch_train.sh"
        ):
            findings.append(Finding("private-training-run-artifact", relative))
    for required in manifest["required_paths"]:
        if not (ROOT / PurePosixPath(required)).is_file():
            findings.append(Finding("missing-required-release-file", required))
    return findings


def _content_findings(files: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []
    patterns = _patterns()
    forbidden_import = re.compile(
        r"^\s*(?:from|import)\s+(?:tmcra_service|tmcra_admin|tmcra_billing|tmcra_staff)(?:\b|\.)"
    )
    for path in files:
        lines = _text_lines(path)
        if lines is None:
            continue
        relative = path.relative_to(ROOT).as_posix()
        for number, line in enumerate(lines, 1):
            if (
                "/tests/" in f"/{relative}"
                and "public-audit: allow-test-fixture" in line
            ):
                continue
            for category, pattern in patterns:
                if pattern.search(line):
                    findings.append(Finding(category, relative, number))
            if relative.startswith("runtime/") and forbidden_import.search(line):
                findings.append(Finding("production-control-plane-import", relative, number))
            for raw_address in IPV4_PATTERN.findall(line):
                try:
                    address = ipaddress.ip_address(raw_address)
                except ValueError:
                    continue
                if not address.is_loopback:
                    findings.append(Finding("non-loopback-ipv4-literal", relative, number))
    return findings


def _history_findings() -> list[Finding]:
    probes = {
        "history-private-key": "BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY",
        "history-production-host": (
            "(api\\." + "tmcra\\.com|gpu" + "home\\.cc|sc01-" + "ssh|119\\." + "28\\.)"
        ),
        "history-openai-secret": "sk-[A-Za-z0-9_-]{20,}",
        "history-huggingface-secret": "hf_[A-Za-z0-9]{20,}",
        "history-github-secret": "(gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})",
    }
    findings: list[Finding] = []
    for category, expression in probes.items():
        output = _git("log", "--all", "--format=%H", "-G", expression, "--", ".")
        commits = sorted(set(output.decode("ascii", "ignore").splitlines()))
        for commit in commits:
            if re.fullmatch(r"[0-9a-f]{40}", commit):
                findings.append(Finding(category, f"git-object:{commit}"))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the TMCRA public release tree")
    parser.add_argument("--history", action="store_true", help="also scan reachable Git history")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    files = _release_files()
    findings = _path_findings(files, manifest) + _content_findings(files)
    if args.history:
        findings.extend(_history_findings())
    findings = sorted(set(findings))
    report = {
        "schema_version": "tmcra.public-release-audit.1",
        "status": "passed" if not findings else "failed",
        "files_checked": len(files),
        "history_checked": bool(args.history),
        "findings": [
            {"category": item.category, "path": item.path, "line": item.line}
            for item in findings
        ],
        "secret_values_printed": False,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"public release audit: {report['status']} ({len(files)} files)")
        for item in findings:
            suffix = f":{item.line}" if item.line else ""
            print(f"- {item.category}: {item.path}{suffix}")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
