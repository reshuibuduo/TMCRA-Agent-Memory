#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

LFS_INCLUDE = "models/tmcra_v4_longmemeval_s500_20260715/*.pt"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_file(root: Path, row: dict[str, object]) -> None:
    destination = (root / str(row["destination"])).resolve()
    expected = str(row["sha256"])
    expected_bytes = int(row["bytes"])
    if destination.is_file() and destination.stat().st_size == expected_bytes and digest(destination) == expected:
        print(f"verified {row['id']}: {destination}")
        return
    if row.get("storage") == "git_lfs":
        detail = "file is missing"
        if destination.is_file():
            with destination.open("rb") as handle:
                prefix = handle.read(128)
            if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                detail = "only a Git LFS pointer is present"
            else:
                detail = f"size or SHA-256 mismatch ({destination.stat().st_size} bytes)"
        raise RuntimeError(
            f"Git LFS checkpoint {row['id']} is unavailable: {detail}: {destination}\n"
            f"run from the repository root: git lfs pull --include=\"{LFS_INCLUDE}\""
        )
    if not row.get("url"):
        raise RuntimeError(f"asset {row['id']} has neither a valid local Git LFS file nor a download URL")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    print(f"downloading {row['id']} -> {destination}")
    urllib.request.urlretrieve(str(row["url"]), temporary)
    if temporary.stat().st_size != expected_bytes:
        raise RuntimeError(f"size mismatch for {row['id']}")
    actual = digest(temporary)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {row['id']}: {actual}")
    os.replace(temporary, destination)


def fetch_model(root: Path, row: dict[str, object]) -> None:
    from huggingface_hub import snapshot_download

    destination = root / str(row["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {row['repo_id']}@{row['revision']} -> {destination}")
    snapshot_download(
        repo_id=str(row["repo_id"]),
        revision=str(row["revision"]),
        local_dir=destination,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("configs/assets.lock.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--kind", choices=("all", "dataset", "checkpoint", "model"), default="all")
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.kind in {"all", "dataset", "checkpoint"}:
        selected = [
            row
            for row in payload["files"]
            if args.kind == "all" or row["kind"] == args.kind
        ]
        # Validate repository-owned LFS files before starting any large network
        # download, so an incomplete clone fails quickly and predictably.
        selected.sort(key=lambda row: row.get("storage") != "git_lfs")
        for row in selected:
            fetch_file(args.root, row)
    if args.kind in {"all", "model"}:
        for row in payload["huggingface_models"]:
            fetch_model(args.root, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
