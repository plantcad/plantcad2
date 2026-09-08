#!/usr/bin/env python3
"""Upload an evaluation artifact tree to an HF model repo and verify it exactly."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="plantcad/marindna-exp472")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--path-in-repo", required=True)
    parser.add_argument("--commit-message", required=True)
    args = parser.parse_args()

    token = os.environ["HUGGING_FACE_HUB_TOKEN"]
    api = HfApi(token=token)
    account = api.whoami()["name"]
    if account != "eczech":
        raise RuntimeError(f"Expected Hugging Face account eczech, got {account}")
    api.repo_info(repo_id=args.repo_id, repo_type="model")

    root = args.local_dir.resolve()
    files = {path.relative_to(root).as_posix(): path.stat().st_size for path in root.rglob("*") if path.is_file()}
    if not files:
        raise RuntimeError(f"No artifact files found under {root}")
    destination = args.path_in_repo.strip("/")
    destination_prefix = f"{destination}/"
    existing = [path for path in api.list_repo_files(args.repo_id, repo_type="model") if path.startswith(destination_prefix)]
    if existing:
        raise RuntimeError(f"Destination is not empty: {existing[:10]}")

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    commit = api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=root, path_in_repo=destination, commit_message=args.commit_message)
    uploaded = {
        entry.path.removeprefix(destination_prefix): entry.size
        for entry in api.list_repo_tree(args.repo_id, path_in_repo=destination, recursive=True, expand=True, repo_type="model")
        if hasattr(entry, "size") and entry.path.startswith(destination_prefix)
    }
    if uploaded != files:
        raise RuntimeError(f"Uploaded tree mismatch: missing={sorted(set(files) - set(uploaded))}, unexpected={sorted(set(uploaded) - set(files))}, size_mismatches={sorted(path for path in set(files) & set(uploaded) if files[path] != uploaded[path])}")
    print(json.dumps({"commit_url": commit.commit_url, "path_in_repo": destination, "file_count": len(files), "bytes": sum(files.values())}, indent=2))


if __name__ == "__main__":
    main()
