"""Hash-bind a committed evaluation tree and explicit per-checkpoint inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


def build(project: Path, destination: Path, attachments: dict[str, Path]) -> Path:
    repository = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=project, text=True
        ).strip()
    )
    relative = project.relative_to(repository)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", str(relative)],
        cwd=repository,
        text=True,
    )
    if dirty:
        raise ValueError(
            "Commit the evaluation implementation before freezing its bundle"
        )
    paths = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--", str(relative)], cwd=repository
        )
        .decode()
        .split("\0")
    )
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_repository": "https://github.com/plantcad/plantcad2",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip(),
        "inference_repository": "https://github.com/Open-Athena/marin-dna",
        "inference_commit": "6c07df025a185c6e39278327ba5b678dcfd93c86",
        "files": {},
    }

    def add(path: Path, name: str) -> None:
        target = destination / name
        if target.exists() or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Conflicting or unsafe bundle path")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        with target.open("rb") as stream:
            manifest["files"][name] = hashlib.file_digest(stream, "sha256").hexdigest()

    for path in paths:
        if not path:
            continue
        name = str(Path(path).relative_to(relative))
        if name.startswith(("artifacts/", "tests/")):
            continue
        add(repository / path, name)
    for name, path in attachments.items():
        add(path, name)
    (destination / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    archive = destination.with_suffix(".tar.gz")
    if archive.exists():
        raise ValueError("Refusing to replace an existing source archive")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(destination)))
    return archive
