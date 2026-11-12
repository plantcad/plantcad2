"""Keep generated SVGs free of trailing whitespace without changing geometry."""

from pathlib import Path


def normalize_svg(path: Path) -> None:
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
