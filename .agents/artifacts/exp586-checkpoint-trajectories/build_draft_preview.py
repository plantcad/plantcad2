"""Render the unpublished overview/checkpoint draft as self-contained HTML."""

import argparse
import base64
import hashlib
import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

from markdown_it import MarkdownIt
from mdit_py_plugins.footnote import footnote_plugin

HERE = Path(__file__).resolve().parent
FIGURES = ["benchmark-comparison", "loss-comparison", "checkpoint-tokens", "checkpoint-progress", "checkpoint-changes-primary"]


class Structure(HTMLParser):
    """Check that the overview is first and only technical figures collapse."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.images: list[tuple[int, str]] = []
        self.tables = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "details":
            self.depth += 1
            assert "open" not in dict(attrs)
        elif tag == "img":
            self.images.append((self.depth, str(dict(attrs)["src"])))
        elif tag == "table":
            self.tables += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "details":
            self.depth -= 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / "figures").mkdir(exist_ok=True)
    markdown = (HERE / "draft.md").read_text()
    assert markdown.startswith("WIP\n\n")
    assert not re.search(r"\b(we|our|ours|us|Natural|likelihood)\b", markdown, re.IGNORECASE)
    assert "Composite −SV" not in markdown and "🤖" not in markdown
    body = MarkdownIt("commonmark", {"html": True}).enable("table").use(footnote_plugin).render(markdown)
    assert 'class="footnote-ref"' in body and 'class="footnotes' in body
    assert 'href="https://proceedings.mlr.press/v162/clark22a.html"' in body
    assert 'href="https://marin.community/analysis/scaling-laws-analysis/"' in body
    structure = Structure()
    structure.feed(body)
    assert structure.depth == 0 and structure.tables == 2
    assert structure.images == [(depth, f"figures/{name}.png") for depth, name in zip([0, 0, 0, 1, 1], FIGURES, strict=True)]
    hashes = {}
    for name in FIGURES:
        for extension in ("png", "svg"):
            path = HERE / "figures" / f"{name}.{extension}"
            relative = path.relative_to(HERE)
            shutil.copy2(path, args.output / relative)
            hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        encoded = base64.b64encode((HERE / "figures" / f"{name}.png").read_bytes()).decode()
        body = body.replace(f'src="figures/{name}.png"', f'src="data:image/png;base64,{encoded}"')
    css = """
body{font:16px/1.55 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;color:#24292f;max-width:1280px;margin:28px auto;padding:0 24px}
img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}
td,th{border:1px solid #d0d7de;padding:8px;text-align:left;vertical-align:top}th{background:#f6f8fa}
details{border:1px solid #d0d7de;border-radius:6px;padding:14px}summary{cursor:pointer;font-weight:600}
code{background:#f6f8fa;padding:2px 4px}.draft{padding:12px;border:1px solid #d0d7de;border-radius:6px;background:#f6f8fa;margin-bottom:20px}
"""
    banner = '<div class="draft">Draft for PlantCAD2 PR #22 — the live results comment has not been updated.</div>'
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Draft: MarinMoE comparisons</title><style>' + css + '</style></head><body>' + banner + body + '</body></html>'
    (args.output / "preview.html").write_text(html)
    (args.output / "post.md").write_text(markdown)
    manifest = {
        "destination": "https://github.com/plantcad/plantcad2/pull/22#issuecomment-6647191065",
        "status": "Unpublished draft; no live comment changes",
        "source": "draft.md",
        "post_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
        "figure_sha256": hashes,
        "image_details_depths": [depth for depth, _ in structure.images],
        "benchmark_provenance": json.loads((HERE / "benchmark-comparison-provenance.json").read_text()),
    }
    (args.output / "draft-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Draft preview: {args.output / 'preview.html'}")


if __name__ == "__main__":
    main()
