"""Build a self-contained review copy without publishing or replacing live text."""

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
PUBLISHED = HERE.with_name("exp586-rank-composites")


class Structure(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.images = []
        self.details = 0
        self.tables = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "details":
            assert "open" not in dict(attrs)
            self.depth += 1
            self.details += 1
        elif tag == "img":
            self.images.append((self.depth, dict(attrs)["src"]))
        elif tag == "table":
            self.tables += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "details":
            self.depth -= 1
            assert self.depth >= 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)
    markdown = (HERE / "draft.md").read_text()
    source = json.loads((HERE / "live-source.json").read_text())["body"].replace("\r\n", "\n")
    assert markdown.split('<p align="center">', 1)[0] == source.split('<p align="center">', 1)[0]
    body = MarkdownIt("commonmark", {"html": True}).enable("table").use(footnote_plugin).render(markdown)
    structure = Structure()
    structure.feed(body)
    assert structure.depth == 0 and structure.details == 6 and structure.tables == 2
    assert [depth for depth, _ in structure.images] == [0, 0, 1, 0, 1, 1]
    assert len(re.findall('class="footnote-item"', body)) == 3
    image_hashes = {}
    for _, url in structure.images:
        name = Path(url).stem
        folder = HERE if name == "benchmark-comparison" else PUBLISHED
        path = folder / "figures" / f"{name}.png"
        data = path.read_bytes()
        image_hashes[name] = hashlib.sha256(data).hexdigest()
        for extension in ["png", "svg"]:
            shutil.copy2(folder / "figures" / f"{name}.{extension}", args.output / "figures" / f"{name}.{extension}")
        encoded = base64.b64encode(data).decode()
        body = body.replace(f'src="{url}"', f'src="data:image/png;base64,{encoded}"')
    assert len(image_hashes) == 6 and body.count('src="data:image/png;base64,') == 6
    css = """
body{font:16px/1.55 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;color:#24292f;max-width:1280px;margin:28px auto;padding:0 24px}
img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}
td,th{border:1px solid #d0d7de;padding:8px;text-align:left;vertical-align:top}th{background:#f6f8fa}
details{border:1px solid #d0d7de;border-radius:6px;padding:14px}details+details{margin-top:12px}summary{cursor:pointer;font-weight:600}
code{background:#f6f8fa;padding:2px 4px}.draft{padding:12px;border:1px solid #d0d7de;border-radius:6px;background:#f6f8fa;margin-bottom:20px}
"""
    banner = '<div class="draft">Unpublished draft for PlantCAD2 PR #22 — Unweighted Avg Rank gives each task family equal weight; Weighted Avg Rank follows the PlantCAD2 leaderboard. The live comment is unchanged.</div>'
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Draft: unweighted and weighted average ranks</title><style>' + css + '</style></head><body>' + banner + body + '</body></html>'
    checksum = hashlib.sha256(html.encode()).hexdigest()
    filename = f"preview-{checksum[:12]}.html"
    (args.output / filename).write_text(html)
    (args.output / "preview.html").write_text(html)
    (args.output / "draft.md").write_text(markdown)
    manifest = {"status": "Unpublished draft", "preview_file": filename, "html_sha256": checksum,
                "figure_sha256": image_hashes, "closed_details": 6, "tables": 2, "footnotes": 3,
                "draft_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
                "source_updated_at": json.loads((HERE / "live-source.json").read_text())["updated_at"]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output / filename)


if __name__ == "__main__":
    main()
