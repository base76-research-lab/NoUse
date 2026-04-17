#!/usr/bin/env python3
"""Render a Markdown note to a standalone HTML preview for Substack copy/paste.

Recommended workflow:
1. Write in VS Code (`.md`)
2. Render to `.html`
3. Open the HTML in a browser
4. Copy the rendered content and paste into the Substack editor
"""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _default_output(input_path: Path) -> Path:
    return input_path.with_suffix(".html")


def _html_title(input_path: Path) -> str:
    text = input_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            if line.startswith("title:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
    return input_path.stem


def _build_style() -> str:
    return """
:root {
  --page: #fcfaf5;
  --surface: #f6f1e6;
  --line: #d9cfbf;
  --ink: #14110f;
  --muted: #6f685f;
  --accent: #daa42d;
}

* { box-sizing: border-box; }

html { background: var(--page); }

body {
  margin: 0;
  color: var(--ink);
  background: var(--page);
  font-family: Georgia, "Times New Roman", serif;
  line-height: 1.68;
}

main {
  max-width: 760px;
  margin: 0 auto;
  padding: 56px 28px 72px;
}

p,
li,
blockquote {
  font-size: 19px;
}

h1,
h2,
h3,
h4 {
  line-height: 1.15;
  font-weight: 600;
  letter-spacing: -0.01em;
}

h1 {
  font-size: 46px;
  margin: 0 0 12px;
}

h2 {
  font-size: 30px;
  margin: 48px 0 16px;
}

h3 {
  font-size: 23px;
  margin: 28px 0 10px;
}

p,
ul,
ol,
blockquote,
pre,
table {
  margin: 0 0 18px;
}

ul,
ol {
  padding-left: 28px;
}

li + li {
  margin-top: 8px;
}

blockquote {
  margin-left: 0;
  padding: 14px 18px;
  border-left: 4px solid var(--accent);
  background: linear-gradient(90deg, rgba(218, 164, 45, 0.10), rgba(218, 164, 45, 0.03));
}

code {
  font-family: "DejaVu Sans Mono", "SFMono-Regular", Consolas, monospace;
  font-size: 0.9em;
  background: rgba(20, 17, 15, 0.06);
  padding: 0.12em 0.32em;
  border-radius: 4px;
}

pre {
  overflow-x: auto;
  padding: 16px 18px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fff;
}

pre code {
  background: transparent;
  padding: 0;
}

hr {
  border: 0;
  border-top: 1px solid var(--line);
  margin: 34px 0;
}

a {
  color: inherit;
}

img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 28px auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 17px;
}

th,
td {
  text-align: left;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}

.substack-meta {
  margin: 0 0 28px;
  color: var(--muted);
  font-size: 15px;
}

.substack-meta p {
  margin: 4px 0;
  font-size: inherit;
}

@media (max-width: 720px) {
  main {
    padding: 40px 20px 56px;
  }

  p,
  li,
  blockquote {
    font-size: 18px;
  }

  h1 {
    font-size: 36px;
  }

  h2 {
    font-size: 26px;
  }
}
""".strip()


def _embed_local_images(body_html: str, base_dir: Path) -> str:
    pattern = re.compile(r'(<img\b[^>]*\bsrc=")([^"]+)(")', re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        prefix, src, suffix = match.groups()
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)

        image_path = (base_dir / src).resolve()
        if not image_path.exists() or not image_path.is_file():
            return match.group(0)

        mime_type, _ = mimetypes.guess_type(image_path.name)
        mime_type = mime_type or "application/octet-stream"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        embedded_src = f"data:{mime_type};base64,{encoded}"
        return f"{prefix}{embedded_src}{suffix}"

    return pattern.sub(replace, body_html)


def render_markdown(input_path: Path, output_path: Path) -> None:
    if shutil.which("pandoc") is None:
        raise RuntimeError("pandoc is required but was not found on PATH")

    style = _build_style()
    page_title = html.escape(_html_title(input_path))

    # Pandoc cannot ingest inline CSS via --css as a data URL reliably across versions.
    # We therefore inject a tiny HTML wrapper with the style inline.
    body_result = subprocess.run(
        [
            "pandoc",
            str(input_path),
            "--from",
            "markdown+yaml_metadata_block",
            "--to",
            "html5",
            "--wrap=none",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    body_html = _embed_local_images(body_result.stdout.strip(), input_path.parent)
    output_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title}</title>
  <style>
{style}
  </style>
</head>
<body>
  <main>
{body_html}
  </main>
</body>
</html>
"""
    output_path.write_text(output_html, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Markdown note to Substack-ready preview HTML."
    )
    parser.add_argument("input", type=Path, help="Path to the Markdown source file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path (defaults to input path with .html suffix)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = args.input.expanduser().resolve()
    output_path = (args.output or _default_output(input_path)).expanduser().resolve()

    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        render_markdown(input_path, output_path)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or str(exc), file=sys.stderr)
        return exc.returncode or 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Rendered {input_path} -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
