# Substack Workflow

Recommended baseline: **semi-automation**

## Why this path

Substack is designed for a write -> render -> paste workflow.

This is the most stable path for us because:

- we keep the source of truth in repo as Markdown
- we get a clean HTML preview before publishing
- Substack's editor handles pasted rich HTML and images well
- we avoid brittle API or headless-editor automation too early

## Workflow

1. Write in VS Code as Markdown
2. Render to HTML
3. Open the HTML in a browser
4. Copy the rendered content
5. Paste into the Substack editor

## Render command

```bash
python scripts/render_substack_html.py docs/lab-notes/2026-04-14-larynx-problem-substack-positioning.md
```

This writes:

```text
docs/lab-notes/2026-04-14-larynx-problem-substack-positioning.html
```

The renderer now embeds local images directly into the HTML file, so the preview is self-contained instead of depending on relative image paths.

## Notes

- Keep images referenced with relative paths from the Markdown file when possible.
- Treat the Markdown file as canonical; HTML is a render artifact for preview and paste.
- Use PDF only when a print-style companion is useful. For Substack itself, HTML is the primary export.
- If Substack formatting looks slightly off after paste, fix it in Substack rather than adding complexity to the renderer.

## Current rule

Do not automate publishing yet.

The correct baseline is:

```text
Markdown in repo -> HTML preview -> copy/paste into Substack
```
