#!/usr/bin/env python3
"""Generate thesis-safe visualization assets for benchmark prompt structure."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "docs" / "visualizations"

sys.path.insert(0, str(REPO_ROOT / "scorer"))
from run_benchmark import (
    COMPONENT_RULES,
    UNDERSTANDING_SYSTEM_PROMPT,
    UNDERSTANDING_PROMPT_TEMPLATE,
    AGENTIC_SYSTEM_PROMPT,
    AGENTIC_PROMPT_TEMPLATE,
)

FIGURE_BASENAME = "benchmark_prompt_visualization"
TITLE = "Benchmark Input Prompt Structure"

MAIN_LAYERS = [
    (
        "System message",
        "Role, constraints, and response discipline supplied outside the user prompt body.",
    ),
    (
        "Board configuration JSON",
        "Canonical board state: geometry, hoppers, catchers, components, bit states, and gear groups.",
    ),
    (
        "Task objective or question",
        "Either a procedural understanding question or an agentic construction objective.",
    ),
    (
        "Component rules",
        "Domain rules for ramps, bits, gear bits, crossovers, triggers, interceptors, and catchers.",
    ),
    (
        "Expected response / tool workflow",
        "JSON answer format for understanding, or required simulator-tool workflow for synthesis.",
    ),
]

VARIANTS = [
    (
        "Procedural understanding",
        ["Question type / expected answer format", "Answer board-behavior questions"],
    ),
    (
        "Agentic synthesis",
        ["Available parts", "Target behavior", "Tool-based place, run, adjust loop"],
    ),
]


COLORS = {
    "background": "#F6F7F8",
    "surface": "#FFFFFF",
    "surface_alt": "#F9FAFB",
    "border": "#C7CCD4",
    "text": "#111827",
    "muted": "#4B5563",
    "quiet": "#6B7280",
    "accent": "#2F6FED",
    "secondary": "#AAB2BD",
}


def _asset_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "html": output_dir / f"{FIGURE_BASENAME}.html",
        "svg": output_dir / f"{FIGURE_BASENAME}.svg",
        "png": output_dir / f"{FIGURE_BASENAME}.png",
        "markdown": output_dir / f"{FIGURE_BASENAME}.md",
    }


def _wrapped_tspans(text: str, *, x: int, y: int, width: int, line_height: int = 22) -> str:
    lines = textwrap.wrap(text, width=width, break_long_words=False)
    return "\n".join(
        f'<tspan x="{x}" y="{y + idx * line_height}">{escape(line)}</tspan>'
        for idx, line in enumerate(lines)
    )


def _svg_text(x: int, y: int, text: str, *, css_class: str, anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" class="{css_class}" text-anchor="{anchor}">{escape(text)}</text>'


def _build_svg() -> str:
    layer_blocks: list[str] = []
    x, width, height = 420, 760, 88
    layer_y = [130, 245, 360, 475, 590]

    for idx, ((label, body), y) in enumerate(zip(MAIN_LAYERS, layer_y)):
        fill = COLORS["surface"] if idx % 2 == 0 else COLORS["surface_alt"]
        accent = COLORS["accent"] if idx == 0 else COLORS["secondary"]
        layer_blocks.append(
            f'''
  <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" class="layer" fill="{fill}"/>
  <rect x="{x}" y="{y}" width="18" height="{height}" rx="18" fill="{accent}"/>
  {_svg_text(x + 42, y + 34, label, css_class="layer-title")}
  <text x="{x + 42}" y="{y + 64}" class="layer-body">{_wrapped_tspans(body, x=x + 42, y=y + 64, width=78, line_height=18)}</text>
'''
        )

    arrows = []
    for start_y, end_y in zip([218, 333, 448, 563], [245, 360, 475, 590]):
        arrows.append(
            f'<line x1="800" y1="{start_y}" x2="800" y2="{end_y - 12}" class="arrow" marker-end="url(#arrowhead)"/>'
        )

    variant_blocks = []
    variant_specs = [(180, 715, 540, VARIANTS[0]), (880, 715, 540, VARIANTS[1])]
    for vx, vy, vw, (title, bullets) in variant_specs:
        bullet_text = "\n".join(
            _svg_text(vx + 34, vy + 66 + idx * 26, f"• {bullet}", css_class="variant-body")
            for idx, bullet in enumerate(bullets)
        )
        variant_blocks.append(
            f'''
  <rect x="{vx}" y="{vy}" width="{vw}" height="140" rx="18" class="variant"/>
  <rect x="{vx}" y="{vy}" width="{vw}" height="6" rx="3" fill="{COLORS['accent']}" opacity="0.85"/>
  {_svg_text(vx + 34, vy + 38, title, css_class="variant-title")}
  {bullet_text}
'''
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" role="img" aria-labelledby="title desc">
  <title id="title">{TITLE}</title>
  <desc id="desc">Layered anatomy of the benchmark input prompt, with procedural understanding and agentic synthesis variants.</desc>
  <defs>
    <marker id="arrowhead" markerWidth="12" markerHeight="8" refX="10" refY="4" orient="auto">
      <path d="M 0 0 L 12 4 L 0 8 z" fill="{COLORS['accent']}"/>
    </marker>
    <style>
      .title {{ font: 700 38px Georgia, 'Times New Roman', serif; fill: {COLORS['text']}; }}
      .subtitle {{ font: 20px Georgia, 'Times New Roman', serif; fill: {COLORS['muted']}; }}
      .scaffold {{ font: 18px Georgia, 'Times New Roman', serif; fill: {COLORS['quiet']}; }}
      .layer {{ stroke: {COLORS['border']}; stroke-width: 2; }}
      .layer-title {{ font: 700 24px Georgia, 'Times New Roman', serif; fill: {COLORS['text']}; }}
      .layer-body {{ font: 18px Arial, sans-serif; fill: {COLORS['muted']}; }}
      .variant {{ fill: {COLORS['surface']}; stroke: {COLORS['border']}; stroke-width: 2; }}
      .variant-title {{ font: 700 23px Georgia, 'Times New Roman', serif; fill: {COLORS['text']}; }}
      .variant-body {{ font: 18px Arial, sans-serif; fill: {COLORS['muted']}; }}
      .arrow {{ stroke: {COLORS['accent']}; stroke-width: 3; opacity: 0.76; }}
      .footer {{ font: 15px Arial, sans-serif; fill: {COLORS['quiet']}; }}
    </style>
  </defs>
  <rect width="1600" height="900" fill="{COLORS['background']}"/>
  {_svg_text(800, 58, TITLE, css_class="title", anchor="middle")}
  <line x1="250" y1="84" x2="1350" y2="84" stroke="{COLORS['accent']}" stroke-width="4" stroke-linecap="round"/>
  {_svg_text(800, 116, "Layered prompt anatomy for two benchmark prompt variants", css_class="subtitle", anchor="middle")}
  {_svg_text(190, 176, "Shared prompt scaffold", css_class="scaffold")}
  <line x1="220" y1="200" x2="220" y2="680" stroke="{COLORS['accent']}" stroke-width="3" opacity="0.55"/>
  {''.join(layer_blocks)}
  {''.join(arrows)}
  {''.join(variant_blocks)}
  {_svg_text(800, 882, "Academic grayscale composition with one blue accent; suitable for direct insertion into thesis slides.", css_class="footer", anchor="middle")}
</svg>
'''


def _draw_matplotlib_text(ax, x: float, y: float, text: str, *, size: float, weight: str = "normal", color: str | None = None, ha: str = "left") -> None:
    ax.text(x, y, text, fontsize=size, fontweight=weight, color=color or COLORS["text"], ha=ha, va="top")


def _write_png(path: Path) -> None:
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS["background"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1600)
    ax.set_ylim(900, 0)
    ax.axis("off")

    _draw_matplotlib_text(ax, 800, 38, TITLE, size=24, weight="bold", ha="center")
    ax.plot([250, 1350], [84, 84], color=COLORS["accent"], linewidth=3, solid_capstyle="round")
    _draw_matplotlib_text(ax, 800, 96, "Layered prompt anatomy for two benchmark prompt variants", size=12, color=COLORS["muted"], ha="center")

    ax.text(190, 176, "Shared prompt scaffold", fontsize=10.5, color=COLORS["quiet"], va="top")
    ax.plot([220, 220], [200, 680], color=COLORS["accent"], linewidth=2, alpha=0.55)

    x, width, height = 420, 760, 88
    for idx, ((label, body), y) in enumerate(zip(MAIN_LAYERS, [130, 245, 360, 475, 590])):
        fill = COLORS["surface"] if idx % 2 == 0 else COLORS["surface_alt"]
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.01,rounding_size=18", linewidth=1.2, edgecolor=COLORS["border"], facecolor=fill))
        ax.add_patch(FancyBboxPatch((x, y), 18, height, boxstyle="round,pad=0.0,rounding_size=18", linewidth=0, facecolor=COLORS["accent"] if idx == 0 else COLORS["secondary"]))
        _draw_matplotlib_text(ax, x + 42, y + 18, label, size=13, weight="bold")
        _draw_matplotlib_text(ax, x + 42, y + 50, textwrap.fill(body, width=82), size=9.5, color=COLORS["muted"])

    for start_y, end_y in zip([218, 333, 448, 563], [245, 360, 475, 590]):
        ax.add_patch(FancyArrowPatch((800, start_y), (800, end_y - 12), arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color=COLORS["accent"], alpha=0.76))

    for vx, vy, vw, (title, bullets) in [(180, 715, 540, VARIANTS[0]), (880, 715, 540, VARIANTS[1])]:
        ax.add_patch(FancyBboxPatch((vx, vy), vw, 140, boxstyle="round,pad=0.01,rounding_size=18", linewidth=1.2, edgecolor=COLORS["border"], facecolor=COLORS["surface"]))
        ax.add_patch(FancyBboxPatch((vx, vy), vw, 6, boxstyle="round,pad=0.0,rounding_size=3", linewidth=0, facecolor=COLORS["accent"], alpha=0.85))
        _draw_matplotlib_text(ax, vx + 34, vy + 22, title, size=13, weight="bold")
        for idx, bullet in enumerate(bullets):
            _draw_matplotlib_text(ax, vx + 34, vy + 56 + idx * 26, f"• {bullet}", size=9.6, color=COLORS["muted"])

    ax.text(800, 882, "Academic grayscale composition with one blue accent; suitable for direct insertion into thesis slides.", fontsize=8.8, color=COLORS["quiet"], ha="center", va="top")
    fig.savefig(path, format="png", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def _write_html(path: Path, svg_text: str) -> None:
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{TITLE}</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ margin: 0; background: {COLORS['background']}; color: {COLORS['text']}; font-family: Georgia, 'Times New Roman', serif; }}
    .page {{ max-width: 1420px; margin: 0 auto; padding: 24px; }}
    .card {{ background: white; border: 1px solid #D7DCE3; box-shadow: 0 10px 30px rgba(17,24,39,0.06); padding: 14px; }}
    h1 {{ margin: 0 0 10px; font-size: 20px; font-weight: 700; }}
    p {{ margin: 10px 0 0; color: {COLORS['muted']}; line-height: 1.45; }}
    svg {{ width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
  <main class=\"page\">
    <h1>{TITLE}</h1>
    <div class=\"card\">{svg_text}</div>
    <p>Visualization of the benchmark prompt anatomy spanning system message, board configuration JSON, task objective or question, component rules, and expected response / tool workflow. Variant callouts show Procedural understanding and Agentic synthesis.</p>
  </main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _write_markdown(path: Path, assets: dict[str, Path]) -> None:
    markdown = f"""# {TITLE}

Generated assets:

- HTML: `{assets['html'].name}`
- SVG: `{assets['svg'].name}`
- PNG: `{assets['png'].name}`
- Markdown: `{assets['markdown'].name}`
- Example Prompt: `{assets['example'].name}`

Generate or refresh the files with:

```bash
uv run python scripts/generate_prompt_visualization.py
```

The diagram uses a layered prompt anatomy:

1. System message
2. Board configuration JSON
3. Task objective or question
4. Component rules
5. Expected response / tool workflow

It highlights two prompt variants:

- Procedural understanding: question type, question text, expected answer format
- Agentic synthesis: available parts, target behavior, simulator-tool workflow

The figure is academic, clean, and thesis-safe: grayscale surfaces with a single blue accent.
"""
    path.write_text(markdown, encoding="utf-8")


def _load_example_challenge() -> tuple[dict, dict]:
    """Load a real challenge and its question for the example prompt."""
    challenges_dir = REPO_ROOT / "tasks" / "official" / "challenges" / "json"
    questions_dir = REPO_ROOT / "tasks" / "official" / "questions"

    challenge_path = challenges_dir / "tt-official-ch01.json"
    question_path = questions_dir / "tt-official-ch01_questions.json"

    challenge = json.loads(challenge_path.read_text(encoding="utf-8"))
    questions = json.loads(question_path.read_text(encoding="utf-8"))

    return challenge, questions["questions"][0]


def _build_example_prompt(target_dir: Path) -> Path:
    """Generate an example prompt with real challenge content."""
    challenge, question = _load_example_challenge()

    board_json = json.dumps(challenge["board"], indent=2)
    question_type = question["type"]
    question_text = question["question"]

    understanding_prompt = UNDERSTANDING_PROMPT_TEMPLATE.format(
        board_json=board_json,
        COMPONENT_RULES=COMPONENT_RULES,
        question_type=question_type,
        question=question_text,
        answer_format='{{"answer": "...", "reasoning": "..."}}',
    )

    available_parts = challenge.get("available_parts", {})
    parts_lines = [f"  - {part}: {count}" for part, count in available_parts.items() if count > 0]
    available_parts_str = "\n".join(parts_lines) if parts_lines else "  (none)"

    agentic_prompt = AGENTIC_PROMPT_TEMPLATE.format(
        board_json=board_json,
        available_parts=available_parts_str,
        target_behavior=challenge["objective"],
        COMPONENT_RULES=COMPONENT_RULES,
    )

    example_md = f'''# Example Benchmark Prompt

This file shows actual prompts generated for challenge **tt-official-ch01** ("{challenge['title']}").

---

## Prompt Variant 1: Procedural Understanding

### System Prompt
```
{UNDERSTANDING_SYSTEM_PROMPT}
```

### Full User Prompt
```
{understanding_prompt}
```

---

## Prompt Variant 2: Agentic Synthesis

### System Prompt
```
{AGENTIC_SYSTEM_PROMPT}
```

### Full User Prompt
```
{agentic_prompt}
```

---

## Challenge Details

- **Task ID:** {challenge['task_id']}
- **Title:** {challenge['title']}
- **Objective:** {challenge['objective']}
- **Question Type (Understanding):** {question_type}
'''

    example_path = target_dir / "benchmark_prompt_example.md"
    example_path.write_text(example_md, encoding="utf-8")
    return example_path


def generate_visualization_assets(output_dir: Path | None = None) -> dict[str, Path]:
    """Generate HTML, SVG, PNG, Markdown, and example prompt assets."""
    target_dir = output_dir or OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    assets = _asset_paths(target_dir)

    example_path = _build_example_prompt(target_dir)
    assets["example"] = example_path

    svg_text = _build_svg()
    assets["svg"].write_text(svg_text, encoding="utf-8")
    _write_png(assets["png"])
    _write_html(assets["html"], svg_text)
    _write_markdown(assets["markdown"], assets)

    return assets


def main() -> None:
    assets = generate_visualization_assets()
    for kind, path in assets.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
