from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.generate_prompt_visualization import OUTPUT_DIR, generate_visualization_assets


EXPECTED_NAMES = {
    "html": "benchmark_prompt_visualization.html",
    "svg": "benchmark_prompt_visualization.svg",
    "png": "benchmark_prompt_visualization.png",
    "markdown": "benchmark_prompt_visualization.md",
    "example": "benchmark_prompt_example.md",
}


def test_default_output_dir_points_to_docs_visualizations():
    assert OUTPUT_DIR == REPO_ROOT / "docs" / "visualizations"


def test_generate_visualization_assets_returns_expected_paths(tmp_path):
    assets = generate_visualization_assets(output_dir=tmp_path)

    assert set(assets) == {"html", "svg", "png", "markdown", "example"}
    assert all(isinstance(path, Path) for path in assets.values())

    for key, name in EXPECTED_NAMES.items():
        assert assets[key] == tmp_path / name


def test_generate_visualization_assets_writes_expected_files(tmp_path):
    assets = generate_visualization_assets(output_dir=tmp_path)

    for path in assets.values():
        assert path.exists()
        assert path.stat().st_size > 0

    assert assets["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_html_and_svg_contain_expected_prompt_labels(tmp_path):
    assets = generate_visualization_assets(output_dir=tmp_path)

    html = assets["html"].read_text(encoding="utf-8")
    svg = assets["svg"].read_text(encoding="utf-8")

    labels = [
        "Benchmark Input Prompt Structure",
        "System message",
        "Board configuration JSON",
        "Task objective or question",
        "Component rules",
        "Expected response / tool workflow",
        "Question type / expected answer format",
        "Available parts",
        "Target behavior",
    ]

    assert "<svg" in svg
    assert "<text" in svg
    for label in labels:
        assert label in html
        assert label in svg


def test_html_embeds_valid_inline_svg_without_xml_preamble(tmp_path):
    assets = generate_visualization_assets(output_dir=tmp_path)

    html = assets["html"].read_text(encoding="utf-8")

    assert "<?xml" not in html
    assert "<!DOCTYPE svg" not in html
    assert html.count("<svg") == 1


def test_variant_cards_have_room_for_all_bullets(tmp_path):
    assets = generate_visualization_assets(output_dir=tmp_path)

    svg = assets["svg"].read_text(encoding="utf-8")

    assert 'height="140"' in svg
    assert 'Tool-based place, run, adjust loop' in svg


def test_generated_text_assets_are_reproducible(tmp_path):
    first = generate_visualization_assets(output_dir=tmp_path)
    first_svg = first["svg"].read_text(encoding="utf-8")
    first_html = first["html"].read_text(encoding="utf-8")

    second = generate_visualization_assets(output_dir=tmp_path)
    second_svg = second["svg"].read_text(encoding="utf-8")
    second_html = second["html"].read_text(encoding="utf-8")

    assert second_svg == first_svg
    assert second_html == first_html
    assert "Created" not in second_svg
    assert "date" not in second_svg.lower()


def test_markdown_mentions_outputs_generation_command_and_prompt_variants(tmp_path):
    assets = generate_visualization_assets(output_dir=tmp_path)

    markdown = assets["markdown"].read_text(encoding="utf-8")

    for name in EXPECTED_NAMES.values():
        assert name in markdown
    assert "uv run python scripts/generate_prompt_visualization.py" in markdown
    assert "two prompt variants" in markdown.lower()
    assert "Procedural understanding" in markdown
    assert "Agentic synthesis" in markdown


def test_example_prompt_contains_real_content(tmp_path):
    assets = generate_visualization_assets(output_dir=tmp_path)

    example_path = tmp_path / "benchmark_prompt_example.md"
    assert example_path.exists()

    content = example_path.read_text(encoding="utf-8")

    assert "tt-official-ch01" in content
    assert "You are an expert Turing Tumble analyst" in content or "You are a Turing Tumble solver agent" in content
    assert "ball_hoppers" in content
    assert "ramp_right" in content or "ramp_left" in content
    assert "blue balls" in content.lower()
    assert "GRAVITY" in content or "Gravity" in content
