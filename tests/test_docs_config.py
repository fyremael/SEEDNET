from pathlib import Path


def test_material_icon_extension_is_enabled():
    root = Path(__file__).resolve().parents[1]
    config = (root / "mkdocs.yml").read_text(encoding="utf-8")
    index = (root / "docs" / "index.md").read_text(encoding="utf-8")

    assert ":material-memory:" in index
    assert "pymdownx.emoji:" in config
    assert "material.extensions.emoji.twemoji" in config
    assert "material.extensions.emoji.to_svg" in config
