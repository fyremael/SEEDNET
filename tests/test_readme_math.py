from pathlib import Path


def test_readme_uses_github_math_delimiters():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")

    assert r"\[" not in text
    assert r"\]" not in text
    assert text.count("$$") >= 2
    assert text.count("$$") % 2 == 0
