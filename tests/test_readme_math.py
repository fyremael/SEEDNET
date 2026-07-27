from pathlib import Path


FORBIDDEN_MATH_FRAGMENTS = (
    r"\[",
    r"\]",
    r"\operatorname",
)


def test_readme_uses_github_math_syntax():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")

    for fragment in FORBIDDEN_MATH_FRAGMENTS:
        assert fragment not in text

    assert text.count("$$") >= 2
    assert text.count("$$") % 2 == 0
