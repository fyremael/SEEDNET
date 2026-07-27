from pathlib import Path


FORBIDDEN_MATH_FRAGMENTS = (
    r"\[",
    r"\]",
    r"\operatorname",
)


def test_readme_uses_github_math_syntax_and_exact_weight_definition():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")

    for fragment in FORBIDDEN_MATH_FRAGMENTS:
        assert fragment not in text

    assert text.count("$$") >= 2
    assert text.count("$$") % 2 == 0
    assert r"u_{n,k}=H(s,nK+k)\in[0,1)" in text
    assert r"\sqrt{\frac{12}{K}}" in text
    assert r"u_{n,k}-\frac{1}{2}" in text
    assert r"Y=XW^\top" in text
