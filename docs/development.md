# Development

## Repository checks

```bash
pip install -e ".[dev,docs]"
pytest
python scripts/check_project.py
mkdocs build --strict
```

CUDA/Triton tests are skipped automatically when the fused environment is not
available.

## Documentation contract

The site is rebuilt when any of these change:

- `docs/**`;
- `src/**`;
- `mkdocs.yml`;
- documentation dependencies in `pyproject.toml`;
- the Pages workflow itself.

Pull requests run a strict documentation build but do not deploy. Pushes to
`main` build and publish the `site/` artifact through GitHub Pages.

## Updating public APIs

When adding a public symbol:

1. export it from `src/seednet/__init__.py`;
2. write a complete Google-style docstring;
3. add tests for its behavioural contract;
4. add or update the appropriate page under `docs/api/`;
5. run `mkdocs build --strict`.

## Local documentation server

```bash
mkdocs serve
```

The development server watches Markdown, configuration, and source docstrings.
