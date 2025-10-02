#! /bin/sh
# for local builds only:

uv clean && rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info
uv build --link-mode=copy
uv pip install .
uv run pytest -v