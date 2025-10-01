#! /bin/sh

uv clean && rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info
uv build
uv pip install .
uv run pytest -v