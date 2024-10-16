#! /usr/bin/env bash

# Install Dependencies
poetry install --with dev

# Install pre-commit hooks
poetry run pre-commit install --install-hooks

git config --global filter.nbstripout.clean /workspaces/gensor/.venv/bin/nbstripout

mkdir -p .logs
