#! /usr/bin/env bash

# Install Dependencies
poetry install --with dev

# Install pre-commit hooks
poetry run pre-commit install --install-hooks

# Set up git nbstripout filter with the correct path
NBSTRIPOUT_PATH="$(poetry run which nbstripout)"  # Get the nbstripout path from the poetry virtual environment

git config --global filter.nbstripout.clean "$NBSTRIPOUT_PATH clean --stdin"
git config --global filter.nbstripout.smudge "cat"
git config --global filter.nbstripout.required true

mkdir -p .logs
