#! /usr/bin/env bash
git config --global --add safe.directory /workspaces/gensor

# Install Dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install --install-hooks

mkdir -p .logs
