#!/bin/bash

set -e

VENV_NAME="omr_venv"
PROJECT_ROOT=$(pwd)
export PYTHONPATH="${PROJECT_ROOT}"

mkdir -p .cache/huggingface
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"

if [ -f .env ]; then
    export $(cat .env | xargs)
fi

if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_NAME
    source $VENV_NAME/bin/activate

    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Virtual environment found. Activating..."
    source $VENV_NAME/bin/activate
fi

if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set in .env"
else
    hf auth login --token "$HF_TOKEN" --add-to-git-credential > /dev/null 2>&1 || echo "Git credential helper skipped."
    export HF_TOKEN=$HF_TOKEN
fi

echo "Starting OMR pipeline..."
python3 main.py
echo "Done."
