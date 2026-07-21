#!/bin/bash

# EmbeddedEarth Start Script
# Activates the virtual environment and launches the Streamlit application.

# Resolve the project directory from the script's location,
# so it works no matter where it's invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 1. Check that the virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found at ./venv"
    echo "Please run: ./install.sh"
    exit 1
fi

# 2. Activate the virtual environment
echo "📦 Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate

# 3. Launch the app via run.py (handles host/port + helpful messages)
echo "🚀 Starting EmbeddedEarth..."
echo "👉 Open http://localhost:8501 in your browser"
python run.py
