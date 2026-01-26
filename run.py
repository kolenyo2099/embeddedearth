#!/usr/bin/env python3
"""
EmbeddedEarth - CLI Entry Point

Run the FastAPI backend that serves the Svelte frontend.

Usage:
    python run.py
"""

import sys
import os


def main():
    """Launch the FastAPI application."""
    print("🚀 Launching EmbeddedEarth API...")

    try:
        import uvicorn
    except ImportError as e:
        print(f"❌ Missing dependency: {e.name}")
        print("Please run: ./install.sh")
        sys.exit(1)

    host = os.getenv("EMBEDDED_EARTH_HOST", "0.0.0.0")
    port = int(os.getenv("EMBEDDED_EARTH_PORT", "8501"))

    print("🛰️  Starting EmbeddedEarth...")
    print(f"👉 API available at http://localhost:{port}")
    print("👉 Frontend served from / if the Svelte build exists.")

    try:
        uvicorn.run("app.api:app", host=host, port=port, reload=False)
    except KeyboardInterrupt:
        print("\n\n👋 EmbeddedEarth stopped.")


if __name__ == "__main__":
    main()
