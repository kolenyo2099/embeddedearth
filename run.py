#!/usr/bin/env python3
"""
ElLocoGIS - CLI Entry Point

Run the Streamlit application with configured settings.

Usage:
    python run.py
    
Or directly:
    streamlit run app/main.py
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    """Launch the Streamlit application."""
    # Get project root
    project_root = Path(__file__).parent.absolute()
    app_path = project_root / "app" / "main.py"
    
    # Ensure we're in the project directory
    os.chdir(project_root)
    
    # Streamlit configuration
    args = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--theme.base=light",
        "--theme.primaryColor=#003262",  # Berkeley Blue
        "--theme.backgroundColor=#FFFFFF",
        "--theme.secondaryBackgroundColor=#F5F5F5",
        "--theme.textColor=#333333",
    ]
    
    print(f"🛰️  Starting ElLocoGIS...")
    print(f"📁 Project root: {project_root}")
    print(f"🌐 Opening browser at http://localhost:8501")
    print("\nPress Ctrl+C to stop the server.\n")
    
    try:
        subprocess.run(args, check=True)
    except KeyboardInterrupt:
        print("\n\n👋 ElLocoGIS stopped.")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install with: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
