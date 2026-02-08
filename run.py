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
    """Run the EmbeddedEarth application."""
    print("🚀 Launching EmbeddedEarth...")
    
    # Check dependencies
    try:
        import streamlit
        import geemap
        import open_clip
    except ImportError as e:
        print("\n" + "=" * 70)
        print(f"❌ Missing critical dependency: {e.name}")
        print("=" * 70)
        if e.name == 'open_clip':
            print("\nThe custom 'open_clip' fork (DOFA-CLIP) is required.")
            print("\nTo fix this, run the installation script:")
            print("  ./install.sh")
            print("\nOr manually install:")
            print("  git clone https://github.com/xiong-zhitong/DOFA-CLIP.git")
            print("  cd DOFA-CLIP/open_clip")
            print("  pip install -e .")
        else:
            print(f"\nPlease install missing dependency:")
            print("  pip install -r requirements.txt")
        print("\nFor detailed diagnostics, run:")
        print("  python validate_installation.py")
        print("=" * 70 + "\n")
        sys.exit(1)
        
    # Run Streamlit
    import streamlit.web.cli as stcli
    app_path = os.path.join(os.path.dirname(__file__), "app/main.py")
    
    sys.argv = ["streamlit", "run", app_path, "--server.port=8501", "--server.address=0.0.0.0"]
    
    print(f"🛰️  Starting EmbeddedEarth...")
    print(f"👉 Open http://localhost:8501 in your browser")
    
    try:
        sys.exit(stcli.main())
    except KeyboardInterrupt:
        print("\n\n👋 EmbeddedEarth stopped.")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install with: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
