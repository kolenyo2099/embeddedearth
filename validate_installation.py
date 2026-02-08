#!/usr/bin/env python3
"""
EmbeddedEarth Installation Validator

Checks if all dependencies and configuration are properly set up.
Run this after installation to diagnose issues.
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

def check_open_clip() -> Tuple[bool, str]:
    """Check if custom open_clip fork is installed."""
    try:
        import open_clip
        return True, "✅ open_clip installed (DOFA-CLIP fork)"
    except ImportError:
        return False, "❌ open_clip NOT installed - run install.sh or manually clone DOFA-CLIP"

def check_core_dependencies() -> List[Tuple[bool, str]]:
    """Check if core Python dependencies are installed."""
    deps = {
        'streamlit': 'Streamlit web framework',
        'torch': 'PyTorch deep learning',
        'ee': 'Google Earth Engine API',
        'geemap': 'Interactive mapping',
        'faiss': 'Vector similarity search',
        'transformers': 'Hugging Face transformers',
        'rasterio': 'Geospatial raster I/O',
    }

    results = []
    for module, description in deps.items():
        try:
            __import__(module)
            results.append((True, f"✅ {description} ({module})"))
        except ImportError:
            results.append((False, f"❌ {description} ({module}) - run: pip install -r requirements.txt"))

    return results

def check_environment_variables() -> List[Tuple[bool, str]]:
    """Check if required environment variables are set."""
    results = []

    # Check for .env file
    env_file = Path('.env')
    if env_file.exists():
        results.append((True, "✅ .env file exists"))
    else:
        results.append((False, "⚠️  .env file not found - copy .env.example to .env and configure"))

    # Check GEE_PROJECT_ID
    gee_id = os.getenv('GEE_PROJECT_ID')
    if gee_id:
        results.append((True, f"✅ GEE_PROJECT_ID set: {gee_id}"))
    else:
        results.append((False, "⚠️  GEE_PROJECT_ID not set - required for fetching satellite imagery"))

    # Check USE_GPU (optional)
    use_gpu = os.getenv('USE_GPU', 'false')
    results.append((True, f"ℹ️  USE_GPU={use_gpu}"))

    # Check HF_TOKEN (optional)
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        results.append((True, "✅ HF_TOKEN set"))
    else:
        results.append((True, "ℹ️  HF_TOKEN not set (optional, only needed for gated models)"))

    return results

def check_directories() -> List[Tuple[bool, str]]:
    """Check if required directories exist."""
    results = []

    # Check DOFA-CLIP directory
    dofa_dir = Path('DOFA-CLIP')
    if dofa_dir.exists():
        results.append((True, "✅ DOFA-CLIP directory exists"))
    else:
        results.append((False, "❌ DOFA-CLIP directory not found - run install.sh"))

    # Check cache directories (created automatically but good to verify)
    cache_dir = Path('cache')
    models_cache_dir = Path('models_cache')

    if cache_dir.exists():
        results.append((True, "✅ cache/ directory exists"))
    else:
        results.append((True, "ℹ️  cache/ will be created on first run"))

    if models_cache_dir.exists():
        results.append((True, "✅ models_cache/ directory exists"))
    else:
        results.append((True, "ℹ️  models_cache/ will be created on first run"))

    return results

def check_gee_credentials() -> Tuple[bool, str]:
    """Check if GEE credentials are configured."""
    gee_creds_dir = Path.home() / '.config' / 'earthengine'
    alt_creds_dir = Path('.gee_credentials')

    if gee_creds_dir.exists() or alt_creds_dir.exists():
        return True, "✅ GEE credentials found"
    return False, "⚠️  GEE credentials not found - you'll need to authenticate on first run"

def main():
    """Run all validation checks."""
    print("=" * 70)
    print("EmbeddedEarth Installation Validator")
    print("=" * 70)
    print()

    all_passed = True

    # Python version
    print("📌 Python Version:")
    passed, msg = check_python_version()
    print(f"  {msg}")
    all_passed = all_passed and passed
    print()

    # Critical: open_clip
    print("📌 Critical Dependency (DOFA-CLIP):")
    passed, msg = check_open_clip()
    print(f"  {msg}")
    all_passed = all_passed and passed
    print()

    # Core dependencies
    print("📌 Core Python Dependencies:")
    results = check_core_dependencies()
    for passed, msg in results:
        print(f"  {msg}")
        all_passed = all_passed and passed
    print()

    # Directories
    print("📌 Required Directories:")
    results = check_directories()
    for passed, msg in results:
        print(f"  {msg}")
        if not passed:
            all_passed = False
    print()

    # Environment variables
    print("📌 Environment Configuration:")
    results = check_environment_variables()
    for passed, msg in results:
        print(f"  {msg}")
        if not passed:
            all_passed = False
    print()

    # GEE credentials
    print("📌 Google Earth Engine Authentication:")
    passed, msg = check_gee_credentials()
    print(f"  {msg}")
    # Don't fail validation for missing GEE creds - will prompt during app
    print()

    # Summary
    print("=" * 70)
    if all_passed:
        print("✅ All critical checks passed! You can run the app with:")
        print("   streamlit run app/main.py")
        print()
        print("ℹ️  Note: You may need to authenticate with Google Earth Engine")
        print("   on first run via the browser.")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print()
        print("Quick fix steps:")
        print("  1. Run: ./install.sh")
        print("  2. Copy: cp .env.example .env")
        print("  3. Edit .env and add your GEE_PROJECT_ID")
        print("  4. Run this script again to verify")
    print("=" * 70)

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
