# EmbeddedEarth - AI-Powered Satellite Imagery Search

A semantic search engine for satellite imagery using DOFA-CLIP (Dynamic One-For-All CLIP) for multimodal vision-language understanding of Earth Observation data.

## Features

- **Text-to-Image Search**: Describe what you're looking for (e.g., "solar panels", "industrial facility near river")
- **DOFA-CLIP Integration**: Uses wavelength-aware dynamic encoding for multispectral Sentinel-2 imagery
- **Explainable AI**: Grad-CAM heatmaps show what the model is focusing on
- **Interactive Map**: Draw areas of interest directly on the map

## Installation

### Quick Start (Recommended)

The easiest way to install is using the provided script (requires [uv](https://github.com/astral-sh/uv)):

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/EmbeddedEarth.git
cd EmbeddedEarth

# 2. Run installation script
chmod +x install.sh
./install.sh

# 3. Configure environment variables
cp .env.example .env
# Edit .env and add your GEE_PROJECT_ID

# 4. Validate installation
python validate_installation.py

# 5. Run the app
streamlit run app/main.py
```

### Manual Installation (Alternative)

If you prefer manual setup or are on Windows:

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install DOFA-CLIP fork (CRITICAL - must be done first!)
git clone https://github.com/xiong-zhitong/DOFA-CLIP.git
cd DOFA-CLIP/open_clip
pip install -e .
cd ../..

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Configure environment
cp .env.example .env
# Edit .env and add your GEE_PROJECT_ID

# 6. Validate installation
python validate_installation.py
```

## Usage

```bash
# Activate your virtual environment first (if not already active)
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Run the app
streamlit run app/main.py

# Or use the convenience script
python run.py
```

## Requirements

- Python 3.8+
- Google Earth Engine account (for satellite data)
- CUDA-capable GPU recommended for faster inference

## Google Earth Engine Setup

EmbeddedEarth uses Google Earth Engine (GEE) to fetch satellite imagery. You'll need a Google Cloud Project with the Earth Engine API enabled.

1.  **Sign up** for Earth Engine at [earthengine.google.com](https://earthengine.google.com).
2.  **Create a Cloud Project**:
    - Go to the [Google Cloud Console](https://console.cloud.google.com).
    - Create a new project (e.g., `embedded-earth`).
    - Note down your **Project ID**.
3.  **Enable Earth Engine API**:
    - In your project dashboard, search for "Earth Engine API" and enable it.
4.  **Connect in the App**:
    - Enter your Project ID in the app sidebar.
    - Click "Connect to GEE".
    - Follow the browser authentication flow.

## Troubleshooting

### "ModuleNotFoundError: No module named 'open_clip'"

This is the most common issue. The custom DOFA-CLIP fork must be installed separately:

```bash
git clone https://github.com/xiong-zhitong/DOFA-CLIP.git
cd DOFA-CLIP/open_clip
pip install -e .
cd ../..
```

### App works on one machine but not another

Common causes:
1. **Missing DOFA-CLIP installation** - Must be cloned and installed on each machine
2. **Missing .env file** - Copy `.env.example` to `.env` and configure `GEE_PROJECT_ID`
3. **Missing GEE authentication** - Run the app once and follow browser authentication
4. **Different Python versions** - Requires Python 3.8+

**Solution:** Run `python validate_installation.py` to diagnose issues

### "Earth Engine not initialized"

You need to authenticate with Google Earth Engine:
1. Get your Project ID from [Google Cloud Console](https://console.cloud.google.com)
2. Add it to `.env` as `GEE_PROJECT_ID=your-project-id`
3. Run the app and follow the browser authentication flow

### Slow performance

By default, the app uses CPU. To enable GPU acceleration:
1. Install CUDA-compatible PyTorch
2. Set `USE_GPU=true` in your `.env` file

### Installation validation

To check if everything is set up correctly:

```bash
python validate_installation.py
```

This script will identify missing dependencies, configuration issues, and provide fix instructions.

## License

MIT
