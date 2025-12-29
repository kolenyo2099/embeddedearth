# EmbeddedEarth - AI-Powered Satellite Imagery Search

A semantic search engine for satellite imagery using DOFA-CLIP (Dynamic One-For-All CLIP) for multimodal vision-language understanding of Earth Observation data.

## Features

- **Text-to-Image Search**: Describe what you're looking for (e.g., "solar panels", "industrial facility near river")
- **DOFA-CLIP Integration**: Uses wavelength-aware dynamic encoding for multispectral Sentinel-2 imagery
- **Explainable AI**: Grad-CAM heatmaps show what the model is focusing on
- **Interactive Map**: Draw areas of interest directly on the map

## Installation

The easiest way to install is using the provided script (requires [uv](https://github.com/astral-sh/uv)):

```bash
# Clone the repository
git clone https://github.com/yourusername/EmbeddedEarth.git
cd EmbeddedEarth

# Run installation script
chmod +x install.sh
./install.sh
```

Alternatively, manual installation:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install DOFA-CLIP fork
git clone https://github.com/xiong-zhitong/DOFA-CLIP.git
cd DOFA-CLIP/open_clip
pip install -e .
cd ../..

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the Streamlit app
streamlit run app/main.py
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

## License

MIT
