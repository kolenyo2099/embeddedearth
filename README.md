# ElLocoGis - AI-Powered Satellite Imagery Search

A semantic search engine for satellite imagery using DOFA-CLIP (Dynamic One-For-All CLIP) for multimodal vision-language understanding of Earth Observation data.

## Features

- **Text-to-Image Search**: Describe what you're looking for (e.g., "solar panels", "industrial facility near river")
- **DOFA-CLIP Integration**: Uses wavelength-aware dynamic encoding for multispectral Sentinel-2 imagery
- **Explainable AI**: Grad-CAM heatmaps show what the model is focusing on
- **Super-Resolution Refinement**: 4x upscaling with Real-ESRGAN for fine-grained search
- **Interactive Map**: Draw areas of interest directly on the map

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ElLocoGis.git
cd ElLocoGis

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install DOFA-CLIP's custom open_clip fork
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

## License

MIT
