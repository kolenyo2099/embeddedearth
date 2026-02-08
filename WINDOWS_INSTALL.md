# Windows Installation Guide

## The Windows DLL Problem

If you see errors like:
```
DLL load failed while importing _base: The operating system cannot run %1
```

This is caused by **GDAL/rasterio** dependencies that require native C++ DLLs. On Windows, these are difficult to install via pip.

## ✅ Recommended Solution: Use Conda/Mamba

The easiest and most reliable way to install EmbeddedEarth on Windows is using **conda** or **mamba**, which handles binary dependencies automatically.

### Option 1: Automated Installation (Recommended)

1. **Install Mambaforge** (includes both conda and mamba):
   - Download from: https://github.com/conda-forge/miniforge#mambaforge
   - Run the installer (mambaforge-xxx-Windows-x86_64.exe)
   - During installation, check "Add to PATH" (optional but convenient)

2. **Run the Windows installation script**:
   ```cmd
   install-windows.bat
   ```

3. **Configure and run**:
   ```cmd
   conda activate embeddedearth
   notepad .env          # Add your GEE_PROJECT_ID
   python validate_installation.py
   streamlit run app\main.py
   ```

### Option 2: Manual Installation

If you prefer manual control:

```cmd
# 1. Create conda environment
conda create -n embeddedearth python=3.10 -y
conda activate embeddedearth

# 2. Install GDAL and geospatial packages via conda (NOT pip!)
conda install -c conda-forge gdal rasterio geopandas fiona -y

# 3. Install PyTorch
conda install -c pytorch pytorch torchvision cpuonly -y

# 4. Clone and install DOFA-CLIP
git clone https://github.com/xiong-zhitong/DOFA-CLIP.git
cd DOFA-CLIP\open_clip
pip install -e .
cd ..\..

# 5. Install remaining dependencies
pip install -r requirements.txt --no-deps

# 6. Configure environment
copy .env.example .env
notepad .env  # Add your GEE_PROJECT_ID

# 7. Validate
python validate_installation.py
```

## ❌ Why Not Use Pip?

Using `pip install -r requirements.txt` on Windows often fails because:
- `rasterio` requires pre-compiled GDAL DLLs
- `geopandas` requires Fiona which requires GDAL
- These DLLs depend on specific Visual C++ runtime versions
- Binary compatibility issues are common

**Conda/mamba solves this** by providing pre-compiled packages with all DLLs included.

## Alternative: Pip with Pre-built Wheels

If you absolutely must use pip (not recommended):

1. Install Visual C++ Build Tools:
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "Desktop development with C++"

2. Install OSGeo4W (for GDAL):
   - Download from: https://trac.osgeo.org/osgeo4w/
   - Install GDAL and dependencies

3. Set environment variables:
   ```cmd
   set GDAL_DATA=C:\OSGeo4W\share\gdal
   set PROJ_LIB=C:\OSGeo4W\share\proj
   ```

4. Try pip installation:
   ```cmd
   pip install -r requirements.txt
   ```

**Warning:** This approach is error-prone and not officially supported.

## Troubleshooting

### "conda not found"
Install Miniconda or Mambaforge first (see links above).

### "git not found"
Install Git for Windows: https://git-scm.com/download/win

### Still getting DLL errors?
1. Make sure you're in the conda environment: `conda activate embeddedearth`
2. Check which rasterio was installed: `conda list rasterio`
3. It should show `channel: conda-forge`, not `pypi`
4. If it shows pypi, reinstall: `conda install -c conda-forge rasterio --force-reinstall`

### Performance Issues
By default, uses CPU mode. To enable GPU:
1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Install GPU-enabled PyTorch:
   ```cmd
   conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
3. Set in .env: `USE_GPU=true`

## Verification

After installation, run:
```cmd
python validate_installation.py
```

This will check:
- ✓ Python version
- ✓ GDAL/rasterio imports
- ✓ DOFA-CLIP (open_clip) installed
- ✓ All required packages
- ✓ Environment variables configured

If all checks pass, you're ready to run:
```cmd
streamlit run app\main.py
```
