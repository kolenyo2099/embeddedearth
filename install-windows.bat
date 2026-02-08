@echo off
REM EmbeddedEarth Windows Installation Script
REM Uses conda/mamba for reliable GDAL installation on Windows

echo ========================================================================
echo  EmbeddedEarth Windows Installation
echo ========================================================================
echo.

REM Check if conda/mamba is available
where mamba >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set CONDA_CMD=mamba
    echo [OK] Using mamba for faster installation
) else (
    where conda >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set CONDA_CMD=conda
        echo [OK] Using conda
    ) else (
        echo [ERROR] Neither conda nor mamba found!
        echo.
        echo Windows installation requires conda/mamba to handle GDAL dependencies.
        echo Please install Miniconda or Anaconda first:
        echo   https://docs.conda.io/en/latest/miniconda.html
        echo.
        echo Or install Mambaforge (recommended):
        echo   https://github.com/conda-forge/miniforge#mambaforge
        echo.
        pause
        exit /b 1
    )
)

echo.
echo Step 1: Creating conda environment 'embeddedearth'...
%CONDA_CMD% create -n embeddedearth python=3.10 -y
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create conda environment
    pause
    exit /b 1
)

echo.
echo Step 2: Activating environment...
call conda activate embeddedearth
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate environment
    pause
    exit /b 1
)

echo.
echo Step 3: Installing GDAL and geospatial dependencies via conda...
echo (This avoids Windows DLL issues)
%CONDA_CMD% install -c conda-forge gdal rasterio geopandas fiona -y
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install GDAL dependencies
    pause
    exit /b 1
)

echo.
echo Step 4: Installing PyTorch...
%CONDA_CMD% install -c pytorch pytorch torchvision cpuonly -y
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to install PyTorch via conda, trying pip...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo Step 5: Cloning DOFA-CLIP repository...
if not exist "DOFA-CLIP" (
    git clone https://github.com/xiong-zhitong/DOFA-CLIP.git
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to clone DOFA-CLIP
        pause
        exit /b 1
    )
) else (
    echo [OK] DOFA-CLIP already exists
)

echo.
echo Step 6: Installing custom open_clip fork...
cd DOFA-CLIP\open_clip
pip install -e .
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install DOFA-CLIP
    cd ..\..
    pause
    exit /b 1
)
cd ..\..

echo.
echo Step 7: Installing remaining Python dependencies...
pip install -r requirements.txt --no-deps
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some dependencies may have failed, but core packages should work
)

echo.
echo Step 8: Creating .env file from template...
if not exist ".env" (
    copy .env.example .env
    echo [OK] Created .env file - please edit it and add your GEE_PROJECT_ID
) else (
    echo [OK] .env file already exists
)

echo.
echo ========================================================================
echo  Installation Complete!
echo ========================================================================
echo.
echo Next steps:
echo   1. Edit .env file and add your GEE_PROJECT_ID
echo   2. Validate installation: python validate_installation.py
echo   3. Run the app: streamlit run app\main.py
echo.
echo To activate this environment later:
echo   conda activate embeddedearth
echo.
pause
