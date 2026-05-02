@echo off
REM Auto-download script for UCI HAR dataset (Windows)
REM This script downloads and extracts the dataset automatically

echo ==========================================
echo   UCI HAR Dataset Auto-Download Script
echo ==========================================
echo.

REM Check if dataset already exists
if exist "data\UCI HAR Dataset" (
    echo [OK] Dataset already exists.
    echo       Location: .\data\UCI HAR Dataset\
    echo.
    echo To re-download, delete the data folder first:
    echo   rmdir /s /q data
    exit /b 0
)

REM Create data directory
echo [1/3] Creating data directory...
if not exist "data" mkdir data
cd data

REM Download dataset using Python (cross-platform compatible)
echo [2/3] Downloading UCI HAR Dataset...
echo       Source: UCI ML Repository
echo       Size: ~30 MB
echo.

python -c "import urllib.request; urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%%20HAR%%20Dataset.zip', 'UCI_HAR_Dataset.zip'); print('[OK] Download completed.')"

if not exist "UCI_HAR_Dataset.zip" (
    echo [ERROR] Download failed!
    echo        Please check your internet connection
    echo        or download manually from:
    echo        https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
    cd ..
    exit /b 1
)

REM Extract dataset
echo.
echo [3/3] Extracting dataset...
echo       This may take a moment...

python -c "import zipfile; import os; zip_ref = zipfile.ZipFile('UCI_HAR_Dataset.zip', 'r'); zip_ref.extractall('.'); print('[OK] Extraction completed.'); os.remove('UCI_HAR_Dataset.zip'); print('[OK] Removed zip file')"

cd ..

echo.
echo ==========================================
echo [SUCCESS] Setup completed!
echo.
echo You can now run: python main.py
echo ==========================================
pause
