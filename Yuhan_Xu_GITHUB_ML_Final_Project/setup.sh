#!/bin/bash
# Auto-download script for UCI HAR dataset
# This script downloads and extracts the dataset automatically

echo "=========================================="
echo "  UCI HAR Dataset Auto-Download Script"
echo "=========================================="
echo ""

# Check if dataset already exists
if [ -d "data/UCI HAR Dataset" ]; then
    echo "[OK] Dataset already exists."
    echo "      Location: ./data/UCI HAR Dataset/"
    echo ""
    echo "To re-download, delete the data folder first:"
    echo "  rm -rf data/"
    exit 0
fi

# Create data directory
echo "[1/3] Creating data directory..."
mkdir -p data
cd data

# Download dataset
echo "[2/3] Downloading UCI HAR Dataset..."
echo "      Source: UCI ML Repository"
echo "      Size: ~30 MB"
echo ""

# URL for UCI HAR Dataset
URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"

# Try with curl first (most systems)
if command -v curl &> /dev/null; then
    curl -L "$URL" -o "UCI_HAR_Dataset.zip"
# Fallback to wget
elif command -v wget &> /dev/null; then
    wget "$URL" -O "UCI_HAR_Dataset.zip"
# Fallback to Python
else
    python -c "import urllib.request; urllib.request.urlretrieve('$URL', 'UCI_HAR_Dataset.zip')"
fi

if [ ! -f "UCI_HAR_Dataset.zip" ]; then
    echo "[ERROR] Download failed!"
    echo "Please check your internet connection"
    echo "or download manually from:"
    echo "https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones"
    exit 1
fi

echo "[OK] Download completed."

# Extract dataset
echo ""
echo "[3/3] Extracting dataset..."
echo "      This may take a moment..."

# Use Python for cross-platform extraction
python -c "
import zipfile
import os

zip_path = 'UCI_HAR_Dataset.zip'
print(f'Extracting {zip_path}...')

try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    print('[OK] Extraction completed.')
    print(f'      Extracted {len(zip_ref.namelist())} files')

    # Clean up zip file
    os.remove(zip_path)
    print('[OK] Removed zip file')

    # Check if extraction was successful
    if os.path.exists('UCI HAR Dataset'):
        print('')
        print('[SUCCESS] Dataset ready!')
        print('          Location: ./data/UCI HAR Dataset/')
    else:
        print('[ERROR] Extraction failed - expected folder not found')

except Exception as e:
    print(f'[ERROR] Extraction failed: {e}')
    exit(1)
"

cd ..
echo ""
echo "=========================================="
echo "[SUCCESS] Setup completed!"
echo ""
echo "You can now run: python main.py"
echo "=========================================="
