import os
import zipfile
import subprocess
import sys

KAGGLE_DATASET = "jessicali9530/axolotl-dataset"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ZIP_PATH = os.path.join(DATA_DIR, "axolotl-dataset.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "axolotl_images")
KAGGLE_JSON = os.path.join(os.path.dirname(__file__), "kaggle.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Check for kaggle.json
if not os.path.exists(KAGGLE_JSON):
    print("ERROR: kaggle.json not found in backend/. Please download it from your Kaggle account and place it here.")
    sys.exit(1)

# Set Kaggle config environment variable
os.environ["KAGGLE_CONFIG_DIR"] = os.path.dirname(KAGGLE_JSON)

# Download dataset if not already downloaded
if not os.path.exists(ZIP_PATH):
    print("Downloading axolotl dataset from Kaggle...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "kaggle"
    ], check=True)
    subprocess.run([
        "kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", DATA_DIR
    ], check=True)
else:
    print("Dataset zip already exists. Skipping download.")

# Unzip dataset
if not os.path.exists(EXTRACT_DIR):
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print(f"Extracted to {EXTRACT_DIR}")
else:
    print("Dataset already extracted.")

print("Done.")
