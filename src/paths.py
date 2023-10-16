from pathlib import Path
import os

# Get the parent directory of the current file (__file__)
PARENT_DIR = Path(__file__).parent.resolve().parent

# Define paths for data directories relative to the parent directory
DATA_DIR = PARENT_DIR / "data"
RAW_DATA_DIR = PARENT_DIR / "data" / "raw"
TRANSFORMED_DATA_DIR = PARENT_DIR / "data" / "transformed"

# Check if the general data directory exists, if not, create it
if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

# Check if the raw data directory exists, if not, create it
if not Path(RAW_DATA_DIR).exists():
    os.mkdir(RAW_DATA_DIR)

# Check if the transformed data directory exists, if not, create it
if not Path(TRANSFORMED_DATA_DIR).exists():
    os.mkdir(TRANSFORMED_DATA_DIR)
