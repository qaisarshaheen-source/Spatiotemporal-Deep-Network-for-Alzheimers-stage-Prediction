"""
Configuration for Alzheimer's detection model training and deployment.
On Kaggle: dataset = /kaggle/input/alzheimers-detection-dataset/dataset, output = /kaggle/working/
Works in both script (__file__) and Jupyter/Kaggle notebook (no __file__).
"""
import os

# Paths: use __file__ when running as script; use cwd in Jupyter/Kaggle notebooks
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Kaggle paths (dataset is read-only input, output goes to working)
KAGGLE_DATASET_DIR = "/kaggle/input/alzheimers-detection-dataset/dataset"
KAGGLE_WORKING_DIR = "/kaggle/working"

def _is_kaggle():
    return os.path.isdir("/kaggle/input")

if _is_kaggle():
    DATASET_DIR = KAGGLE_DATASET_DIR
    MODELS_DIR = os.path.join(KAGGLE_WORKING_DIR, "models")
else:
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

SAVED_MODEL_PATH = os.path.join(MODELS_DIR, "best_alzheimer_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.txt")

# Image and training
IMG_SIZE = (224, 224)  # Compatible with MobileNetV2, EfficientNet, and custom CNN
IMG_SHAPE = (*IMG_SIZE, 3)
BATCH_SIZE = 32
SEED = 42
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
EPOCHS = 50
PATIENCE = 10
# Imbalance: oversample so each class has at least this many training samples
OVERSAMPLE_MIN_PER_CLASS = 800
# Cap class weight so one class does not dominate (e.g. Moderate_Demented had 31+)
CLASS_WEIGHT_MAX = 10.0

# Class names (must match folder names in dataset/)
CLASS_NAMES = ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"]
NUM_CLASSES = len(CLASS_NAMES)

# Supported image extensions
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
