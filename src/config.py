"""
Configuration file for Rice Variety Classification project
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ORIGINAL_DATA_DIR = PROJECT_ROOT / "data" / "Original"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Image processing
IMAGE_SIZE = (224, 224)  # Standard size for transfer learning models
BATCH_SIZE = 32
NUM_CHANNELS = 3
RANDOM_SEED = 42

# Training parameters
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2

# Model parameters
USE_TRANSFER_LEARNING = True
PRETRAINED_MODEL = "efficientnet_b0"  # Options: resnet50, efficientnet_b0, vgg16, mobilenet
FREEZE_BASE_LAYERS = True
NUM_FREEZE_LAYERS = -30  # Negative means freeze from the end

# Rice varieties (classes)
RICE_VARIETIES = [
    "1_Subol_Lota",
    "2_Bashmoti",
    "3_Ganjiya",
    "4_Shampakatari",
    "5_Katarivog",
    "6_BR28",
    "7_BR29",
    "8_Paijam",
    "9_Bashful",
    "10_Lal_Aush",
    "11_Jirashail",
    "12_Gutisharna",
    "13_Red_Cargo",
    "14_Najirshail",
    "15_Katari_Polao",
    "16_Lal_Biroi",
    "17_Chinigura_Polao",
    "18_Amon",
    "19_Shorna5",
    "20_Lal_Binni"
]

NUM_CLASSES = len(RICE_VARIETIES)

# ML Models
ML_MODELS = {

}

# Output files
MODEL_SAVE_NAME = "rice_classifier_model.h5"
BEST_MODEL_NAME = "rice_classifier_best.h5"
HISTORY_FILE = "training_history.pkl"
METRICS_FILE = "metrics.json"
CONFUSION_MATRIX_FILE = "confusion_matrix.png"
