from pathlib import Path


# Project Directories

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
FIGURE_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "cache"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Dataset Paths

DEEPFASHION_DIR = DATASET_DIR / "deepfashion"
POLYVORE_DIR = DATASET_DIR / "polyvore"
FASHIONIQ_DIR = DATASET_DIR / "fashioniq"
AMAZON_FASHION_DIR = DATASET_DIR / "amazon_fashion"


# Model Configuration

VISION_MODEL = "EVA02-L-14"
VISION_PRETRAINED = "merged2b_s4b_b131k"

TEXT_MODEL = "microsoft/Phi-3.5-mini-instruct"


# Embedding Dimensions

IMAGE_EMBED_DIM = 768
TEXT_EMBED_DIM = 3072

PROJECTOR_INPUT_DIM = IMAGE_EMBED_DIM
PROJECTOR_HIDDEN_DIM = 1536
PROJECTOR_OUTPUT_DIM = IMAGE_EMBED_DIM


# Training

BATCH_SIZE = 64
NUM_EPOCHS = 3

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

NUM_WORKERS = 4

SEED = 42


# Hybrid Loss

TEMPERATURE = 0.07

LAMBDA_NMSE = 0.10


# Optimization

OPTIMIZER = "AdamW"

GRADIENT_CLIP = 1.0

USE_MIXED_PRECISION = True


# Retrieval

TOP_K = 5

COSINE_THRESHOLD = 0.50

OOD_THRESHOLD = 0.35


# Evaluation

RECALL_VALUES = (1, 5, 10)


# Checkpoints

PROJECTOR_CHECKPOINT = (
    CHECKPOINT_DIR / "projector_eva02_phi35_hybrid.pt"
)

MSE_PROJECTOR_CHECKPOINT = (
    CHECKPOINT_DIR / "projector_mse.pt"
)

REGRESSION_PROJECTOR_CHECKPOINT = (
    CHECKPOINT_DIR / "projector_regression.pt"
)

LORA_CHECKPOINT = (
    CHECKPOINT_DIR / "lora_rank4"
)


# Device

try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

except ImportError:
    DEVICE = "cpu"
