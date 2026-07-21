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

TEXT_MODEL = "CLIP"


# Embedding and Projector Configuration

VISION_EMBED_DIM = 768
TEXT_EMBED_DIM = 768

PROJECTOR_INPUT_DIM = VISION_EMBED_DIM
PROJECTOR_HIDDEN_DIM = 1536
PROJECTOR_OUTPUT_DIM = VISION_EMBED_DIM

PROJECTOR_ACTIVATION = "relu"
PROJECTOR_DROPOUT = 0.0
PROJECTOR_NORMALIZE = True

# Training

BATCH_SIZE = 64
PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 3
PHASE3_EPOCHS = 5

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

NUM_WORKERS = 4

SEED = 42


# Hybrid Loss

TEMPERATURE = 0.07

TEMPERATURE_MIN = 0.03

TEMPERATURE_DECAY = 0.98

LAMBDA_NMSE = 0.10

LAMBDA_INFO_NCE = 0.90


# Optimization

OPTIMIZER = "AdamW"

GRADIENT_CLIP = 1.0

USE_MIXED_PRECISION = True

SCHEDULER = "CosineAnnealingLR"

EARLY_STOPPING = False

SAVE_BEST_ONLY = True


# Similarity Evaluation

COSINE_THRESHOLD = 0.50


# Evaluation

COMPUTE_ALIGNMENT_GAP = True

COMPUTE_STD = True


# Checkpoints

CHECKPOINTS = {
    "phase1": CHECKPOINT_DIR / "projector_phase1.pt",
    "phase2": CHECKPOINT_DIR / "projector_phase2.pt",
    "phase3": CHECKPOINT_DIR / "projector_phase3.pt",
}


# Device

try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

except ImportError:
    DEVICE = "cpu"
