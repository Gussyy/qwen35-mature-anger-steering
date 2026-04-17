"""Project-wide config. Loads HF token from .env before any model call."""
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

HF_TOKEN = os.environ.get("huggingface_hub_token") or os.environ.get("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

MODEL_LARGE = "Qwen/Qwen3.5-2B"
MODEL_SMALL = "Qwen/Qwen3.5-0.8B"
MODEL_XLARGE = "Qwen/Qwen3.5-4B"

D_MODEL_LARGE = 2048
D_MODEL_SMALL = 1024
D_MODEL_XLARGE = 2560
N_LAYERS = 24           # 2B and 0.8B
N_LAYERS_XLARGE = 32    # 4B

SWEEP_LAYERS = [6, 10, 14, 18, 22]               # for 24-layer models
SWEEP_LAYERS_XLARGE = [8, 13, 18, 23, 28]        # ~proportional for 32-layer 4B
SWEEP_COEFS = [-2.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]

MODEL_IDS = {"large": MODEL_LARGE, "small": MODEL_SMALL, "xlarge": MODEL_XLARGE}
MODEL_TAGS = {"large": "qwen_large", "small": "qwen_small", "xlarge": "qwen_xlarge"}
SWEEP_LAYER_SETS = {"large": SWEEP_LAYERS, "small": SWEEP_LAYERS, "xlarge": SWEEP_LAYERS_XLARGE}

DATA_DIR = ROOT / "data"
VECTORS_DIR = ROOT / "vectors"
SAES_DIR = ROOT / "saes"
for d in (DATA_DIR, VECTORS_DIR, SAES_DIR):
    d.mkdir(exist_ok=True)
