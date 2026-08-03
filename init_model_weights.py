"""
scripts/init_model_weights.py
------------------------------
Utility to create a starter checkpoint so the API doesn't run on an untrained,
randomly-initialized model. Run this once before first deployment, and re-run
your real training pipeline (on labeled/benign flow data) to replace it later.

Usage:
    python -m scripts.init_model_weights
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from config import get_settings  # noqa: E402
from model import NeuralNetwork  # noqa: E402


def main() -> None:
    settings = get_settings()
    model = NeuralNetwork(input_dim=settings.MODEL_INPUT_DIM, latent_dim=settings.MODEL_LATENT_DIM)

    out_path = Path(settings.MODEL_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Wrote starter (untrained) checkpoint to {out_path}")
    print("Replace this with a checkpoint trained on real benign-flow data before production use.")


if __name__ == "__main__":
    main()
