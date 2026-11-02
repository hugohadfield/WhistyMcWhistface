"""Safe, versioned MLP checkpoint loading for training and gameplay."""
from pathlib import Path

import torch

from alphawhist.learning.encoding import FEATURE_DIM, FEATURE_SCHEMA_VERSION
from alphawhist.learning.model import DistributionalWhistMLP

CHECKPOINT_SCHEMA = 1


def choose_device(requested="auto"):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS device requested, but MPS is unavailable")
    return device


def save_checkpoint(path, model, *, metadata):
    """Store only tensors and primitive types; no executable Python objects."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "hidden_size": model.head.in_features,
        "residual_blocks": sum(
            type(module).__name__ == "ResidualBlock" for module in model.modules()
        ),
    }
    checkpoint = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "model_config": config,
        "state_dict": model.state_dict(),
        "metadata": dict(metadata),
    }
    temporary = out.with_suffix(out.suffix + ".part")
    torch.save(checkpoint, temporary)
    temporary.replace(out)


def load_checkpoint(path, device="cpu"):
    """Reject mismatched inputs and load weights without unsafe pickle globals."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (blob.get("checkpoint_schema") != CHECKPOINT_SCHEMA
            or blob.get("feature_schema") != FEATURE_SCHEMA_VERSION
            or blob.get("feature_dim") != FEATURE_DIM):
        raise ValueError("Incompatible checkpoint or neural feature representation")
    cfg = blob["model_config"]
    if set(cfg) != {"hidden_size", "residual_blocks"}:
        raise ValueError("Unsupported MLP architecture")
    model = DistributionalWhistMLP(**cfg)
    model.load_state_dict(blob["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, blob["metadata"]
