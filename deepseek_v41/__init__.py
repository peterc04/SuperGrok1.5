"""DeepSeek-V4.1-Flash text backbone for training, in plain PyTorch (see model.py for the port notes)."""

from .config import PRESETS, Config, get_config
from .model import ParamRole, Transformer, build_model

__all__ = ["Config", "PRESETS", "ParamRole", "Transformer", "build_model", "get_config"]
