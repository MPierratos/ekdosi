from enum import Enum

import torch

__all__ = ["get_optimizer"]


class SupportedOptimizers(str, Enum):
    """Supported optimizer names"""

    ADAM: str = "adam"
    ADAMW: str = "adamw"
    SGD: str = "sgd"


def get_optimizer(name: SupportedOptimizers):
    if name == SupportedOptimizers.ADAM:
        return torch.optim.Adam
    if name == SupportedOptimizers.ADAMW:
        return torch.optim.AdamW
    if name == SupportedOptimizers.SGD:
        return torch.optim.SGD

    raise ValueError(
        f"{name} is not supported. Supported: {[o.value for o in SupportedOptimizers]}"
    )
