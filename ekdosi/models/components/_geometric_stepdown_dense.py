import math
from typing import Generator, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_geometric_series(
    min_n: int, max_n: int, reverse: bool = False
) -> List[int]:
    """
    Iterates over a geometric series.

    min_n: the minimum to start with
    max_n: the maximum to end at
    reverse: if the series should be inverted

    i.e. if min_n = 5, and the max_n = 8, then the series would be calculated like so:

    2^5 = 32
    2^8 = 256

    the output would be:
    [32, 64, 128, 256] if not reversed
    [256, 128, 64, 32] if reversed
    """

    def geo(min_n: int, max_n: int, r: float = 2.0) -> Generator[int, None, None]:
        """
        Generate a geometric series.
        """
        for iii in range(int(min_n), int(max_n) + 1):
            yield int(r**iii)

    series = list(geo(min_n=min_n, max_n=max_n))
    if reverse:
        series.reverse()
    return series


class GeometricStepDownDense(nn.Module):
    """Creates a dense set of layers that steps down gradually (by log2)

    The dense layers include a linear layer, a dropout layer, and a batch norm layer.

    For example: if your input dim is 128, min_n=5 would be 2**5 or 32 output_dim

    Therefore, Geometric step down will create dense layers (128->64->32)
        Linear(128)
        Dropout(0.05)
        BatchNorm(64)

        Linear(64)
        Dropout(0.05)
        BatchNorm(64)

        Linear(32)
        Dropout(0.05)
        BatchNorm(32)
    """

    def __init__(
        self,
        input_dim: int,
        min_n: int = 5,
        dropout: float = 0.05,
        batch_norm: bool = True,
    ):
        super(GeometricStepDownDense, self).__init__()
        max_n = math.floor(math.log2(input_dim))
        min_n = int(min_n)
        series = generate_geometric_series(min_n, max_n, reverse=True)
        self.layers = nn.ModuleList()
        for units in series:
            self.layers.append(nn.Linear(input_dim, units))
            if dropout is not None:
                self.layers.append(nn.Dropout(p=dropout))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(units))
            input_dim = units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


class GeometricStepUpDense(nn.Module):
    """Creates a dense set of layers that steps up gradually (by log2)

    The dense layers include a linear layer, a dropout layer, and a batch norm layer.

    For example: if your input dim is 32, max_n=8 would be 2**8 or 256 output_dim

    Therefore, Geometric step up will create dense layers (32->64->128->256)
        Linear(32)
        Dropout(0.05)
        BatchNorm(32)

        Linear(64)
        Dropout(0.05)
        BatchNorm(64)

        Linear(128)
        Dropout(0.05)
        BatchNorm(128)

        Linear(256)
        Dropout(0.05)
        BatchNorm(256)
    """

    def __init__(
        self,
        input_dim: int,
        max_n: int = 8,
        dropout: float = 0.05,
        batch_norm: bool = True,
    ):
        super(GeometricStepUpDense, self).__init__()
        max_n = int(max_n)
        min_n = math.floor(math.log2(input_dim))
        series = generate_geometric_series(min_n, max_n, reverse=False)
        self.layers = nn.ModuleList()
        for units in series:
            self.layers.append(nn.Linear(input_dim, units))
            if dropout is not None:
                self.layers.append(nn.Dropout(p=dropout))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(units))
            input_dim = units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


def test_geometric_stepdown_dense() -> None:
    # Define the input dimension, i.e. input channel is 128
    input_dim = 128

    # Create an instance of GeometricStepdownDense
    # Because min_n = 5, we know the final layer will have a shape of 2**5, or 32
    model = GeometricStepDownDense(
        input_dim=input_dim, min_n=5, dropout=0.05, batch_norm=True
    )

    # Create a random tensor to represent your input data
    x = torch.randn(64, input_dim)  # 64 is the batch size

    # Pass the input data through the model
    output = model(x)

    print(output.shape)
    print(model)


def test_geometric_stepup_dense() -> None:
    # Define the input dimension, i.e. input channel is 32
    input_dim = 32

    # Create an instance of GeometricStepdownDense
    # Because max_n = 8, we know the final layer will have a shape of 2**8, or 256
    model = GeometricStepUpDense(
        input_dim=input_dim, max_n=8, dropout=0.05, batch_norm=True
    )

    # Create a random tensor to represent your input data
    x = torch.randn(64, input_dim)  # 64 is the batch size

    # Pass the input data through the model
    output = model(x)

    print(output.shape)
    print(model)
