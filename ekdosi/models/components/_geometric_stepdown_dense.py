import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Generator

def geo(min_n: int, max_n: int, r: float = 2.0) -> Generator[float, None, None]:
    for iii in range(int(min_n), int(max_n)):
        if iii == 1:
            yield r
        else:
            yield (-(r ** iii)) / (1 - r)

def generate_geometric_series(min_n: int, max_n: int) -> List[int]:
    series = list(geo(min_n=min_n, max_n=max_n))
    series = [int(xxx) for xxx in series]
    series.reverse()
    return series

class GeometricStepdownDense(nn.Module):
    """Creates a dense set of layers that steps down gradually (by log2)
    
    The dense layers include a linear layer, a dropout layer, and a batch norm layer.

    For example: if your input dim is 128, min_n=5 would be 2**5 or 32 output_dim

    Therefore, Geometric step down will create dense layers (128->64->32)
        Linear(128)
        Dropout(0.05)  
        BatchNorm(64)
        Linear(64)
        Dropout(0.05)
        BatchNorm(32)
    """
    def __init__(self, input_dim: int, min_n: int = 5, dropout: float = 0.05, batch_norm: bool = True):
        super(GeometricStepdownDense, self).__init__()
        max_n = math.floor(math.log2(input_dim))
        min_n = int(min_n)
        series = generate_geometric_series(min_n, max_n)
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
    model = GeometricStepdownDense(input_dim=input_dim, min_n=5, dropout=0.05, batch_norm=True)

    # Create a random tensor to represent your input data
    x = torch.randn(64, input_dim)  # 64 is the batch size

    # Pass the input data through the model
    output = model(x)

    print(output.shape)
    print(model)