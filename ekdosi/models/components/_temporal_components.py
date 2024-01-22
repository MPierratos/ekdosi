import torch
import torch.nn as nn
import numpy as np

class DifferenceLayer(nn.Module):
    """Assumes <batch>, <time>, <channels>,
    
    This difference layer is a non-trainable that takes the difference between 
    parameters across the time dimension. This is useful for building at first and
    second derivatives.
    """

    def __init__(self, use_float64: bool = False):
        super(DifferenceLayer, self).__init__()
        self.use_float64 = use_float64
        self.diff_convolution = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(2,1),
            bias=False
        )

        self.diff_convolution.weight.requires_grad=False
        self.diff_convolution.weight.data = self.kernel_initializer(self.diff_convolution.weight.shape)

    def kernel_initializer(self, shape):
        """initializes the kernel to that creates a diff across a direction"""
        weights = torch.tensor([-1.0, 1.0]).reshape(shape).float()
        if self.use_float64:
            weights = weights.double()
        return weights

    def forward(self, input):
        input = input.unsqueeze(1)
        return self.diff_convolution(input)
    

def test_DifferenceLayer():
    def checkerboard(shape):
        return np.indices(shape).sum(axis=0) % 2
    
    df = torch.tensor(checkerboard(shape=(3, 7, 5)), dtype=torch.float32)
    correct = torch.diff(df, dim=1)
   
    d1 = DifferenceLayer()
    res = torch.squeeze(d1(df))
    assert res.dtype == correct.dtype
    test = res == correct
    if test.all():
        return True
    else:
        return False

