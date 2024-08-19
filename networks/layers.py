import torch
from torch import nn


class Layer(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, **kwargs):
        super(Layer, self).__init__()
        self.dropout = None
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Relu(Layer):
    """A linear layer followed by a ReLU activation function."""
    def __init__(self, in_size, out_size, dropout=False, **kwargs):
        super(Relu, self).__init__(in_size, out_size, **kwargs)
        self.linear = nn.Linear(in_size, out_size)
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    
class ResBlock(nn.Module):
    """A residual block with two convolutional layers and a ReLU activation function."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x
