import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from torch.utils.data import DataLoader, TensorDataset
from utils.visualization_utils import plot_training_history, count_parameters
from utils.training_utils import train_model
from utils.comparison_utils import get_mnist_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        logger.debug("CustomConvLayer: forward pass")
        out = F.conv2d(x, self.weight, self.bias, padding=1)
        return F.relu(out + 0.1 * torch.sin(out))


class CustomActivation(nn.Module):
    def forward(self, x):
        logger.debug("CustomActivation: forward pass")
        return x * torch.tanh(F.relu(x))

class CustomPooling(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        logger.debug("CustomPooling: forward pass")
        max_pooled = F.max_pool2d(x, self.kernel_size)
        avg_pooled = F.avg_pool2d(x, self.kernel_size)
        return 0.5 * (max_pooled + avg_pooled)

class SimpleAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        logger.debug("SimpleAttention: forward pass")
        b, c, h, w = x.shape
        q = self.query(x).view(b, c, -1)
        k = self.key(x).view(b, c, -1)
        v = self.value(x).view(b, c, -1)

        attn = self.softmax(torch.bmm(q.transpose(1, 2), k))
        out = torch.bmm(attn, v.transpose(1, 2)).transpose(1, 2).view(b, c, h, w)
        return out + x

class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        logger.debug("BasicResidualBlock: forward pass")
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.conv = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.expand = nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        logger.debug("BottleneckBlock: forward pass")
        identity = x
        out = self.relu(self.reduce(x))
        out = self.relu(self.conv(out))
        out = self.expand(out)
        return self.relu(out + identity)

class WideResidualBlock(nn.Module):
    def __init__(self, in_channels, widen_factor=2):
        super().__init__()
        widened_channels = in_channels * widen_factor
        self.conv1 = nn.Conv2d(in_channels, widened_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(widened_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        logger.debug("WideResidualBlock: forward pass")
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)