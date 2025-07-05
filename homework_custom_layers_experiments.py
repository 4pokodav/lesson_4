import torch
import torch.nn as nn
import logging
import os
from models.custom_layers import (
    CustomConvLayer, CustomActivation, CustomPooling, SimpleAttention,
    BasicResidualBlock, BottleneckBlock, WideResidualBlock
)
from utils.visualization_utils import plot_training_history, count_parameters
from utils.training_utils import train_model
from utils.comparison_utils import get_mnist_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestNet(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            block(16),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


def run_experiment(block_class, block_args=None, name="block"):
    logger.info(f"Эксперимент с {name}")
    block_args = block_args or {}
    model = TestNet(lambda c: block_class(c, **block_args)).to(device)
    logger.info(f"Параметров в модели: {count_parameters(model)}")

    history = train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device=device)
    plot_training_history(history, save_path="homework_4/plots/", filename=f"{name}_history.png")
    return history


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    # Запуск экспериментов
    run_experiment(BasicResidualBlock, name="basic_residual")
    run_experiment(BottleneckBlock, block_args={"bottleneck_channels": 8}, name="bottleneck_residual")
    run_experiment(WideResidualBlock, block_args={"widen_factor": 2}, name="wide_residual")

    # Пример использования кастомных слоёв
    dummy_input = torch.randn(2, 1, 28, 28)
    logger.info("Тест кастомных слоёв...")
    custom_conv = CustomConvLayer(1, 8, 3)
    attn = SimpleAttention(8)
    custom_act = CustomActivation()
    custom_pool = CustomPooling(2)

    x = custom_conv(dummy_input)
    x = attn(x)
    x = custom_act(x)
    x = custom_pool(x)

    logger.info(f"Размер выходного тензора после кастомных слоёв: {x.shape}")