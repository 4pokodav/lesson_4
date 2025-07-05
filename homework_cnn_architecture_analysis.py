import torch
import torch.nn as nn
import logging
import time
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils.visualization_utils import count_parameters
from utils.visualization_utils import plot_accuracy
from utils.training_utils import train_model
from utils.visualization_utils import visualize_activations, visualize_feature_maps
from utils.comparison_utils import get_mnist_loaders
from models.cnn_models import CNNWithResidual

RESULTS_DIR = 'homework_4/results/architecture_analysis/'
PLOTS_DIR = 'homework_4/plots/'
'''
logging.basicConfig(filename=RESULTS_DIR + 'analysis.log',
                    level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
'''

logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(RESULTS_DIR + 'analysis.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

train_loader, test_loader = get_mnist_loaders()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2.1 Влияние размера ядра
class KernelCNN(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class ComboKernelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Conv2d(1, 16, kernel_size=1)
        self.conv3x3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv3x3(torch.relu(self.conv1x1(x)))))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def log_receptive_field(kernel_size, num_layers):
    rf = 1
    for _ in range(num_layers):
        rf += (kernel_size - 1)
    logging.info(f"Оценка рецептивного поля: ядро {kernel_size}x{kernel_size}, {num_layers} conv слоёв - receptive field ~ {rf}x{rf}")

def log_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            logging.info(f"Градиент слоя {name}: {grad_norm:.6f}")

def run_kernel_size_experiment():
    kernels = [3, 5, 7]
    results = {}

    for k in kernels:
        model = KernelCNN(kernel_size=k).to(device)
        logging.info(f"Начало обучения для kernel={k}")
        params = count_parameters(model)
        logging.info(f"Параметров в модели: {params}")
        log_receptive_field(kernel_size=k, num_layers=2)
        start = time.time()
        history = train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device=device)
        end = time.time()
        results[f'{k}x{k}'] = history
        logging.info(f"Kernel={k} завершено за {end - start:.2f} секунд")

        # Визуализация активаций
        sample = next(iter(test_loader))[0][:1].to(device)
        visualize_activations(model.conv1, sample, PLOTS_DIR + f'kernel_{k}_activations.png')

    # Комбинированная архитектура
    model = ComboKernelCNN().to(device)
    logging.info("Начало обучения для комбинированной архитектуры (1x1 + 3x3)")
    start = time.time()
    history = train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device=device)
    end = time.time()
    results['1x1+3x3'] = history
    logging.info(f"Комбинированная архитектура завершена за {end - start:.2f} секунд")

    sample = next(iter(test_loader))[0][:1].to(device)
    visualize_activations(model.conv1x1, sample, PLOTS_DIR + 'combo_activations.png')

    return results

# 2.2 Влияние глубины сети
class DepthCNN(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        layers = [nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.conv(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def run_depth_experiment():
    depths = [2, 4, 6]
    results = {}

    for d in depths:
        model = DepthCNN(num_layers=d).to(device)
        logging.info(f"Начало обучения для глубины {d} слоёв")
        params = count_parameters(model)
        logging.info(f"Параметров в модели: {params}")
        log_receptive_field(kernel_size=3, num_layers=d)
        start = time.time()
        history = train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device=device)
        end = time.time()
        results[f'depth_{d}'] = history
        logging.info(f"Глубина={d} завершено за {end - start:.2f} секунд")
        log_gradients(model)

        sample = next(iter(test_loader))[0][:1].to(device)
        visualize_feature_maps(model.conv, sample, PLOTS_DIR + f'depth_{d}_features.png')

    logging.info("Начало обучения с Residual связями")
    model = CNNWithResidual(input_channels=1).to(device)
    params = count_parameters(model)
    logging.info(f"Параметров в модели: {params}")
    log_receptive_field(3, 6)
    start = time.time()
    history = train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device=device)
    end = time.time()
    results['residual'] = history
    logging.info(f"Residual CNN завершено за {end - start:.2f} секунд")
    log_gradients(model)

    sample = next(iter(test_loader))[0][:1].to(device)
    with torch.no_grad():
        x = F.relu(model.bn1(model.conv1(sample)))
        activation = model.res1(x)                  
    visualize_feature_maps(activation, sample, PLOTS_DIR + f'residual_features.png', is_tensor=True)
    
    return results


if __name__ == "__main__":
    logging.info("Старт экспериментов по архитектурам CNN")

    kernel_results = run_kernel_size_experiment()
    depth_results = run_depth_experiment()

    plot_accuracy(kernel_results, PLOTS_DIR + 'kernel_size_accuracy.png', 'Влияние размера ядра свёртки')
    plot_accuracy(depth_results, PLOTS_DIR + 'depth_accuracy.png', 'Влияние глубины CNN')

    logging.info("Все эксперименты завершены")