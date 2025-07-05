import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from utils.comparison_utils import get_mnist_loaders, get_cifar_loaders
from models.cnn_models import SimpleCNN, CNNWithResidual, CIFARCNN
from utils.training_utils import train_model
from utils.visualization_utils import count_parameters, plot_training_history
from utils.comparison_utils import compare_models
from models.fc_models import SimpleFC

def evaluate_time(model, loader, device):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            _ = model(data)
    end = time.time()

    return end - start

def plot_confusion_matrix(model, loader, device, class_names, save_path="homework_4/plots/", filename=""):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    cm = confusion_matrix(np.concatenate(all_labels), np.concatenate(all_preds))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig(f"{save_path}{filename}")
    plt.close()

def plot_gradient_flow(model):
    ave_grads = []
    layers = []
    for n, p in model.named_parameters():
        if p.requires_grad and "bias" not in n:
            if p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
    plt.figure(figsize=(10, 4))
    plt.plot(ave_grads, alpha=0.7, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0, len(ave_grads)), layers, rotation='vertical')
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=0)
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("homework_4/plots/gradient_flow.png")
    plt.close()

def run_experiment(model_class, dataset_fn, input_size=None, input_channels=1, num_classes=10, device='cpu', epochs=5):
    train_loader, test_loader = dataset_fn(batch_size=64)
    if model_class == SimpleFC:
        model = model_class(input_size=input_size, num_classes=num_classes).to(device)
    elif model_class == CIFARCNN:
        model = model_class(num_classes=num_classes).to(device)
    else:
        model = model_class(input_channels=input_channels, num_classes=num_classes).to(device)

    print(f"Модель: {model_class.__name__}, Параметры: {count_parameters(model)}")
    start_train = time.time()
    history = train_model(model, train_loader, test_loader, epochs=epochs, device=device)
    end_train = time.time()
    plot_gradient_flow(model)
    inference_time = evaluate_time(model, test_loader, device)
    print(f"Время обучения: {end_train - start_train:.2f} секунд")
    print(f"Время работы: {inference_time:.2f} секунд")
    plot_training_history(history, save_path="homework_4/plots/", filename=f"{model_class.__name__}_mnist.png")

    return model, history


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    print("\nMNIST")
    fc_model, fc_history = run_experiment(SimpleFC, get_mnist_loaders, input_size=28*28, input_channels=1, num_classes=10, device=device)
    simple_cnn_model, simple_cnn_history = run_experiment(SimpleCNN, get_mnist_loaders, input_channels=1, num_classes=10, device=device)
    residual_model, residual_history = run_experiment(CNNWithResidual, get_mnist_loaders, input_channels=1, num_classes=10, device=device)

    print("\nCNN vs FC on MNIST")
    compare_models(fc_history, simple_cnn_history, save_path="homework_4/plots/", filename="fc_snn_mnist_comparison.png")
    compare_models(fc_history, residual_history, save_path="homework_4/plots/", filename="fc_residual_mnist_comparison.png")

    print("\nCIFAR-10")
    cifar_fc_model, cifar_fc_history = run_experiment(SimpleFC, get_cifar_loaders, input_size=32*32*3, input_channels=3, num_classes=10, device=device)
    cifar_res_model, cifar_res_history = run_experiment(CNNWithResidual, get_cifar_loaders, input_channels=3, num_classes=10, device=device)
    cifar_reg_res_model, cifar_reg_res_history = run_experiment(CIFARCNN, get_cifar_loaders, num_classes=10, device=device)

    print("\nCNN vs FC on CIFAR-10")
    compare_models(cifar_fc_history, cifar_res_history, save_path="homework_4/plots/", filename="fc_snn_cifar_comparison.png")
    compare_models(cifar_fc_history, cifar_reg_res_history, save_path="homework_4/plots/", filename="fc_residual_cifar_comparison.png")

    print("\nConfusion Matrix for CIFAR-10 CNN:")
    plot_confusion_matrix(cifar_reg_res_model, get_cifar_loaders(batch_size=64)[1], device, class_names=[
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    ], save_path="homework_4/plots/", filename="confusion_mat_cifar_cnn.png")