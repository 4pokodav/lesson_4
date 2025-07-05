import torch
import matplotlib.pyplot as plt

def visualize_activations(layer, input_tensor, save_path):
    with torch.no_grad():
        output = layer(input_tensor)
        fig, axes = plt.subplots(1, min(8, output.shape[1]), figsize=(15, 4))
        for i in range(min(8, output.shape[1])):
            axes[i].imshow(output[0, i].cpu(), cmap='viridis')
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def visualize_feature_maps(conv_block, input_tensor, save_path, is_tensor=False):
    if not is_tensor:
        with torch.no_grad():
            x = input_tensor
            for layer in conv_block:
                x = layer(x)
            fig, axes = plt.subplots(1, min(8, x.shape[1]), figsize=(15, 4))
            for i in range(min(8, x.shape[1])):
                axes[i].imshow(x[0, i].cpu(), cmap='inferno')
                axes[i].axis('off')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
    else:
        activations = conv_block.squeeze(0).cpu()
        fig, axes = plt.subplots(1, min(8, activations.shape[0]), figsize=(15, 5))
        for i, ax in enumerate(axes):
            ax.imshow(activations[i], cmap='viridis')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()        

def plot_accuracy(results_dict, save_path, title):
    """
    Строит графики accuracy по эпохам для разных моделей.
    """
    plt.figure(figsize=(10, 6))

    for label, history in results_dict.items():
        if 'train_acc' in history:
            plt.plot(history['train_acc'], linestyle='--', label=f'{label} - train')
        if 'test_acc' in history:
            plt.plot(history['test_acc'], linestyle='-', label=f'{label} - test')

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_history(history, save_path=None, filename=None):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}{filename}")
    plt.close()

def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path):
    """Сохраняет модель"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Загружает модель"""
    model.load_state_dict(torch.load(path))
    return model