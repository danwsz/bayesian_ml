import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from omegaconf import OmegaConf
import json


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_config(filename='config.json'):
    with open(filename, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    main_conf = OmegaConf.create(json_data)
    main_conf.cuda = torch.cuda.is_available()
    return main_conf


def plot_reconstructions(original, reconstructed, n=8):
    fig, axes = plt.subplots(2, n, figsize=(n, 2))
    for i in range(n):
        axes[0, i].imshow(original[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].axis('off')

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstructed")
    plt.tight_layout()
    plt.savefig("mnist_reconstructions.png") 
    plt.show()
