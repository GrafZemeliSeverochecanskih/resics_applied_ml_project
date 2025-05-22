import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

def compute_saliency_map(input_tensor, model, target_class=None):
    model.eval()
    input_tensor.requires_grad_()

    # Forward pass
    output = model(input_tensor)
    
    if target_class is None:
        target_class = output.argmax(dim=1)

    loss = output[0, target_class]
    model.zero_grad()
    loss.backward()

    saliency = input_tensor.grad.data.abs()
    saliency, _ = torch.max(saliency, dim=1)
    return saliency.squeeze().cpu().numpy()

def visualize_saliency_map(saliency, image_tensor):
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())  # Normalize for display

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Saliency Map")
    plt.imshow(saliency, cmap='hot')
    plt.axis("off")

    plt.tight_layout()
    plt.show()