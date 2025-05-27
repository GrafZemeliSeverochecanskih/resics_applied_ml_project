import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

class SaliencyGenerator:
    def __init__(self, model):
        """
        Initializes the SaliencyGenerator.
        Args:
            model: The PyTorch model (e.g., the actual ResNet50 instance) for which to generate saliency.
        """
        self.__model = model
        self.__model.eval()
    def compute_saliency_map_data(self, input_tensor, target_class=None):
        """
        Computes the saliency map data for a given input tensor.
        Args:
            input_tensor (torch.Tensor): The preprocessed input image tensor (requires_grad should be set).
            target_class (int, optional): The target class index for saliency. If None, uses the predicted class.
        Returns:
            np.array: The computed saliency map as a NumPy array.
            int: The class index used for generating the saliency map.
        """
        input_tensor.requires_grad_()

        output = self.__model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        
        self.__model.zero_grad()
        score.backward()

        saliency = input_tensor.grad.data.abs()
        
        saliency_max_channels, _ = torch.max(saliency, dim=1)
        
        return saliency_max_channels.squeeze().cpu().numpy(), target_class

    def create_saliency_visualization_base64(self, original_image_tensor, saliency_map_data):
        """
        Generates a single image containing the original photo and its saliency map,
        and returns it as a Base64 encoded string.
        Args:
            original_image_tensor (torch.Tensor): The original preprocessed image tensor (for display).
            saliency_map_data (np.array): The saliency map data.
        Returns:
            str: Base64 encoded string of the visualization PNG image.
        """
        image_for_display = original_image_tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        image_for_display = (image_for_display - image_for_display.min()) / (image_for_display.max() - image_for_display.min() + 1e-6)


        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image_for_display)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        axes[1].imshow(image_for_display) 
        axes[1].imshow(saliency_map_data, cmap='hot', alpha=0.6)
        axes[1].set_title("Saliency Map")
        axes[1].axis('off')

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_base64