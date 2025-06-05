import torch
from torch import nn
import torchvision
from torchinfo import summary
from PIL import Image
from pathlib import Path
import io
from collections import OrderedDict
import numpy as np
import base64
import matplotlib.cm as cm

class UploadModel(torch.nn.Module):
    def __init__(self,
                 weights_path, 
                 dropout_rate: float = 0.5):
        super().__init__()
        self.__weights_path = Path(weights_path)
        self.__class_names = sorted([
            "airplane", "airport", "baseball_diamond", 
            "basketball_court", "beach", "bridge", "chaparral", 
            "church", "circular_farmland", "cloud", "commercial_area",
            "dense_residential", "desert", "forest", "freeway",
            "golf_course", "ground_track_field", "harbor", 
            "industrial_area", "intersection", "island", "lake", 
            "meadow", "medium_residential", "mobile_home_park", "mountain", 
            "overpass", "palace", "parking_lot", "railway", "railway_station",
            "rectangular_farmland", "river", "roundabout", "runway", 
            "sea_ice", "ship", "snowberg", "sparse_residential", "stadium", 
            "storage_tank", "tennis_court", "terrace", "thermal_power_station", 
            "wetland"
        ])
        self.__dropout_rate = dropout_rate
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__weights_transformed = None
        self.__initializeModel()


    def __initializeModel(self):    
        weights_enum = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.__weights_transformed = weights_enum.transforms()
        self.__model = torchvision.models.resnet50(
            weights=weights_enum
            ).to(self.__device)
        num_ftrs = self.__model.fc.in_features
        
        new_classifier = nn.Sequential(
            nn.Dropout(p=self.__dropout_rate),
            nn.Linear(num_ftrs, 45)
        ).to(self.__device)
        
        self.__model.fc = new_classifier
        load_state_dict = torch.load(self.__weights_path, 
                                    map_location=torch.device(self.__device), 
                                    weights_only=True)
        new_state_dict = OrderedDict()
        prefix_to_remove = "_ResNet50CustomModel__model."
        for key, value in load_state_dict.items():
            if key.startswith(prefix_to_remove):
                new_key = key[len(prefix_to_remove):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        try:
            self.__model.load_state_dict(new_state_dict)
        except RuntimeError as e:
            print("Still encountering a RuntimeError after attempting to fix keys.")
            print("This might be due to differences in model architecture beyond key names.")
            print("Original error:", e)
            raise e
        
        self.__model.eval()

    def forward(self, x):
        """This is the forward pass of the model."""
        return self.__model(x)
     
    @property
    def class_names(self):
        return self.__class_names
    
    @property
    def weights(self):
        return self.__weights_transformed

    def predict_single_image(self, image_path: Path):
        """
        Takes a path to a single image, processes it, and returns the predicted class name.
        """
        self.__model.eval()
        img = Image.open(image_path).convert("RGB")
        transform = self.weights
        img_tensor = transform(img).unsqueeze(0).to(self.__device)

        with torch.no_grad():
            output_logits = self.__model(img_tensor)
    
        pred_index = torch.argmax(output_logits, dim=1).item()
        predicted_class_name = self.class_names[pred_index]
        
        return predicted_class_name

    def __denormalize_tensor_to_pil(self, tensor_batch: torch.Tensor) -> Image.Image:
        """
        Denormalizes an image tensor (normalized with ImageNet stats) and converts it to a PIL Image.
        Assumes tensor_batch is [1, C, H, W].
        """
        if tensor_batch is None or tensor_batch.numel() == 0:
            return Image.new('RGB', (224, 224), color = 'grey')

        img_tensor = tensor_batch.squeeze(0).cpu().clone()
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device)

        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)
        to_pil = torchvision.transforms.ToPILImage()
        pil_image = to_pil(img_tensor)
        return pil_image

    def get_saliency_visualization(self, input_tensor: torch.Tensor, target_class_idx: int) -> str:
        """
        Generates a vanilla gradient saliency map, applies a colormap,
        overlays it on the original image, and returns a base64 encoded PNG image string.
        """
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()

        input_tensor_clone = input_tensor.clone().detach().requires_grad_(True)
        self.__model.eval()
        output = self.__model(input_tensor_clone)

        if target_class_idx is None:
            target_class_idx = torch.argmax(output, dim=1).item()

        score_for_target_class = output[0, target_class_idx]
        self.__model.zero_grad()
        score_for_target_class.backward()
        saliency_gradients = input_tensor_clone.grad.data
        
        if saliency_gradients is None:
            print("Warning: Gradients for saliency map are None.")
            return ""

        saliency_gradients = torch.abs(saliency_gradients)
        saliency_map, _ = torch.max(saliency_gradients, dim=1) 
        saliency_map_np = saliency_map.squeeze().cpu().numpy()
        min_val = np.min(saliency_map_np)
        max_val = np.max(saliency_map_np)
        saliency_map_0_to_1 = saliency_map_np
        if max_val - min_val > 1e-8:
            saliency_map_0_to_1 = (saliency_map_np - min_val) / (max_val - min_val)
        
        heatmap_rgba_norm = cm.jet(saliency_map_0_to_1)
        heatmap_rgba_uint8 = (heatmap_rgba_norm * 255).astype(np.uint8)
        saliency_pil_rgba = Image.fromarray(heatmap_rgba_uint8, mode='RGBA')

        original_pil_rgb = self.__denormalize_tensor_to_pil(input_tensor)
        original_pil_rgba = original_pil_rgb.convert('RGBA')
        if saliency_pil_rgba.size != original_pil_rgba.size:
            saliency_pil_rgba = saliency_pil_rgba.resize(original_pil_rgba.size, Image.Resampling.LANCZOS)

        alpha_intensity_scale = 25
        
        r, g, b, a = saliency_pil_rgba.split()
        new_alpha_data = (saliency_map_0_to_1 * alpha_intensity_scale * 255).astype(np.uint8)
        new_alpha_channel = Image.fromarray(new_alpha_data, mode='L')
        saliency_pil_rgba.putalpha(new_alpha_channel)

        blended_image_pil = Image.alpha_composite(original_pil_rgba, saliency_pil_rgba)

        buffered = io.BytesIO()
        blended_image_pil.convert('RGB').save(buffered, format="PNG")
        saliency_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return saliency_base64
    
    def predict_and_explain(self, image_bytes: bytes):
        """
        Processes image bytes, predicts, gets probabilities, and generates saliency visualization.
        """
        self.__model.eval()

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        if self.__weights_transformed is None:
            weights_enum = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            transform = weights_enum.transforms()
        else:
            transform = self.__weights_transformed 
        
        input_tensor = transform(img).unsqueeze(0).to(self.__device)

        with torch.no_grad():
            output_logits = self.__model(input_tensor.clone().detach()) 
        
        probabilities = torch.softmax(output_logits, dim=1)
        _, top_pred_index_tensor = torch.max(probabilities, dim=1)
        top_pred_index = top_pred_index_tensor.item()
        predicted_class_name = self.class_names[top_pred_index]
        
        all_probs_numpy = probabilities[0].cpu().numpy()
        class_probabilities = sorted(
            [(self.class_names[i], all_probs_numpy[i] * 100) for i in range(len(self.class_names))],
            key=lambda item: item[1],
            reverse=True
        )

        saliency_viz_b64 = self.get_saliency_visualization(
            input_tensor, top_pred_index
        )
        
        return predicted_class_name, class_probabilities, saliency_viz_b64