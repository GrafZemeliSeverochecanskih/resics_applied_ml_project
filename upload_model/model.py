import torch
from torch import nn
import torchvision
from torchinfo import summary
from PIL import Image
from pathlib import Path
import io
from collections import OrderedDict, Counter
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

        self.__feature_maps = None
        self.__gradients = None

    def __initializeModel(self):    
        weights_enum = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.__weights_transformed = weights_enum.transforms()
        self.__model = torchvision.models.resnet50(
            weights=weights_enum
            ).to(self.__device)
        
        target_layer = self.__model.layer4[-1]
        target_layer.register_forward_hook(self.__forward_hook)
        target_layer.register_full_backward_hook(self.__backward_hook)

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

    def __forward_hook(self, module, input, output):
        self.__feature_maps = output.detach()

    def __backward_hook(self, module, grad_in, grad_out):
        self.__gradients = grad_out[0].detach()

    def __create_saliency_overlay(self, saliency_map_np: np.ndarray, input_tensor: torch.Tensor) -> str:
        """
        Helper function to create a base64-encoded saliency map overlay.
        """
        # Normalize the saliency map
        min_val, max_val = np.min(saliency_map_np), np.max(saliency_map_np)
        if max_val - min_val > 1e-8:
            saliency_map_0_to_1 = (saliency_map_np - min_val) / (max_val - min_val)
        else:
            saliency_map_0_to_1 = saliency_map_np

        # Apply colormap and convert to PIL image
        heatmap_rgba_norm = cm.jet(saliency_map_0_to_1)
        heatmap_rgba_uint8 = (heatmap_rgba_norm * 255).astype(np.uint8)
        saliency_pil_rgba = Image.fromarray(heatmap_rgba_uint8, mode='RGBA')

        # Denormalize original image and convert to PIL
        original_pil_rgb = self.__denormalize_tensor_to_pil(input_tensor)
        original_pil_rgba = original_pil_rgb.convert('RGBA')
        
        if saliency_pil_rgba.size != original_pil_rgba.size:
            saliency_pil_rgba = saliency_pil_rgba.resize(original_pil_rgba.size, Image.Resampling.LANCZOS)

        # Blend the images
        blended_image_pil = Image.alpha_composite(original_pil_rgba, saliency_pil_rgba)
        
        # Convert to base64
        buffered = io.BytesIO()
        blended_image_pil.convert('RGB').save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def get_grad_cam_visualization(self, input_tensor: torch.Tensor, target_class_idx: int) -> str:
        """
        Generates a Grad-CAM visualization.
        """
        self.__model.eval()
        output = self.__model(input_tensor)

        if target_class_idx is None:
            target_class_idx = torch.argmax(output, dim=1).item()

        score_for_target_class = output[0, target_class_idx]
        self.__model.zero_grad()
        score_for_target_class.backward(retain_graph=True)

        if self.__gradients is None or self.__feature_maps is None:
            return ""

        pooled_gradients = torch.mean(self.__gradients, dim=[2, 3])
        
        for i in range(self.__feature_maps.shape[1]):
            self.__feature_maps[:, i, :, :] *= pooled_gradients[0, i]
            
        heatmap = torch.mean(self.__feature_maps, dim=1).squeeze()
        heatmap = nn.functional.relu(heatmap)
    
        heatmap_resized = nn.functional.interpolate(heatmap.unsqueeze(0).unsqueeze(0), 
                                        size=(input_tensor.shape[2], input_tensor.shape[3]), 
                                        mode='bilinear', align_corners=False)

        return self.__create_saliency_overlay(heatmap_resized.squeeze().cpu().numpy(), input_tensor)

    def get_normgrad_visualization(self, input_tensor: torch.Tensor, target_class_idx: int) -> str:
        """
        Generates a NormGrad (Gradient * Input) saliency map.
        """
        input_tensor_clone = input_tensor.clone().detach().requires_grad_(True)
        self.__model.eval()
        output = self.__model(input_tensor_clone)

        if target_class_idx is None:
            target_class_idx = torch.argmax(output, dim=1).item()

        score_for_target_class = output[0, target_class_idx]
        self.__model.zero_grad()
        score_for_target_class.backward()
        
        if input_tensor_clone.grad is None:
            return ""

        saliency_gradients = input_tensor_clone.grad.data * input_tensor_clone.data
        saliency_map, _ = torch.max(saliency_gradients.abs(), dim=1)
        
        return self.__create_saliency_overlay(saliency_map.squeeze().cpu().numpy(), input_tensor)
        
    def predict_and_explain(self, image_bytes: bytes) -> tuple[str, list[tuple[str, float]], str, str]:
        """
        Processes image, predicts, and generates Grad-CAM and NormGrad visualizations.
        """
        self.__model.eval()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.__weights_transformed(img).unsqueeze(0).to(self.__device)

        with torch.no_grad():
            output_logits = self.__model(input_tensor.clone().detach()) 
        
        probabilities = torch.softmax(output_logits, dim=1)
        top_pred_index = torch.argmax(probabilities, dim=1).item()
        predicted_class_name = self.class_names[top_pred_index]
        
        class_probabilities = sorted(
            [(self.class_names[i], probabilities[0, i].item() * 100) for i in range(len(self.class_names))],
            key=lambda item: item[1],
            reverse=True
        )

        grad_cam_viz_b64 = self.get_grad_cam_visualization(input_tensor, top_pred_index)
        normgrad_viz_b64 = self.get_normgrad_visualization(input_tensor, top_pred_index)
        
        return predicted_class_name, class_probabilities[:5], grad_cam_viz_b64, normgrad_viz_b64

    def __denormalize_tensor_to_pil(self, tensor_batch: torch.Tensor) -> Image.Image:
        """
        Denormalizes an image tensor and converts it to a PIL Image.
        """
        if tensor_batch is None or tensor_batch.numel() == 0:
            return Image.new('RGB', (224, 224), color='grey')
        
        img_tensor = tensor_batch.squeeze(0).cpu().clone()
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(-1, 1, 1)
        
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)
        return torchvision.transforms.ToPILImage()(img_tensor)

    def __enable_dropout(self):
        """
        Enables dropout layers during evaluation, which is necessary for
        Monte Carlo Dropout. By default, `model.eval()` disables dropout.
        """
        for m in self.__model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def predict_with_mc_dropout(self, image_bytes: bytes, num_samples: int = 25) -> tuple[str, float, list[tuple[str, float]]]:
        """
        Performs inference using Monte Carlo Dropout to estimate model uncertainty.

        This method runs multiple forward passes on the input image with dropout enabled.
        It then calculates the most frequent prediction (mode) and its probability.

        Args:
            image_bytes (bytes): The input image in bytes format.
            num_samples (int, optional): The number of forward passes to perform. Defaults to 25.

        Returns:
            A tuple containing:
            - most_common_prediction (str): The name of the most frequently predicted class.
            - probability (float): The probability of the most common prediction, calculated as (count / num_samples).
            - mean_probabilities (List[Tuple[str, float]]): A sorted list of the top 5 classes and their mean probabilities across all samples.
        """
        self.__model.eval()
        self.__enable_dropout()

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.__weights_transformed(img).unsqueeze(0).to(self.__device)
        
        predictions = []
        all_probs = []

        with torch.no_grad():
            for _ in range(num_samples):
                output_logits = self.__model(input_tensor)
                output_probs = torch.softmax(output_logits, dim=1)
                
                all_probs.append(output_probs)
                
                pred_index = torch.argmax(output_probs, dim=1).item()
                predictions.append(self.class_names[pred_index])

        prediction_counts = Counter(predictions)
        most_common_prediction, count = prediction_counts.most_common(1)[0]
        probability = count / num_samples
        mean_probs_tensor = torch.mean(torch.cat(all_probs), dim=0)
        mean_probs_numpy = mean_probs_tensor.cpu().numpy()
        
        all_mean_probabilities = sorted(
            [(self.class_names[i], mean_probs_numpy[i]) for i in range(len(self.class_names))],
            key=lambda item: item[1],
            reverse=True
        )
        top_5_probabilities = all_mean_probabilities[:5]

        return most_common_prediction, probability, top_5_probabilities
    
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

        with torch.inference_mode():
            output_logits = self.__model(img_tensor)
            output_probs = torch.softmax(output_logits, dim=1)
            confidence, pred_index = torch.max(output_probs, dim=1)
                
        predicted_class_name = self.class_names[pred_index.item()]
        confidence_score = confidence.item()
        
        return predicted_class_name, confidence_score

    # def __denormalize_tensor_to_pil(self, tensor_batch: torch.Tensor) -> Image.Image:
    #     """
    #     Denormalizes an image tensor (normalized with ImageNet stats) and converts it to a PIL Image.
    #     Assumes tensor_batch is [1, C, H, W].
    #     """
    #     if tensor_batch is None or tensor_batch.numel() == 0:
    #         return Image.new('RGB', (224, 224), color = 'grey')

    #     img_tensor = tensor_batch.squeeze(0).cpu().clone()
    #     mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device)
    #     std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device)

    #     mean = mean.view(-1, 1, 1)
    #     std = std.view(-1, 1, 1)
        
    #     img_tensor = img_tensor * std + mean
    #     img_tensor = torch.clamp(img_tensor, 0, 1)
    #     to_pil = torchvision.transforms.ToPILImage()
    #     pil_image = to_pil(img_tensor)
    #     return pil_image

    # def get_saliency_visualization(self, input_tensor: torch.Tensor, target_class_idx: int) -> str:
    #     """
    #     Generates a vanilla gradient saliency map, applies a colormap,
    #     overlays it on the original image, and returns a base64 encoded PNG image string.
    #     """
    #     if input_tensor.grad is not None:
    #         input_tensor.grad.zero_()

    #     input_tensor_clone = input_tensor.clone().detach().requires_grad_(True)
    #     self.__model.eval()
    #     output = self.__model(input_tensor_clone)

    #     if target_class_idx is None:
    #         target_class_idx = torch.argmax(output, dim=1).item()

    #     score_for_target_class = output[0, target_class_idx]
    #     self.__model.zero_grad()
    #     score_for_target_class.backward()
    #     saliency_gradients = input_tensor_clone.grad.data
        
    #     if saliency_gradients is None:
    #         print("Warning: Gradients for saliency map are None.")
    #         return ""

    #     saliency_gradients = torch.abs(saliency_gradients)
    #     saliency_map, _ = torch.max(saliency_gradients, dim=1) 
    #     saliency_map_np = saliency_map.squeeze().cpu().numpy()
    #     min_val = np.min(saliency_map_np)
    #     max_val = np.max(saliency_map_np)
    #     saliency_map_0_to_1 = saliency_map_np
    #     if max_val - min_val > 1e-8:
    #         saliency_map_0_to_1 = (saliency_map_np - min_val) / (max_val - min_val)
        
    #     heatmap_rgba_norm = cm.jet(saliency_map_0_to_1)
    #     heatmap_rgba_uint8 = (heatmap_rgba_norm * 255).astype(np.uint8)
    #     saliency_pil_rgba = Image.fromarray(heatmap_rgba_uint8, mode='RGBA')

    #     original_pil_rgb = self.__denormalize_tensor_to_pil(input_tensor)
    #     original_pil_rgba = original_pil_rgb.convert('RGBA')
    #     if saliency_pil_rgba.size != original_pil_rgba.size:
    #         saliency_pil_rgba = saliency_pil_rgba.resize(original_pil_rgba.size, Image.Resampling.LANCZOS)

    #     alpha_intensity_scale = 25
        
    #     r, g, b, a = saliency_pil_rgba.split()
    #     new_alpha_data = (saliency_map_0_to_1 * alpha_intensity_scale * 255).astype(np.uint8)
    #     new_alpha_channel = Image.fromarray(new_alpha_data, mode='L')
    #     saliency_pil_rgba.putalpha(new_alpha_channel)

    #     blended_image_pil = Image.alpha_composite(original_pil_rgba, saliency_pil_rgba)

    #     buffered = io.BytesIO()
    #     blended_image_pil.convert('RGB').save(buffered, format="PNG")
    #     saliency_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    #     return saliency_base64
    
    # def predict_and_explain(self, image_bytes: bytes):
    #     """
    #     Processes image bytes, predicts, gets probabilities, and generates saliency visualization.
    #     """
    #     self.__model.eval()

    #     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
    #     if self.__weights_transformed is None:
    #         weights_enum = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    #         transform = weights_enum.transforms()
    #     else:
    #         transform = self.__weights_transformed 
        
    #     input_tensor = transform(img).unsqueeze(0).to(self.__device)

    #     with torch.no_grad():
    #         output_logits = self.__model(input_tensor.clone().detach()) 
        
    #     probabilities = torch.softmax(output_logits, dim=1)
    #     _, top_pred_index_tensor = torch.max(probabilities, dim=1)
    #     top_pred_index = top_pred_index_tensor.item()
    #     predicted_class_name = self.class_names[top_pred_index]
        
    #     all_probs_numpy = probabilities[0].cpu().numpy()
    #     class_probabilities = sorted(
    #         [(self.class_names[i], all_probs_numpy[i] * 100) for i in range(len(self.class_names))],
    #         key=lambda item: item[1],
    #         reverse=True
    #     )

    #     saliency_viz_b64 = self.get_saliency_visualization(
    #         input_tensor, top_pred_index
    #     )
        
    #     return predicted_class_name, class_probabilities, saliency_viz_b64