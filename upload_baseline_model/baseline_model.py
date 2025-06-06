import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from typing import List, Tuple

class UploadCNN:
    """
    A class to load a pre-trained CNN model and make predictions on images.
    
    This class is designed to work with the CNN architecture defined in
    the original `CNNBaselineModel`. It handles model initialization,
    loading of saved weights, and image preprocessing for inference.
    """
    def __init__(self, 
                 model_weights_path: str,
                 num_filters: int = 16,
                 input_size: Tuple[int, int] = (224, 224)):
        """
        Initializes the UploadCNN predictor.

        Args:
            model_weights_path (str): Path to the saved model weights (.pth file).
            num_filters (int): The number of filters used in the convolutional
                               layers. This must match the trained model's architecture.
            input_size (Tuple[int, int]): The height and width the model expects
                                          for input images. Should be (224, 224) for the provided model.
        """
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
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
        self.__num_classes = 45
        self.__input_size = input_size
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=num_filters * (self.__input_size[0] // 2) * (self.__input_size[1] // 2), 
                      out_features=self.__num_classes)
        ).to(self.__device)

        print(f"Loading model weights from: {model_weights_path}")
        try:
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.__device, weights_only=True))
        except FileNotFoundError:
            print(f"Error: Model weights file not found at '{model_weights_path}'.")
            raise
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Please ensure the model architecture (num_filters, input_size) matches the saved weights.")
            raise

        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(self.__input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Loads and preprocesses an image for model inference.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return self.transform(image).unsqueeze(0)
        except FileNotFoundError:
            print(f"Error: Image file not found at '{image_path}'.")
            raise

    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Makes a prediction on a single image.
        """
        image_tensor = self.__preprocess_image(image_path)
        image_tensor = image_tensor.to(self.__device)

        with torch.inference_mode():
            output_logits = self.model(image_tensor)
            output_probs = torch.softmax(output_logits, dim=1)
            confidence, pred_index = torch.max(output_probs, dim=1)
            
        predicted_class = self.__class_names[pred_index.item()]
        confidence_score = confidence.item()

        return predicted_class, confidence_score