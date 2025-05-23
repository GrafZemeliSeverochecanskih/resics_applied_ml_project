from model.abstract.abstract_model import AbstractModel
import torch
import torchvision
from model.new_model.resnet50_custom_model import ResNet50CustomModel
from pathlib import Path
from PIL import Image

class UploadResNet50Model(AbstractModel):
    """This class is responsible for creating a ResNet-50 Base model basing 
    on the existing weights. 

    Args:
        AbstractModel (_type_): Abstract model.
    """
    def __init__(self,
                 weight_path: str,
                 image_path: str,
                 dropout_rate: float = 0.5,
                 unfreeze_specific_blocks: list[str] = None,
                 unfreeze_classifier: bool = False
                 ):
        """This is the UploadResNet50Model constructor.

        Args:
            weight_path (str): path to the file with the UploadResNet50Model weights.
            dropout_rate (float, optional): dropout rate. Defaults to 0.5.
            unfreeze_specific_blocks (list[str], optional): list of blocks that 
            need to be unfreezed. Defaults to None.
            unfreeze_classifier (bool, optional): whether the classifier should 
            be unfreezed or not. Defaults to True.
        """
        super().__init__()
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model = ResNet50CustomModel(
            image_path,
            dropout_rate=dropout_rate,
            unfreeze_specific_blocks = unfreeze_specific_blocks,
            unfreeze_classifier = unfreeze_classifier
        ) 
        self.__model.load_state_dict(torch.load(weight_path, 
                                                map_location=torch.device(self.__device), 
                                                weights_only=False))
        self.__model.to(self.__device)
        print(f"Model loaded successfully from '{weight_path}' onto '{self.__device}' and set to evaluation mode.")
    
    @property
    def model(self):
        return self.__model
    
    @property
    def summary(self):
        return self.__model.model_summary
    
    @property
    def test_accuracy(self):
        return self.__model.test_accuracy
    
    @property
    def class_names(self):
        return sorted(self.__model.class_names)
    
    def get_image_path(self, class_index: int, image_index: int):
        class_list = self.class_names
        if not 0 <= class_index < len(class_list):
            raise IndexError("Class index is out of range.")
        
        target_class_name = class_list[class_index]
        class_folder_path = self.__model.test_directory / target_class_name

        if not class_folder_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {class_folder_path}")
        
        image_files = sorted([p for p in class_folder_path.glob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if not 0 <= image_index < len(image_files):
            raise IndexError("Image index is out of range for this class.")
            
        return image_files[image_index]
    
    def predict_single_image(self, image_path: Path):
        """
        Takes a path to a single image, processes it, and returns the predicted class name.
        """
        # Set the model to evaluation mode
        self.__model.eval()

        # Open image with Pillow
        img = Image.open(image_path).convert("RGB")
        
        # Get the correct transformations from the model
        transform = self.__model.weights

        # Apply transformations and add a batch dimension (B, C, H, W)
        img_tensor = transform(img).unsqueeze(0).to(self.__device)

        with torch.no_grad():
            # Get raw model output (logits)
            output_logits = self.__model(img_tensor)
        
        # Get the predicted class index
        pred_index = torch.argmax(output_logits, dim=1).item()
        
        # Convert index to class name
        predicted_class_name = self.class_names[pred_index]
        
        return predicted_class_name
