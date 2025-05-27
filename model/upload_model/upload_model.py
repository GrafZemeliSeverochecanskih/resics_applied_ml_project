from model.abstract.abstract_model import AbstractModel
import torch
import torchvision
from model.new_model.resnet50_custom_model import ResNet50CustomModel
from pathlib import Path
from PIL import Image
import io

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
        self.__model.eval()
        img = Image.open(image_path).convert("RGB")
        transform = self.__model.weights
        img_tensor = transform(img).unsqueeze(0).to(self.__device)

        with torch.no_grad():
            output_logits = self.__model(img_tensor)
    
        pred_index = torch.argmax(output_logits, dim=1).item()
        predicted_class_name = self.class_names[pred_index]
        
        return predicted_class_name

    def predict_image_with_probabilities(self, image_bytes: bytes):
        self.__model.eval()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = self.__model.weights
        img_tensor = transform(img).unsqueeze(0).to(self.__device)
        with torch.no_grad():
            output_logits = self.__model(img_tensor)
        probabilities = torch.softmax(output_logits, dim = 1)
        top_prob, top_pred_idx = torch.max(probabilities, dim = 1)
        predicted_class_name = self.class_names[top_pred_idx.item()]

        all_probs = probabilities[0].cpu().numpy()
        class_probabilities = sorted(
            [(self.class_names[i], all_probs[i] * 100) for i in range(len(self.class_names))],
            key = lambda item: item[1],
            reverse = True
        )
        return predicted_class_name, class_probabilities
    
    def predict_and_explain(self, image_bytes: bytes):
        """
        Processes image bytes, predicts, gets probabilities, and generates saliency visualization.
        """
        self.__model.eval()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = self.__model.weights 
        input_tensor = transform(img).unsqueeze(0).to(self.__device)

        self.__model.model.eval()
        with torch.no_grad():
            output_logits = self.__model.model(input_tensor.clone().detach())
        
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

        saliency_viz_b64 = self.__model.get_saliency_visualization(
            input_tensor, top_pred_index
        )
        
        return predicted_class_name, class_probabilities, saliency_viz_b64