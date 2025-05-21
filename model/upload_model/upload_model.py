from abstract.abstract_model import AbstractModel
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from new_model.resnet50_custom_model import ResNet50CustomModel
from utils.mc_dropout import MonteCarloDropout

class UploadResNet50Model(AbstractModel):
    """This class is responsible for creating a ResNet-50 Base model basing 
    on the existing weights. 

    Args:
        AbstractModel (_type_): Abstract model.
    """
    def __init__(self,
                 weight_path: str,
                 dropout_rate: float = 0.5,
                 unfreeze_specific_blocks: list[str] = None,
                 unfreeze_classifier: bool = True
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
            dropout_rate=dropout_rate,
            unfreeze_specific_blocks = unfreeze_specific_blocks,
            unfreeze_classifier = unfreeze_classifier
        ) 
        self.__model.load_state_dict(torch.load(weight_path, map_location=torch.device(self.__device)))
        self.__model.to(self.__device)
        print(f"Model loaded successfully from '{weight_path}' onto '{self.__device}' and set to evaluation mode.")
    
    @property
    def model(self):
        return self.__model
    
    def run_mc_dropout_single_image(self, image_tensor: torch.Tensor, num_samples: int = 30):
        """
        Run MC Dropout on a single image using the loaded model.

        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor [3, 224, 224]
            num_samples (int): Number of forward passes to perform

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor]: predicted class index, mean probs, and uncertainty
        """
        mc_dropout = MonteCarloDropout(model=self.__model, data_loader=None, num_samples=num_samples)
        return mc_dropout.predict_single_image(image_tensor)
