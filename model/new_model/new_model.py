from abstract_model import AbstractModel
import torch
import torchvision
from torchinfo import summary
from torch import nn

class NewResNet50Model(AbstractModel):
    """This class allows user to create new model basing on the 
        ResNet-50 architecture. User is able to select what layers to unfreeze and select the 
        rate of dropout.

    Args:
        AbstractModel (_type_): abstract model.
    """
    def __init__(self, 
                 dropout_rate: float = 0.5,
                 unfreeze_specific_blocks: list[str] = None,
                 unfreeze_classifier: bool = True
                 ):
        """This is NewResNet50Model class constructor.

        Args:
            dropout_rate (float, optional): dropout rate. Defaults to 0.5.
            unfreeze_specific_blocks (list[str], optional): layers that need to be unfreezed. Defaults to None.
            unfreeze_classifier (bool, optional): whether unfreeze classifier or not. Defaults to True.
        """
        super().__init__()
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__unfreeze_classifier = unfreeze_classifier
        self.__unfreeze_specific_blocks = unfreeze_specific_blocks
        self.__dropout_rate = dropout_rate

        weights_enum = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.__weights_transformed = weights_enum.transforms()
        self.__model =  torchvision.models.resnet50(weights=weights_enum).to(self.__device)
        
        self.__freeze_layers()
        self.__modify_last_layer()

    def __freeze_layers(self):
        """This method freezes the layers, in case user does not want to trani model."""
        for param in self.__model.parameters():
            param.requires_grad = False

        if self.__unfreeze_specific_blocks:
            for block_name in self.__unfreeze_specific_blocks:
                if hasattr(self.__model, block_name):
                    print(f"Unfreezing parameters for: {block_name}")
                    for param in getattr(self.__model, block_name).parameters():
                        param.requires_grad = True
                else:
                    print(f"Warning: Block '{block_name}' not found in model. Available top-level blocks: "
                          f"{[name for name, _ in self.__model.named_children()]}")

    def __modify_last_layer(self):
        """This method modifies last layers of ResNet-50 architecture."""
        num_ftrs = self.__model.fc.in_features
        
        new_classifier = nn.Sequential(
            nn.Dropout(p=self.__dropout_rate),
            nn.Linear(num_ftrs, 45)
        ).to(self.__device)

        self.__model.fc = new_classifier

        if self.__unfreeze_classifier:
            print("Ensuring the new classifier head is trainable.")
            for param in self.__model.fc.parameters():
                param.requires_grad = True
        else:
            print("New classifier head will be frozen (parameters will not be updated during training).")
            for param in self.__model.fc.parameters():
                param.requires_grad = False

    def forward(self, x):
        """This is the forward pass of the model."""
        return self.__model(x)
    
    @property
    def model_summary(self):
        """This method provides the summary of the model."""
        print(summary(model=self.__model, 
                input_size=(32, 3, 224, 224),
                # col_names=["input_size"],
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
        ))
        return None
    
    @property
    def weights(self):
        """This is a getter for weights."""
        return self.__weights_transformed
    
    @property
    def model(self):
        """This is a getter for the model."""
        return self.__model    