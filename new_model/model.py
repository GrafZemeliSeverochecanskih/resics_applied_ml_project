import torch
from torch import nn
from pathlib import Path
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import *
import torchvision
from new_model.trainer import Trainer
from new_model.plot_creator import PlotCreator
from new_model.utils import Utilities
from new_model.data_handler import DataHandler

class NewModel(torch.nn.Sequential):
    def __init__(self,
                 image_path: str,
                 output_dir: str,
                 filename: str = "resnet_50_custom_model_weights",
                 train: str = "train/",
                 val: str = "val/",
                 test: str = "test/",
                 epochs: int = 10,
                 unfreeze_classifier: bool = True,
                 unfreeze_specific_blocks: list[str] = None,
                 batch_size = 32,
                 transform: transforms.Compose = None,
                 learning_rate: float = 0.001,
                 weight_decay = 0.01,
                 loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
                 dropout_rate: float = 0.5
                 ):
        super().__init__()
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__image_path = Path(image_path)
        self.__train_dir = self.__image_path / train
        self.__val_dir = self.__image_path / val
        self.__test_dir = self.__image_path / test
        self.__output_dir = Path(output_dir)
        self.__filename = filename + ".pth"
        self.__dropout_rate = dropout_rate
        
        self.__unfreeze_classifier = unfreeze_classifier
        self.__unfreeze_specific_blocks = unfreeze_specific_blocks
        weights_enum = torchvision.models.\
            ResNet50_Weights.IMAGENET1K_V2
        self.__weights_transformed = weights_enum.transforms()
        self.__model = torchvision.models.resnet50(
            weights=weights_enum
            ).to(self.__device)
        self.__freeze_layers()
        self.__modify_last_layer()

        self.__batch_size = batch_size

        data_hander = DataHandler(
            train_dir=self.__train_dir, 
            val_dir=self.__val_dir,
            test_dir=self.__test_dir,
            num_workers=os.cpu_count(),
            batch_size=self.__batch_size,
            transform = transform
        )

        self.__train_dataloader = data_hander.train_dataloader
        self.__val_dataloader = data_hander.val_dataloader
        self.__test_dataloader = data_hander.test_dataloader

        self.__lr = learning_rate
        self.__weight_decay = weight_decay
        self.__optimizer = AdamW(
            list(filter(lambda p: p.requires_grad, 
                        self.__model.parameters())),
            lr=self.__lr,
            weight_decay=self.__weight_decay
            )
        
        self.__loss_fn = loss_fn
        
        self.__epochs = epochs
        
        self.__trainer = Trainer(
            model = self.__model,
            optimizer=self.__optimizer,
            train_dataloader = self.__train_dataloader,
            val_dataloader = self.__val_dataloader,
            loss_fn = self.__loss_fn,
            learning_rate = self.__lr,
            epochs = self.__epochs
        )

        self.__results = self.__trainer.run()
        
        plot_creator = PlotCreator(self.__results, self.__output_dir)
        plot_creator.run()


        utils = Utilities(
            results=self.__results,
            output_dir=self.__output_dir,
            model=self.__model,
            filename=self.__filename,
            val_dataloader=self.__val_dataloader,
            test_dataloader=self.__test_dataloader
        )
        utils.run()

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
    def test_accuracy(self):
        """Evaluates the model on the given test dataset and returns
          accuracy.
        """
        self.__model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.__test_dataloader:
                images = images.to(self.__device)
                labels = labels.to(self.__device)

                outputs = self.__model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        return test_accuracy
    
    @property
    def weights(self):
        return self.__weights_transformed