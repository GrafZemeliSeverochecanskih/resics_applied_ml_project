import torch
from torch import nn
from pathlib import Path
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import *
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np 

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
        weights_enum = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.__weights_transformed = weights_enum.transforms()
        self.__model = torchvision.models.resnet50(
            weights=weights_enum
            ).to(self.__device)
        self.__freeze_layers()
        self.__modify_last_layer()

        self.__num_workers = os.cpu_count()
        self.__batch_size = batch_size

        if transform is None:
            self.__transform = self.__create_default_transform()

        self.__train_data = datasets.ImageFolder(self.__train_dir, transform = self.__transform)
        self.__val_data = datasets.ImageFolder(self.__val_dir, transform=self.__transform)
        self.__test_data = datasets.ImageFolder(self.__test_dir, transform=self.__transform)
        
        self.__class_names = self.__train_data.classes
        
        self.__train_dataloader = self.__create_dataloader(self.__train_data, shuffle=True)
        self.__val_dataloader = self.__create_dataloader(self.__val_data)
        self.__test_dataloader = self.__create_dataloader(self.__test_data)

        self.__lr = learning_rate
        self.__optimizer = Adam(
            list(filter(lambda p: p.requires_grad, self.__model.parameters())),
            lr=self.__lr
            )
        self.__loss_fn = loss_fn
        self.__epochs = epochs
        
        self.__results = self.__run()
        self.__create_graphs()
        self.__create_and_save_val_confusion_matrix()
        self.__save_accuracies_json()
        self.__save_model_weights()

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
    
    def __create_default_transform(self) -> transforms.Compose:
        """This function created default transform.

        Returns:
            transforms.Compose: basic transofrmations applied to images.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean =[0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])
        ])

        return transform
    
    def __create_dataloader(self, data: datasets, shuffle = False) -> DataLoader:
        """This function creates data loaders."""
        dataloader = DataLoader(
            data,
            batch_size = self.__batch_size,
            shuffle = shuffle,
            num_workers=self.__num_workers,
            pin_memory=True
        )
        return dataloader
    
    def __train_step(self):
        """This function implements training step."""
        self.__model.train()
        train_loss, train_acc = 0, 0
        for batch, (X, y) in enumerate(self.__train_dataloader):
            X, y = X.to(self.__device), y.to(self.__device)
            y_pred = self.__model(X)
            loss = self.__loss_fn(y_pred, y)
            train_loss += loss.item()
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim = 1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        train_loss = train_loss / len(self.__train_dataloader)
        train_acc = train_acc / len(self.__train_dataloader)
        return train_loss, train_acc

    def __val_step(self):
        """This function implements valiadtion step."""
        self.__model.eval()
        val_loss, val_acc = 0, 0
        with torch.inference_mode():
            for batch, (X, y) in enumerate(self.__val_dataloader):
                X, y = X.to(self.__device), y.to(self.__device)
                val_pred_logits = self.__model(X)
                loss = self.__loss_fn(val_pred_logits, y)
                val_loss += loss.item()
                val_pred_labels = val_pred_logits.argmax(dim=1)
                val_acc += ((val_pred_labels == y).sum().item()/len(val_pred_labels))
        val_loss = val_loss / len(self.__val_dataloader)
        val_acc = val_acc / len(self.__val_dataloader)
        return val_loss, val_acc
    
    def __run(self):
        """This function implements a trainer run"""
        results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        self.__model.to(self.__device)
        for epoch in tqdm(range(self.__epochs)):
            train_loss, train_acc = self.__train_step()
            val_loss, val_acc = self.__val_step()
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc:.4f}"
                )
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)
        return results
    
    def __create_graphs(self):
        """This function creates graphs with loss and accuracy."""
        self.__visualize_loss()
        self.__visualize_accuracy()

    def __visualize_loss(self):
        if self.__results and all(k in self.__results for k in ["train_loss", "val_loss"]):
            epochs_ran = len(self.__results["train_loss"])
            epoch_ticks = range(1, epochs_ran + 1)

            plt.figure(figsize=(15, 7))
            plt.plot(epoch_ticks, self.__results["train_loss"], label="train_loss")
            plt.plot(epoch_ticks, self.__results["val_loss"], label="val_loss")
            plt.title("Loss Curves")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.xticks(epoch_ticks)
            plt.legend()

            graph_path = self.__output_dir / "loss_curves.png"
            plt.savefig(graph_path)
            print(f"Loss curves saved to {graph_path}")
            plt.close()
        else:
            print("Skipping plotting loss curves as training results are not available or incomplete.")

    def __visualize_accuracy(self):
        if self.__results and all(k in self.__results for k in ["train_acc", "val_acc"]):
            epochs_ran = len(self.__results["train_acc"])
            epoch_ticks = range(1, epochs_ran + 1)

            plt.figure(figsize=(15, 7))
            plt.plot(epoch_ticks, self.__results["train_acc"], label="train_accuracy")
            plt.plot(epoch_ticks, self.__results["val_acc"], label="val_accuracy")
            plt.title("Accuracy Curves")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.xticks(epoch_ticks)
            plt.legend()

            graph_path = self.__output_dir / "accuracy_curves.png"
            plt.savefig(graph_path)
            print(f"Accuracy curves saved to {graph_path}")
            plt.close()
        else:
            print("Skipping plotting accuracy curves as training results are not available or incomplete.")

    def __save_accuracies_json(self):
        print("Saving accuracies to JSON...")
        if self.__results and \
           self.__results.get("train_acc") and \
           self.__results.get("val_acc"):
            
            # Ensure output directory exists
            os.makedirs(self.__output_dir, exist_ok=True)
            
            accuracies = {
                "last_train_accuracy": self.__results["train_acc"][-1] if self.__results["train_acc"] else None,
                "last_val_accuracy": self.__results["val_acc"][-1] if self.__results["val_acc"] else None,
                "test_accuracy": self.test_accuracy # Call the property
            }
            
            json_path = self.__output_dir / "accuracies_summary.json"
            
            try:
                with open(json_path, 'w') as f:
                    json.dump(accuracies, f, indent=4)
                print(f"Accuracies summary saved to {json_path}")
            except Exception as e:
                print(f"Error saving accuracies to {json_path}: {e}")
        else:
            print("Skipping saving accuracies as training results are not available or incomplete.")

    def __create_and_save_val_confusion_matrix(self):
        """Creates, plots, and saves a confusion matrix for the validation dataset."""
        print("Generating and saving validation confusion matrix...")
        self.__model.eval()
        all_labels = []
        all_preds = []
        
        # Ensure output directory exists
        os.makedirs(self.__output_dir, exist_ok=True)

        with torch.no_grad():
            for X, y in tqdm(self.__val_dataloader, desc="Generating Validation Confusion Matrix"):
                X, y_true = X.to(self.__device), y.to(self.__device) # y_true for clarity
                outputs = self.__model(X)
                _, predicted_labels = torch.max(outputs, 1)
                
                all_preds.extend(predicted_labels.cpu().numpy())
                all_labels.extend(y_true.cpu().numpy())
        
        if not all_labels or not all_preds:
            print("No data found in validation set to generate confusion matrix.")
            return

        try:
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(max(10, len(self.__class_names)*0.5), max(8, len(self.__class_names)*0.4))) # Dynamic sizing
            
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=self.__class_names, 
                        yticklabels=self.__class_names)
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title("Validation Confusion Matrix")
            plt.tight_layout()
            
            cm_path = self.__output_dir / "validation_confusion_matrix.png"
            plt.savefig(cm_path)
            print(f"Validation confusion matrix saved to {cm_path}")
            plt.close()
        except Exception as e:
            print(f"Error generating or saving confusion matrix: {e}")

    def __save_model_weights(self):
        if self.__model is not None:
            try:
                os.makedirs(self.__output_dir, exist_ok=True)
                model_save_path = self.__output_dir / self.__filename
                torch.save(self.__model.state_dict(), model_save_path)
            except AttributeError:
                 print(f"AttributeError: Could not save model weights. \
                       Ensure self.__model is a valid PyTorch nn.Module \
                       and has state_dict method.")
            except Exception as e:
                print(f"Error saving model weights to \
                       {model_save_path}: {e}")
        else:
            print("Model not initialized")
    
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