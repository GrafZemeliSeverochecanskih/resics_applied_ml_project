from torch.optim import * 
from tqdm import tqdm
from torch import nn
import torch

class Trainer:
    """This class implements all functions needed for training a modified
        ResNet-50 architecture from the scratch.
        Code adapted from https://www.learnpytorch.io/06_pytorch_transfer_learning/ 
        https://www.learnpytorch.io/05_pytorch_going_modular/
    """
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer, 
                 train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
                 learning_rate: float = 0.001,
                 epochs = 10
                 ):
        self.__model = model
        self.__train_dataloader = train_dataloader
        self.__val_dataloader = val_dataloader
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__loss_fn = loss_fn
        self.__lr = learning_rate
        self.__epochs = epochs
        self.__optimizer = optimizer

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
        
    def run(self):
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
        