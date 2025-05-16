from new_model import *
from torch.optim import * 
from tqdm import tqdm
from torch import nn
import torch
from new_model.new_model import NewResNet50Model

class Trainer:
    """This class implements all functions needed for training a modified
        ResNet-50 architecture from the scratch.
        Code adapted from https://www.learnpytorch.io/06_pytorch_transfer_learning/ 
        https://www.learnpytorch.io/05_pytorch_going_modular/
    """
    def __init__(self, 
                 model: NewResNet50Model, 
                 train_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
                 learning_rate: int = 0.001,
                 epochs = 10
                 ):
        """This is the Trainer class constructor.
        Args:
            model (NewResNet50Model): a modified ResNet-50 architecture
            train_dataloader (torch.utils.data.DataLoader): training data 
            loader
            test_dataloader (torch.utils.data.DataLoader): test data loader
            loss_fn (torch.nn.Module, optional): loss function used during
            the trainig. Defaults to nn.CrossEntropyLoss().
            learning_rate (int, optional): learning rate. Defaults to 0.001.
            epochs (int, optional): number of epochs. Defaults to 10.
        """
        self.__lr = learning_rate
        self.__model = model
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__optimizer = Adam(
            list(filter(lambda p: p.requires_grad, model.parameters())),
            lr=self.__lr
            )
        
        self.__train_dataloader = train_dataloader
        self.__test_dataloader = test_dataloader
        self.__loss_fn = loss_fn
        self.__epochs = epochs
    
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
    
    def __test_step(self):
        """This function implements test step."""
        self.__model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for batch, (X, y) in enumerate(self.__test_dataloader):
                X, y = X.to(self.__device), y.to(self.__device)
                test_pred_logits = self.__model(X)
                loss = self.__loss_fn(test_pred_logits, y)
                test_loss += loss.item()
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
        test_loss = test_loss / len(self.__test_dataloader)
        test_acc = test_acc / len(self.__test_dataloader)
        return test_loss, test_acc
    
    def run(self):
        """This function implements a trainer run"""
        results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        self.__model.to(self.__device)
        for epoch in tqdm(range(self.__epochs)):
            train_loss, train_acc = self.__train_step()
            test_loss, test_acc = self.__test_step()
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
                )
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
        return results    
    