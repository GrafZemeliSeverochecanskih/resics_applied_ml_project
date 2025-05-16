import torch
from torch import nn

class MonteCarloDropout:
    """This class implements all needed functionality for MC Dropout.
    """
    def __init__(self, 
                 model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 num_samples: int =  20
                 ):
        """This is a MonteCarloDropout class constructor.

        Args:
            model (torch.nn.Module): ResNet-50 model.
            data_loader (torch.utils.data.DataLoader): data loader
            num_samples (int, optional): number of samples. Defaults to 20.
        """
        self.__model = model
        self.__dataloader = data_loader
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__num_samples = num_samples

        self.__all_predictions_softmax_tensor, self.__mean_predictions, \
            self.__predictive_uncertainty = self.__calculate_prediction()
        
    def __enable_dropout(self):
        """This method enables dropout."""
        for m in self.__model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def __calculate_prediction(self):
        """This function calculates prediction."""
        self.__model.eval()
        self.__enable_dropout()
        all_predictions_softmax = []
        with torch.no_grad():
            for X, y in self.__dataloader:
                X, y = X.to(self.__device), y.to(self.__device)
                batch_predictions_softmax = []
                for _ in range(self.__num_samples):
                    pred_logits = self.__model(X)
                    batch_predictions_softmax.append(
                        torch.softmax(pred_logits, dim=1)
                        )
                stacked_batch_preds = torch.stack(batch_predictions_softmax)
                all_predictions_softmax.append(stacked_batch_preds)
        
        all_predictions_softmax_tensor = torch.cat(
            all_predictions_softmax, dim=1
            )
        mean_predictions = torch.mean(
            all_predictions_softmax_tensor, dim=0
            )
        predictive_uncertainty = torch.var(
            all_predictions_softmax_tensor, dim = 0
            )
        return all_predictions_softmax_tensor, mean_predictions, predictive_uncertainty

    def run(self):
        """This function runs the MC-Dropout"""
        self.__final_prediction = torch.argmax(self.__mean_predictions, dim=1)
        self.__overall_image_uncertainty = torch.sum(self.__predictive_uncertainty, dim = 1)

    @property
    def model(self):
        """This is a getter for model."""
        return self.__model
    
    @property
    def dataloader(self):
        """This is a getter for dataloader."""
        return self.__dataloader

    @property
    def device(self):
        """This is a getter for device."""
        return self.__device 
    
    @property
    def num_samples(self):
        """This is a getter for num_samples."""
        return self.__num_samples

    @property
    def mean_predictions(self):
        """This is a getter for mean of predictions."""
        return self.__mean_predictions

    @property
    def predictive_uncertainty(self):
        """This is a getter for predictive uncertainty."""
        return self.__predictive_uncertainty

    @property
    def all_predictions_softmax_tensor(self):
        """This is a getter for all predictions softmax storde as tensor."""
        return self.__all_predictions_softmax_tensor
    
    @property
    def final_prediction(self):
        """This is a getter for final prediction."""
        return self.__final_prediction
    
    @property
    def overall_image_uncertainty(self):
        """This is a getter for overall uncertainty image."""
        return self.__overall_image_uncertainty
