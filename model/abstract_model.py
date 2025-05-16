import torch

class AbstractModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        """This is the forward pass of the model"""
        pass
    
    @property
    def device(self):
        """This is a getter of the device"""
        return self._device
    