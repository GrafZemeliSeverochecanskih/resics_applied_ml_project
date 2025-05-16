import torch

class AbstractModel(torch.nn.Module):
    """This is an abstract class for models.

    Args:
        torch (_type_): torch base class for all neural networks. From:
        https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
    """
    def __init__(self):
        """This is the AbstractModel class constructor."""
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        """This is the forward pass of the model."""
        pass
    
    @property
    def device(self):
        """This is a getter of the device."""
        return self._device
    