import matplotlib.pyplot as plt
class PlotCreator:
    """This class creates the plots."""
    def __init__(self, 
                 results: list[float]
                 ):
        """This is PlotCreator class constructor.

        Args:
            results (list[float]): the array with results.
        """
        self.__results = results
        self.__loss = results["train_loss"]
        self.__test_loss = results["test_loss"]
        self.__accuracy = results["train_acc"]
        self.__test_accuracy = results["test_acc"]
        self.__epochs = range(len(results["train_loss"]))
    
    def visualize_loss(self):
        """This function creates a plot for loss.

        Returns:
            _type_: displays plots
        """
        if self.__results and all(k in self.__results for k in ["train_loss", "test_loss", "train_acc", "test_acc"]):
            plt.figure(figsize=(15, 7))
            plt.plot(self.__epochs, self.__loss, label="train_loss")
            plt.plot(self.__epochs, self.__test_loss, label="test_loss")
            plt.title("Loss")
            plt.xlabel("Epochs")
            plt.legend()
            plt.show()
        else:
            print("Skipping plotting loss curves as training results are not available or incomplete.")
        return None
    
    def visualize_accuracy(self):
        """This function creates plots for accuracy."""
        if self.__results and all(k in self.__results for k in ["train_loss", "test_loss", "train_acc", "test_acc"]):
            plt.figure(figsize=(15, 7))
            plt.plot(self.__epochs, self.__accuracy, label="train_accuracy")
            plt.plot(self.__epochs, self.__test_accuracy, label="test_accuracy")
            plt.title("Accuracy")
            plt.xlabel("Epochs")
            plt.legend()
            plt.show()
        else:
            print("Skipping plotting loss curves as training results are not available or incomplete.")
        return None

    @property
    def results(self):
        """This is a getter for results."""
        return self.__results
    
    @property
    def loss(self):
        """This is a getter for training loss values."""
        return self.__loss
    
    @property
    def test_loss(self):
        """This is a getter for test loss values."""
        return self.__test_loss
    
    @property
    def accuracy(self):
        """This is a getter for training accuracy."""
        return self.__accuracy
    
    @property
    def test_accuracy(self):
        """This is a getter for test accuracy."""
        return self.__test_accuracy
    
    @property
    def epochs(self):
        """This is a getter for the number of epochs."""
        return self.__epochs
    
