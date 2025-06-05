import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class PlotCreator:
    """This class creates the plots."""
    def __init__(self, 
                 results: list[float],
                 output_dir,
                 ):
        """This is PlotCreator class constructor.

        Args:
            results (list[float]): the array with results.
        """
        self.__results = results
        self.__output_dir = output_dir

    def run(self):
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
    