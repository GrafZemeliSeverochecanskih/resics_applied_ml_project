from new_baseline_model.model import CNNBaselineModel
from pathlib import Path
from new_baseline_model.data_handler import DataHandler
import os
import json

class NewCnnRunner:
    def __init__(self,
                 data_path:str,
                 output_dir:str,
                 model_filename:str = "my_cnn_baseline",
                 batch_size:int = 32,
                 epochs: int = 5,
                 learning_rate: float = 0.001,
                 num_filters_conv_layers = 16):
        self.__data_path = Path(data_path)
        self.__train_dir = self.__data_path / "train"
        self.__val_dir = self.__data_path / "val"
        self.__test_dir = self.__data_path / "test"
        self.__output_dir = Path(output_dir) 
        self.__model_filename = model_filename
        self.__batch_size = batch_size
        self.__num_workers = os.cpu_count() // 2 if os.cpu_count() else 0
        self.__epochs = epochs
        self.__learning_rate = learning_rate
        self.__num_filters_conv_layers = num_filters_conv_layers
        self.__output_dir.mkdir(parents=True, exist_ok=True)

        self.__data_handler = DataHandler(
            train_dir=str(self.__train_dir),
            val_dir=str(self.__val_dir),
            test_dir=str(self.__test_dir),
            num_workers=self.__num_workers,
            batch_size=self.__batch_size
        )
        print("DataHandler initialized.")
        print(f"Training classes: {self.__data_handler.classes}")

        print("Initializing CNNBaselineModel...")
        self.__cnn_model = CNNBaselineModel(
            data_handler=self.__data_handler,
            output_dir=str(self.__output_dir),
            filename=self.__model_filename,
            epochs=self.__epochs,
            learning_rate=self.__learning_rate,
            num_filters=self.__num_filters_conv_layers
        )
        print("CNNBaselineModel initialized.")

        print("Starting model training (fit method)...")
        training_history = self.__cnn_model.fit()
        print("Model training finished.")
        print("Training results:", training_history)

        print("Evaluating model on the test set...")
        test_loss, test_accuracy = self.__cnn_model.evaluate()
        print(f"Evaluation complete. Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
        self.__save_results(training_history, test_accuracy)

    def __save_results(self, training_history: dict, test_accuracy: float):
        """
        Saves the training, validation, and test accuracies to a JSON file.

        Args:
            training_history (dict): The history dictionary returned by the model's fit method.
            test_accuracy (float): The accuracy of the model on the test set.
        """
        results = {
            "train_accuracy": training_history.get('train_acc', [])[-1],
            "val_accuracy": training_history.get('val_acc', [])[-1],
            "test_accuracy": test_accuracy
        }
        
        results_filepath = self.__output_dir / "accuracy_results.json"
        
        print(f"Saving accuracy results to {results_filepath}...")
        with open(results_filepath, 'w') as f:
            json.dump(results, f, indent=4)
        print("Accuracy results saved successfully.")

    @property
    def model(self):
        return self.__cnn_model