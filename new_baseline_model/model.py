import torch
from torch import nn
from pathlib import Path
from torch.optim import Adam
from new_baseline_model.data_handler import DataHandler



class CNNBaselineModel:
    def __init__(self,
                 data_handler: DataHandler,
                 output_dir: str,
                 filename: str = "cnn_baseline_model_weights",
                 epochs: int = 10,
                 learning_rate: float = 0.001,
                 loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
                 num_filters: int = 16
                 ):

        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.__device}")
        self.__data_handler = data_handler
        self.__output_dir = Path(output_dir)
        self.__output_dir.mkdir(parents=True, exist_ok=True)
        self.__filename = filename + ".pth"
        self.__epochs = epochs
        self.__lr = learning_rate
        self.__loss_fn = loss_fn
        self.__num_filters = num_filters

        self.__train_dataloader = self.__data_handler.train_dataloader
        self.__val_dataloader = self.__data_handler.val_dataloader
        self.__test_dataloader = self.__data_handler.test_dataloader

        self.__input_channels = 3
        if not self.__data_handler.classes:
             raise ValueError("DataHandler has no classes. Ensure data is loaded correctly and paths are valid.")
        self.__num_classes = len(self.__data_handler.classes)
        print(f"Number of classes determined from DataHandler: {self.__num_classes}")

        self.__model = nn.Sequential(
            nn.Conv2d(in_channels=self.__input_channels,
                      out_channels=self.__num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=1, out_features=self.__num_classes)
        ).to(self.__device)
        self.__determine_fc_input_features()

        self.__optimizer = Adam(
            self.__model.parameters(),
            lr=self.__lr
        )
        print("CNNBaselineModel initialized successfully.")

    def __get_sample_input_shape(self):
        """Gets the shape of a sample input from the train_dataloader after transforms."""
        if not self.__train_dataloader:
            print("Warning: Train dataloader is not available. Falling back to default input shape.")
            return self.__input_channels, 224, 224
        try:
            sample_batch, _ = next(iter(self.__train_dataloader))
            return sample_batch.shape[1], sample_batch.shape[2], sample_batch.shape[3]
        except StopIteration:
            print("Warning: Train dataloader is empty. Falling back to default input shape for FC layer calculation.")
            return self.__input_channels, 224, 224

    def __determine_fc_input_features(self):
        """
        Passes a dummy tensor (or a real sample) through the convolutional part
        to determine the input features for the fully connected layer.
        """
        channels, height, width = self.__get_sample_input_shape()
        dummy_input = torch.randn(1, channels, height, width).to(self.__device)
        conv_part = nn.Sequential(*list(self.__model.children())[:-1])

        with torch.no_grad():
            dummy_output = conv_part(dummy_input)

        flattened_features = dummy_output.numel()
        print(f"Dynamically determined flattened features for Linear layer: {flattened_features}")

        self.__model[-1] = nn.Linear(in_features=flattened_features, out_features=self.__num_classes).to(self.__device)


    def train_step(self, dataloader: torch.utils.data.DataLoader):
        self.__model.train()
        train_loss, train_acc = 0, 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.__device), y.to(self.__device)

            y_pred_logits = self.__model(X)
            loss = self.__loss_fn(y_pred_logits, y)
            train_loss += loss.item()
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

            y_pred_class = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred_logits)
        train_loss /= len(dataloader)
        train_acc /= len(dataloader)
        return train_loss, train_acc

    def val_step(self, dataloader: torch.utils.data.DataLoader):
        self.__model.eval()
        val_loss, val_acc = 0, 0
        with torch.inference_mode():
            for X, y in dataloader:
                X, y = X.to(self.__device), y.to(self.__device)

                y_pred_logits = self.__model(X)

                loss = self.__loss_fn(y_pred_logits, y)
                val_loss += loss.item()

                y_pred_class = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)
                val_acc += (y_pred_class == y).sum().item() / len(y_pred_logits)

        val_loss /= len(dataloader)
        val_acc /= len(dataloader)
        return val_loss, val_acc

    def fit(self):
        """Trains the model and saves the weights."""
        results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        print(f"Starting training for {self.__epochs} epochs on {self.__device}...")
        print(f"Model architecture: \n{self.__model}")
        print(f"Number of classes: {self.__num_classes}")
        print(f"Optimizer: {self.__optimizer}")


        for epoch in range(self.__epochs):
            train_loss, train_acc = self.train_step(self.__train_dataloader)
            val_loss, val_acc = self.val_step(self.__val_dataloader)

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

        model_save_path = self.__output_dir / self.__filename
        print(f"Saving model weights to: {model_save_path}")
        torch.save(obj=self.__model.state_dict(), f=model_save_path)
        print(f"Model weights saved successfully to {model_save_path}")

        return results

    def evaluate(self):
        """Evaluates the model on the test dataset."""
        print(f"Evaluating model on test data using {self.__device}...")
        self.__model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for X, y in self.__test_dataloader:
                X, y = X.to(self.__device), y.to(self.__device)
                outputs_logits = self.__model(X)
                loss = self.__loss_fn(outputs_logits, y)
                test_loss += loss.item()

                pred_labels = torch.argmax(torch.softmax(outputs_logits, dim=1), dim=1)
                test_acc += (pred_labels == y).sum().item() / len(outputs_logits)

        test_loss /= len(self.__test_dataloader)
        test_acc /= len(self.__test_dataloader)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        return test_loss, test_acc

    def predict(self, image_tensor: torch.Tensor):
        """Makes a prediction on a single image tensor."""
        self.__model.eval()
        with torch.inference_mode():
            image_tensor = image_tensor.to(self.__device)
            if len(image_tensor.shape) == 3:
                 image_tensor = image_tensor.unsqueeze(0)
            output_logits = self.__model(image_tensor)
            output_probs = torch.softmax(output_logits, dim=1)
            pred_prob_val, pred_label_idx = torch.max(output_probs, dim=1)

        class_name = self.__data_handler.classes[pred_label_idx.item()]
        return class_name, pred_prob_val.item()
