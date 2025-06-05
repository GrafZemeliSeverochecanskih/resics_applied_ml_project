import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import torch
import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix

class Utilities:
    def __init__(self,
                 results,
                 output_dir,
                 model,
                 filename,
                 val_dataloader,
                 test_dataloader):
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__filename = filename + ".pth"
        self.__results = results
        self.__output_dir = output_dir
        self.__model = model
        self.__filename = filename
        self.__val_dataloader = val_dataloader
        self.__test_dataloader = test_dataloader
        self.__class_names = sorted([
            "airplane", "airport", "baseball_diamond", 
            "basketball_court", "beach", "bridge", "chaparral", 
            "church", "circular_farmland", "cloud", "commercial_area",
            "dense_residential", "desert", "forest", "freeway",
            "golf_course", "ground_track_field", "harbor", 
            "industrial_area", "intersection", "island", "lake", 
            "meadow", "medium_residential", "mobile_home_park", "mountain", 
            "overpass", "palace", "parking_lot", "railway", "railway_station",
            "rectangular_farmland", "river", "roundabout", "runway", 
            "sea_ice", "ship", "snowberg", "sparse_residential", "stadium", 
            "storage_tank", "tennis_court", "terrace", "thermal_power_station", 
            "wetland"
        ])

    def run(self):
        self.__save_accuracies_json()
        self.__create_and_save_val_confusion_matrix()
        self.__save_model_weights()

    def __save_accuracies_json(self):
        print("Saving accuracies to JSON...")
        if self.__results and \
           self.__results.get("train_acc") and \
           self.__results.get("val_acc"):
            
            os.makedirs(self.__output_dir, exist_ok=True)
            
            accuracies = {
                "last_train_accuracy": 100 * self.__results["train_acc"][-1] if self.__results["train_acc"] else None,
                "last_val_accuracy": 100 * self.__results["val_acc"][-1] if self.__results["val_acc"] else None,
                "test_accuracy": self.__test_accuracy()
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
        
        os.makedirs(self.__output_dir, exist_ok=True)

        with torch.no_grad():
            for X, y in tqdm.tqdm(self.__val_dataloader, desc="Generating Validation Confusion Matrix"):
                X, y_true = X.to(self.__device), y.to(self.__device)
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

    
    def __test_accuracy(self):
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
    