#  Satellite Image Classification with Explainability and Uncertainty Estimation
Applied Machine Learning Project Group 31
#### Contributors: Adelina Mazilu (S5484669), Simion Polivencu (S5480183), Heleen Van Asselt (S5588324), Raha Torabihaghighi (S5618762)

# Satellite Image Classification with Explainability and Uncertainty Estimation
### Applied Machine Learning Project

This project, developed for the Applied Machine Learning course, focuses on classifying high-resolution satellite images from the **RESISC-45 dataset** into 45 distinct categories. Beyond simple classification, it integrates modern machine learning techniques to provide model explainability through saliency maps and to estimate prediction uncertainty using Monte Carlo (MC) Dropout.

The core of the project is an interactive web application that allows users to upload a satellite image and receive a detailed analysis from two different models: a simple baseline CNN and a more sophisticated, fine-tuned ResNet-50.

## ✨ Features

- **Dual-Model Architecture**: Compares a lightweight, custom-built Convolutional Neural Network (CNN) against a powerful, pre-trained ResNet-50 model fine-tuned for this specific task.
- **Uncertainty Estimation**: The ResNet-50 model uses **Monte Carlo Dropout** during inference to provide a confidence score, helping to identify predictions where the model is uncertain.
- **Explainable AI (XAI)**: Generates **Grad-CAM** and **NormGrad** saliency maps to visualize which parts of the image were most influential in the ResNet-50 model's decision-making process.
- **Interactive Web Interface**: A user-friendly web application built with **FastAPI** where users can upload an image and instantly see the classification results, uncertainty estimates, and saliency map visualizations.
- **Performance Visualizations**: API endpoints provide access to detailed model performance metrics, including test accuracies and pre-generated plots of training/validation loss and accuracy curves.

## 🛠️ Model Architecture

Two models are implemented and compared in this project:

1.  **Baseline CNN (`UploadCNN`)**: A simple, scratch-built CNN with a single convolutional layer followed by a linear layer. It serves as a performance baseline to demonstrate the advantages of transfer learning.

2.  **Fine-Tuned ResNet-50 with MC Dropout (`UploadModel`)**: A ResNet-50 model, pre-trained on ImageNet, with its final classification layer replaced. A `Dropout` layer is added to this new classifier, which is uniquely kept active during inference to enable Monte Carlo Dropout for uncertainty estimation. The model has been fine-tuned on the RESISC-45 dataset.

## 📂 Project Structure

The project is organized into modules for data handling, model training, inference, and the web application.
<details>
<summary> Project Structure (click to expand)</summary>

```
├── README.md
├── requirements.txt
├── app/
│ └── init.py
├── model/
│ ├── main.py
│ ├── abstract/
│ │ └── abstract_model.py
│ ├── new_model/
│ │ ├── create_new_model.py
│ │ ├── resnet50_custom_model.py
│ │ └── trainer.py
│ ├── upload_model/
│ │ └── upload_model.py
│ ├── utils/
│ │ ├── data_handler.py
│ │ ├── data_shuffler.py
│ │ ├── mc_dropout.py
│ │ ├── plot_creator.py
│ │ └── saliency_utils.py
│ └── weights/
│ ├── resnet_50_custom_model_weights.pth
│ └── trained_resnet50_mc_dropout_model.pth
├── notebooks/
│ ├── experiment_1.ipynb
│ ├── experiment_2.ipynb
│ ├── experiment_2_1.ipynb
│ ├── experiment_2_loading_model.ipynb
│ ├── experiment_4.ipynb
│ └── vit-base-oxford-iiit-pets/
│ └── runs/
│ └── May21_13-34-05_DESKTOP-OEICJQE/
│ └── events.out.tfevents.*
└── tests/
└── init.py
```
</details>


## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

-   Python 3.8 or higher
-   The **RESISC-45 dataset**. You can download it from [here](https://paperswithcode.com/dataset/resisc45). Once downloaded, place the image folders into the `data/train`, `data/val`, and `data/test` directories accordingly.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/GrafZemeliSeverochecanskih/resics_applied_ml_project.git](https://github.com/GrafZemeliSeverochecanskih/resics_applied_ml_project.git)
    cd resics_applied_ml_project
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Start the FastAPI server:**
    ```bash
    uvicorn app:app --reload
    ```
    The `--reload` flag enables hot-reloading for development.

2.  **Open the web interface:**
    Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser. You can now upload an image and see the classification results.

## 📡 API Endpoints

The FastAPI application exposes the following endpoints:

-   `GET /`: Serves the main HTML page for image uploads.
-   `POST /predict`: Handles image submission, runs predictions with both models, and returns an HTML page with the results.
-   `GET /metrics/accuracies`: Returns a JSON response with the test accuracy summaries for both models.
-   `GET /metrics/accuracy-plot`: Returns the pre-generated accuracy curves plot as a PNG image.
-   `GET /metrics/loss-plot`: Returns the pre-generated loss curves plot as a PNG image.

---
