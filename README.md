#  Satellite Image Classification with Explainability and Uncertainty Estimation
Applied Machine Learning Project Group 31
#### Contributors: Adelina Mazilu (S5484669), Simion Polivencu (S5480183), Heleen Van Asselt (S5588324), Raha Torabihaghighi (S5618762)

This repository is created for the computer vision project based on the RESISC-45 dataset as part of Applied Machine Learning course (https://ocasys.rug.nl/current/catalog/course/WBAI065-05)

This project is focused on classifying satellite images into 45 categories (e.g., airport, island, etc.,) using deep learning. It also utilizes saliency maps and uncertainty estimation to enhance transparency and explainability. Our goal is to accurately classify high-resolution satellite images into 45 categories while providing visual explanations using saliency maps and identify low confidence outputs using uncertainty estimation.

## Model Architecture

Our baseline model is a basic Convolutional Neural Network (CNN). We used a custom ResNet-50 model with added dropout layers for uncertainty estimation via Monte Carlo Dropout. As an alternative advanced model, we also explored fine-tuning a Visual Transformer (ViT). Explainability is achieved through saliency maps using gradient-based methods.

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


## Installation

1. Clone the repository:
git clone https://github.com/GrafZemeliSeverochecanskih/resics_applied_ml_project.git
cd resics_applied_ml_project
3. This project uses the RESICS-45 dataset, it can be found at https://paperswithcode.com/dataset/resisc45