from model.upload_model.upload_model import UploadResNet50Model
from fastapi import FastAPI
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "model" / "weights" / "resnet_50_custom_model_weights.pth"
IMAGE_DATA_PATH = PROJECT_ROOT / "model" / "data"

model = UploadResNet50Model(
    weight_path=WEIGHTS_PATH,
    image_path=IMAGE_DATA_PATH
    )
app = FastAPI(
    title="RESICS Model API",
    description="API for testing the RESICS ResNet50 model."
)

@app.get("/")
def main_page_message():
    return "API for testing the RESICS ResNet50 model."

@app.get("/accuracy_score")
def test_accuracy():
    accuracy = model.test_accuracy
    return {"test accuracy": str(accuracy)}