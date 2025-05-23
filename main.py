from model.upload_model.upload_model import UploadResNet50Model
from fastapi import FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel, Field
import base64

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "model" / "weights" / "resnet_50_custom_model_weights.pth"
IMAGE_DATA_PATH = PROJECT_ROOT / "model" / "data"

model = UploadResNet50Model(
    weight_path=WEIGHTS_PATH,
    image_path=IMAGE_DATA_PATH
    )
print("DEBUG: Checking the structure of class_names...")
class_names_from_model = model.class_names
print(f"Type of class_names: {type(class_names_from_model)}")
print(f"Content of class_names: {class_names_from_model}")

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

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="The model's predicted class for the image.")
    image_filename: str = Field(..., description="The filename of the image.")
    # image_base64: str = Field(..., description="The image file encoded as a Base64 string.")

@app.get("/predict/index/{folder_number}/{index_num}", response_model=PredictionResponse)
def predict_indexed_image(folder_number: int, index_num: int):
    """
    Selects an image by its class folder index and file index,
    returns the model's prediction and the image as a Base64 string.
    """
    try:
        image_path = model.get_image_path(class_index=folder_number, image_index=index_num)
        prediction = model.predict_single_image(image_path=image_path)
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        # image_base64_string = base64.b64encode(image_bytes).decode("utf-8")
        return PredictionResponse(
            prediction=prediction,
            image_filename=image_path.name,
            # image_base64=image_base64_string
        )

    except (IndexError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")