from model.upload_model.upload_model import UploadResNet50Model
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from pathlib import Path
from pydantic import BaseModel, Field
import base64
from fastapi.templating import Jinja2Templates

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "model" / "weights" / "resnet_50_custom_model_weights.pth"
IMAGE_DATA_PATH = PROJECT_ROOT / "model" / "data"

model = UploadResNet50Model(
    weight_path=WEIGHTS_PATH,
    image_path=IMAGE_DATA_PATH
    )

templates = Jinja2Templates(directory="templates")

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
    image_base64: str = Field(..., description="The image file encoded as a Base64 string.")

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
        image_base64_string = base64.b64encode(image_bytes).decode("utf-8")
        return PredictionResponse(
            prediction=prediction,
            image_filename=image_path.name,
            image_base64=image_base64_string
        )

    except (IndexError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.get("/show_prediction/index/{folder_number}/{index_num}", response_class= HTMLResponse)
async def show_prediction_page(request: Request, folder_number: int, index_num: int):
    """
    Selects an image, performs prediction, and renders an HTML page
    to display the result and the image.
    """
    try:
        image_path = model.get_image_path(class_index=folder_number, image_index=index_num)
        prediction = model.predict_single_image(image_path=image_path)
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        image_base64_string = base64.b64encode(image_bytes).decode("utf-8")
        file_extension = image_path.suffix.lower()
        if file_extension in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif file_extension == ".png":
            mime_type = "image/png"
        else:
            mime_type = "image/jpeg"
        
        return templates.TemplateResponse("prediction_result.html", {
            "request": request,
            "prediction": prediction,
            "image_filename": image_path.name,
            "image_data": image_base64_string,
            "mime_type": mime_type
        })
    except (IndexError, FileNotFoundError) as e:
        return templates.TemplateResponse("error.html", {"request": request, "detail": str(e)}, status_code=404)
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "detail": f"An error occurred: {e}"}, status_code=500)
    
@app.get("/upload", response_class=HTMLResponse)
async def get_upload_form(request: Request):
    """Serves the HTML page with the upload form."""
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.post("/upload-predict", response_class=HTMLResponse)
async def handle_prediction_upload(request: Request, file: UploadFile = File(...)):
    """Handles the image upload, predicts its class, and displays the result."""
    try:
        image_bytes = await file.read()
        prediction, probabilities = model.predict_image_with_probabilities(image_bytes)
        image_base64_string = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = file.content_type
        return templates.TemplateResponse("upload_result.html", {
            "request": request,
            "prediction": prediction,
            "probabilities": probabilities,
            "image_data": image_base64_string,
            "mime_type": mime_type
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "detail": f"An error occurred during prediction: {e}"}, status_code=500)