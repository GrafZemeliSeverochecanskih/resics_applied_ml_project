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

@app.get("/upload", response_class=HTMLResponse)
async def get_upload_form(request: Request):
    """Serves the HTML page with the upload form."""
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.post("/upload-predict", response_class=HTMLResponse)
async def handle_prediction_upload(request: Request, file: UploadFile = File(...)):
    """Handles the image upload, predicts its class, and displays the result."""
    try:
        image_bytes = await file.read()
        prediction, probabilities, saliency_b64 = model.predict_and_explain(image_bytes)

        original_image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        return templates.TemplateResponse("upload_result.html", {
            "request": request,
            "prediction": prediction,
            "probabilities": probabilities,
            "original_image_data": original_image_b64,
            "saliency_image_data": saliency_b64,
            "mime_type": file.content_type
        })
    except FileNotFoundError as e:
        return templates.TemplateResponse("error.html", {"request": request, "detail": str(e)}, status_code=404)
    except IOError as e:
        return templates.TemplateResponse("error.html", {"request": request, "detail": f"Image processing error: {str(e)}"}, status_code=400)
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred during prediction: {e}")
        traceback.print_exc()
        return templates.TemplateResponse("error.html", {"request": request, "detail": f"An unexpected error occurred. Please try again."}, status_code=500)
