import base64
import io
import json
from pathlib import Path

import torch
from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from PIL import Image
import torch
from upload_model.model import UploadModel
from upload_baseline_model.baseline_model import UploadCNN

BASE_DIR = Path(__file__).resolve().parent
MODEL_WEIGHTS_PATH = BASE_DIR /"resnet_50_weight_decay.pth"
BASELINE_MODEL_WEIGHTS_PATH = BASE_DIR / "my_cnn_baseline.pth"
ACCURACY_PLOT_PATH = BASE_DIR / "accuracy_curves.png"
LOSS_PLOT_PATH = BASE_DIR / "loss_curves.png"

BASELINE_ACCURACY_PATH = BASE_DIR / "accuracy_results.json"
TRANSFER_ACCURACY_PATH = BASE_DIR / "accuracies_summary.json"

app = FastAPI(
    title="Image Classification API",
    description="Compares a baseline CNN and a fine-tuned ResNet50 model using MC Dropout for inference.",
    version="1.1.0"
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

class AccuracyResponse(BaseModel):
    """Defines the structure for the accuracies JSON response."""
    baseline_model_accuracy: float = Field(..., example=0.32, description="Test accuracy of the baseline CNN model.")
    resnet_model_accuracy: float = Field(..., example=0.91, description="Test accuracy of the fine-tuned ResNet50 model.")

try:
    transfer_model = UploadModel(weights_path=str(MODEL_WEIGHTS_PATH))
    baseline_model = UploadCNN(model_weights_path=str(BASELINE_MODEL_WEIGHTS_PATH))
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not load model weights. {e}")
    transfer_model = None
    baseline_model = None
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    transfer_model = None
    baseline_model = None

@app.get("/", response_class=HTMLResponse, summary="Image Upload Page", 
         description="Serves the main HTML page containing a form to upload an image for classification by both models.")
async def get_upload_page(request: Request):
    """Serves the main HTML page with a form for uploading an image."""
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse, summary="Process Image and Show Results",
          description=(
              "Accepts an uploaded image and returns an HTML page with predictions from both models: "
              "the ResNet50 model using MC Dropout for uncertainty estimation, and a baseline CNN model. "
              "Includes class predictions, probabilities, and saliency map visualizations."
          ))
async def handle_prediction(request: Request, file: UploadFile = File(..., description="Image to classify")):
    """
    Accepts an uploaded image, gets predictions from both models (using MC Dropout for the main one),
    generates saliency maps, and renders the results on an HTML page.
    """
    if not transfer_model or not baseline_model:
        raise HTTPException(status_code=503, detail="Models are not available due to a loading error.")

    image_bytes = await file.read()

    try:
        mc_prediction, mc_probability, _ = transfer_model.predict_with_mc_dropout(image_bytes, num_samples=25)

        _, top5_probs, grad_cam_b64, normgrad_b64 = transfer_model.predict_and_explain(image_bytes)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during ResNet50 prediction: {e}")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = baseline_model.transform(image).unsqueeze(0).to(baseline_model._UploadCNN__device)

        with torch.inference_mode():
            output_logits = baseline_model.model(image_tensor)
            output_probs = torch.softmax(output_logits, dim=1)
            confidence, pred_index = torch.max(output_probs, dim=1)

        pred_baseline = baseline_model._UploadCNN__class_names[pred_index.item()]
        confidence_baseline = confidence.item()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Baseline CNN prediction: {e}")

    original_image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    context = {
        "request": request,
        "mc_prediction": mc_prediction,
        "mc_probability": mc_probability,
        "top5_probabilities": top5_probs,
        "prediction_baseline": pred_baseline,
        "confidence_baseline": confidence_baseline,
        "original_image_b64": original_image_b64,
        "grad_cam_viz_b64": grad_cam_b64,
        "normgrad_viz_b64": normgrad_b64,
        "mime_type": file.content_type
    }

    return templates.TemplateResponse("results.html", context)


@app.get("/metrics/accuracies", response_class=JSONResponse, summary="Get Full Model Accuracy Summaries",
         description=(
             "Returns a JSON object containing detailed accuracy summary information for both the baseline CNN "
             "and the fine-tuned ResNet50 models. This includes test accuracies and other evaluation metrics "
             "loaded from JSON summary files."
         ))
async def get_accuracies():
    """
    Returns a JSON object containing the full content of the accuracy
    summary files for both models.
    """
    try:
        with open(BASELINE_ACCURACY_PATH, 'r') as f:
            baseline_data = json.load(f)
        with open(TRANSFER_ACCURACY_PATH, 'r') as f:
            transfer_data = json.load(f)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Accuracy summary file not found: {e.filename}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding one of the accuracy JSON files.")

    full_accuracy_data = {
        "baseline_model_summary": baseline_data,
        "resnet_model_accuracy": transfer_data
    }
    return JSONResponse(content=full_accuracy_data)


@app.get("/metrics/accuracy-plot", response_class=FileResponse, summary="Get Accuracy Curve Plot",
         description="Returns a PNG image of the pre-generated accuracy curve plot showing model performance over training epochs.")
async def get_accuracy_plot():
    """
    Returns the pre-generated accuracy curve plot as a PNG image.
    """
    if not ACCURACY_PLOT_PATH.is_file():
        raise HTTPException(status_code=404, detail="Accuracy plot image not found.")
    return FileResponse(ACCURACY_PLOT_PATH, media_type="image/png")


@app.get("/metrics/loss-plot", response_class=FileResponse, summary="Get Loss Curve Plot",
         description="Returns a PNG image of the pre-generated loss curve plot visualizing model loss over training epochs.")
async def get_loss_plot():
    """
    Returns the pre-generated loss curve plot as a PNG image.
    """
    if not LOSS_PLOT_PATH.is_file():
        raise HTTPException(status_code=404, detail="Loss plot image not found.")
    return FileResponse(LOSS_PLOT_PATH, media_type="image/png")
