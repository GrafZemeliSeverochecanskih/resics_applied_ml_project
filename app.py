import fastapi
from fastapi import File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import torch
import uvicorn
from pathlib import Path
import shutil
import tempfile
import io

from upload_model.model import UploadModel

app = fastapi.FastAPI()
MODEL_WEIGHTS_PATH = r"D:\project_images — копия\resics_applied_ml_project\resnet_50_custom_model_weights.pth"

if not Path(MODEL_WEIGHTS_PATH).is_file():
    print(f" WARNING: Model weights file not found at {MODEL_WEIGHTS_PATH}")
    print(" Please ensure the MODEL_WEIGHTS_PATH in your script is correct.")
model = None
try:
    model = UploadModel(weights_path=MODEL_WEIGHTS_PATH)
    print(" Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure your model.py and weights path are correct and all dependencies are installed.")
    model = None

@app.get("/", response_class=HTMLResponse)
async def get_upload_form():
    try:
        html_file_path = Path(__file__).parent / "index.html"
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found</h1><p>Please ensure index.html is in the same directory as the server script.</p>", status_code=500)
    except Exception as e:
        return HTMLResponse(content=f"<h1>An error occurred</h1><p>{str(e)}</p>", status_code=500)

@app.post("/predict/")
async def predict_image_original(image_file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

    if not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    tmp_file_path = None
    try:
        suffix = Path(image_file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(image_file.file, tmp_file)
            tmp_file_path = Path(tmp_file.name)
        
        predicted_class_name = model.predict_single_image(image_path=tmp_file_path)
        
        return {"filename": image_file.filename, "prediction": predicted_class_name}

    except Exception as e:
        print(f"Error during original prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        if tmp_file_path and tmp_file_path.exists():
            tmp_file_path.unlink()
        await image_file.close()

@app.post("/predict_and_explain/")
async def predict_image_and_explain(image_file: UploadFile = File(...)):
    """
    Receives an image, uses the UploadModel's predict_and_explain method,
    and returns prediction, probabilities, and saliency map.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

    if not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await image_file.read()
        
        predicted_class, class_probs, saliency_b64 = model.predict_and_explain(image_bytes=image_bytes)
        
        return {
            "filename": image_file.filename,
            "prediction": predicted_class,
            "probabilities": class_probs,
            "saliency_map_base64": saliency_b64
        }

    except AttributeError as e:
        print(f"AttributeError during predict_and_explain: {e}")
        print("Ensure 'predict_and_explain' method and its requirements (e.g., saliency generator) are correctly implemented in UploadModel.")
        raise HTTPException(status_code=500, detail=f"Model method error: {str(e)}. Check server logs and model.py.")
    except Exception as e:
        print(f"Error during predict_and_explain: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image for explanation: {str(e)}")
    finally:
        await image_file.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
