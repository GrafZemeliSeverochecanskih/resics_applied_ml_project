# <<<<<<< HEAD
from upload_model.model import UploadModel
# from new_model.model import NewModel
from pathlib import Path
import os
from new_baseline_model.data_handler import DataHandler
from new_baseline_model.model import CNNBaselineModel
from new_cnn_runner import NewCnnRunner 
from upload_baseline_model.model import UploadCNN

if __name__ == "__main__":
    model = UploadCNN(
        r"D:\project_images\resics_applied_ml_project\my_cnn_baseline.pth",
    )
    print(model.predict(r"D:\project_images\resics_applied_ml_project\data\test\airplane\airplane_692.jpg"))
    # model = NewCnnRunner(
    #     r"D:\project_images\resics_applied_ml_project\data",
    #     r"D:\project_images\resics_applied_ml_project",
    #     epochs = 2
    # )
    # print("Starting main execution...")
    # BASE_DATA_PATH = Path(r"D:\project_images\resics_applied_ml_project\data")
    # TRAIN_DIR = BASE_DATA_PATH / "train"
    # VAL_DIR = BASE_DATA_PATH / "val"
    # TEST_DIR = BASE_DATA_PATH / "test"
    # OUTPUT_DIR = Path(r"D:\project_images\resics_applied_ml_project")
    # MODEL_FILENAME = "my_cnn_baseline"

    # BATCH_SIZE = 32
    # NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 0
    # EPOCHS = 5
    # LEARNING_RATE = 0.001
    # NUM_FILTERS_CONV_LAYER = 16

    # data_handler = DataHandler(
    #     train_dir=str(TRAIN_DIR),
    #     val_dir=str(VAL_DIR),
    #     test_dir=str(TEST_DIR),
    #     num_workers=NUM_WORKERS,
    #     batch_size=BATCH_SIZE
    # )
    # print("DataHandler initialized.")
    # print(f"Training classes: {data_handler.classes}")

    # print("Initializing CNNBaselineModel...")
    # cnn_model = CNNBaselineModel(
    #     data_handler=data_handler,
    #     output_dir=str(OUTPUT_DIR),
    #     filename=MODEL_FILENAME,
    #     epochs=EPOCHS,
    #     learning_rate=LEARNING_RATE,
    #     num_filters=NUM_FILTERS_CONV_LAYER
    # )
    # print("CNNBaselineModel initialized.")

    # print("Starting model training (fit method)...")
    # training_results = cnn_model.fit()
    # print("Model training finished.")
    # print("Training results:", training_results)

    # print("Evaluating model on the test set...")
    # test_loss, test_accuracy = cnn_model.evaluate()
    # print(f"Evaluation complete. Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Before writing the path try to add r
    # in case you get an error
    # For instance r"D:\"
    # model = UploadModel(r"resnet_50.pth")
    # print(model.predict_single_image(r"D:\project_images — копия\resics_applied_ml_project\data\test\basketball_court\basketball_court_027.jpg"))
    # # model = NewModel(image_path=r"D:\project_images — копия\resics_applied_ml_project\data",
    #     output_dir=r"D:\project_images\resics_applied_ml_project",
    #     filename="resnet_50",
    #     unfreeze_specific_blocks = ["layer1", "layer2", "layer3", "layer4"],
    #     unfreeze_classifier=True,
    #     epochs=10
    #     )
# =======
# from model.upload_model.upload_model import UploadResNet50Model
# from fastapi import FastAPI, HTTPException, Request, File, UploadFile
# from fastapi.responses import HTMLResponse
# from pathlib import Path
# from pydantic import BaseModel, Field
# import base64
# from fastapi.templating import Jinja2Templates
# from typing import Dict

# PROJECT_ROOT = Path(__file__).resolve().parent
# WEIGHTS_PATH = PROJECT_ROOT / "model" / "weights" / "resnet_50_custom_model_weights.pth"
# IMAGE_DATA_PATH = PROJECT_ROOT / "model" / "data"

# class AccuracyResponse(BaseModel):
#     test_accuracy: str = Field(..., example="0.85")

# class PredictionResponse(BaseModel):
#     prediction: str = Field(..., example="airplane")
#     probabilities: dict = Field(..., example={"airplane": 0.87, "forest": 0.05})
#     original_image_data: str = Field(..., example="iVBORw0KGgoAAAANSUhEUgAA...")  # base64 image string
#     saliency_image_data: str = Field(..., example="iVBORw0KGgoAAAANSUhEUgAA...")  # base64 saliency map string


# model = UploadResNet50Model(
#     weight_path=WEIGHTS_PATH,
#     image_path=IMAGE_DATA_PATH
#     )

# templates = Jinja2Templates(directory="templates")

# app = FastAPI(
#     title="RESICS Model API",
#     description="API for testing the RESICS ResNet50 model."
# )

# @app.get("/", summary="Root endpoint", description="Simple message confirming API is up")
# def main_page_message():
#     return {"message": "API for testing the RESICS ResNet50 model."}

# @app.get("/accuracy_score", response_model=AccuracyResponse, summary="Get test accuracy",
#          description="Returns the model's test accuracy on validation data.")
# def test_accuracy():
#     accuracy = model.test_accuracy
#     return AccuracyResponse(test_accuracy=str(accuracy))

# @app.get("/upload", response_class=HTMLResponse, summary="Get upload form",
#          description="Returns an HTML form for uploading an image to classify.")
# async def get_upload_form(request: Request):
#     """Serves the HTML page with the upload form."""
#     return templates.TemplateResponse("upload_form.html", {"request": request})

# @app.post("/upload-predict", response_class=HTMLResponse, summary="Predict image class",
#           description="Upload an image file, get predicted class, class probabilities, and saliency map.")
# async def handle_prediction_upload(request: Request, file: UploadFile = File(..., description="Image file to classify")):
#     """Handles the image upload, predicts its class, and displays the result."""
#     try:
#         image_bytes = await file.read()
#         prediction, probabilities, saliency_b64 = model.predict_and_explain(image_bytes)

#         original_image_b64 = base64.b64encode(image_bytes).decode("utf-8")

#         return templates.TemplateResponse("upload_result.html", {
#             "request": request,
#             "prediction": prediction,
#             "probabilities": probabilities,
#             "original_image_data": original_image_b64,
#             "saliency_image_data": saliency_b64,
#             "mime_type": file.content_type
#         })
#     except FileNotFoundError as e:
#         return templates.TemplateResponse("error.html", {"request": request, "detail": str(e)}, status_code=404)
#     except IOError as e:
#         return templates.TemplateResponse("error.html", {"request": request, "detail": f"Image processing error: {str(e)}"}, status_code=400)
#     except Exception as e:
#         import traceback
#         print(f"An unexpected error occurred during prediction: {e}")
#         traceback.print_exc()
#         return templates.TemplateResponse("error.html", {"request": request, "detail": f"An unexpected error occurred. Please try again."}, status_code=500)
    
# @app.post("/upload-predict-json", response_model=PredictionResponse, summary="Predict image class (JSON)",
#           description="Upload image, get JSON response with prediction and saliency map in base64.")
# async def handle_prediction_json(file: UploadFile = File(..., description="Image file to classify")):
#     image_bytes = await file.read()
#     prediction, probabilities, saliency_b64 = model.predict_and_explain(image_bytes)
#     original_image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
#     return PredictionResponse(
#         prediction=prediction,
#         probabilities=probabilities,
#         original_image_data=original_image_b64,
#         saliency_image_data=saliency_b64
#     )
# >>>>>>> fb4d3a6e04252646ff9482085b54e7e4a3824d02
