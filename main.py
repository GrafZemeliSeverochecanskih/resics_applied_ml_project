from upload_model.model import UploadModel
from new_model.model import NewModel
from pathlib import Path
import os
from new_baseline_model.data_handler import DataHandler
from new_baseline_model.model import CNNBaselineModel
from new_cnn_runner import NewCnnRunner 
from upload_baseline_model.baseline_model import UploadCNN
from upload_model.utils import Utilities
import uvicorn
from app import app

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
    # model = UploadCNN(
    #     r"D:\project_images\resics_applied_ml_project\my_cnn_baseline.pth",
    # )
    # print(model.predict(r"D:\project_images\resics_applied_ml_project\data\test\airplane\airplane_692.jpg"))
    # model = NewCnnRunner(
    #     r"D:\project_images\resics_applied_ml_project\data",
    #     r"D:\project_images\resics_applied_ml_project",
    #     epochs = 2
    # )
    # print("Starting main execution...")

    # Before writing the path try to add r
    # in case you get an error
    # For instance r"D:\"
    # model = UploadModel(r"resnet_50_weight_decay.pth")
    # img = Utilities().convert_image_to_bytes(r"D:\project_images\resics_applied_ml_project\data\test\palace\palace_001.jpg")
    # print(model.predict_with_mc_dropout(image_bytes=img))
    # predicted_class_name, probs, grad_cam_viz_b64, normgrad_viz_b64 = model.predict_and_explain(image_bytes=img)
    # print(predicted_class_name, probs)
    # grad_cam_viz_image = Utilities().base64_to_image(grad_cam_viz_b64)
    # normgrad_viz_image = Utilities().base64_to_image(normgrad_viz_b64)
    # grad_cam_viz_image.show()
    # normgrad_viz_image.show()
    # print(model.predict_single_image(r"D:\project_images\resics_applied_ml_project\data\test\palace\palace_001.jpg"))
    # model = NewModel(image_path=r"D:\project_images — копия\resics_applied_ml_project\data",
    #     output_dir=r"D:\project_images\resics_applied_ml_project",
    #     filename="resnet_50_weight_decay",
    #     unfreeze_specific_blocks = ["layer1", "layer2", "layer3", "layer4"],
    #     unfreeze_classifier=True,
    #     epochs=10
    #     )
