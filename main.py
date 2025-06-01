from upload_model.model import UploadModel
from new_model.model import NewModel

if __name__ == "__main__":
    # Before writing the path try to add r
    # in case you get an error
    # For instance r"D:\"
    model = UploadModel(r"resnet_50_custom_model_weights.pth")
    print(model.predict_single_image(r"D:\project_images\resics_applied_ml_project\model\data\test\basketball_court\basketball_court_027.jpg"))
    # model = NewModel(image_path=r"D:\project_images\resics_applied_ml_project\model\data",
    #     output_dir=r"D:\project_images — копия\resics_applied_ml_project",
    #     filename="resnet_50",
    #     unfreeze_specific_blocks = ["layer4"]
    #     )