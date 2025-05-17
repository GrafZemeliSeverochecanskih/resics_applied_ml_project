from new_model.create_new_model import CreateNewModel
from upload_model.upload_model import UploadResNet50Model
import sys
import io


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    uploader = UploadResNet50Model(
        weight_path="resics_applied_ml_project\\model\\weights\\resnet_50_custom_model_weights.pth",
    )
    uploader.model.model_summary
    # model_creator = CreateNewModel(
    #     epochs=10,
    #     output_dir="D:\\project_images\\resics_applied_ml_project\\model\\weights",
    #     image_path="D:\\Study\\resics_applied_ml_project\\splitted_data",
    #     unfreeze_specific_blocks = ["layer4"]
    #     )
    # if model_creator.plot_creator and model_creator.results:
    #     print("Visualizing training loss...")
    #     model_creator.plot_creator.visualize_loss()
    #     print("Visualizing training accuracy...")
    #     model_creator.plot_creator.visualize_accuracy()
    #     print("Plot visualization complete.")
    #     print("If plots are not showing, ensure Matplotlib backend is correctly configured for your environment (e.g., running in a GUI environment or saving plots to files).")
    # elif not model_creator.results:
    #     print("Plot creator might be initialized, but no training results found. Skipping visualization.")
    # else:
    #     print("Plot creator not initialized, skipping visualization. This might happen if an error occurred during training or result generation.")

    # print("Process finished.")
    