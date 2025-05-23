from fastapi import FastAPI
from model.upload_model.upload_model import UploadResNet50Model
# from pydantic import BaseModel, Field
import os, sys


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    w_path = "resics_applied_ml_project\\model\\weights\\resnet_50_custom_model_weights.pth"
    im_path = "resics_applied_ml_project\\model\\data"
    model = UploadResNet50Model(
        weight_path=w_path,
        image_path=im_path
    )
    # class LayerInfo(BaseModel):
    #     name: str = Field(..., description="The name of the layer.")
    #     type: str = Field(..., description="The type of the layer (e.g., Linear, Conv2d).")
    #     output_shape: list[int] = Field(..., description="The output shape of the layer.")
    #     param_count: int = Field(..., description="The number of parameters in the layer.")

    # class ModelSummary(BaseModel):
    #     total_params: int = Field(..., description="Total parameters in the model.")
    #     trainable_params: int = Field(..., description="Number of trainable parameters.")
    #     non_trainable_params: int = Field(..., description="Number of non-trainable parameters.")
    #     input_size_mb: float = Field(..., description="Estimated size of the input in Megabytes.")
    #     forward_pass_size_mb: float = Field(..., description="Estimated size of the forward pass in Megabytes.")
    #     total_size_mb: float = Field(..., description="Estimated total model size in Megabytes.")
    #     layers: list[LayerInfo] = Field(..., description="A list containing details for each layer.")
        
    app = FastAPI(
        title="RESICS Model API",
        description="API for testing the RESICS ResNet50 model."
    )
    
    @app.get("/test_acc")
    def test_accuracy():
        accuracy = model.test_accuracy
        return {"test accuracy":accuracy}
    

    
    # @app.get("/summary")
    # def model_summary():
    #     model_stats = model.summary

    #     # Prepare the list of layer information
    #     layers_info = []
    #     for layer in model_stats.summary_list:
    #         # Skip layers with no parameters (like ReLU, MaxPool2d, etc.) if desired
    #         if layer.num_params > 0:
    #             layer_data = LayerInfo(
    #                 name=layer.layer_name,
    #                 type=str(layer.class_name),
    #                 output_shape=list(layer.output_size),
    #                 param_count=layer.num_params
    #             )
    #             layers_info.append(layer_data)

    #     # Create the final response object using the Pydantic model
    #     summary_response = ModelSummary(
    #         total_params=model_stats.total_params,
    #         trainable_params=model_stats.trainable_params,
    #         non_trainable_params=model_stats.non_trainable_params,
    #         input_size_mb=model_stats.to_megabytes(model_stats.input_size),
    #         forward_pass_size_mb=model_stats.to_megabytes(model_stats.forward_pass_size),
    #         total_size_mb=model_stats.to_megabytes(model_stats.total_mult_adds + model_stats.total_input_size),
    #         layers=layers_info
    #     )

    #     return summary_response
