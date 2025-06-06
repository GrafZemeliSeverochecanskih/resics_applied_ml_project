from new_model.model import NewModel
class UploadResNetModelRunner():
    def __init__(self,
                 image_path,
                 output_dir,
                 filename="resnet_50",
                 unfreeze_specific_blocks = ["layer1", "layer2", "layer3", "layer4"],
                 unfreeze_classifier=True,
                 epochs=10
                 ):
        self.__model = NewModel(image_path=image_path,
        output_dir=output_dir,
        filename=filename,
        unfreeze_specific_blocks = unfreeze_specific_blocks,
        unfreeze_classifier=unfreeze_classifier,
        epochs=epochs()
        )
    
    @property
    def model(self):
        return self.__model