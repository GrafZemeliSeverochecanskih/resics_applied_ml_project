from new_model.create_model import CreateNewModel

if __name__ == "__main__":
    model_creator = CreateNewModel(
        epochs=2,
        image_path="D:\\Study\\resics_applied_ml_project\\splitted_data",
        monte_carlo_inference = False
        )
    print(model_creator.model.model_summary)
    