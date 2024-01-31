import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)


def evaluate_model(test_data_path, model_path):
    # Load the test data
    with h5py.File(test_data_path, "r") as hf:
        test_features = torch.tensor(hf["test"]["features"][:])
        test_embeddings = torch.tensor(hf["test"]["embeddings"][:])
        # Load features_index
        feature_index = torch.tensor(hf["test"]["feature_names"][:])
        true_labels = torch.tensor(hf["test"]["labels"][:])
        # Convert features_index to a list for easier handling
        features_index_list = list(hf["test"]["feature_index"][:])

    # Load the model
    model = torch.load(model_path)
    model.eval()

    # Make predictions
    with torch.no_grad():
        predictions = model(test_features, test_embeddings)

    # Calculate evaluation metrics for the entire dataset
    metrics = calculate_metrics(true_labels, predictions)

    # Group by "Material" and "Product Code"
    material_index = features_index_list.index(b"Material")
    product_code_index = features_index_list.index(b"Product Code")
    material_codes = test_features[:, material_index]
    product_codes = test_features[:, product_code_index]

    # Convert to DataFrame for easier group by operations
    df = pd.DataFrame(
        {
            "Material": material_codes.numpy().astype(np.int),
            "Product Code": product_codes.numpy().astype(np.int),
            "True Labels": true_labels.numpy(),
            "Predictions": predictions.numpy().reshape(-1),
        }
    )

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter("model_evaluation.xlsx", engine="xlsxwriter")

    # Convert the dataframe to an XlsxWriter Excel object.
    pd.DataFrame(metrics, index=[0]).to_excel(
        writer, sheet_name="Overall Metrics", index=False
    )

    material_metrics = (
        df.groupby("Material")
        .apply(lambda x: calculate_metrics(x["True Labels"], x["Predictions"]))
        .unstack()
    )
    material_metrics.to_excel(writer, sheet_name="Material Metrics")

    product_metrics = (
        df.groupby("Product Code")
        .apply(lambda x: calculate_metrics(x["True Labels"], x["Predictions"]))
        .unstack()
    )
    product_metrics.to_excel(writer, sheet_name="Product Metrics")

    material_product_metrics = (
        df.groupby(["Material", "Product Code"])
        .apply(lambda x: calculate_metrics(x["True Labels"], x["Predictions"]))
        .unstack()
    )
    material_product_metrics.to_excel(writer, sheet_name="Material_Product Metrics")

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    print(
        "Metrics for the entire dataset, grouped by Material, Product Code, and both have been saved to 'model_evaluation.xlsx'"
    )


def calculate_metrics(true_labels, predictions):
    evs = explained_variance_score(true_labels, predictions)
    mae = mean_absolute_error(true_labels, predictions)
    mse = mean_squared_error(true_labels, predictions)
    medae = median_absolute_error(true_labels, predictions)
    sample_size = len(true_labels)
    average_predicted = np.mean(predictions)
    average_actual = np.mean(true_labels)
    average_difference = np.mean(np.abs(predictions - true_labels))
    return {
        "Explained Variance Score": evs,
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "Median Absolute Error": medae,
        "Sample Size": sample_size,
        "Average Predicted": average_predicted,
        "Average Actual": average_actual,
        "Average Difference": average_difference,
    }


# Assuming the path to the h5 file and the model path
test_data_path = "path/to/your/test_data.h5"
model_path = "model.pt"

evaluate_model(test_data_path, model_path)
