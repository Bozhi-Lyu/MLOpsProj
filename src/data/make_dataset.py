# Standard library imports
import logging
import sys

# Related third-party imports
import torch
import numpy as np
import pandas as pd

# Configure logging to output to standard output with debug level information
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def change_to_tensor(df: pd.DataFrame) -> torch.Tensor:
    """
    Convert the pixels column in the DataFrame to a PyTorch Tensor of shape (n, 48, 48).

    Args:
        df (pd.DataFrame): DataFrame with a 'pixels' column containing space-separated pixel values.

    Returns:
        torch.Tensor: A tensor representation of the images.
    """
    img = df["pixels"].apply(lambda x: x.split(" "))
    x = np.array([np.array(x) for x in img])
    data = torch.tensor(x.astype(int)).reshape(-1, 48, 48).unsqueeze(1)
    return data


def make_dataset(raw_dir="./data/raw/", processed_dir="./data/processed/"):
    """
    Process the dataset from a raw CSV file and save tensors to specified directories.

    Args:
        raw_dir (str): The directory where the raw data is stored.
        processed_dir (str): The directory where processed data will be saved.
    """

    logger.info("Processing dataset...")
    logger.info(f"Raw directory: {raw_dir}")

    # Reading dataset from CSV
    df = pd.read_csv(raw_dir + "fer2013.csv")

    # Splitting the dataset into training, validation, and test sets and converting to tensors
    train = change_to_tensor(df[df["Usage"] == "Training"]) / 255.0
    validation = change_to_tensor(df[df["Usage"] == "PublicTest"]) / 255.0
    test = change_to_tensor(df[df["Usage"] == "PrivateTest"]) / 255.0

    # Extracting target labels for each set
    train_target = torch.tensor(df[df["Usage"] == "Training"]["emotion"].values)
    validation_target = torch.tensor(df[df["Usage"] == "PublicTest"]["emotion"].values)
    test_target = torch.tensor(df[df["Usage"] == "PrivateTest"]["emotion"].values)

    # Logging the shapes of the datasets for verification
    logger.info(f"Train shape: {train.shape}")
    logger.info(f"Validation shape: {validation.shape}")
    logger.info(f"Test shape: {test.shape}")

    logger.info(f"Train target shape: { train_target.shape}")
    logger.info(f"Validation target shape: { validation_target.shape}")
    logger.info(f"Test target shape: { test_target.shape}")

    # Specify directory
    logger.info(f"Processed directory: { processed_dir}")

    # Saving the processed data to the specified directory
    torch.save(train, processed_dir + "train_images.pt")
    torch.save(validation, processed_dir + "validation_images.pt")
    torch.save(test, processed_dir + "test_images.pt")
    torch.save(train_target, processed_dir + "train_target.pt")
    torch.save(validation_target, processed_dir + "validation_target.pt")
    torch.save(test_target, processed_dir + "test_target.pt")


# Execution
if __name__ == "__main__":
    # Get the data and process it
    make_dataset()
