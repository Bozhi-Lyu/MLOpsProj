# Standard library imports
import logging
import sys

# Related third-party imports
import torch
import numpy as np
import pandas as pd
from src.data.make_dataset import change_to_tensor


# Configure logging to output to standard output with debug level information
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def make_subset(raw_dir="./data/raw/", processed_dir="./data/processed/", num=100):
    """
    Process the subset for prediction from a raw CSV file and save tensors to specified directories.

    Args:
        raw_dir (str): The directory where the raw data is stored.
        processed_dir (str): The directory where processed data will be saved.
        num(int): Number of samples for prediction.
    """

    logger.info("Processing dataset...")
    logger.info(f"Raw directory: {raw_dir}")

    # Reading dataset from CSV
    df = pd.read_csv(raw_dir + "fer2013.csv")
    random_samples = df.sample(n=num)

    pred_images = change_to_tensor(random_samples) / 255.0
    pred_target = torch.tensor(random_samples["emotion"].values)

    # Specify directory
    logger.info(f"Processed directory: { processed_dir}")
    logger.info(f"pred_images shape: { pred_images.shape}")
    logger.info(f"pred_target shape: { pred_target.shape}")

    # Saving the processed data to the specified directory
    torch.save(pred_images, processed_dir + "pred_images.pt")
    torch.save(pred_target, processed_dir + "pred_target.pt")

    logger.info(f"Prediction dataset saved.")


# Execution
if __name__ == "__main__":
    # Get the data and process it
    make_subset()
