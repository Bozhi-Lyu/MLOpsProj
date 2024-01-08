import torch
import pandas as pd
import torch
import os
import logging 
import numpy as np 
import sys 
from torchvision import transforms

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def change_to_tensor(df: pd.DataFrame)-> torch.Tensor:
    """ 
    Changes the pixels column to a tensor of shape (n, 48, 48)
    """
    
    img = df['pixels'].apply(lambda x: x.split(' '))
    x = np.array([np.array(x) for x in img])
    data = torch.tensor(x.astype(int)).reshape(-1, 48, 48).unsqueeze(1)
    return data 


def make_dataset(raw_dir = "./data/raw/", processed_dir = "./data/processed/"):
    
    logger.info("Processing dataset...")
    logger.info(f"Raw directory: {raw_dir}")

    df = pd.read_csv(raw_dir + "fer2013.csv")

    train = change_to_tensor(df[df['Usage'] == 'Training']) / 255.0
    validation = change_to_tensor(df[df['Usage'] == 'PublicTest']) / 255.0
    test = change_to_tensor(df[df['Usage'] == 'PrivateTest']) / 255.0

    train_target = torch.tensor(df[df['Usage'] == 'Training']['emotion'].values)
    validation_target = torch.tensor(df[df['Usage'] == 'PublicTest']['emotion'].values)
    test_target = torch.tensor(df[df['Usage'] == 'PrivateTest']['emotion'].values)

    # # change targets to one-hot encoding
    # train_target = torch.nn.functional.one_hot(train_target, num_classes=7)
    # validation_target = torch.nn.functional.one_hot(validation_target, num_classes=7)
    # test_target = torch.nn.functional.one_hot(test_target, num_classes=7)

    logger.info(f"Train shape: {train.shape}")
    logger.info(f"Validation shape: {validation.shape}")
    logger.info(f"Test shape: {test.shape}")

    logger.info(f"Train target shape: { train_target.shape}")
    logger.info(f"Validation target shape: { validation_target.shape}")
    logger.info(f"Test target shape: { test_target.shape}")

    logger.info(f"Processed directory: { processed_dir}")
    
    torch.save(train, processed_dir + "train_images.pt")
    torch.save(validation, processed_dir + "validation_images.pt")
    torch.save(test, processed_dir + "test_images.pt")
    torch.save(train_target, processed_dir + "train_target.pt")
    torch.save(validation_target, processed_dir + "validation_target.pt")
    torch.save(test_target, processed_dir + "test_target.pt")
    


if __name__ == '__main__':
    # Get the data and process it
    make_dataset()

