import os
import torch
from omegaconf import OmegaConf
from src.train_model import main, CustomTensorDataset
from tests import _TEST_ROOT, _PATH_DUMMY


def example_data():
    processed_dir = os.path.join(_PATH_DUMMY, "processed/")
    train_images = torch.load(os.path.join(processed_dir, "train_images.pt"))
    train_target = torch.load(os.path.join(processed_dir, "train_target.pt"))
    return train_images, train_target


def test_CustomTensorDataset():
    train_images, train_target = example_data()
    dataset = CustomTensorDataset([train_images, train_target])

    for sample in dataset:
        assert sample[0].shape == torch.Size([1, 48, 48])
        assert sample[1].shape == torch.Size([])


def test_main_training_process():
    config = OmegaConf.load(os.path.join(_TEST_ROOT, "test_config.yaml"))
    main(config)
    assert True
