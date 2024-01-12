import torch
from tests import _PATH_DATA

def test_data():

    train_images = torch.load(_PATH_DATA + "/processed/train_images.pt") 
    train_target = torch.load(_PATH_DATA + "/processed/train_target.pt")

    validation_images = torch.load(_PATH_DATA + "/processed/validation_images.pt")
    validation_target = torch.load(_PATH_DATA + "/processed/validation_target.pt")

    test_images = torch.load(_PATH_DATA + "/processed/test_images.pt")
    test_target = torch.load(_PATH_DATA + "/processed/test_target.pt")

    assert train_images.shape == torch.Size([28709, 1, 48, 48])
    assert train_target.shape == torch.Size([28709])

    assert validation_images.shape == torch.Size([3589, 1, 48, 48])
    assert validation_target.shape == torch.Size([3589])

    assert test_images.shape == torch.Size([3589, 1, 48, 48])
    assert test_target.shape == torch.Size([3589])