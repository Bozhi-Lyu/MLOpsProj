import os
import torch
from tests import _PATH_DUMMY
from src.data.make_dataset import make_dataset


def test_data():
    # _TEST_ROOT = os.path.dirname(__file__)  # root of test folder
    # _PATH_DUMMY = os.path.join(_TEST_ROOT, "testdata")  # root of dummy data
    raw_dir = os.path.join(_PATH_DUMMY, "raw", "testdata.csv")
    processed_dir = os.path.join(_PATH_DUMMY, "processed/")
    print("raw_dir", raw_dir)
    print("processed_dir", processed_dir)
    make_dataset(raw_dir=raw_dir, processed_dir=processed_dir)

    train_images = torch.load(os.path.join(processed_dir, "train_images.pt"))
    train_target = torch.load(os.path.join(processed_dir, "train_target.pt"))

    validation_images = torch.load(os.path.join(processed_dir, "validation_images.pt"))
    validation_target = torch.load(os.path.join(processed_dir, "validation_target.pt"))

    test_images = torch.load(os.path.join(processed_dir, "test_images.pt"))
    test_target = torch.load(os.path.join(processed_dir, "test_target.pt"))

    assert train_images.shape[-3:] == torch.Size([1, 48, 48])
    assert validation_images.shape[-3:] == torch.Size([1, 48, 48])
    assert test_images.shape[-3:] == torch.Size([1, 48, 48])
    assert len(list(train_target.size())) == 1
    assert len(list(validation_target.size())) == 1
    assert len(list(test_target.size())) == 1

    assert list(train_target.size())[0] + list(validation_target.size())[0] + list(test_target.size())[0] == 100


if __name__ == "__main__":
    # Get the data and process it
    test_data()
