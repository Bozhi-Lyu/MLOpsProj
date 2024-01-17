import os
import torch
from tests import _PATH_MODEL, _PATH_DUMMY
from src.models.model import DeiTClassifier


def test_model():
    testmodel = DeiTClassifier()

    model_path = os.path.join(_PATH_MODEL, "saved_models", "model.pt")
    testmodel.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    processed_dir = os.path.join(_PATH_DUMMY, "processed/")

    train_images = torch.load(os.path.join(processed_dir, "train_images.pt"))
    train_target = torch.load(os.path.join(processed_dir, "train_target.pt"))

    output_tensor = testmodel(train_images)

    assert output_tensor.size(dim=0) == train_target.size(dim=0)
    max_value = torch.max(train_target)
    assert max_value < train_target.size()[0]


if __name__ == "__main__":
    # Get the data and process it
    test_model()
