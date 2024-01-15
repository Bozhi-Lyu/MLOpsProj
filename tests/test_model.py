import os
import torch
from tests import _PATH_MODEL
from src.models.model import DeiTClassifier


def test_model():
    testmodel = DeiTClassifier()

    model_path = os.path.join(_PATH_MODEL, "saved_models", "model.pt")
    testmodel.load_state_dict(torch.load(model_path))
    input_shape = (1, 1, 48, 48)
    expected_output_shape = (1, 7)

    input_tensor = torch.randn(*input_shape)
    output_tensor = testmodel(input_tensor)

    assert output_tensor.size() == torch.Size(expected_output_shape)
