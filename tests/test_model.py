import torch
from tests import _PATH_MODEL
from src.models.model import DeiTClassifier

def test_model():

    testmodel = DeiTClassifier()
    
    testmodel.load_state_dict(torch.load(_PATH_MODEL + "/saved_models/model.pt"))
    input_shape = (1, 1, 48, 48) 
    expected_output_shape = (1, 7) 

    input_tensor = torch.randn(*input_shape)
    output_tensor = testmodel(input_tensor)

    assert output_tensor.size() == torch.Size(expected_output_shape)