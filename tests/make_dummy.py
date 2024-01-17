import os
import torch
import random
import pandas as pd

from collections import OrderedDict



def make_dummyset(csv_file, output_file, n=100):
    """
    Real raw data required.
    """
    df = pd.read_csv(csv_file)
    sample = df.sample(n)
    sample.to_csv(output_file, index=False)


def make_dummymodel(model_path, dummymodel_path):
    """
    Real model_dict required.
    """

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    dummy_state_dict = OrderedDict()
    for name, param in state_dict.items():
        random_factor = random.uniform(0.5, 1.5)
        dummy_state_dict[name] = param * random_factor
    torch.save(dummy_state_dict, dummymodel_path)

if __name__ == "__main__":
    _TEST_ROOT = os.path.dirname(__file__)  # root of test folder
    _PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
    _PATH_MODEL = os.path.join(_PROJECT_ROOT, "models")  # root of model
    _PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of real data
    _PATH_DUMMY = os.path.join(_TEST_ROOT, "testdata")  # root of dummy data

    csv_file = os.path.join(_PATH_DATA, "raw", "fer2013.csv")
    output_file = os.path.join(_PATH_DUMMY, "raw", "testdata.csv")
    make_dummyset(csv_file=csv_file, output_file=output_file)

    model_path = os.path.join(_PATH_MODEL, "saved_models", "model.pt")
    dummymodel_path = os.path.join(_TEST_ROOT, "dummymodel.pt")
    make_dummymodel(model_path, dummymodel_path)
