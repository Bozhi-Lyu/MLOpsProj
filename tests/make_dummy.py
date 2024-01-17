import os
import pandas as pd


def make_dummy(csv_file, output_file, n=100):
    """
    Real raw data required.
    """
    df = pd.read_csv(csv_file)
    sample = df.sample(n)
    sample.to_csv(output_file, index=False)


if __name__ == "__main__":
    _TEST_ROOT = os.path.dirname(__file__)  # root of test folder
    _PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
    _PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of real data
    _PATH_DUMMY = os.path.join(_TEST_ROOT, "testdata")  # root of dummy data

    print(_PATH_DATA)
    csv_file = os.path.join(_PATH_DATA, "raw", "fer2013.csv")
    output_file = os.path.join(_PATH_DUMMY, "raw", "testdata.csv")
    make_dummy(csv_file=csv_file, output_file=output_file)
