import os

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of real data
_PATH_MODEL = os.path.join(_PROJECT_ROOT, "models")  # root of model
_PATH_DUMMY = os.path.join(_TEST_ROOT, "testdata")  # root of dummy data
