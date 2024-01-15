import pytest
import torch
from src.train_model import main, CustomTensorDataset


@pytest.fixture
def example_data():
    # Example data for testing
    train_images = torch.randn(100, 3, 32, 32)
    train_target = torch.randint(0, 5, (100,))
    return train_images, train_target


def test_CustomTensorDataset(example_data):
    train_images, train_target = example_data
    dataset = CustomTensorDataset([train_images, train_target])

    assert len(dataset) == 100

    sample = dataset[0]
    assert len(sample) == 2
    assert isinstance(sample[0], torch.Tensor)
    assert isinstance(sample[1], torch.Tensor)


def test_main_training_process():
    # Mocking hydra.main() to prevent script execution during testing
    with pytest.raises(SystemExit):
        main()

    # assert "Processing dataset completed." in caplog.text
    # assert "Training completed." in caplog.text
