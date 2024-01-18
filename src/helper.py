import torch
from torch.utils.data import Dataset
from typing import Iterator
import warnings

def extract_hyperparameters(config):
    """
    Extract hyperparameters from the provided config.
    Prioritises hyperparameters based on W&B sweeps, but uses default
    parameters in the train_config.yaml file otherwise.

    Parameters:
    - config: The configuration dictionary.

    Returns:
    - learning_rate: Extracted learning rate.
    - epochs: Extracted number of epochs.
    - optimizer: Extracted optimizer.
    - batch_size: Extracted batch size.
    """
    # Check if "sweep_learning_rate" is present in the configuration
    if "sweep_learning_rate" in config:
        learning_rate = config["sweep_learning_rate"]
    else:
        # Fall back to using "learning_rate" from the original configuration
        learning_rate = config["learning_rate"]

    # Check if "sweep_epochs" is present in the configuration
    if "sweep_epochs" in config:
        epochs = config["sweep_epochs"]
    else:
        # Fall back to using "epochs" from the original configuration
        epochs = config["epochs"]

    # Check if "sweep_optimizer" is present in the configuration
    if "sweep_optimizer" in config:
        optimizer = config["sweep_optimizer"]
    else:
        # Fall back to using "optimizer" from the original configuration
        optimizer = config["optimizer"]

    # Check if "sweep_batch_size" is present in the configuration
    if "sweep_batch_size" in config:
        batch_size = config["sweep_batch_size"]
    else:
        # Fall back to using "batch_size" from the original configuration
        batch_size = config["batch_size"]

    return learning_rate, epochs, optimizer, batch_size

def parse_optimizer(
        input: str, 
        parameters: Iterator[torch.nn.parameter.Parameter], 
        lr: float
) -> torch.optim.Optimizer:
    """
    Parses string inputs to PyTorch optimizers.
    Args:
        input:         A string that specifies which optimizer should be used. 
        parameters:    An iterator of model parameters. Intended to be the output
                       of model.parameters()
        lr: The learning rate used in the optimizer

    Returns:
        torch.optim.Optimizer: A optimizer as according to the input parameter.
                               Defaults to using Adam if the input string
                               is unrecognised.
    """
    if input == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    elif input == "adamw":
        return torch.optim.AdamW(parameters, lr=lr)
    elif input == "adagrad":
        return torch.optim.Adagrad(parameters, lr=lr)
    elif input == "adadelta":
        return torch.optim.Adadelta(parameters, lr=lr)
    elif input == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    else: #default to adam
        warnings.warn("Warning: input string in parse_optimizer unrecognised. Defaulting to Adam")
        return torch.optim.Adam(parameters, lr=lr)
    
class CustomTensorDataset(Dataset):
    """
    TensorDataset with support of transforms.

    Extends the standard PyTorch Dataset to include transform capabilities,
    which enables the data to be preprocessed bvia various transformations 
    before the data is input into the model.   
    """

    def __init__(self, tensors: torch.Tensor, transform=None) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)