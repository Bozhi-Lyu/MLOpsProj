import logging
import os
# Related third-party imports
import torch

import torch
from torch.utils.data import DataLoader
from models.model import DeiTClassifier

import hydra
import wandb

log = logging.getLogger(__name__)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For CUDA 10.1
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8" # For CUDA >= 10.2
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

torch.use_deterministic_algorithms(True)

@hydra.main(config_path="config", config_name="default_config.yaml")
def predict(
    config,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    model_checkpoint: str = "/models/saved_models/model.pt",
    ) -> None:
    """
    Run prediction for a given model and dataloader.

    Args:
        model (nn.Module): Model to use for prediction.
        dataloader (data.DataLoader): Dataloader with batches of data.

    Returns:
        torch.Tensor: A tensor of shape [N, d], where N is the number of samples
                      and d is the output dimension of the model.
    """
    
    config = config['hyperparameters']  
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=config.project_name,
               entity=config.user)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    wandb.config = config

    model = DeiTClassifier().to(device)
    wandb.watch(model)
    model.load_state_dict(torch.load(model_checkpoint))
    


    return torch.cat([model(batch) for batch in dataloader], 0)
    