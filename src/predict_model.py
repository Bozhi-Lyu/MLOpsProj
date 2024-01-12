# Related third-party imports
import torch


# def predict(
#   model: torch.nn.Module,
#    dataloader: torch.utils.data.DataLoader
# ) -> None:
def predict(model: nn.Module, dataloader: data.DataLoader) -> torch.Tensor:
    """
    Run prediction for a given model and dataloader.

    Args:
        model (nn.Module): Model to use for prediction.
        dataloader (data.DataLoader): Dataloader with batches of data.

    Returns:
        torch.Tensor: A tensor of shape [N, d], where N is the number of samples
                      and d is the output dimension of the model.
    """
    return torch.cat([model(batch) for batch in dataloader], 0)
