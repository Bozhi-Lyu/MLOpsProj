import torch
import os
from src.models.model import DeiTClassifier
from src.train_model import CustomTensorDataset

def predict_image(
        model: torch.nn.Module,
        image: torch.Tensor
) -> None:
    return model(image)

def predict(
) -> torch.Tensor:
    """
    Run prediction for a given model and dataloader.

    Args:
        model (nn.Module): Model to use for prediction.
        dataloader (data.DataLoader): Dataloader with batches of data.

    Returns:
        torch.Tensor: A tensor of shape [N, d], where N is the number of samples
                      and d is the output dimension of the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_checkpoint: str = "models/saved_models/model.pt",
    model = DeiTClassifier().to(device)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    pred_images = torch.load(os.path.join("data/processed/"+ "test_images.pt"))
    pred_target = torch.load("data/processed/" + "pred_target.pt")

    pred_set = CustomTensorDataset((pred_images, None))

    '''
    # Including this here in case that we need a dataloader
    predloader = torch.utils.data.DataLoader(
        pred_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    '''

    return torch.cat([model(batch) for batch in pred_set], 0)

if __name__ == "__name__":
    predict()
