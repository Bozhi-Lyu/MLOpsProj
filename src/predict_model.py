import logging
import os

# Related third-party imports
import torch

import torch
from torchvision import transforms

from src.models.model import DeiTClassifier
from src.train_model import CustomTensorDataset

import hydra
import wandb


@hydra.main(config_path=".", config_name="pred_config.yaml", version_base="1.2")
def predict(
    config,
    model_checkpoint: str = "models/saved_models/model.pt",
) -> None:
    """
    Run prediction for a given model and dataloader.
    Run src/data/make_subset.py first.

    Args:
        model_checkpoint (str): Saved model to use for prediction.
        data_path (str): Data used for prediction.

    """

    config = config["hyperparameters"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project=config.project_name, entity=config.user)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    wandb.config = config

    pred_images = torch.load(os.path.join(config["data_path"] + "pred_images.pt"))
    pred_target = torch.load(config["data_path"] + "pred_target.pt")
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of applying a horizontal flip
            transforms.RandomRotation(10),  # Rotate the image by up to 10 degrees
            transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),  # Zoom in on the image
        ]
    )

    pred_set = CustomTensorDataset((pred_images, pred_target), transform=transform)

    model = DeiTClassifier().to(device)
    wandb.watch(model)
    model.load_state_dict(torch.load(model_checkpoint))

    trainloader = torch.utils.data.DataLoader(
        pred_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    correct = 0
    total = 0

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    logger.info(f"Prediction Accuracy: {(100 * accuracy):.2f}%")
    wandb.log({"Prediction_Accuracy": accuracy})


# Execution
if __name__ == "__main__":
    predict()
