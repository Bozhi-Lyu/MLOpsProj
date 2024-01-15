# Standard library imports
import os
import logging

# Related third-party imports
import torch
import tqdm
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import hydra
from src.models.model import DeiTClassifier


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


@hydra.main(config_name="train_config.yaml", config_path=".", version_base="1.2")
def main(config):
    """
    Main function for training the model.

    Initializes the model and dataloaders, then continues to train and validate.
    Evaluates the model's performance and saves the results as well as the final trained model.
    """

    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For CUDA 10.1
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # For CUDA >= 10.2
    # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    torch.use_deterministic_algorithms(True)
    # make sure reproducibility.

    config = config["hyperparameters"]
    torch.manual_seed(config["seed"])
    wandb.init(project=config.project_name, entity=config.user)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    wandb.config = config

    logger.info("Training FER model")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Training on GPU")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of applying a horizontal flip
            transforms.RandomRotation(10),  # Rotate the image by up to 10 degrees
            transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),  # Zoom in on the image
        ]
    )

    train_images = torch.load(config["data_path"] + "train_images.pt")
    train_target = torch.load(config["data_path"] + "train_target.pt")

    validation_images = torch.load(config["data_path"] + "validation_images.pt")
    validation_target = torch.load(config["data_path"] + "validation_target.pt")

    test_images = torch.load(config["data_path"] + "test_images.pt")
    test_target = torch.load(config["data_path"] + "test_target.pt")

    # train_set = TensorDataset(train_images, train_target)
    # validation_set = TensorDataset(validation_images, validation_target)
    # test_set = TensorDataset(test_images, test_target)

    # Create datasets
    train_set = CustomTensorDataset((train_images, train_target), transform=transform)
    validation_set = CustomTensorDataset((validation_images, validation_target), transform=transform)
    test_set = CustomTensorDataset((test_images, test_target), transform=transform)

    # Initialize model and dataloaders
    model = DeiTClassifier().to(device)
    wandb.watch(model)
    logger.info("Processing dataset completed.")

    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    validationloader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    testloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Initialize optimizer and loss criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    history = []

    logger.info("Starting training...")

    # Loop
    for epoch in range(config.epochs):
        train_loss = 0
        val_loss = 0
        model.train()
        for images, labels in tqdm.tqdm(trainloader, total=len(trainloader), position=0, leave=True):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            history.append(loss.item())

            wandb.log({"train_loss": loss})

            logger.debug("In epoch loss: {}".format(loss.item()))

        logger.info(f"Epoch {epoch} - Training loss: {train_loss/len(trainloader)}")

        model.eval()
        with torch.no_grad():
            correct = 0
            for images, labels in validationloader:
                images = images.to(device)
                labels = labels.to(device)

                output = model(images)
                loss = criterion(output, labels)
                logger.debug("Validation loss: {}".format(loss.item()))

                # Measure accuracy
                _, pred = torch.max(output, 1)
                correct += (pred == labels).sum().item()
                logger.debug("Validation accuracy: {}".format(correct / len(labels)))

        accuracy = correct / len(validation_set)
        wandb.log({"val_accuracy": accuracy})

    val_loss += loss.item()

    logger.info(f"Epoch {epoch} - Validation loss: {val_loss/len(validationloader)}")

    logger.info("Training completed.")

    if not os.path.exists("models/saved_models"):
        os.makedirs("models/saved_models")

    torch.save(model.state_dict(), "models/saved_models/model.pt")

    # Plot training curve
    plt.plot(range(len(history)), history, label="Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.savefig("reports/figures/training_curve.png")

    # Test the model
    logger.info("Testing model...")

    model.eval()
    with torch.no_grad():
        correct = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)
            logger.debug("Test loss: {}".format(loss.item()))

            # Measure accuracy
            _, pred = torch.max(output, 1)
            correct += (pred == labels).sum().item()

        accuracy = correct / len(test_set)

        logger.info("Test accuracy: {}".format(accuracy))
        wandb.log({"test_accuracy": accuracy})

        logger.info(f"Test accuracy: {accuracy * 100}%")


# Execution
if __name__ == "__main__":
    main()
