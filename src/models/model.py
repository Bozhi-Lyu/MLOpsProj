# Related third-party imports
import torch
from torch import nn
import torch.nn.functional as F
from transformers import DeiTForImageClassification
from transformers import DeiTConfig
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("src/models/model_config.yaml")["hyperparameters"]

# Define model configuration
model_config = DeiTConfig(
    image_size=config.image_size,
    patch_size=config.patch_size,
    num_classes=config.num_classes,
    num_channels=config.num_channels,
    num_attention_heads=config.num_attention_heads,
    num_hidden_layers=config.num_hidden_layers,
    hidden_size=config.hidden_size,
)


class DeiTClassifier(nn.Module):
    """
    Custom classifier that uses the DeiT (Data-efficient Image Transformer) model.

    This classifier modifies the original DeiT model by replacing the classifier
    with a custom sequence of layers to accommodate specific output requirements.

    Methods:
        forward(x): Performs a forward pass through the model.
        extract_features(x): Extracts features from the input data using the model.
    """

    def __init__(self) -> None:
        """
        Initialize the model.
        """
        super().__init__()
        self.model = DeiTForImageClassification(model_config)

        # Update the classifier in the model
        n_inputs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(n_inputs, config.fully_connected_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.fully_connected_size, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output logits from the model.
        """
        x = self.model(x).logits
        x = F.log_softmax(x, dim=1)
        return x

    def extract_features(self, x: torch.Tensor) -> nn.Module:
        """
        Extract features from the input data using the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Any: Model output before the classification layer.
        """
        return self.model(x)
