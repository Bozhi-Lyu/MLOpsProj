import torch
from torch import nn
from transformers import DeiTForImageClassification, DeiTConfig
import torch.nn.functional as F
from omegaconf import OmegaConf

config = OmegaConf.load('src/models/model_config.yaml')['hyperparameters']

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
    def __init__(self):
        super().__init__()
        self.model = DeiTForImageClassification(model_config)

        n_inputs = self.model.classifier.in_features

        self.model.classifier = nn.Sequential(
                    nn.Linear(n_inputs, config.fully_connected_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(config.fully_connected_size, config.num_classes)
                )

    def forward(self, x):
        x = self.model(x).logits
        x = F.log_softmax(x, dim=1)
        return x

    def extract_features(self, x):
        return self.model(x)
