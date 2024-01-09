import torch
from torch import nn
from transformers import AutoImageProcessor, DeiTForImageClassification, DeiTConfig
import torch.nn.functional as F
import timm 

model_config = DeiTConfig(
    image_size=48,
    patch_size=16,
    num_classes=7, 
    num_channels=1,
    num_attention_heads=4,
    num_hidden_layers=4,
    hidden_size=256,
)


class DeiTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DeiTForImageClassification(model_config)

        n_inputs = self.model.classifier.in_features

        self.model.classifier = nn.Sequential(
                    nn.Linear(n_inputs, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 7)
                )
        

    def forward(self, x):
        x = self.model(x).logits
        x = F.log_softmax(x, dim=1)
        return x

    def extract_features(self, x):
        return self.model(x)
