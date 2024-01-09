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


class DeiTClassifierPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model =  DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")
        
        n_inputs = self.model.classifier.in_features

        self.model.classifier = nn.Sequential(
                    nn.Linear(n_inputs, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 7)
                )

        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last layer (the "head")
        for param in self.model.classifier.parameters():
            param.requires_grad = True


    def forward(self, x):
        x = self.model(x).logits
        x = F.log_softmax(x, dim=1)
        return x

    def extract_features(self, x):
        return self.model(x)

# Python
def freeze_model_except_head(model):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last layer (the "head")
    for param in model.model.classifier.parameters():
        param.requires_grad = True

class EfficientNet(nn.Module):

    def __init__(self, num_classes=7):
        super().__init__()
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.model.fc =  nn.Sequential(
                    nn.Linear(1280, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes)
                )

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x

    def extract_features(self, x):
        return self.model(x)

class ResNet(nn.Module):
    
        def __init__(self, num_classes=7):
            super().__init__()
            self.model = timm.create_model('resnest14d', pretrained=True)
            self.model.fc =  nn.Sequential(
                        nn.Linear(2048, 512),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(512, num_classes)
                    )
    
        def forward(self, x):
            x = self.model(x)
            x = F.log_softmax(x, dim=1)
            return x
    
        def extract_features(self, x):
            return self.model(x)