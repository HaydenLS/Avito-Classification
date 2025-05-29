# Project libs
from app.imports import *

# Torch libs
import torchvision.models as models

MODEL_DIR = 'trained_models/'

def get_model(out_channels):
    """Данный метод позволяет получить саму модель.
        Он воспроизводит архитектуру модели и возвращает её"""

    model_resnet = models.resnet34(pretrained=True).to(DEVICE)

    class fc_layers(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

            self.out = nn.Linear(256, out_channels)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            logits = self.out(x)
            return logits
    
    model_resnet.fc = fc_layers()

    return model_resnet


def load_model(out_channels, path='trainded_models/model_resnet.pt'):
    model = get_model(out_channels)

    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model



def save_model():
    pass