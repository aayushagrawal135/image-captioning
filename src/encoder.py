import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, encoder_dim=14):
        super(Encoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.set_encoder()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoder_dim, encoder_dim))

    def set_encoder(self):
        modules = list(models.resnet18(pretrained=True).children())[:-2]
        self.encoder = nn.Sequential(*modules)

        for param in self.encoder.parameters():
            param.requires_grad = False

    # images: (batch_size, 3, 224, 224)
    def forward(self, images):
        features = self.encoder(images)  # (batch_size, 512, 7, 7)
        features = self.adaptive_pool(features)  # (batch_size, 512, 14, 14)
        features = features.permute(0, 2, 3, 1)  # (batch_size, 14, 14, 512)
        return features
