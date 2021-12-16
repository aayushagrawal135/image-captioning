import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, model_name, encoder_dim = 2048):
        super(EncoderCNN, self).__init__()
        self.encoder_dim = encoder_dim
        self.encoder = self.get_encoder(model_name)

    def get_encoder(self,model_name):
        if model_name == "resnet50":
            resnet = models.resnet50(pretrained=True)
            for param in resnet.parameters():
                param.requires_grad_(False)
            modules = list(resnet.children())[:-2]
            # resnet.fc = nn.Linear()
            resnet = nn.Sequential(*modules)
            return resnet

        elif model_name == "vgg16":
            vgg = models.vgg16(pretrained=True)
            for param in vgg.parameters():
                param.requires_grad_(False)
            modules = list(vgg.children())[:-2]
            vgg = nn.Sequential(*modules)
            return vgg

    def forward(self, images):
        features = self.encoder(images)                                    #(batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)                           #(batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,49,2048)
        return features