import torch
import torch.nn as nn
from decoder import DecoderRNN
from encoder import EncoderCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderDecoder(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, model_name, drop_prob=0.3):
        super().__init__()

        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.model_name = model_name

        self.encoder = EncoderCNN(model_name, encoder_dim)
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size = vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )

        self.resize_encoding = None

    def encode(self, images):
        features = self.encoder(images)

        if self.resize_encoding is None:
            in_feat = features.size()[-1]
            self.resize_encoding = nn.Linear(in_feat, self.encoder_dim)
            
            for param in self.resize_encoding.parameters():
                param.requires_grad_(False)
        
        features = self.resize_encoding(features)
        return features

    def forward(self, images, captions):
        features = self.encode(images)
        outputs = self.decoder(features, captions)
        return outputs

    def save(self, model, num_epochs):
        model_state = {
            'num_epochs':num_epochs,
            'embed_size':self.embed_size,
            'vocab_size':self.vocab_size,
            'attention_dim':self.attention_dim,
            'encoder_dim':self.encoder_dim,
            'decoder_dim':self.decoder_dim,
            'state_dict':model.state_dict()
        }
        torch.save(model_state,'../models/encoder_{self.model_name}.pth')

def get_model(vocab_size, model_name = "resnet50"):
    return EncoderDecoder(
        embed_size=300,
        vocab_size=vocab_size,
        attention_dim=256,
        encoder_dim=2048,
        decoder_dim=512,
        model_name=model_name).to(device)

    #vocab_size = len(dataset.vocab),
