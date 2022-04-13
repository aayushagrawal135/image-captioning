import torch
import torch.nn as nn
from attention import *


# Attention Decoder
class Decoder(nn.Module):
    def __init__(self, embed_dim, vocab_size, attention_dim, encoder_dim, decoder_dim, dropout=0.3):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)

        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(dropout)

    # encoder_out: (batch_size, 14*14, 512)
    # encoded_captions: (batch_size, x)
    # caption_lens: (batch_size, 1)
    def forward(self, encoder_out, encoded_captions, caption_lens):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)  # 512 (number of output channels)
        vocab_size = self.vocab_size

        # Flatten
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, 14*14, 512)
        num_pixels = encoder_out.size(1)  # 14*14

        caption_lens, indices = caption_lens.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[indices]
        encoded_captions = encoded_captions[indices]

        embeddings = self.embedding(encoded_captions)  # (batch_size, x, embedding_dim)
        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = (caption_lens - 1).tolist()
        max_seq_length = max(decode_lengths)

        predictions = torch.zeros(batch_size, max_seq_length, self.vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max_seq_length, num_pixels).to(self.device)

        for s in range(max_seq_length):
            alpha, weighted_encoder_out = self.attention(encoder_out, h)
            lstm_input = torch.cat((embeddings[:, s, :], weighted_encoder_out), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            predictions[:, s] = output
            alphas[:, s] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, indices

    # encoder_out: (batch_size, 14*14, 512)
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)  # (batch_size, 512)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim)
        return h, c
