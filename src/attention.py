import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Idea of Attetion:
        Instead of giving the entire image feature (output of encoder) to the decoder,
        we can select which region should be "highlighted/blurred" for producing the current word.

        We take weighted average of all regions in the image features for this.

        The weights of the average are determined with the hidd output of the previous layer
        which act as context for next word.

        Thus, we have Linear layers coming from both encoder and decoder.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention_dim = attention_dim

        self.decoder_to_attention = nn.Linear(decoder_dim, attention_dim)
        self.encoder_to_attention = nn.Linear(encoder_dim, attention_dim)
        self.attention = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # sum of values in a row (across columns) is 1

    # encoder out is flattened to (batch_size, 14*14, 512)
    # decoder (#batch_size, 256)
    def forward(self, encoder_out, decoder_hidden):
        regions = self.encoder_to_attention(encoder_out)  # (batch_size, 14*14, attention_dim)
        context = self.decoder_to_attention(decoder_hidden)  # (batch_size, attention_dim)

        # plug a dimension `1` in between so that addition can happen with broadcast
        combined_states = self.relu(regions + context.unsqueeze(1))  # (batch_size, 14*14, attention_dim)
        attention_val = self.attention(combined_states).squeeze(2)  # before squeeze (batch_size, 14*14, 1),
        # after squeeze (batch_size, 14*14)

        alpha = self.softmax(attention_val)  # (batch_size, 14*14)
        # broadcacast
        weighted_encoder_out = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        # alpha: (batch_size, 14*14)
        # weighted_encoder_out: (batch_size, 512)
        return alpha, weighted_encoder_out
