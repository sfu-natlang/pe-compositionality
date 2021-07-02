import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ImageEmbed import *

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, textual, visual, image_size=64, dropout=0.5, bidirectional=False):
        super(RNNModel, self).__init__()
        self.visual  = visual
        self.textual = textual
        self.ntoken = ntoken
        self.bidirectional = bidirectional

        self.drop = nn.Dropout(dropout)

        emb_dim = ninp
        if self.visual and self.textual:
            # If the model uses two inputs, allocate 
            # half of the input dimensions to each one
            emb_dim = emb_dim//2
        if self.visual:
            self.image_encoder = ImageEmbedding(image_size, emb_dim)
        if self.textual:
            self.text_encoder = nn.Embedding(ntoken, emb_dim)

        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, bidirectional=self.bidirectional)

        # In a bidirectional model, there are nhid dimensions for both directions:
        if self.bidirectional:
            nhid *= 2
        self.nhid = nhid
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        if self.textual:
            nn.init.uniform_(self.text_encoder.weight, -initrange, initrange)
        if self.visual:
            self.image_encoder.init_weights(initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        text, images = input

        if self.visual:
            emb_v = self.drop(self.image_encoder(images))
        if self.textual:
            emb_t = self.drop(self.text_encoder(text))

        if self.visual and self.textual:
            emb = torch.cat([emb_v, emb_t], dim=2)
        elif self.visual:
            emb = emb_v
        elif self.textual:
            emb = emb_t
        else:
            raise Exception('Model has no inputs! Please set at least one of model.visual or model.textual to true')

        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        nhid = self.nhid
        if self.bidirectional:
            nhid = nhid // 2
        nlayers = self.nlayers
        if self.bidirectional:
            nlayers *= 2
        return (weight.new_zeros(nlayers, bsz, nhid),
                weight.new_zeros(nlayers, bsz, nhid))
