import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    """
    Helper layer implementing the flatten operation.
    """
    def forward(self, input):
        return input.view(input.size(0), -1)

class ImageEmbedding(nn.Module):
    """
    Module which acts like an Embedding layer, but for a sequence of images rather than
    token ids. Each image is run through a convolution and the layer returns a sequence
    of the resulting image embeddings. All timesteps share the same CNN weights.
    """
    def __init__(self, image_size, output_dim, kernel_size=10, stride=2, n_filters=96, pool_dim=2):
        super(ImageEmbedding, self).__init__()

        # Compute size of the final dense layer:
        size = image_size
        size = int( ((size - kernel_size)/stride) + 1 )
        size = size // pool_dim
        dense_dim = size*size*n_filters

        self.conv = nn.Conv2d( 1, n_filters, kernel_size=kernel_size, stride=(stride,stride) )
        self.pool = nn.MaxPool2d( pool_dim )
        self.flat  = Flatten()
        self.dense = nn.Linear( dense_dim, output_dim )

    def forward(self, input):
        # Expects input of shape 
        # (seq_len, batch_size, n_channels, img_height, img_width)

        # Apply the convolution to each image in 
        # the sequence, as in an Embedding layer 
        embs = []
        for batch in range(input.shape[1]):
            emb = input[:,batch,:,:,:]
            emb = nn.ReLU()(self.conv(emb))
            emb = self.pool(emb)
            emb = self.flat(emb)
            emb = self.dense(emb)
            embs.append(emb)
        emb = torch.stack(embs).permute(1,0,2)
        return emb

    def init_weights(self, initrange):
        initrange = math.sqrt(2)
        nn.init.xavier_uniform_(self.conv.weight,  initrange)
        nn.init.xavier_uniform_(self.dense.weight, initrange)
