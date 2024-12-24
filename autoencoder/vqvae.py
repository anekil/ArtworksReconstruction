import torch
import torch.nn as nn
import numpy as np
from encoder import Encoder
from quantizer import VectorQuantizer
from decoder import Decoder


class VQVAE(nn.Module):
    def __init__(self, h_dim=128, res_h_dim=32, n_res_layers=2, prequantizer_dim=8,
                 n_embeddings=512, embedding_dim=8, beta=0.25):
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(3, h_dim, res_h_dim, prequantizer_dim)
        
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        
        self.decoder = Decoder(embedding_dim, 3, h_dim, res_h_dim)

    def forward(self, x):

        z_e = self.encoder(x)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        return x_hat, embedding_loss, perplexity

if __name__ == "__main__":
    from torchinfo import summary
    vqvae = VQVAE(
        # h_dim=128,
        # res_h_dim=32,
        # n_res_layers=3,
        # n_embeddings=512,
        # embedding_dim=8,
        # beta=.25
    )
    summary(vqvae, input_size=(4, 3, 256, 256))
