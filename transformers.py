import torch 
from torch import nn 
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        """
        Rotary Positional Embedding (RoPE)
        
        Args:
            dim: The dimensionality of the token embeddings (must be even).
        """
        super(RotaryEmbedding, self).__init__()
        
        self.dim = dim
        self.inv_freq = 1. / (10000 ** (torch.arange(0., dim, 2.0) / dim))
        
    def forward(self, x):
        """
        Apply rotary embedding to the input tensor x.
        
        Args:
            x: A tensor of shape (batch_size, seq_len, dim)
        
        Returns:
            The tensor after applying rotary embedding, shape (batch_size, seq_len, dim)
        """
        seq_len, device = x.size(1), x.device
        # Create position indices for the rotary embedding
        freqs = self.inv_freq.to(device) * torch.arange(seq_len, device=device)
        
        # Apply the rotation (cosine/sine) to the even and odd dimensions
        rot_sin = torch.sin(freqs).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        rot_cos = torch.cos(freqs).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        
        # Prepare input tensor x (shape: batch_size, seq_len, dim)
        x_1 = x[..., ::2]  # Even-indexed dimensions
        x_2 = x[..., 1::2]  # Odd-indexed dimensions
        
        # Rotate the even and odd-indexed components separately
        x_1_rot = x_1 * rot_cos - x_2 * rot_sin
        x_2_rot = x_1 * rot_sin + x_2 * rot_cos
        
        # Concatenate the rotated parts back together
        x_rot = torch.cat([x_1_rot, x_2_rot], dim=-1)
        
        return x_rot
class Attention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.K = nn.Linear(embed_dim,embed_dim)
        self.Q = nn.Linear(embed_dim,embed_dim)
        self.V = nn.Linear(embed_dim,embed_dim)
        
        self.positional_encoder = RotaryEmbedding(embed_dim)
        
    def forward(self, x):
        K = self.positional_encoder(self.K(x))
        Q = self.positional_encoder(self.Q(x))
        V = self.V(x)