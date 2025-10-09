import torch
import torch.nn as nn
from .base import Embedding

class RowScaleShift(nn.Module):
    def __init__(self):
        super().__init__()
        init_scale = torch.tensor([0.4, 0.0035, 0.009, 2]).view(1, 1, 4)
        self.scale = torch.nn.Parameter(init_scale.clone())
        # self.scale = nn.Parameter(torch.ones(1, 1, 4))  # (1, 1, h)
        self.shift = nn.Parameter(torch.zeros(1, 1, 4))  # (1, 1, h)

    def forward(self, x):
        return x * self.scale + self.shift

class ModifiedLinearEmbedding(Embedding):
    def __init__(self, input_dim, embedding_dim, keep_input=True):
        """
        Args:
            input_dim: Input feature dimension
            embedding_dim: Output embedding dimension
            keep_input: If True, concatenate original input with linear projection.
                       embedding_dim must be > input_dim in this case.
        """
        self.keep_input = keep_input
        
        if keep_input:
            if embedding_dim <= input_dim:
                raise ValueError(
                    f"When keep_input=True, embedding_dim ({embedding_dim}) "
                    f"must be greater than input_dim ({input_dim})"
                )
            # Linear layer only needs to produce the additional dimensions
            self.projection_dim = embedding_dim - input_dim - 8
        else:
            self.projection_dim = embedding_dim
        
        super().__init__(input_dim, embedding_dim)
    
    def initialize_layers(self):
        # Linear projection for additional dimensions
        self.layers = nn.Linear(self.input_dim, self.projection_dim)
        self.norm = RowScaleShift()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Tensor of shape (..., embedding_dim)
        """

        values = self.norm(x[..., :4])
        time_embeddings = x[..., -8:]

        if self.keep_input:
            # Project to additional dimensions
            projected = self.layers(values)

            # Concatenate original input with projection
            return torch.cat([values, projected, time_embeddings], dim=-1)
        else:
            # Standard linear projection
            return self.layers(x)
