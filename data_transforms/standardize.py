import torch

from .affine import AffineTransform


class StandardizeTransform(AffineTransform):
    """
    Standardize lag terms, i.e., z = (x - mean(x)) / std(x)
    - Computed as (1 / std(x)) * x - mean(x) * (1 / std(x)) to fit with inverse call,
      which does (z + (mean(x) / std(x))) * std(x) = z * std(x) + mean(x)
    """
    def __init__(self, lag):
        super().__init__(lag=lag)
        
    def forward(self, x):
        n, l, f = x.shape

        if f > 8:
          self.a = torch.ones(n, 1, f, device=x.device, dtype=x.dtype)
          self.b = torch.zeros(n, 1, f, device=x.device, dtype=x.dtype)

          std_features = x[:, :self.lag, :-8]  # Shape: (n, lag, f-8)

          std_vals = torch.std(std_features, dim=1)  # Shape: (n, f-8)
          mean_vals = torch.mean(std_features, dim=1)  # Shape: (n, f-8)
          
          std_vals = std_vals + 1e-6

          # Set a and b for the standardized features
          self.a[:, :, :-8] = (1. / std_vals)[:, None, :]
          self.b[:, :, :-8] = mean_vals[:, None, :] * self.a[:, :, :-8]

        else:
          self.a = 1. / torch.std(x[:, :self.lag, :], dim=1)[:, None, :]
          self.b = torch.mean(x[:, :self.lag, :], dim=1)[:, None, :] * self.a

        return self.a * x - self.b
