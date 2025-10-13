
import torch
import torch.nn as nn
from .affine import AffineTransform

class MeanPercentageTransform(AffineTransform):
    """
    Convert to percentage change relative to reference point,
    then zero-center the percentage changes
    """
    def __init__(self, lag):
        super().__init__(lag=lag)
        
    def forward(self, x):
        # Use the value at position (lag-1) as reference for percentage change
        # Shape: [batch, 1, features]
        self.reference = x[:, 0:1, :]
        # self.reference = x[:, self.lag - 1:self.lag, :]
        
        # Convert to percentage change: (x - ref) / ref
        pct_change = (x - self.reference) / (self.reference + 1e-8)  # add epsilon to avoid division by zero
        
        # Calculate mean of percentage changes over the first lag timesteps
        self.b = pct_change[:, :self.lag, :].mean(dim=1)[:, None, :]
        self.a = torch.ones(1)
        
        # Zero-center the percentage changes
        return self.a * pct_change - self.b
    
class PercentageTransform(AffineTransform):
    """
    Convert to percentage change relative to reference point
    """
    def __init__(self, lag):
        super().__init__(lag=lag)
        
    def forward(self, x):
        # Use the first element as reference for percentage change
        # Shape: [batch, 1, features]
        self.reference = x[:, 0:1, :]
        
        # Set affine parameters for percentage change
        # (x - ref) / (ref + eps) = (1/(ref+eps)) * x - (ref/(ref+eps))
        self.a = 1.0 / (self.reference + 1e-8)
        self.b = torch.ones(1)
        
        # Apply affine transform
        return self.a * x - self.b