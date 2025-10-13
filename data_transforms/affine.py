import torch.nn as nn


class AffineTransform(nn.Module):
    def __init__(self, lag=None):
        """
        Transform data: f(x) = ax - b  
        """
        super().__init__()
        self.lag = lag
        
    def forward(self, x):
        # Assume x.shape is B x L x D
        raise NotImplementedError
    
    
class InverseAffineTransform(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform  # AffineTransform object
        
    def forward(self, x):
        return ((x + self.transform.b.to(x.device)) /
                self.transform.a.to(x.device))
    
class InverseMeanPercentageAffineTransform(nn.Module):
    """
    Inverse transform for MeanPercentageTransform
    """
    def __init__(self, transform):
        super().__init__()
        self.transform = transform  # MeanPercentageTransform object
        
    def forward(self, x):
        # First, undo the mean centering
        # x is the zero-centered percentage change
        pct_change = (x + self.transform.b.to(x.device)) / self.transform.a.to(x.device)
        
        # Convert percentage change back to absolute values
        # original = reference * (1 + pct_change)
        reference = self.transform.reference.to(x.device)
        original = reference * (1 + pct_change)
        
        return original

class InversePercentageAffineTransform(nn.Module):
    """
    Inverse transform for MeanPercentageTransform
    """
    def __init__(self, transform):
        super().__init__()
        self.transform = transform  # MeanPercentageTransform object
        
    def forward(self, x):
        # First, undo the mean centering
        # x is the zero-centered percentage change
        pct_change = (x + self.transform.b.to(x.device)) / self.transform.a.to(x.device)
        
        # Convert percentage change back to absolute values
        # original = reference * (1 + pct_change)
        reference = self.transform.reference.to(x.device)
        original = reference * (1 + pct_change)
        
        return original