import torch
import torch.nn as nn
from .base import Embedding

class FourierChannelAttention(torch.nn.Module):
    def __init__(self, c, latent_dim, out_channels, bands = 11, norm = False):
        super().__init__()

        self.bands = bands
        self.norm = norm
        self.c = c

        self.alpha = torch.nn.Parameter(torch.full((1, c, 1, 1), 1.0), requires_grad=True)

        self.precomp = torch.nn.Sequential(
            torch.nn.Conv2d(c + 2, c, 3, 1, 1),
            torch.nn.PReLU(c, 0.2),
            torch.nn.Conv2d(c, c, 3, 1, 1),
            torch.nn.PReLU(c, 0.2),
        )

        self.encoder = torch.nn.Sequential(
            torch.nn.AdaptiveMaxPool2d((bands, bands)),
            torch.nn.Conv2d(c, out_channels, 1, 1, 0),
            torch.nn.PReLU(out_channels, 0.2),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(bands * bands * out_channels, latent_dim),
            torch.nn.PReLU(latent_dim, 0.2)
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, bands * bands * c),
            torch.nn.Sigmoid(),
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, c),
            torch.nn.Sigmoid(),
        )

    def normalize_fft_magnitude(self, mag, sh, sw, target_size=(64, 64)):
        """
        mag: [B, C, sh, sw]
        Returns: [B, C, Fy, Fx]
        """
        B, C, _, _ = mag.shape
        Fy, Fx = target_size

        mag_reshaped = mag.view(B * C, 1, sh, sw)
        norm_mag = torch.nn.functional.interpolate(
            mag_reshaped, size=(Fy, Fx), mode='bilinear', align_corners=False
        )
        norm_mag = norm_mag.view(B, C, Fy, Fx)
        return norm_mag

    def denormalize_fft_magnitude(self, norm_mag, sh, sw):
        """
        norm_mag: [B, C, Fy, Fx]
        Returns: [B, C, sh, sw]
        """
        B, C, Fy, Fx = norm_mag.shape

        norm_mag = norm_mag.view(B * C, 1, Fy, Fx)
        mag = torch.nn.functional.interpolate(
            norm_mag, size=(sh, sw), mode='bilinear', align_corners=False
        )
        mag = mag.view(B, C, sh, sw)
        return mag
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')  # [B, C, H, W//2 + 1]
        _, _, sh, sw = x_fft.shape

        mag = x_fft.abs()
        phase = x_fft.angle()

        if self.norm:
            mag_n = self.normalize_fft_magnitude(mag, sh, sw, target_size=(64, 64))
        else:
            mag_n = torch.nn.functional.interpolate(
                mag, 
                size=(64, 64), 
                mode="bilinear",
                align_corners=False, 
                )

        mag_n = torch.log1p(mag_n) + self.alpha * mag_n
        grid_x = torch.linspace(0, 1, 64, device=x.device).view(1, 1, 1, 64).expand(B, 1, 64, 64)
        grid_y = torch.linspace(0, 1, 64, device=x.device).view(1, 1, 64, 1).expand(B, 1, 64, 64)
        mag_n = self.precomp(torch.cat([mag_n, grid_x, grid_y], dim=1))

        latent = self.encoder(mag_n)

        spat_at = self.fc1(latent).view(-1, self.c, self.bands, self.bands)
        spat_at = spat_at / 0.4 + 0.5
        if self.norm:
            spat_at = self.denormalize_fft_magnitude(spat_at, sh, sw)
        else:
            spat_at = torch.nn.functional.interpolate(
                spat_at,
                size=(sh, sw),
                mode="bilinear",
                align_corners=False,
                )

        mag = mag * spat_at.clamp(min=1e-6)
        x_fft = torch.polar(mag, phase)
        x = torch.fft.irfft2(x_fft, s=(H, W), norm='ortho')

        chan_scale = self.fc2(latent).view(-1, self.c, 1, 1) + 0.1
        x = x * chan_scale.clamp(min=1e-6) 
        return x

class ResConv(torch.nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'zeros', bias=True)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
        self.relu = torch.nn.PReLU(c, 0.2)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)

class ResConvAtt(torch.nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = torch.nn.PReLU(c, 0.2)
        self.attn = FourierChannelAttention(c, c, 11)

    def forward(self, x):
        return self.relu(self.conv(self.attn(x)) * self.beta + x)

class RowScaleShift(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.scale = nn.Parameter(torch.ones(1, 1, 4))  # (1, 1, h)
        init_scale = torch.tensor([0.4, 0.0035, 0.009, 2]).view(1, 1, 4)
        self.scale = torch.nn.Parameter(init_scale.clone())
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
        
        '''
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
        '''
        self.projection_dim = embedding_dim
        super().__init__(input_dim, embedding_dim)
    
    def initialize_layers(self):
        # Linear projection for additional dimensions
        # self.layers = nn.Linear(self.input_dim, self.projection_dim)
        self.norm = RowScaleShift()

        c = 48
        self.layers = nn.Linear(self.input_dim, self.projection_dim)
        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(4, c - 4, 3, 1, 1),
            torch.nn.PReLU(),
        )
        self.convblock0 = torch.nn.Sequential(
            ResConvAtt(c),
            ResConvAtt(c),
        )
        self.convblock1 = torch.nn.Sequential(
            ResConv(c + 8),
            ResConv(c + 8),
            ResConv(c + 8),
            ResConv(c + 8),
        )

        '''
        self.expand = nn.ConvTranspose2d(
            in_channels=c,
            out_channels=c // 2,
            kernel_size=(1, 8),   # only along width
            stride=(1, 8),        # expand width by 8×
            padding=(0, 0),
            bias=True
        )
        '''

        self.lastconv = torch.nn.Conv2d(c + 8, 1, 1, 1, 0)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Tensor of shape (..., embedding_dim)
        """

        values = self.norm(x[..., :4])
        time_embeddings = x[..., -8:]

        # projected = self.layers(values)
        # projected = projected.unsqueeze(1)
        projected = values.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, 16)
        # projected = values.repeat(1, 1, 64)

        feat = self.conv0(projected)
        feat = torch.cat((projected, feat), 1)
        feat = self.convblock0(feat)

        # feat = self.expand(feat)
        time_embeddings = time_embeddings.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, 16)
        feat = torch.cat((feat, time_embeddings), 1)
        feat = self.convblock1(feat)
        feat = self.lastconv(feat).squeeze(1)
        feat = feat.repeat(1, 1, 16)

        return feat
        
        '''
        if self.keep_input:
            # Project to additional dimensions
            projected = self.layers(values)

            # Concatenate original input with projection
            return torch.cat([values, projected, time_embeddings], dim=-1)
        else:
            # Standard linear projection
            return self.layers(x)
        '''
