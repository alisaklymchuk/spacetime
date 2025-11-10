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

class FourierChannelAttention1d(torch.nn.Module):
    def __init__(self, c, latent_dim, out_channels, bands=11, norm=False):
        super().__init__()
        self.bands = bands
        self.norm = norm
        self.c = c
        self.alpha = torch.nn.Parameter(torch.full((1, c, 1), 1.0), requires_grad=True)
        
        # 1D convolutions instead of 2D
        self.precomp = torch.nn.Sequential(
            torch.nn.Conv1d(c + 1, c, 3, 1, 1),  # +1 for grid instead of +2
            torch.nn.PReLU(c, 0.2),
            torch.nn.Conv1d(c, c, 3, 1, 1),
            torch.nn.PReLU(c, 0.2),
        )
        
        self.encoder = torch.nn.Sequential(
            torch.nn.AdaptiveMaxPool1d(bands),  # 1D pooling
            torch.nn.Conv1d(c, out_channels, 1, 1, 0),
            torch.nn.PReLU(out_channels, 0.2),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(bands * out_channels, latent_dim),  # bands instead of bands*bands
            torch.nn.PReLU(latent_dim, 0.2)
        )
        
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, bands * c),  # bands instead of bands*bands
            torch.nn.Sigmoid(),
        )
        
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, c),
            torch.nn.Sigmoid(),
        )
    
    def normalize_fft_magnitude(self, mag, st, target_size=64):
        """
        mag: [B, C, st]
        Returns: [B, C, target_size]
        """
        B, C, _ = mag.shape
        mag_reshaped = mag.view(B * C, 1, st)
        norm_mag = torch.nn.functional.interpolate(
            mag_reshaped, size=target_size, mode='linear', align_corners=False
        )
        norm_mag = norm_mag.view(B, C, target_size)
        return norm_mag
    
    def denormalize_fft_magnitude(self, norm_mag, st):
        """
        norm_mag: [B, C, target_size]
        Returns: [B, C, st]
        """
        B, C, target_size = norm_mag.shape
        norm_mag = norm_mag.view(B * C, 1, target_size)
        mag = torch.nn.functional.interpolate(
            norm_mag, size=st, mode='linear', align_corners=False
        )
        mag = mag.view(B, C, st)
        return mag
    
    def forward(self, x):
        B, C, T = x.shape
        
        # 1D FFT
        x_fft = torch.fft.rfft(x, norm='ortho')  # [B, C, T//2 + 1]
        _, _, st = x_fft.shape
        
        mag = x_fft.abs()
        phase = x_fft.angle()
        
        if self.norm:
            mag_n = self.normalize_fft_magnitude(mag, st, target_size=64)
        else:
            mag_n = torch.nn.functional.interpolate(
                mag,
                size=64,
                mode="linear",
                align_corners=False,
            )
        
        mag_n = torch.log1p(mag_n) + self.alpha * mag_n
        
        # 1D grid instead of 2D
        grid = torch.linspace(0, 1, 64, device=x.device).view(1, 1, 64).expand(B, 1, 64)
        
        # Concatenate with grid
        mag_n = self.precomp(torch.cat([mag_n, grid], dim=1))
        
        # Encode
        latent = self.encoder(mag_n)
        
        # Spatial attention (now 1D)
        spat_at = self.fc1(latent).view(-1, self.c, self.bands)
        spat_at = spat_at / 0.4 + 0.5
        
        if self.norm:
            spat_at = self.denormalize_fft_magnitude(spat_at, st)
        else:
            spat_at = torch.nn.functional.interpolate(
                spat_at,
                size=st,
                mode="linear",
                align_corners=False,
            )
        
        # Apply spatial attention
        mag = mag * spat_at.clamp(min=1e-6)
        
        # Reconstruct FFT
        x_fft = torch.polar(mag, phase)
        
        # Inverse FFT
        x = torch.fft.irfft(x_fft, n=T, norm='ortho')
        
        # Channel scale
        chan_scale = self.fc2(latent).view(-1, self.c, 1) + 0.1
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
        init_scale = torch.tensor([
            0.4,        # plot0
            0.0035,     # plot1
            0.009,      # plot2
            2,          # plot3
            0.01,       # plot4
            0.04,       # plot5
            0.00028,    # plot6
            0.005,      # plot7
            0.00125,    # plot8
            0.004,      # plot9
            2.5,        # plot10
            0.05        # plot11
            ]).view(1, 1, 12)
        self.scale = torch.nn.Parameter(init_scale.clone())
        self.shift = nn.Parameter(torch.zeros(1, 1, 12))  # (1, 1, h)

    def forward(self, x):
        return x * self.scale + self.shift

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=4, time_dim=8, hidden_dim=256, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x, time_enc):
        # x: (n, l, 4), time_enc: (n, l, 8)
        combined = torch.cat([x, time_enc], dim=-1)  # (n, l, 12)
        return self.encoder(combined)  # (n, l, 256)

class AttentionEncoder(nn.Module):
    def __init__(self, input_dim=12, time_dim=8, hidden_dim=128, 
                 output_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        
        # Project inputs
        self.input_proj = nn.Linear(input_dim + time_dim, hidden_dim)
        # self.time_proj = nn.Linear(time_dim, hidden_dim)
        
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 8,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers,
            enable_nested_tensor=False,
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x, time_enc):
        # x: (n, l, 12), time_enc: (n, l, 8)

        inp = torch.cat((x, time_enc), -1)

        combined = self.input_encoder(inp) + self.input_proj(inp)

        # Apply transformer
        encoded = self.transformer(combined)  # (n, l, 256)
        
        # Output projection
        output = self.output_proj(encoded)  # (n, l, 256)
        return self.norm(output)

class TimeSeriesEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        time_dim: int = 8,
        hidden_dim: int = 384,
        num_layers: int = 12,
        num_heads: int = 16,
        ffn_mult: int = 8,
        max_len: int = 4096,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        output_dim: int = 64,
        use_film: bool = True,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        token_dim = input_dim + time_dim

        # Pre-encoder (pre-norm + residual MLP)
        self.in_norm = nn.LayerNorm(token_dim)
        self.in_proj = nn.Linear(token_dim, hidden_dim)
        self.in_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_drop = nn.Dropout(dropout)

        # Optional FiLM from time_enc
        self.use_film = use_film
        if use_film:
            self.film = nn.Sequential(
                nn.LayerNorm(time_dim),
                nn.Linear(time_dim, 2 * hidden_dim)
            )

        # Positional embeddings
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        # Transformer encoder (pre-LN) with large FFN and attention dropout
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        # set attention dropout on the underlying MHA
        for m in enc_layer.modules():
            if isinstance(m, nn.MultiheadAttention):
                m.dropout = attn_dropout

        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
            )

        # Output projection
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,        # (N, L, input_dim)
        time_enc: torch.Tensor  # (N, L, time_dim)
    ) -> torch.Tensor:
        N, L, _ = x.shape
        tok = torch.cat([x, time_enc], dim=-1)               # (N, L, input+time)

        # Token mixer with residual
        h = self.in_mlp(self.in_norm(tok)) + self.in_proj(tok)
        h = self.in_drop(h)

        # FiLM (optional)
        if self.use_film:
            gamma_beta = self.film(time_enc)                 # (N, L, 2H)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            h = h * (1 + gamma) + beta

        # Positional embeddings
        pos_ids = torch.arange(L, device=h.device).unsqueeze(0)  # (1, L)
        h = h + self.pos_emb(pos_ids)

        # Encode (no masks at all)
        enc = self.encoder(h)                                   # (N, L, H)

        # Project per-timestep to output_dim
        seq_out = self.out_proj(self.out_norm(enc))             # (N, L, output_dim)
        return seq_out

class TemporalConv1d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size, padding='same')
    
    def forward(self, x):
        # x: (batch, time, features)
        x = x.transpose(1, 2)      # -> (batch, features, time)
        x = self.conv(x)
        x = x.transpose(1, 2)      # -> (batch, time, features)
        return x

'''
class ResConv1d(torch.nn.Module):
    def __init__(self, c, dropout=0.1):
        super().__init__()
        self.conv = torch.nn.Conv1d(c, c, 3, 1, 1, padding_mode = 'reflect', bias=True)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1)), requires_grad=True)
        self.norm = torch.nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(
            embed_dim=c,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.PReLU(c, 0.2)

    def forward(self, x):
        # x: (batch, time, features)
        resudial = x.transpose(1, 2)
        # x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2)
        x = x * self.beta
        x = self.relu(x + resudial)
        x = x.transpose(1, 2)   # -> (batch, time, features)
        return x
'''

class ResConv1d(torch.nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = torch.nn.Conv1d(c, c, 3, 1, 1, padding_mode = 'zeros', bias=True)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1)), requires_grad=True)        
        self.relu = torch.nn.PReLU(c, 0.2)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)

class ResConvAtt1d(torch.nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = torch.nn.Conv1d(c, c, 3, 1, 1, padding_mode = 'zeros', bias=True)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1)), requires_grad=True)        
        self.relu = torch.nn.PReLU(c, 0.2)
        self.attn = FourierChannelAttention1d(c, c, 11)

    def forward(self, x):
        x = self.attn(x)
        return self.relu(self.conv(x) * self.beta + x)

class UpMix1d(nn.Module):
    def __init__(self, c, cd):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(cd, c, 4, 2, 1)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1)), requires_grad=True)
        self.relu = torch.nn.PReLU(c, 0.2)

    def forward(self, x, x_deep):
        return self.relu(self.conv(x_deep) * self.beta + x)

class DownMix1d(nn.Module):
    def __init__(self, c, cd):
        super().__init__()
        self.conv = torch.nn.Conv1d(c, cd, 3, 2, 1, padding_mode = 'reflect', bias=True)
        self.beta = torch.nn.Parameter(torch.ones((1, cd, 1)), requires_grad=True)
        self.relu = torch.nn.PReLU(cd, 0.2)

    def forward(self, x, x_deep):
        return self.relu(self.conv(x) * self.beta + x_deep)

class Mix1d(nn.Module):
    def __init__(self, c, cd):
        super().__init__()
        self.conv0 = torch.nn.ConvTranspose1d(cd, c, 4, 2, 1)
        self.conv1 = torch.nn.Conv1d(c, c, 3, 1, 1)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1)), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones((1, c, 1)), requires_grad=True)
        self.relu = torch.nn.PReLU(c, 0.2)

    def forward(self, x, x_deep):
        return self.relu(self.conv0(x_deep) * self.beta + self.conv1(x) * self.gamma)

class DoubleConv(nn.Module):
    def __init__(self, input_dim=12, time_dim=8, hidden_dim=1152, 
                 output_dim=512, kernel_size = 3, dropout=0.2):
        super().__init__()
        self.encode = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim + time_dim, hidden_dim, 11, 2, 5),
            torch.nn.PReLU(hidden_dim, 0.2),
            torch.nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1),
            torch.nn.PReLU(hidden_dim, 0.2),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1),
            torch.nn.PReLU(hidden_dim, 0.2),
            torch.nn.ConvTranspose1d(hidden_dim, output_dim, 4, 2, 1)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=16,
            dropout=dropout,
            batch_first=True
        )


    def forward(self, x, time_enc):
        x = torch.cat((x, time_enc), -1)
        x = x.transpose(1, 2)
        x = self.encode(x)
        x = x.transpose(1, 2)
        # x, _ = self.attn(x, x, x)
        return x

class FlownetDeep(nn.Module):
    def __init__(self, input_dim=2, time_dim=8, hidden_dim=384, output_dim=64):
        super().__init__()
        c = hidden_dim
        cd = 1 * round(1.618 * c) + 2 - (1 * round(1.618 * c) % 2)

        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim + time_dim, c//2, 5, 2, 2, padding_mode = 'zeros'),
            torch.nn.PReLU(c//2),
            )                
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(c//2, c, 5, 2, 2, padding_mode = 'reflect'),
            torch.nn.PReLU(c, 0.2),
            )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(c, cd, 3, 2, 1, padding_mode = 'reflect'),
            torch.nn.PReLU(cd, 0.2),     
        )

        self.convblock1f = torch.nn.Sequential(
            ResConvAtt1d(c//2),
            ResConv1d(c//2),
            ResConv1d(c//2),
            ResConv1d(c//2),
        )

        self.convblock1 = torch.nn.Sequential(
            ResConvAtt1d(c),
            ResConv1d(c),
            ResConv1d(c),
            ResConv1d(c),
        )

        self.convblock_deep1 = torch.nn.Sequential(
            ResConvAtt1d(cd),
            ResConv1d(cd),
            ResConv1d(cd),
            ResConv1d(cd),
        )

        self.convblock2f = torch.nn.Sequential(
            ResConvAtt1d(c//2),
            ResConv1d(c//2),
            ResConv1d(c//2),
        )

        self.convblock2 = torch.nn.Sequential(
            ResConvAtt1d(c),
            ResConv1d(c),
            ResConv1d(c),
        )

        self.convblock_deep2 = torch.nn.Sequential(
            ResConvAtt1d(cd),
            ResConv1d(cd),
            ResConv1d(cd),
        )

        self.convblock3f = torch.nn.Sequential(
            ResConv1d(c//2),
            ResConv1d(c//2),
        )

        self.convblock3 = torch.nn.Sequential(
            ResConv1d(c),
            ResConv1d(c),
        )

        self.convblock_deep3 = torch.nn.Sequential(
            ResConv1d(cd),
            ResConv1d(cd),
        )

        self.convblock_last_shallow = torch.nn.Sequential(
            ResConv1d(c//2),
            ResConv1d(c//2),
            ResConv1d(c//2),
            ResConv1d(c//2),
        )

        self.convblock_last = torch.nn.Sequential(
            ResConv1d(c),
            ResConv1d(c),
            ResConv1d(c),
            ResConv1d(c),
        )



        self.mix1 = UpMix1d(c, cd)
        self.mix1f = DownMix1d(c//2, c)
        self.revmix1 = DownMix1d(c, cd)
        self.revmix1f = UpMix1d(c//2, c)

        self.mix2 = UpMix1d(c, cd)
        self.mix2f = DownMix1d(c//2, c)
        self.revmix2 = DownMix1d(c, cd)
        self.revmix2f = UpMix1d(c//2, c)

        self.mix3 = Mix1d(c, cd)
        self.mix3f = DownMix1d(c//2, c)
        self.revmix3f = UpMix1d(c//2, c)

        self.mix4 = Mix1d(c//2, c)

        self.lastconv = torch.nn.ConvTranspose1d(c//2, output_dim, 4, 2, 1)

    def forward(self, x, time_enc):
        x = torch.cat((x, time_enc), -1)
        x = x.transpose(1, 2)

        feat = self.conv0(x)
        featF = self.convblock1f(feat)
        feat = self.conv1(feat)
        feat_deep = self.conv2(feat)

        feat = self.convblock1(feat)
        feat_deep = self.convblock_deep1(feat_deep)

        feat = self.mix1f(featF, feat)
        feat_tmp = self.mix1(feat, feat_deep)
        feat_deep = self.revmix1(feat, feat_deep)
        featF = self.revmix1f(featF, feat_tmp)

        featF = self.convblock2f(featF)
        feat = self.convblock2(feat_tmp)
        feat_deep = self.convblock_deep2(feat_deep)

        feat = self.mix2f(featF, feat)
        feat_tmp = self.mix2(feat, feat_deep)
        feat_deep = self.revmix2(feat, feat_deep)
        featF = self.revmix2f(featF, feat_tmp)

        featF = self.convblock3f(featF)
        feat = self.convblock3(feat_tmp)
        feat_deep = self.convblock_deep3(feat_deep)

        feat = self.mix3f(featF, feat)
        feat = self.mix3(feat, feat_deep)
        featF = self.revmix3f(featF, feat)

        feat = self.convblock_last(feat)
        featF = self.convblock_last_shallow(featF)

        feat = self.mix4(featF, feat)

        out = self.lastconv(feat)

        return out.transpose(1, 2)

class CrossModalEncoder(nn.Module):
    def __init__(self, input_dim=12, time_dim=8, hidden_dim=128, 
                 output_dim=512, num_heads=16, dropout=0.1):
        super().__init__()
        
        # Separate pathways for each modality
        '''
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        '''

        self.input_encoder = nn.Sequential(
            TemporalConv1d(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.input_convblock = nn.Sequential(
            ResConv1d(hidden_dim),
            ResConv1d(hidden_dim),
            ResConv1d(hidden_dim),
            ResConv1d(hidden_dim),
        )

        self.output_convblock = nn.Sequential(
            ResConv1d(output_dim),
            ResConv1d(output_dim),
            ResConv1d(output_dim),
            ResConv1d(output_dim),
        )

        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Cross-attention: time attends to input
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention for refinement
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x, time_enc):
        # x: (n, l, 4), time_enc: (n, l, 8)
        x_emb = self.input_encoder(x)  # (n, l, 128)
        t_emb = self.time_encoder(time_enc)  # (n, l, 128)
        
        x_emb = self.input_convblock(x_emb)

        # Cross-attention
        t_attended, _ = self.cross_attn(t_emb, x_emb, x_emb)  # (n, l, 128)
        
        # Concatenate
        combined = torch.cat([x_emb, t_attended], dim=-1)  # (n, l, 256)
        
        # Self-attention
        refined, _ = self.self_attn(combined, combined, combined)  # (n, l, 256)
        
        # FFN with residual
        output = self.ffn(refined + combined)  # (n, l, 256)
        output = self.output_convblock(output)

        return output

class SingleCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.repeat(1, 1, 32)

class FourCopyTimeEmb(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, values, time_emb):
        values = values.repeat(1, 1, 120)
        return torch.cat((values, time_emb), -1)

class TimewarpStyle(torch.nn.Module):
    def __init__(self, c=192, exp=16, output_dim=128):
        super().__init__()

        self.exp = exp
        self.c = c
        self.output_dim = output_dim

        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(4, c - 4, 3, 1, 1),
            torch.nn.Mish(),
            torch.nn.Conv2d(c - 4, c - 4, 3, 1, 1),
            torch.nn.Mish(),
        )
        self.convblock0 = torch.nn.Sequential(
            ResConvAtt(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.convblock1 = torch.nn.Sequential(
            ResConvAtt(c + 8),
            ResConv(c + 8),
            ResConv(c + 8),
            ResConv(c + 8),
        )

        self.dimconv = torch.nn.Conv2d(exp, 1, 3, 1, 1)
        self.lastconv = torch.nn.Conv2d(c + 8, output_dim, 1, 1, 0)
        self.norm = torch.nn.LayerNorm(output_dim),

    def forward(self, x, time_embeddings):
        x = x.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, self.exp)

        feat = self.conv0(x)
        feat = torch.cat((x, feat), 1)

        feat = self.convblock0(feat)

        time_embeddings = time_embeddings.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, self.exp)
        feat = torch.cat((feat, time_embeddings), 1)

        feat = self.convblock1(feat)

        feat = feat.permute(0, 3, 2, 1)
        feat = self.dimconv(feat)

        feat = feat.permute(0, 3, 2, 1)

        feat = self.lastconv(feat)

        feat = feat.permute(0, 3, 2, 1)

        feat = feat.squeeze(1)

        feat = self.norm(feat)

        return feat

class ModifiedLinearEmbedding(Embedding):
    def __init__(self, input_dim, embedding_dim):
        super().__init__(input_dim, embedding_dim)
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
    
    def initialize_layers(self):
        # Linear projection for additional dimensions
        # self.layers = nn.Linear(self.input_dim, self.projection_dim)
        self.norm = RowScaleShift()
        self.expert01 = TimeSeriesEncoder(input_dim=self.input_dim, output_dim=self.embedding_dim//2)
        self.expert02 = FlownetDeep(input_dim=self.input_dim, output_dim=self.embedding_dim//2)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Tensor of shape (..., embedding_dim)
        """

        # values = self.norm(x[..., :1])
        values = x[..., :self.input_dim]
        time_embeddings = x[..., -8:]

        a = self.expert01(values, time_embeddings)
        b = self.expert02(values, time_embeddings)

        out = torch.stack((a, b), dim=-1).reshape(a.shape[0], a.shape[1], -1)

        return out