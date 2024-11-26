import torch.nn as nn
import torch
import torch.nn.functional as F


# Taken from https://github.com/dome272/Diffusion-Models-pytorch
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class ConvBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),  # Convolution
            nn.GroupNorm(32, out_channels),  # Group normalization
            nn.SiLU(),  # SiLU activation
            nn.Dropout(p=0.1),
        )
        
        # Optional 1x1 convolution for residual connection if dimensions mismatch
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x): 
        return self.conv(x) + self.residual_conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(p=0.1),
        )
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )
        self.activation = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, out_channels)  # Project time embedding

    def forward(self, x, temb):
        x_conv = self.conv(x)  # Standard conv operation
        # Project time embedding to spatial dimensions
        temb_proj = self.time_proj(temb)
        temb_proj = temb_proj[:, :, None, None].repeat(1, 1, x_conv.shape[-2], x_conv.shape[-1])
        x_output = x_conv + temb_proj
        residual = self.residual_conv(x)
        return self.activation(x_output + residual)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.res_block_1 = ResBlock(in_channels, out_channels, time_dim)
        self.downSample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.res_block_2 = ResBlock(out_channels, out_channels, time_dim)

    def forward(self, x, temb):
        x = self.res_block_1(x, temb)
        x_down = self.downSample(x)
        x_down = self.res_block_2(x_down, temb)
        return x_down


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.upSample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.res_block_1 = ResBlock(in_channels * 2, in_channels, time_dim)
        self.res_block_2 = ResBlock(in_channels, out_channels, time_dim)

    def forward(self, x, x_skip, temb):
        x_up = self.upSample(x)
        x_cat = torch.cat([x_skip, x_up], dim=1)
        x_up = self.res_block_1(x_cat, temb)
        x_up = self.res_block_2(x_up, temb)
        return x_up


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.dense1 = nn.Linear(embedding_dim, projection_dim)
        self.dense2 = nn.Linear(projection_dim, projection_dim)
        self.activation = nn.SiLU()

    def forward(self, t):
        temb = self.dense1(t)
        temb = self.activation(temb)
        temb = self.dense2(temb)
        return temb