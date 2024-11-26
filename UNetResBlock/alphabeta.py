import torch.nn as nn
import torch
import torch.nn.functional as F

def compute_linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Linear schedule for betas from beta_start to beta_end over timesteps.
    """
    betas = torch.linspace(beta_start, beta_end, timesteps)
    return betas

def compute_alpha_schedule(betas):
    """
    Computes alpha values for each timestep using the betas.
    alpha_t = 1 - beta_t
    """
    return 1.0 - betas

def compute_alpha_cumulative_product(alpha_t):
    """
    Computes the cumulative product of alpha_t over all timesteps.
    This gives us the alpha_cumprod needed for the reverse process.
    """
    return torch.cumprod(alpha_t, dim=0)
