# Inspired from https://github.com/mikonvergence/DiffusionFastForward/blob/master/src/DenoisingDiffusionProcess/backbones/unet_convnext.py
# Inspired from https://github.com/mikonvergence/DiffusionFastForward/blob/master/src/DenoisingDiffusionProcess/forward.py
from torch import Tensor
import torch.nn as nn
import typing as t
import einops
import torch
import math

from .samplers import *
from diffusers.configuration_utils import ConfigMixin
from tqdm import tqdm

class SinusoidalPosEmb(nn.Module):
    """
        Based on transformer-like embedding from 'Attention is all you need'
        Note: 10,000 corresponds to the maximum sequence length
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionModel(nn.Module, ConfigMixin):
    def __init__(
        self,
        dim_mlp: int,
        dim_time: int,
        num_points: int,
        dim_signal: int,
        sampler: DDPM_Sampler,
        n_heads: int=4,
        k_evecs: int=10,
        schedule='linear',
        dropout: float=0.1,
        num_timesteps=1000,
        num_encoder_layers: int=4,
    ) -> None:
        super(DiffusionModel, self).__init__()
        self.features: int = k_evecs + dim_signal + dim_time
        self.num_timesteps: int = num_timesteps
        self.num_points: int = num_points
        self.dim_signal: int = dim_signal
        self.schedule: str = schedule
        self.sampler = sampler

        self.register_buffer('betas', get_beta_schedule(self.schedule,self.num_timesteps))
        self.register_buffer('betas_sqrt',self.betas.sqrt())
        self.register_buffer('alphas',1-self.betas)
        self.register_buffer('alphas_cumprod',torch.cumprod(self.alphas,0))
        self.register_buffer('alphas_cumprod_sqrt',self.alphas_cumprod.sqrt())
        self.register_buffer('alphas_one_minus_cumprod_sqrt',(1-self.alphas_cumprod).sqrt())
        self.register_buffer('alphas_sqrt',self.alphas.sqrt())

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim_time),
            nn.Linear(dim_time, dim_time * 4),
            nn.GELU(),
            nn.Linear(dim_time * 4, dim_time)
        )

        c_encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.features,
            nhead=n_heads,
            dim_feedforward=dim_mlp,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.c_encoder = nn.TransformerEncoder(
            encoder_layer=c_encoder_layers,
            num_layers=num_encoder_layers,
        )

        q_encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.features,
            nhead=n_heads,
            dim_feedforward=dim_mlp,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.q_encoder = nn.TransformerEncoder(
            encoder_layer=q_encoder_layers,
            num_layers=num_encoder_layers,
        )

        self.out_mlp = nn.Linear(self.features * 2, dim_signal)

    @torch.no_grad()
    def decode(self, q_emb: Tensor, subset: int) -> Tensor:
        # Unpack dimensions for visibility
        B, N, E = q_emb.size()
        assert 0 < subset <= N, 'subset of q_sig must be in [0, N]'

        # Use a custom sampler on device
        self.sampler = self.sampler.to(q_emb.device)

        # Randomly sample a noise array representing the query signal at time step T
        q_sig_t: Tensor = torch.randn((B, N, self.dim_signal), device=q_emb.device)

        # Select a random subset from query for the context
        c_subset: Tensor = torch.randint(low=0, high=N, size=(B, subset), device=q_emb.device)
        c_sig_t: Tensor = einops.repeat(c_subset, 'B S -> B S E', E=self.dim_signal)
        c_sig_t: Tensor = torch.gather(q_sig_t, dim=1, index=c_sig_t)
        c_emb: Tensor = einops.repeat(c_subset, 'B S -> B S E', E=E)
        c_emb: Tensor = torch.gather(q_emb, dim=1, index=c_emb)

        # Aggregate information for each set
        q_t: Tensor = torch.cat([q_emb, q_sig_t], dim=-1)
        c_t: Tensor = torch.cat([c_emb, c_sig_t], dim=-1)

        for timestep in tqdm(reversed(range(self.num_timesteps))):
            # Use the same timestep across all samples in the batch
            t: Tensor = torch.full((q_emb.size(0),), timestep, device=q_emb.device)

            # Predict the signal noise for the query signal
            q_sig_z = self.noise(c_t, q_t, t)
            q_sig_t = self.sampler.step(q_sig_t, q_sig_z, t)

            # Select a random subset from query for the context
            c_subset: Tensor = torch.randint(low=0, high=N, size=(B, subset), device=q_emb.device)
            c_sig_t: Tensor = einops.repeat(c_subset, 'B S -> B S E', E=self.dim_signal)
            c_sig_t: Tensor = torch.gather(q_sig_t, dim=1, index=c_sig_t)
            c_emb: Tensor = einops.repeat(c_subset, 'B S -> B S E', E=E)
            c_emb: Tensor = torch.gather(q_emb, dim=1, index=c_emb)

            # Aggregate information for each set
            q_t: Tensor = torch.cat([q_emb, q_sig_t], dim=-1)
            c_t: Tensor = torch.cat([c_emb, c_sig_t], dim=-1)
            # print(f'Timestep: {timestep}')

        return q_sig_t

    def noise(self, c_t: Tensor, q_t: Tensor, t: Tensor) -> Tensor:
        # Use cosine positional time embeddings
        t = self.time_mlp(t)

        # Concatenate time embedding to context and query
        t_c: Tensor = einops.repeat(t, 'B T -> B V T', V=c_t.size(1))
        c_t = torch.cat([c_t, t_c], dim=-1)
        t_q: Tensor = einops.repeat(t, 'B T -> B V T', V=q_t.size(1))
        q_t = torch.cat([q_t, t_q], dim=-1)

        # Apply multiple transformer-encoder layers
        c_t = self.c_encoder(c_t)
        q_t = self.q_encoder(q_t)
        c_t = c_t if c_t.size(1) == q_t.size(1) else c_t.repeat((1, 2, 1))

        # Concatenate context & query for final layer
        x: Tensor = torch.cat([c_t, q_t], dim=-1)

        # Predict the nosie for the query signal
        return self.out_mlp(x)

    @torch.no_grad()
    def encode(self, c_0_emb: Tensor, c_0_sig: Tensor, q_0_emb: Tensor, q_0_sig, t: Tensor) -> t.Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
            Get noisy sample at t given x_0

            q(x_t | x_0)=N(x_t; alphas_cumprod_sqrt(t)*x_0, 1-alpha_cumprod(t)*I)
        """
        assert (t < self.num_timesteps).all()

        # Add noise to the signal components
        c_t_sig, c_sig_z = self.encode_signal(c_0_sig, t)
        q_t_sig, q_sig_z = self.encode_signal(q_0_sig, t)

        # Concatenate positional embedding with signal
        c_t: Tensor = torch.cat([c_0_emb, c_t_sig], dim=-1)
        q_t: Tensor = torch.cat([q_0_emb, q_t_sig], dim=-1)
        return c_t, c_sig_z, q_t, q_sig_z

    @torch.no_grad()
    def encode_signal(self, sig_0: Tensor, t: Tensor) -> t.Tuple[Tensor, Tensor]:
        """
            Get noisy sample at t given x_0

            q(x_t | x_0)=N(x_t; alphas_cumprod_sqrt(t)*x_0, 1-alpha_cumprod(t)*I)
        """
        assert (t < self.num_timesteps).all()

        mean: Tensor = sig_0 * einops.rearrange(self.alphas_cumprod_sqrt[t],  'B -> B 1 1')
        std: Tensor = einops.rearrange(self.alphas_one_minus_cumprod_sqrt[t], 'B -> B 1 1')

        sig_z: Tensor = torch.randn_like(sig_0)
        sig_t: Tensor = mean + std * sig_z

        return sig_t, sig_z
