# Inspired from https://github.com/mikonvergence/DiffusionFastForward/blob/master/src/DenoisingDiffusionProcess/samplers/DDPM.py
# Inspired from https://huggingface.co/blog/annotated-diffusion
from torch import Tensor
import torch.nn as nn
import typing as t
import einops
import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2


def get_beta_schedule(variant, timesteps) -> Tensor:
    if variant=='cosine':
        return cosine_beta_schedule(timesteps)
    elif variant=='linear':
        return linear_beta_schedule(timesteps)
    elif variant=='quadratic':
        return quadratic_beta_schedule(timesteps)
    elif variant=='sigmoid':
        return sigmoid_beta_schedule(timesteps)
    else:
        raise NotImplemented


def cosine_beta_schedule(timesteps, s=0.008) -> Tensor:
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class DDPM_Sampler(nn.Module):
    def __init__(
        self,
        num_timesteps=1000,
        schedule='linear',
    ) -> None:
        super().__init__()

        self.num_timesteps: int = num_timesteps
        self.schedule: str = schedule

        self.register_buffer('betas', get_beta_schedule(self.schedule,self.num_timesteps))
        self.register_buffer('betas_sqrt', self.betas.sqrt())
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas,0))
        self.register_buffer('alphas_cumprod_sqrt', self.alphas_cumprod.sqrt())
        self.register_buffer('alphas_one_minus_cumprod_sqrt', (1 - self.alphas_cumprod).sqrt())
        self.register_buffer('alphas_sqrt', self.alphas.sqrt())
        self.register_buffer('alphas_sqrt_recip', 1 / (self.alphas_sqrt))

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Tensor:
        return self.step(*args, **kwargs)

    @torch.no_grad()
    def step(self, sig_t: Tensor, z_t: Tensor, t: Tensor) -> Tensor:
        """
            Given approximation of noise z_t in x_t predict x_(t-1)
        """
        assert (t < self.num_timesteps).all()

        # 2. Approximate Distribution of Previous Sample in the chain
        mean_pred, std_pred = self.posterior_params(sig_t, t, z_t)

        # 3. Sample from the distribution
        z: Tensor = torch.randn_like(sig_t) if any(t > 0) else torch.zeros_like(sig_t)
        return mean_pred + std_pred * z

    def posterior_params(self, x_t: Tensor, t: Tensor, noise_pred: Tensor) -> t.Tuple[Tensor, Tensor]:
        assert (t < self.num_timesteps).all()

        beta_t: Tensor = einops.rearrange(self.betas[t], 'B -> B 1 1')
        alpha_one_minus_cumprod_sqrt_t = einops.rearrange(self.alphas_one_minus_cumprod_sqrt[t], 'B -> B 1 1')
        alpha_sqrt_recip_t = einops.rearrange(self.alphas_sqrt_recip[t], 'B -> B 1 1')

        mean: Tensor = alpha_sqrt_recip_t * (x_t - beta_t * noise_pred / alpha_one_minus_cumprod_sqrt_t)
        std: Tensor = einops.rearrange(self.betas_sqrt[t], 'B -> B 1 1')

        return mean, std
