import os
import torch
import inspect
from src.ssdm.conf.configuration import parse_args
from diffusers import DDPMPipeline, EDMEulerScheduler, EulerDiscreteScheduler
from src.ssdm.utils.build_model_and_data import build_model_and_data
from tqdm import tqdm
from src.ssdm.utils.evaluations import qualitative_eval
from src.ssdm.utils.tools import check_dir
from safetensors.torch import load_file
from torch.utils.data import DataLoader

# from src.ssdm.models.diffusion_net import EDMPrecond
from src.ssdm.models.diffusion_net import EDMPrecond, EDMPrecond2

import numpy as np


def edm_sampler(
    latents, net, massvec, L, evals, evecs, gradX, gradY,
    class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    # S_churn=40, S_min=0.05, S_max=50, S_noise=1.003,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, massvec, L, evals, evecs, gradX, gradY).to(torch.float64)
        # denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, massvec, L, evals, evecs, gradX, gradY).to(torch.float64)
            # denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def edm_sampler_v2(
    noise, net, massvec, L, evals, evecs, gradX, gradY, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    # Guided denoiser.
    def denoise(x, t):
        # Dx = net(x, t, labels).to(dtype)
        Dx = net(x, t, massvec, L, evals, evecs, gradX, gradY).to(dtype)
        print(Dx.shape, torch.min(Dx), torch.max(Dx))
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, signal, massvec, L, evals, evecs, gradX, gradY):
        images = signal
        rnd_normal = torch.randn([images.shape[0], 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = images, None
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, massvec, L, evals, evecs, gradX, gradY)
        loss = weight * ((D_yn - y) ** 2)
        loss = loss.mean()
        return loss
    
class EDM2Loss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, signal, massvec, L, evals, evecs, gradX, gradY):
        images = signal
        rnd_normal = torch.randn([images.shape[0], 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        denoised, logvar = net(images + noise, sigma, massvec, L, evals, evecs, gradX, gradY, return_logvar=True)
        loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
        loss = loss.mean()
        return loss
    

def main(args):
    dataset, model = build_model_and_data(args)
    model = EDMPrecond(model=model)
    # model = EDMPrecond2(model=model)
    state_dict = load_file(args.infer_ckp_path)
    exp_dir = os.path.dirname(args.infer_ckp_path)
    print(exp_dir)
    model.load_state_dict(state_dict, strict=True)

    training_dataloader = DataLoader(dataset,
                          batch_size=args.train.train_batch_size,
                          num_workers=args.train.train_num_workers,
                          shuffle=True)
    
    for batch in training_dataloader:
        signal, image = batch
        signal = signal.cuda()
        image = image.cuda()
        frame, massvec, L, evals, evecs, gradX, gradY = \
        training_dataloader.dataset.frames, training_dataloader.dataset.massvec, training_dataloader.dataset.L, \
            training_dataloader.dataset.evals, training_dataloader.dataset.evecs, training_dataloader.dataset.gradX, training_dataloader.dataset.gradY
        frame = torch.stack([frame for _ in range(args.train.train_batch_size)], dim=0).cuda()
        massvec = torch.stack([massvec for _ in range(args.train.train_batch_size)], dim=0).cuda()
        L = torch.stack([L for _ in range(args.train.train_batch_size)], dim=0).cuda()
        evals = torch.stack([evals for _ in range(args.train.train_batch_size)], dim=0).cuda()
        evecs = torch.stack([evecs for _ in range(args.train.train_batch_size)], dim=0).cuda()
        gradX = torch.stack([gradX for _ in range(args.train.train_batch_size)], dim=0).cuda()
        gradY = torch.stack([gradY for _ in range(args.train.train_batch_size)], dim=0).cuda()
        break
    
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(total=args.exp.num_sample)
        batch_seeds = torch.arange(signal.shape[0])
        rnd = StackedRandomGenerator(signal.device, batch_seeds)
        if args.infer_show:
            vis_output_dir = os.path.join(exp_dir, "visualize",)
            check_dir(vis_output_dir)
        if args.infer_save:
            save_output_dir = os.path.join(exp_dir, "save_signals",)
            check_dir(save_output_dir)    
            
        for i in range(args.exp.num_sample):
            # noise_pred = model(latent_model_input, time_step, massvec, L, evals, evecs, gradX, gradY)
            
            signal_t = rnd.randn(signal.shape, device=signal.device)
            signal_t = edm_sampler(signal_t, model, massvec, L, evals, evecs, gradX, gradY, randn_like=rnd.randn_like, num_steps=args.diffusion.sampling_steps)
            # signal_t = edm_sampler_v2(signal_t, model, massvec, L, evals, evecs, gradX, gradY, randn_like=rnd.randn_like)

            sampled_signal = signal_t.cpu()
            if args.infer_show:
                qualitative_eval(sampled_signal, dataset.mesh ,output_dir=vis_output_dir, wandb_log=args.exp.wandb, sample_id=i)
            if args.infer_save:
                for j in range(signal.shape[0]):
                    torch.save(sampled_signal, os.path.join(save_output_dir, "signal_{}_{}.pt".format(i, j)))

            progress_bar.update(1)

if __name__ == '__main__':
    args = parse_args()
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    main(args)