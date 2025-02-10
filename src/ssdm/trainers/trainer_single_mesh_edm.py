import torch
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import os
from src.ssdm.utils.tools import check_dir
from src.ssdm.utils.evaluations import qualitative_eval
import numpy as np

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


def train_loop(args, accelerator, model, optimizer, lr_scheduler, training_dataloader, exp_dir):
    loss_fn = EDMLoss()
    
    global_step = 0
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
    
    
    # import matplotlib.pyplot as plt
    # plt.figure()
    
    for epoch in range(args.train.num_train_epochs):
        progress_bar = tqdm(total=len(training_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(training_dataloader):
            with accelerator.accumulate(model):
                model.train()
                signal, image = batch
                
                loss = loss_fn(model, signal, massvec, L, evals, evecs, gradX, gradY)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.exp.wandb:
                if accelerator.is_main_process:
                    if global_step % 50 == 0:
                        wandb.log({"loss":loss.detach().item(),
                                "lr": lr_scheduler.get_last_lr()[0]})
                        
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            
            # if accelerator.is_main_process:
            #     print(lr_scheduler.get_last_lr()[0])
            #     plt.plot([global_step], [lr_scheduler.get_last_lr()[0]], 'ro')
            
        if accelerator.is_main_process:
            if epoch % args.train.save_interval == 0 and epoch != 0:
                print(f"Save checkpoint for epoch {epoch}")
                accelerator.save_state(os.path.join(exp_dir, "accelerator", f"epoch_{epoch}_steps_{step}"))
        
