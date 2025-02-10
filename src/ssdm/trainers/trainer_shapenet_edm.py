import torch
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import os
from src.ssdm.utils.tools import check_dir
from src.ssdm.utils.evaluations import qualitative_eval
import numpy as np

def print_memory_usage(step=""):
    print(f"{step} - Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"{step} - Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, signal, massvec, L=None, evals=None, evecs=None, gradX=None, gradY=None, mask=False, pad_mask=None):
        # images = signal
        rnd_normal = torch.randn([signal.shape[0], 1, 1], device=signal.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = signal, None
        
        n = torch.randn_like(y) * sigma
        
        D_yn = net(x=(y + n), sigma=sigma, massvec=massvec, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)
        
        if mask:

            loss = weight * ((D_yn - y) ** 2)
            loss = pad_mask * loss
        else:
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
    latents, net, massvec, L=None, evals=None, evecs=None, gradX=None, gradY=None,
    class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.module.sigma_min)
    sigma_max = min(sigma_max, net.module.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.module.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.module.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, massvec, L, evals, evecs, gradX, gradY).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, massvec, L, evals, evecs, gradX, gradY).to(torch.float64)
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
    print("Start training...")
    loss_fn = EDMLoss()

    global_step = 0

    for epoch in range(args.train.num_train_epochs):
        progress_bar = tqdm(total=len(training_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(training_dataloader):
            with accelerator.accumulate(model):
                model.train()

                loss = loss_fn(net=model, 
                               signal=batch['verts_color'], 
                               massvec=batch['mass'], 
                               evals=batch['evals'], 
                               evecs=batch['evecs'], 
                               gradX=batch['gradX'], 
                               gradY=batch['gradY'], 
                               mask=True, 
                               pad_mask = batch['verts_color_padded_mask'])

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            progress_bar.update(1)
            logs = {"loss": float(loss.detach().item()), "lr": float(lr_scheduler.get_last_lr()[0]), "step": global_step}
            if args.exp.wandb:
                if accelerator.is_main_process:
                    if global_step % 50 == 0:
                        wandb.log({"loss":float(loss.detach().item()),
                                "lr": float(lr_scheduler.get_last_lr()[0])})
                        
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            
        if accelerator.is_main_process:
            if epoch % args.train.save_interval == 0 and epoch != 0:
                print(f"Save checkpoint for epoch {epoch}")
                accelerator.save_state(os.path.join(exp_dir, "accelerator", f"epoch_{epoch}_steps_{step}"))
        

                print(f"Save visualize result for the last batch at epoch {epoch}")
                model.eval()
                with torch.no_grad():
                    progress_bar = tqdm(total=args.exp.num_sample)
                    batch_seeds = torch.arange(batch['verts_color'].shape[0])
                    rnd = StackedRandomGenerator(batch['verts_color'].device, batch_seeds)
                    
                    # render gt
                    vis_output_dir=os.path.join(exp_dir, "visualize", f"epoch_{epoch}", f"gt")
                    check_dir(vis_output_dir)
                    qualitative_eval(batch['verts_color'], mesh=batch, output_dir=vis_output_dir, shapenet=True)
                    
                    for i in range(args.exp.num_sample):
                        
                        signal_t = rnd.randn(batch['verts_color'].shape, device=batch['verts_color'].device)
                        signal_t = edm_sampler(latents=signal_t, net=model, massvec=batch['mass'], evals=batch['evals'], evecs=batch['evecs'], gradX=batch['gradX'], gradY=batch['gradY'], randn_like=rnd.randn_like)

                        sampled_signal = signal_t.cpu()
                        vis_output_dir = os.path.join(exp_dir, "visualize",f"epoch_{epoch}","generated")
                        check_dir(vis_output_dir)
                        print(f"Visualization save to {vis_output_dir}.")
                        qualitative_eval(sampled_signal, mesh=batch ,output_dir=vis_output_dir, sample_id=i, shapenet=True)
                model.train()