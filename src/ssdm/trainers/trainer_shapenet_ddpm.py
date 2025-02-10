import torch
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import os
from src.ssdm.utils.tools import check_dir
from src.ssdm.utils.evaluations import qualitative_eval

def train_loop(args, accelerator, model, noise_scheduler, optimizer, lr_scheduler, training_dataloader, exp_dir):
    
    global_step = 0

    for epoch in range(args.train.num_train_epochs):
        progress_bar = tqdm(total=len(training_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(training_dataloader):
            with accelerator.accumulate(model):
                model.train()
                verts_color = batch['verts_color']
                verts_color_padded_mask = batch['verts_color_padded_mask'] # 1: non-padded entries 
                massvec = batch['mass']
                evals = batch['evals']
                evecs = batch['evecs']
                gradX = batch['gradX']
                gradY = batch['gradY']
                if epoch == 0:
                    vis_output_dir=os.path.join(exp_dir, "visualize", f"gt_{step}")
                    check_dir(vis_output_dir)
                    qualitative_eval(verts_color, mesh=batch, output_dir=vis_output_dir, shapenet=True)
                
                # sample noise to add to the images
                noise = torch.randn(verts_color.shape, device=verts_color.device)
                bs = verts_color.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,),device=verts_color.device, dtype=torch.int64)
                noisy_signal = noise_scheduler.add_noise(verts_color, noise, timesteps)
                noise_pred = model(x_in=noisy_signal,timestep=timesteps, mass=massvec, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)
                
                # calculate the loss
                loss = F.mse_loss(noise_pred, noise, reduction='none')
                masked_loss = loss * verts_color_padded_mask
                sum_masked_loss = torch.sum(masked_loss)
                num_unpadded_elements = torch.sum(verts_color_padded_mask)
                mse_loss = sum_masked_loss / num_unpadded_elements

                accelerator.backward(mse_loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            
            logs = {"loss": mse_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.exp.wandb:
                if accelerator.is_main_process:
                    if global_step % 50 == 0:
                        wandb.log({"loss":loss.detach().item(),
                                "lr": lr_scheduler.get_last_lr()[0]})
                    
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        
            if accelerator.is_main_process:
                if epoch % args.train.save_interval == 0 and epoch != 0:
                    print(f"Save checkpoint for epoch {epoch}")
                    accelerator.save_state(os.path.join(exp_dir, "accelerator", f"epoch_{epoch}_steps_{step}"))

                    print(f"Save visualize result for epoch {epoch}")
                    model.eval()
                    with torch.no_grad():
                        generator = torch.Generator(device='cpu').manual_seed(args.train.seed)
                        noise_scheduler.set_timesteps(args.diffusion.sampling_steps)
                        for i in range(args.exp.num_sample):
                            signal_t = torch.randn(verts_color.shape, device=verts_color.device)
                            for time_step in tqdm(noise_scheduler.timesteps): # reverse sampling
                                model_output = model(x_in=signal_t,timestep=timesteps, mass=massvec, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)
                                signal_t = noise_scheduler.step(model_output, time_step, signal_t, generator=generator).prev_sample
                            

                            sampled_signal = signal_t.cpu()
                            vis_output_dir = os.path.join(exp_dir, "visualize",f"epoch_{epoch}")
                            check_dir(vis_output_dir)
                            qualitative_eval(sampled_signal, mesh=batch, output_dir=vis_output_dir, shapenet=True)

                    model.train()


