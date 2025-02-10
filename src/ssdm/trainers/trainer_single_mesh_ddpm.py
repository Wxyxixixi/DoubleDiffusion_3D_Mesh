import torch
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import os
from src.ssdm.utils.tools import check_dir
from src.ssdm.utils.evaluations import qualitative_eval

def train_loop(args, accelerator, model, noise_scheduler, optimizer, lr_scheduler, training_dataloader, exp_dir):
    
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
    
    for epoch in range(args.train.num_train_epochs):
        progress_bar = tqdm(total=len(training_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(training_dataloader):
            with accelerator.accumulate(model):
                model.train()
                signal, image = batch
                
                # sample noise to add to the images
                noise = torch.randn(signal.shape, device=signal.device)
                bs = signal.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,),device=signal.device, dtype=torch.int64)
                noisy_signal = noise_scheduler.add_noise(signal, noise, timesteps)
                noise_pred = model(noisy_signal,timesteps, massvec, L, evals, evecs, gradX, gradY)
                
                # calculate the loss
                loss = F.mse_loss(noise_pred, noise)
                
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
        
        if accelerator.is_main_process:
            if epoch % args.train.save_interval == 0 and epoch != 0:
                print(f"Save checkpoint for epoch {epoch}")
                accelerator.save_state(os.path.join(exp_dir, "accelerator", f"epoch_{epoch}_steps_{step}"))
            
