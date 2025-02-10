import os
import time 
import numpy as np
from tqdm import tqdm
import wandb
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator

from src.ssdm.conf.configuration import parse_args
from src.ssdm.utils.build_model_and_data import build_model_and_data
from src.ssdm.data.diffusionnet_data import ManifoldDataset as ManifoldDatasetOurs
from src.ssdm.utils.evaluations import qualitative_eval
from src.ssdm.utils.tools import check_dir
from src.ssdm.trainers.trainer_ddpm import train_loop as ddpm_train_loop
from src.ssdm.trainers.trainer_edm import train_loop as edm_train_loop
from src.ssdm.models.diffusion_net import EDMPrecond, EDMPrecond2

from src.ssdm.utils.preprocessing import get_bunny_path
from src.baseline.diff import DiffusionModel
from src.baseline.samplers import DDPM_Sampler
from src.baseline.data import DataManager, ManifoldDataset

import math

def main(args):
    # seeds
    np.random.seed(args.train.seed)
    torch.manual_seed(args.train.seed)
    torch.cuda.manual_seed(args.train.seed)
    
    # loggings 
    # TODO: wandb
    current_time = str(time.time()).split(".")[0]
    if args.exp.name == None:
        args.exp.name = current_time
        exp_dir = os.path.join(args.exp.output_dir, args.exp.group, args.exp.job, args.exp.name)
    else:
        exp_dir = os.path.join(args.exp.output_dir, current_time)
    
    # accelerator to handle the training
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accelerate.gradient_accumulation_steps, mixed_precision='no')
    
    if args.exp.wandb:
        if accelerator.is_main_process:
            wandb.init(
                project=args.exp.project,
                group=args.exp.group,
                job_type=args.exp.job,
                name=args.exp.name,
                config=args,
            )
            
    # load the models and datasets
    # dataset, model = build_model_and_data(args, accelerator)
    args.data.object_path = get_bunny_path(args.data)
    output_path = os.path.join(args.data.preprocessed_data, args.data.obj_cat, args.data.precision_level)
    
    if args.model_type == 'diffusion_net':
        if accelerator is not None:
            if accelerator.is_main_process:
                print("Dataset Configuration: ")
                print("object_path: {}".format(args.data.object_path))
                print("image_path: {}".format(args.data.image_path))
                print("k_eig: {}".format(args.dfn_model.k_eig))
                print("overfit: {}".format(args.debug.overfit))
                print("overfit_size: {}".format(args.debug.overfit_size))
                print("op_cache_dir: {}".format(output_path))
                print("split_file: {}".format(args.data.split_file))
                
        dataset = ManifoldDatasetOurs(args.data.object_path,
                                      args.data.image_path,
                                      args.dfn_model.k_eig,
                                      overfit=args.debug.overfit,
                                      overfit_size=args.debug.overfit_size,
                                      op_cache_dir=output_path,
                                      split_file=args.data.split_file)
        
        frame, massvec, L, evals, evecs, gradX, gradY = \
            dataset.frames, dataset.massvec, dataset.L, \
                dataset.evals, dataset.evecs, dataset.gradX, dataset.gradY
        
        # print(frame, massvec, L, evals, evecs, gradX, gradY)
        
        # Handle various types of data and unify them
        data: DataManager = DataManager(data_dir=None, cache_dir=output_path, overfit=args.debug.overfit,)

        # Fix a manifold and allow sampling function datapoints f : M -> Y
        num_samples = 4000
        k_evecs = 128
        dataset: ManifoldDataset = data.dataset('stanford-bunny', 'celeba', n_samples=num_samples, k_evecs=k_evecs, device="cuda")
    
    # build model here:
    schedule = 'linear'
    sampler = DDPM_Sampler(num_timesteps=args.diffusion.training_time_steps, schedule=schedule)
    model = DiffusionModel(
        dim_mlp=2048,
        # dim_time=64,
        dim_time=64,
        num_points=None, # no use
        sampler=sampler,
        dim_signal=3,
        # n_heads=4,
        n_heads=5,
        k_evecs=k_evecs,
        schedule=schedule,
        dropout=0.1,
        num_timesteps=args.diffusion.training_time_steps,
        num_encoder_layers=6,)
    model.cuda()
    
    eff_bs = args.train.train_batch_size * accelerator.num_processes * args.accelerate.gradient_accumulation_steps
    lr = args.train.lr * eff_bs / 256
    args.train.lr = lr
    
    if accelerator.is_main_process:
        print(f"Learning rate: {args.train.lr}")
        print(f"Effective batch size: {eff_bs}")
    
    training_dataloader = DataLoader(
        dataset,
        batch_size=args.train.train_batch_size,
        num_workers=args.train.train_num_workers,
        pin_memory=True,
        shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.train.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
                                    optimizer=optimizer,
                                    num_warmup_steps = 1000,
                                    # num_warmup_steps = 10,
                                    num_training_steps = int(len(training_dataloader) * args.train.num_train_epochs))
    
    if accelerator.is_main_process:
        check_dir(exp_dir) # to make sure the folder path is exist, or else create one.
        
    global_step = 0

    
    min_emb: float = math.sqrt(dataset.sampler.verts.size(0)) * dataset.sampler.lbo_embedder.evecs.min().item()
    max_emb: float = math.sqrt(dataset.sampler.verts.size(0)) * dataset.sampler.lbo_embedder.evecs.max().item()

    model, optimizer, training_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, training_dataloader, lr_scheduler)
    
    for epoch in range(args.train.num_train_epochs):
        progress_bar = tqdm(total=len(training_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(training_dataloader):
            with accelerator.accumulate(model):
                model.train()
                
                emb = batch['pos_embedding'].cuda()
                sig = batch['signal'].cuda()
                
                # Scale data to [-1, 1] (sig has already been -1 to 1)
                emb = 2 * ((emb - min_emb) / (max_emb - min_emb)) - 1
                sig = 2 * sig - 1
                
                # Split batch in half for context and query
                c_emb, q_emb = torch.chunk(emb, chunks=2, dim=1)
                c_sig, q_sig = torch.chunk(sig, chunks=2, dim=1)

                # Sample timestep and noise vectors for context and query
                t = torch.randint(0, args.diffusion.training_time_steps, size=(args.train.train_batch_size,), device=sig.device)

                # Create noisy context & query at timestep t
                c_t, c_sig_z, q_t, q_sig_z = model.module.encode(c_emb, c_sig, q_emb, q_sig, t) 

                # Predict the noise for the score network
                q_sig_z_m = model.module.noise(c_t, q_t, t) 
                
                # calculate the loss
                loss = F.mse_loss(q_sig_z_m, q_sig_z)
                
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
        
    print(f"Save checkpoint for epoch at the end of training")
    accelerator.save_state(os.path.join(exp_dir, "accelerator", f"final"))

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.train.visible_devices
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    main(args)