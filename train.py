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
from src.ssdm.utils.evaluations import qualitative_eval
from src.ssdm.utils.tools import check_dir, shapenet_collate_fn
from src.ssdm.trainers.trainer_single_mesh_ddpm import train_loop as ddpm_train_loop
from src.ssdm.trainers.trainer_single_mesh_edm import train_loop as edm_train_loop
from src.ssdm.models.diffusion_net import EDMPrecond, EDMPrecond2

def main(args):
    # seeds
    np.random.seed(args.train.seed)
    torch.manual_seed(args.train.seed)
    torch.cuda.manual_seed(args.train.seed)
    
    # loggings 
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
    dataset, model = build_model_and_data(args, accelerator)
    
    eff_bs = args.train.train_batch_size * accelerator.num_processes * args.accelerate.gradient_accumulation_steps
    lr = args.train.lr * eff_bs / 256
    args.train.lr = lr
    
    if accelerator.is_main_process:
        print(f"Learning rate: {args.train.lr}")
        print(f"Effective batch size: {eff_bs}")
    
    if args.data.obj_cat == 'shapenet_core':
        collate_fn = shapenet_collate_fn
    else:
        collate_fn = None 
    
    training_dataloader = DataLoader(
        dataset,
        batch_size=args.train.train_batch_size,
        num_workers=args.train.train_num_workers,
        shuffle=True,
        pin_memory=False,
        collate_fn=collate_fn)
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.train.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
                                    optimizer=optimizer,
                                    num_warmup_steps = 1000,
                                    num_training_steps = int(len(training_dataloader) * args.train.num_train_epochs))
    
    if accelerator.is_main_process:
        check_dir(exp_dir) # to make sure the folder path is exist, or else create one.
    
    if args.train.pretrain:
        print(f"pretrain with {args.accelerate.save_dir}")
        accelerator.load_state(args.accelerate.save_dir)
        
    if args.diffusion.pipe_type == "ddpm":
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.diffusion.training_time_steps)
        model, optimizer, training_dataloader, noise_scheduler, lr_scheduler = accelerator.prepare(
            model, optimizer, training_dataloader, noise_scheduler, lr_scheduler)
        ddpm_train_loop(args, accelerator, model, noise_scheduler, optimizer, lr_scheduler, training_dataloader, exp_dir)
    elif args.diffusion.pipe_type == "edm":
        model = EDMPrecond(model=model)
        model, optimizer, training_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, training_dataloader, lr_scheduler)
        edm_train_loop(args, accelerator, model, optimizer, lr_scheduler, training_dataloader, exp_dir)
        
    print(f"Save checkpoint for epoch at the end of training")
    accelerator.save_state(os.path.join(exp_dir, "accelerator", f"final"))


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.train.visible_devices
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    main(args)