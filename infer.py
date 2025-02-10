import os
import torch
from src.ssdm.conf.configuration import parse_args
from diffusers import DDPMPipeline, DDPMScheduler, DDIMScheduler
from src.ssdm.utils.build_model_and_data import build_model_and_data
from tqdm import tqdm
from src.ssdm.utils.evaluations import qualitative_eval
from src.ssdm.utils.tools import check_dir
from safetensors.torch import load_file
from torch.utils.data import DataLoader

def main(args):
    dataset, model = build_model_and_data(args, None)
    state_dict = load_file(args.infer_ckp_path)
    exp_dir = os.path.dirname(args.infer_ckp_path)
    print(exp_dir)
    model.load_state_dict(state_dict, strict=True)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.diffusion.training_time_steps)
    # noise_scheduler = DDIMScheduler(num_train_timesteps=args.diffusion.training_time_steps)

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
        generator = torch.Generator(device='cpu').manual_seed(621)
        noise_scheduler.set_timesteps(args.diffusion.sampling_steps)
        if args.infer_show:
            vis_output_dir = os.path.join(exp_dir, "visualize",)
            check_dir(vis_output_dir)
        if args.infer_save:
            save_output_dir = os.path.join(exp_dir, "save_signals",)
            check_dir(save_output_dir)    
            
        for i in range(args.exp.num_sample):
            signal_t = torch.randn(signal.shape, device=signal.device)
            for time_step in tqdm(noise_scheduler.timesteps): # reverse sampling
                model_output = model(signal_t, time_step, massvec, L, evals, evecs, gradX, gradY)
                signal_t = noise_scheduler.step(model_output, time_step, signal_t, generator=generator).prev_sample
            
            sampled_signal = signal_t.cpu()
            # vis_output_path = os.path.join(exp_dir, "visualize",f"{epoch}.jpg")
            
            if args.infer_show:
                qualitative_eval(sampled_signal, dataset.mesh ,output_dir=vis_output_dir, wandb_log=args.exp.wandb, sample_id=i)
            if args.infer_save:
                for j in range(signal.shape[0]):
                    torch.save(sampled_signal, os.path.join(save_output_dir, "signal_{}_{}.pt".format(i, j)))




if __name__ == '__main__':
    args = parse_args()
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    main(args)
    
    