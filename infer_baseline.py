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

from src.baseline.diff import DiffusionModel
from src.baseline.samplers import DDPM_Sampler
from src.baseline.data import DataManager, ManifoldDataset
import math

def main(args):
    output_path = os.path.join(args.data.preprocessed_data, args.data.obj_cat, args.data.precision_level)
    
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
    
    state_dict = load_file(args.infer_ckp_path)
    # state_dict = torch.load(args.infer_ckp_path)
    exp_dir = os.path.dirname(args.infer_ckp_path)
    print(exp_dir)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    min_emb: float = math.sqrt(dataset.sampler.verts.size(0)) * dataset.sampler.lbo_embedder.evecs.min().item()
    max_emb: float = math.sqrt(dataset.sampler.verts.size(0)) * dataset.sampler.lbo_embedder.evecs.max().item()
    
    with torch.no_grad():
        if args.infer_show:
            vis_output_dir = os.path.join(exp_dir, "visualize",)
            check_dir(vis_output_dir)
        if args.infer_save:
            save_output_dir = os.path.join(exp_dir, "save_signals",)
            check_dir(save_output_dir)    
            
        for i in range(args.exp.num_sample):
            emb = []
            pts = []
        
            pos, faces, coefs = dataset.sampler.mesh_sampler.sample(n=4000)
            pos_embedding = dataset.sampler.lbo_embedder(faces, coefs)

            emb.append(pos_embedding)
            pts.append(pos)
    
            p_pts = torch.stack(pts)
            p_emb = torch.stack(emb).cuda()

            with torch.inference_mode():
                q_emb = 2 * ((p_emb - min_emb) / (max_emb - min_emb)) - 1
                q_sig = model.decode(q_emb, subset=q_emb.size(1))
            
            sampled_signal = q_sig.cpu()

            if args.infer_show:
                mesh = pv.PolyData(p_pts[i].numpy())
                qualitative_eval(sampled_signal, mesh ,output_dir=vis_output_dir, wandb_log=args.exp.wandb, sample_id=i)
            if args.infer_save:
                for j in range(p_pts.shape[0]):
                    torch.save(sampled_signal, os.path.join(save_output_dir, "signal_{}_{}.pt".format(i, j)))

if __name__ == '__main__':
    args = parse_args()
    if args.infer_show:
        import pyvista as pv
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    main(args)