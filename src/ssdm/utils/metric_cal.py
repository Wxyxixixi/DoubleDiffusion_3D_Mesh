import os
import torch.nn as nn
from src.ssdm.data.diffusionnet_data import ManifoldDataset
import glob
import torch
import numpy as np
from tqdm import tqdm

def iterate_in_chunks(data, batch_size):
    """
    Helper function to iterate over data in chunks of specified batch size.
    Args:
        data: The dataset to iterate over
        batch_size: The size of each chunk
    Yields:
        Chunks of the dataset
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def l2_distance(tensor1, tensor2):
    """
    Computes the L2 distance between two sets of point clouds.
    Args:
        tensor1, tensor2: Tensors of shape (batch_size, n_points, 3)
    Returns:
        dist: L2 distance between points in tensor1 and tensor2
    """
    return torch.sqrt(torch.sum((tensor1 - tensor2) ** 2, dim=-1))


def minimum_mathing_distance(sample_pcs, ref_pcs, batch_size, normalize=True):
    """
    Computes the Minimum Matching Distance (MMD) between two sets of point-clouds using L2 distance.
    
    Args:
        sample_pcs (numpy array SxKx3): the S point-clouds, each of K points to be matched and compared to a 
            set of reference point-clouds.
        ref_pcs (numpy array RxKx3): the R point-clouds, each of K points that constitute the set of reference point-clouds.
        batch_size (int): specifies how large the batches should be for comparisons.
        normalize (boolean): if True, the distances are normalized by dividing them by the number of points.
        
    Returns:
        A tuple containing the MMD and the matched distances.
    """
    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    _, n_pc_points_s, pc_dim_s = sample_pcs.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError("Incompatible size of point-clouds.")

    matched_dists = []

    for i in tqdm(range(n_ref)):
        best_in_all_batches = []
        for sample_chunk in iterate_in_chunks(sample_pcs, batch_size):
            ref_tensor = torch.tensor(ref_pcs[i]).unsqueeze(0)  # Shape (1, n_pc_points, 3)
            sample_tensor = torch.tensor(sample_chunk)  # Shape (batch_size, n_pc_points, 3)

            # Compute L2 distance
            dist_ref_to_sample = l2_distance(ref_tensor, sample_tensor)

            if normalize:
                dist_ref_to_sample = torch.mean(dist_ref_to_sample, dim=1)  # Normalize over point count
            else:
                dist_ref_to_sample = torch.sum(dist_ref_to_sample, dim=1)
            
            best_in_batch = torch.min(dist_ref_to_sample).item()  # Best match in the batch
            best_in_all_batches.append(best_in_batch)

        matched_dists.append(min(best_in_all_batches))

    mmd = np.mean(matched_dists)
    return mmd, matched_dists

def coverage(sample_pcs, ref_pcs, batch_size, normalize=True, ret_dist=False):
    """
    Computes the Coverage (COV) between two sets of point-clouds using L2 distance.
    
    Args:
        sample_pcs (numpy array SxKx3): the S point-clouds, each of K points that will be matched and 
            compared to the reference point-clouds.
        ref_pcs (numpy array RxKx3): the R point-clouds, each of K points that constitute the reference point-clouds.
        batch_size (int): specifies the batch size used to compare sample vs ref point-clouds.
        normalize (boolean): if True, the distances are normalized by dividing them with the number of points.
        ret_dist (boolean): if True, the function also returns the matched distances.
        
    Returns:
        cov (float): Coverage score (proportion of unique reference points matched).
        matched_gt (list): Indices of the reference point-clouds that are matched with each sample point-cloud.
        matched_dist (optional): Matched distances of the sample point-clouds.
    """
    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    n_sam, n_pc_points_s, pc_dim_s = sample_pcs.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError("Incompatible Point-Clouds.")

    matched_gt = []
    matched_dist = []

    for i in tqdm(range(n_sam)):
        best_in_all_batches = []
        loc_in_all_batches = []

        for ref_chunk in iterate_in_chunks(ref_pcs, batch_size):
            sample_tensor = torch.tensor(sample_pcs[i]).unsqueeze(0)  # Shape (1, n_pc_points, 3)
            ref_tensor = torch.tensor(ref_chunk)  # Shape (batch_size, n_pc_points, 3)

            # Compute L2 distance
            dist_sample_to_ref = l2_distance(sample_tensor, ref_tensor)

            if normalize:
                dist_sample_to_ref = torch.mean(dist_sample_to_ref, dim=1)  # Normalize over point count
            else:
                dist_sample_to_ref = torch.sum(dist_sample_to_ref, dim=1)

            best_in_batch = torch.min(dist_sample_to_ref).item()
            location_of_best = torch.argmin(dist_sample_to_ref).item()

            best_in_all_batches.append(best_in_batch)
            loc_in_all_batches.append(location_of_best)

        best_in_all_batches = np.array(best_in_all_batches)
        b_hit = np.argmin(best_in_all_batches)  # In which batch the minimum occurred
        matched_dist.append(np.min(best_in_all_batches))
        hit = np.array(loc_in_all_batches)[b_hit]
        matched_gt.append(batch_size * b_hit + hit)

    cov = len(np.unique(matched_gt)) / float(n_ref)

    if ret_dist:
        return cov, matched_gt, matched_dist
    else:
        return cov, matched_gt

dataset = ManifoldDataset(
    'datasets/objects/manifold_example/manifold_processed_bunny.obj',
    'datasets/images/celeba_hq',
    128,
    overfit=False,
    overfit_size=1,
    op_cache_dir='datasets/preprocessed/textured_objs/bunny/manifold',
    split_file='./splits/test_set.txt')

signal_path = '/home/liz0l/workspace/ddpm_diffusionNet/work_dir/ckps/edm_bs8_gpu4_acc1_lr3e-2_e48_fixnetbug/epoch_26_steps_755/save_signals'
signal_list = sorted(glob.glob(os.path.join(signal_path, "*.pt")))

signal_item = []
for signal in signal_list:
    signal_item_ts = torch.load(signal)
    signal_item.append(signal_item_ts)
sample_pcs = torch.cat(signal_item, dim=0)
sample_pcs = torch.clamp(sample_pcs, min=-1, max=1)
sample_pcs = (sample_pcs + 1) / 2
sample_pcs = sample_pcs.cuda()

# ref_num = 2824
# ref_pcs = []
# for i in range(ref_num):
#     signal, image = dataset[i]
#     ref_pcs.append(signal)
# ref_pcs = torch.stack(ref_pcs, dim=0)
# ref_pcs = (ref_pcs + 1) / 2
# torch.save(ref_pcs, './work_dir/gt.pt')
ref_pcs = torch.load('./work_dir/gt.pt')
ref_pcs = ref_pcs.cuda()

mmd, _ = minimum_mathing_distance(sample_pcs, ref_pcs, batch_size=100)
cov, _ = coverage(sample_pcs, ref_pcs, batch_size=100)
print("MMD: {}; COV: {}".format(mmd, cov))

