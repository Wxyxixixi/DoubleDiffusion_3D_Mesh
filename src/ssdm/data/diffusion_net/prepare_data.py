## This file has been deprecated!
import os, sys
import glob
import time
import numpy as np
import pyvista as pv
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import v2
import argparse
import torch
import potpourri3d as pp3d
from src.ssdm.conf.configuration import parse_args
from src.ssdm.utils.preprocessing import simplify_mesh, get_bunny_path
from src.ssdm.utils.tools import parse_mesh_verts_and_faces, check_dir
import lib.models.diffusion_net.geometry as geometry


def prepare_shapenet_core_data(shapenet_dir, op_output_dir, k_eig=128, all_object=True, synset=None):
    
    # Since processing all the 55 synset takes ~55 hours, can choose to process only one categories
    if not all_object:
        if synset == None:
            raise NameError("Synset value is required when processing a single category, but none was provided.")
        synset_ids = [synset]
    else:
        # get all the object categories
        synset_ids = [name for name in os.listdir(shapenet_dir) if os.path.isdir(os.path.join(shapenet_dir, name))]
        
    start = time.time()
    
    # Iterate over the synsets to process them
    for synset_id in synset_ids:
        output_path = os.path.join(op_output_dir, synset_id)
        check_dir(output_path)  # Ensure output directory exists

        # Gather the list of mesh files for this synset
        synset_mesh_list = glob.glob(os.path.join(shapenet_dir, synset_id, "*/models/model_normalized.obj"))
        synset_op_dir_list = []

        verts_list = []
        faces_list = []

        print(f"Loading faces and vertices for synset {synset_id}...")
        
        for mesh_f in tqdm(synset_mesh_list):
            model_id = mesh_f.split('/')[4]  # Extract the model ID
            synset_op_dir_list.append(os.path.join(output_path, model_id))

            if not os.path.exists(os.path.join(output_path,  model_id, "verts_list.pt")):
                # Parse the mesh vertices and faces
                data = parse_mesh_verts_and_faces(mesh_f)
                verts = data["verts"]
                num_verts = verts.shape[0]
                # Exclude meshes that are much smaller than k_eig
                if num_verts <= 10 * k_eig or num_verts > 100000:
                    synset_op_dir_list.remove(os.path.join(output_path, model_id))
                    continue

                # Ensure model directory exists
                check_dir(os.path.join(output_path, model_id))
                
                faces = data["faces"] - 1
                verts_list.append(verts)
                faces_list.append(faces)

                # Save the verts and faces
                torch.save(verts, os.path.join(output_path, model_id, "verts_list.pt"))
                torch.save(faces, os.path.join(output_path, model_id, "faces_list.pt"))
            else:
                # Load verts and exclude small meshes
                verts = torch.load(os.path.join(output_path, model_id, "verts_list.pt"))
                num_verts = verts.shape[0]
                # Exclude meshes that are much smaller than k_eig
                if num_verts <= 10 * k_eig or num_verts > 100000:
                    synset_op_dir_list.remove(os.path.join(output_path, model_id))
                    continue

                check_dir(os.path.join(output_path, model_id))
                verts_list.append(verts)
                
                faces_load = torch.load(os.path.join(output_path, model_id, "faces_list.pt"))

                faces_list.append(faces_load)

        print(f"Processing shape operators for {synset_id}...")
        # Call to geometry.get_all_operators with the required arguments
        _, _, _, _, _, _, _ = geometry.get_all_operators(
            verts_list, faces_list, k_eig=k_eig, op_cache_dir=synset_op_dir_list, shapenet=True
        )
    
    print(f"Finish processing for synset {synset_ids}, elapse time: {(time.time() - start)/60} mins.")


def select_mesh(op_dir='datasets/preprocessed/op_cache/shapenet_core'):
    op_files = glob.glob(os.path.join(op_dir,"*/*/verts_list.pt" ))
    
    for mesh_f in op_files:
        data = parse_mesh_verts_and_faces(mesh_f)
        verts = data["verts"]
        num_verts = verts.shape[0]





def prepare_data(args):    
    # mesh resolution
    if args.data.obj_cat == 'bunny':
        args.data.object_path = get_bunny_path(args.data)
    
    output_path = os.path.join(args.data.preprocessed_data, args.data.obj_cat, args.data.precision_level)
    if not os.path.exists(os.path.join(output_path, 'uv.npz')): 
        print(f"Saving the uv coodinates of {args.data.obj_cat}_{args.dara.precision_level} to {output_path}")
       
        mesh = pv.read(args.data.object_path)
        num_verts = mesh.number_of_points
        pv_uvs = mesh.active_texture_coordinates
        uv = np.zeros_like(pv_uvs)
        uv[:,1] = (pv_uvs[:, 0]) * args.img_size[0] - 1
        uv[:,0] = (-1 * pv_uvs[:, 1] + 1) * args.img_size[1] - 1
        uv = np.array(uv, dtype=int)
        # uv_tensor = torch.from_numpy(uv)
        output_path = os.path.join(args.data.preprocessed_data, args.data.obj_cat, args.data.precision_level)
        np.save(os.path.join(output_path, 'uv.npz'), uv)
    else:
        print(f"uv file for {args.data.obj_cat}_{args.dara.precision_level} has been computed to {output_path}")
    
    # model specific requirments --- laplacian; eigens; etc
    if args.model_type == 'diffusion_net':
        print(f"[{args.model_type}] Compute sptial gradients for {args.data.obj_cat}_{args.dara.precision_level}.")
        from lib.models.diffusion_net import geometry
        verts, faces = pp3d.read_mesh(args.data.object_path)
        verts = torch.tensor(verts).float()
        # only one mesh scenario. 
        faces = [torch.tensor(faces)]
        verts = [geometry.normalize_positions(verts)]
        # op_cache_dir = os.path.join(output_path, 'op_cache')
        frames, massvec, L, evals, evecs, gradX, gradY = geometry.get_all_operators(verts, faces, k_eig=args.dfn_model.k_eig, op_cache_dir=output_path)

    return frames, massvec, L, evals, evecs, gradX, gradY


if __name__ == '__main__':
    # args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    prepare_shapenet_core_data(shapenet_dir='datasets/objects/shapenetCore',
                               op_output_dir='datasets/preprocessed/tmp', 
                               k_eig=128, 
                               all_object=False, 
                               synset='02691156')