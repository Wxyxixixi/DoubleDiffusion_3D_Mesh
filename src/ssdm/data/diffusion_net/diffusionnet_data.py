'''
Adapt from https://github.com/nmwsharp/diffusion-net/blob/master/experiments/rna_mesh_segmentation/rna_mesh_dataset.py
'''
import os, glob,sys
import numpy as np
import pyvista as pv
import potpourri3d as pp3d
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torchvision

import lib.models.diffusion_net as diffusion_net



class ManifoldDataset(Dataset):
    def __init__(self, 
                 object_path, 
                 image_folder,
                 k_eig = 128,
                 op_cache_dir=None,
                 img_size = [1024, 1024],
                 overfit = False,
                 overfit_size = 1,
                 split_file=None,
                ):
        super().__init__()
        self.obj_path = object_path
        self.image_folder = image_folder
        self.split_file = split_file
        with open(self.split_file, 'r') as f:
            # self.image_list = f.readlines()
            lines = f.readlines()
            self.image_list = [os.path.join(self.image_folder, line.strip()) for line in lines]
            
        # self.image_list = sorted(glob.glob(os.path.join(self.image_folder, "*.jpg"))) # per-vertex
        self.k_eig = k_eig
        verts_list = []
        faces_list = []
        
        # Load the meshes & signals
        verts, faces = pp3d.read_mesh(self.obj_path)
        verts = torch.tensor(verts).float()
        faces = torch.tensor(faces)

        # center and unit scale
        verts = diffusion_net.geometry.normalize_positions(verts)
        
        self.verts = verts
        self.faces = faces    
        
        self.mesh = pv.read(self.obj_path)
        self.mesh.texture_map_to_plane(use_bounds=True, inplace=True)
        self.num_vertices = self.mesh.number_of_points
        
        pv_uvs = self.mesh.active_texture_coordinates
        uv = np.zeros_like(pv_uvs)
        uv[:,1] = (pv_uvs[:, 0]) * img_size[0] - 1
        uv[:,0] = (-1 * pv_uvs[:, 1] + 1) * img_size[1] - 1
        uv = np.array(uv, dtype=int)
        self.uv = uv
        self.uv_tensor = torch.from_numpy(uv)

        del pv_uvs
        del uv

        verts_list.append(verts)
        faces_list.append(faces)
        # self.labels_list.append(labels)
        
        # # Precompute operators
        frames_list, massvec_list, L_list, evals_list, evecs_list, gradX_list, gradY_list = diffusion_net.geometry.get_all_operators(verts_list, faces_list, k_eig=self.k_eig, op_cache_dir=op_cache_dir)
        self.frames = frames_list[0]
        self.massvec = massvec_list[0]
        self.L = L_list[0]
        self.evals = evals_list[0]
        self.evecs = evecs_list[0]
        self.gradX = gradX_list[0]
        self.gradY = gradY_list[0]
        
        ## For testing and debugs
        self.overfit = overfit
        self.overfit_size = overfit_size
        
    def __len__(self):
        if self.overfit:
            return self.overfit_size
        else:
            return len(self.image_list)

    
    def __getitem__(self, idx):
        
        image_path = self.image_list[idx]
        
        image = Image.open(image_path)
        image = np.asarray(image)
        image = torch.tensor(image).float()
        image = image / 127.5 - 1
        image = image.permute(2, 0, 1)
        
        signal = image[:, self.uv_tensor[:, 0], self.uv_tensor[:, 1]] # size: (C, N)
        signal = signal.permute(1, 0) # size: (N, C)

        return signal, image
        
    
    
     
    
        
    