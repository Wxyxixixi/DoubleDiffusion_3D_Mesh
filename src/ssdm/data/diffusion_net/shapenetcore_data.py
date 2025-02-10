import os, sys, glob
from natsort import natsorted
os.chdir(os.getcwd())
from os import path
import json
from tqdm import tqdm
import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader
from pytorch3d.datasets import ShapeNetCore
from pytorch3d.io.obj_io import load_obj


import lib.models.diffusion_net.utils as utils
import lib.models.diffusion_net.geometry as geometry
from src.ssdm.utils.vertex_color import get_vertex_color
from src.ssdm.utils.tools import parse_mesh_verts_and_faces, check_dir
from src.ssdm.conf.configuration import parse_args


    
class Shapenet_Dataset(ShapeNetCore):
    def __init__(self, 
                 data_dir, # path to the shapenetcore data
                 k_eig = 128,
                 op_cache_dir=None,
                 synsets=None, 
                 version: int = 2, 
                 load_textures: bool = True, 
                 texture_resolution: int = 4,
                 overfit = False,
                 overfit_size = None,
                 splits = "train",
                 split_file = None,
                 max_verts = None):

        super().__init__(data_dir, synsets=synsets, version=version, load_textures=load_textures, texture_resolution=texture_resolution)
        
        
        if split_file == None:
            raise ValueError("The 'split_file' argument cannot be None. Please provide a valid path or file.")
        
        self.data_dir = data_dir
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir
        
        self.overfit = overfit
        self.overfit_size = overfit_size
        if not isinstance(synsets, list):
            synsets = [synsets]
        if synsets is not None:
            print(f"Dataset contains:")
            print(", ".join([f"{key}: {self.synset_dict[key]}" for key in synsets if key in self.synset_dict]))
        else:
            synsets = self.synset_dict.keys()
        
        with open(split_file,'r') as file:
            split_data = json.load(file)

        self.shape_op_file_list = []
        self.mesh_file_list = []
        for synset in synsets:
            for idx in natsorted(split_data[synset][splits].keys()):
                model_id = split_data[synset][splits][idx]
                self.mesh_file_list.append(os.path.join(data_dir, synset, model_id, "models/model_normalized.obj"))
                self.shape_op_file_list.append(os.path.join(op_cache_dir, synset, model_id,"shape_operator.npz"))

        self.max_verts = max_verts or self.get_max_verts()
        
        
    def get_max_verts(self):
        max_verts = 0
        print("Get maximum verts")
        for model_path in tqdm(self.mesh_file_list):
            verts = parse_mesh_verts_and_faces(model_path, parse_face=False, parse_normal=False, parse_verts=True)["verts"]
            max_verts = max(max_verts, verts.shape[0])
        return max_verts



    def padding_x(self, x_tensor, new_pad_shape, mask=False):
        '''
        padding the input batched features.
        input:
            - x_tensor: torch tensor to be padded
            - new_pad_shape: (*, *)
            - mask: return the padded mask
        
        return:
            - x_pad: torch tensor with padded values
            - pad_mask: torch tensor with zeros and ones, 1 represents the non-padded entries
        '''
        x_pad = torch.zeros(new_pad_shape)
        pad_mask = torch.zeros(new_pad_shape)

        if len(x_tensor.shape) == 1:
            x_pad[:x_tensor.shape[0]] = x_tensor   # pad zeros to the right
            pad_mask[:x_tensor.shape[0]] = 1       # 1: non-padded entries 
        else:
            x_pad[:x_tensor.shape[0], :x_tensor.shape[1]] = x_tensor   # pad zeros to the right
            pad_mask[:x_tensor.shape[0], :x_tensor.shape[1]] = 1       # 1: non-padded entries 

        if mask:
            return x_pad, pad_mask
        
        return x_pad
    
    
    
    def padding_sparse_x(self, x_tensor, new_pad_shape):

        # Get the current sparse tensor's indices and values
        indices = x_tensor._indices()
        values = x_tensor._values()
        
        # Create a new sparse tensor with the given padded shape
        x_pad = torch.sparse_coo_tensor(indices, values, size=new_pad_shape)

        return x_pad
    
    
    
        
    def __len__(self):
        if self.overfit:
            return self.overfit_size
        
        """
        Return number of total models in the loaded dataset.
        """
        return len(self.mesh_file_list)
    


    def load_shape_operators(self, operator_path):
        '''
        Load shape's operator with given model_id from shapenet core v2 dataset.
        '''
        # load shape operators

        npzfile = np.load(operator_path, allow_pickle=True)
        
        def read_sp_mat(prefix):
            data = npzfile[prefix + "_data"]
            indices = npzfile[prefix + "_indices"]
            indptr = npzfile[prefix + "_indptr"]
            shape = npzfile[prefix + "_shape"]
            mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
            return mat
        
        # frames = npzfile["frames"]
        mass = npzfile["mass"]
        # L = read_sp_mat("L")
        evals = npzfile["evals"][:self.k_eig]
        evecs = npzfile["evecs"][:,:self.k_eig]
        gradX = read_sp_mat("gradX")
        gradY = read_sp_mat("gradY")

        # frames = torch.from_numpy(frames)
        mass = torch.from_numpy(mass)
        # L = utils.sparse_np_to_torch(L)
        evals = torch.from_numpy(evals)
        evecs = torch.from_numpy(evecs)
        gradX = utils.sparse_np_to_torch(gradX)
        gradY = utils.sparse_np_to_torch(gradY)
        
        return mass, evals, evecs, gradX, gradY
              
        
    def __getitem__(self, idx):
        """
        Read a model by the given index.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: LongTensor of shape (F, 3) which indexes into the verts tensor.
            - verts_color: FloatTensor of shape [V,C] with values in [0,1], color rgb on the vertices
            - synset_id (str): synset id
            - model_id (str): model id
            - label (str): synset label.
            - shape_operators (tuple): including frames ([V,3]), mass ([V,V]), L ([V,K]), evals ([K]), evecs ([V,K]), gradX ([V,V]), gradY ([V,V])
        """
        model_path = self.mesh_file_list[idx]
        operator_path = self.shape_op_file_list[idx]

        # print(model_path)
        verts, faces, textures = self._load_mesh(model_path)
        
        vertex_color = get_vertex_color(verts, faces, textures, interpolation='area') # [0,1]

        vertex_color = vertex_color.float() * 2 - 1 # scale to [-1,1]
        
        ## exclude frames and L as we are not going to use spectral method.
        mass, evals, evecs, gradX, gradY = self.load_shape_operators(operator_path)
        
        C=3 # color dimension
        k_eig = evecs.shape[1]
        
        vert_color_padded, vert_color_padded_mask = self.padding_x(vertex_color, (self.max_verts, C), mask=True)
        mass_padded = self.padding_x(mass, (self.max_verts))
        evecs_padded = self.padding_x(evecs, (self.max_verts, k_eig))
        gradX_padded = self.padding_sparse_x(gradX, (self.max_verts, self.max_verts))
        gradY_padded = self.padding_sparse_x(gradY, (self.max_verts, self.max_verts))
        

        model = {}
        model['verts'] = verts
        model['faces'] = faces
        model["mesh_file"] = model_path
        model["op_file"] = operator_path
        model["verts_color"] = vert_color_padded
        model["verts_color_padded_mask"] = vert_color_padded_mask
        model["synset_id"] = model_path.split('/')[3]
        model["model_id"] =  model_path.split('/')[4]
        model["label"] = self.synset_dict[model["synset_id"]]
        model["mass"] = mass_padded
        model["evals"] = evals
        model["evecs"] = evecs_padded
        model["gradX"] = gradX_padded
        model["gradY"] = gradY_padded
        
        return model


    
    