import os
import torch
from pytorch3d.io.utils import _make_tensor
from pytorch3d.io.obj_io import _format_faces_indices


def check_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)    


def padding_x(x, new_pad_shape, mask=False):
    '''
    padding the input batched features.
    input:
        - x: list of the torch tensor to be padded, each item in the list is a data sample
        - new_pad_shape: (b, *, *)
        - mask: return the padded mask
    
    return:
        - x_pad: torch tensor with padded values
        - pad_mask: torch tensor with zeros and ones, 1 represents the non-padded entries
    '''
    x_pad = torch.zeros(new_pad_shape)
    pad_mask = torch.zeros(new_pad_shape)
    
    for i, x_tensor in enumerate(x):
        if len(x_tensor.shape) == 1:
            x_pad[i, :x_tensor.shape[0]] = x_tensor   # pad zeros to the right
            pad_mask[i, :x_tensor.shape[0]] = 1       # 1: non-padded entries 
        else:
            x_pad[i, :x_tensor.shape[0], :x_tensor.shape[1]] = x_tensor   # pad zeros to the right
            pad_mask[i, :x_tensor.shape[0], :x_tensor.shape[1]] = 1       # 1: non-padded entries 

    if mask:
        return x_pad, pad_mask
    
    return x_pad



def padding_sparse_x(x, new_pad_shape, mask=False):
    '''
    padding the input batched features.
    input:
        - x: list of the torch tensor to be padded, each item in the list is a data sample
        - new_pad_shape: (b, *, *)
        - mask: return the padded mask
    
    return:
        - x_pad: torch tensor with padded values
        - pad_mask: torch tensor with zeros and ones, 1 represents the non-padded entries
    '''
    # Assuming the batch is a list of sparse tensors
    indices = []
    values = []
    
    for i, x_tensor in enumerate(x):
        # Get the current sparse tensor's indices and values
        batch_indices = x_tensor._indices()
        batch_values = x_tensor._values()

        # Adjust indices to account for the batch dimension (prepend batch index)
        batch_indices = torch.cat([torch.full((1, batch_indices.shape[1]), i), batch_indices], dim=0)  # Add batch dimension
        
        indices.append(batch_indices)
        values.append(batch_values)
        
        
    # Concatenate all indices and values
    batch_indices = torch.cat(indices, dim=1)
    batch_values = torch.cat(values)
    
    # Create a new sparse tensor with the given padded shape
    x_pad = torch.sparse_coo_tensor(batch_indices, batch_values, size=new_pad_shape)

    return x_pad



# def shapenet_collate_fn(batch):
#     '''
#     Padding to the maximum number of vertices in the batch for Shapenet_Dataset
#     vert_color: torch tensor of shape [B,V,C]
#     mass: torch tensor of shape [B,V,V]
#     evecs: torch tensor of shape [B,V,K]
#     gradX: sparse torch tensor of shape [B,V,V]
#     gradY: sparse torch tensor of shape [B,V,V]
#     ** no need to pad evals as it is with shape (B, K, 1), dimensions are the same in a batch
#     '''
#     print(batch)
#     for data in batch: 
#         verts, faces, vertex_color, mass, evals, evecs, gradX, gradY, synset_id, model_id, label = batch
    
#     # Extract inputs
#     # verts = [item["verts"] for item in batch]
#     # faces = [item["faces"] for item in batch]
#     # textures = [item["textures"] for item in batch]
#     # mesh_file = [item["mesh_file"] for item in batch]
#     # op_file = [item["op_file"] for item in batch]
#     # label = [item["label"] for item in batch]
#     # evals = [item["evals"] for item in batch]
#     # vert_color = [item['verts_color'] for item in batch]
#     # mass = [item["mass"] for item in batch]
#     # evecs = [item["evecs"] for item in batch]
#     # gradX = [item["gradX"] for item in batch]
#     # gradY = [item["gradY"] for item in batch]
    
    
#     # paddings the inputs along vertices dimension to the maximum number of vertices in this batch
#     max_num_v = max(x.shape[0] for x in vertex_color)
#     C = 3
#     b = len(vertex_color)
#     k_eig = evecs[0].shape[1]
#     vert_color_padded, vert_color_padded_mask = padding_x(vertex_color, (b, max_num_v, C), mask=True)
#     mass_padded = padding_x(mass, (b, max_num_v))
#     evecs_padded = padding_x(evecs, (b, max_num_v, k_eig))
#     gradX_padded = padding_sparse_x(gradX, (b, max_num_v, max_num_v))
#     gradY_padded = padding_sparse_x(gradY, (b, max_num_v, max_num_v))
    
    
#     return verts, faces, vert_color_padded, vert_color_padded_mask, mass_padded, evals, evecs_padded, gradX_padded, gradY_padded, synset_id, model_id, label
    
def shapenet_collate_fn(batch):
    '''
    Padding to the maximum number of vertices in the batch for Shapenet_Dataset
    vert_color: torch tensor of shape [B,V,C]
    mass: torch tensor of shape [B,V,V]
    evecs: torch tensor of shape [B,V,K]
    gradX: sparse torch tensor of shape [B,V,V]
    gradY: sparse torch tensor of shape [B,V,V]
    ** no need to pad evals as it is with shape (B, K, 1), dimensions are the same in a batch
    '''
    
    # Extract inputs
    verts = [item["verts"] for item in batch]
    faces = [item["faces"] for item in batch]
    # textures = [item["textures"] for item in batch]
    mesh_file = [item["mesh_file"] for item in batch]
    op_file = [item["op_file"] for item in batch]
    label = [item["label"] for item in batch]
    evals = [item["evals"] for item in batch]
    vert_color = [item['verts_color'] for item in batch]
    mass = [item["mass"] for item in batch]
    evecs = [item["evecs"] for item in batch]
    gradX = [item["gradX"] for item in batch]
    gradY = [item["gradY"] for item in batch]
    
    
    # paddings the inputs along vertices dimension to the maximum number of vertices in this batch
    max_num_v = max(x.shape[0] for x in vert_color)
    C = 3
    b = len(vert_color)
    k_eig = evecs[0].shape[1]
    vert_color_padded, vert_color_padded_mask = padding_x(vert_color, (b, max_num_v, C), mask=True)
    mass_padded = padding_x(mass, (b, max_num_v))
    evecs_padded = padding_x(evecs, (b, max_num_v, k_eig))
    gradX_padded = padding_sparse_x(gradX, (b, max_num_v, max_num_v))
    gradY_padded = padding_sparse_x(gradY, (b, max_num_v, max_num_v))
    
    
    return {"verts":verts,
            "faces":faces,
            # "texture":textures,
            "mesh_file":mesh_file,
            "op_file":op_file,
            "verts_color":vert_color_padded,
            "verts_color_padded_mask":vert_color_padded_mask,
            "label": label,
            "mass":mass_padded,
            "evals":torch.stack(evals),
            "evecs":evecs_padded,
            "gradX":gradX_padded,
            "gradY":gradY_padded
            }
    

def parse_mesh_verts_and_faces(mesh_file, parse_face=True, parse_verts=True, parse_normal=True):
    materials_idx = -1
    verts_list = []
    faces_verts_idx = []
    # normal_list = []
    with open(mesh_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if line.startswith("v ") and parse_verts:  # Line is a vertex.
                vert = [float(x) for x in tokens[1:4]]
                if len(vert) != 3:
                    msg = "Vertex %s does not have 3 values. Line: %s"
                    raise ValueError(msg % (str(vert), str(line)))
                verts_list.append(vert)
            elif line.startswith("f ") and parse_face:  # Line is a face.
                face = tokens[1:]
                face_list = [f.split("/") for f in face]
                face_verts = []
                # Update face properties info.
                for vert_props in face_list:
                    # Vertex index.
                    face_verts.append(int(vert_props[0]))
                for i in range(len(face_verts) - 2):
                    faces_verts_idx.append((face_verts[0], face_verts[i + 1], face_verts[i + 2]))
            # elif line.startswith("vn ") and parse_normal:  # Line is a normal.
            #     norm = [float(x) for x in tokens[1:4]]
            #     if len(norm) != 3:
            #         msg = "Normal %s does not have 3 values. Line: %s"
            #         raise ValueError(msg % (str(norm), str(line)))
            #     normal_list.append(norm)
            
    # verts_list = _make_tensor(verts_list, cols=3, dtype=torch.float32)  # (V, 3)
    verts_list = torch.tensor(verts_list,  dtype=torch.float32)
    faces_verts_idx =  torch.tensor(faces_verts_idx, dtype=torch.int64)
    # normal_list =  torch.tensor(normal_list, dtype=torch.float32)
    # faces_verts_idx = _format_faces_indices(
    # faces_verts_idx, verts_list.shape[0], device = verts_list.device
    # )
    return {"verts":verts_list,
            "faces": faces_verts_idx,
            # "normals": normal_list
            }
    
    
    