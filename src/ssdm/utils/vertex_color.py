# author Ziang Cheng
import torch

def get_vertex_color(verts, faces, textures, interpolation="area"):
    '''
    Inputs:
        - item: a dataset item from pytorch3d's shapenet dataset with following fields
            - "verts": torch tensor of shape [V,3]
            - "faces": torch tensor  of shape [F,3]
            - "textures": torch tensor of shape [F,R,R,C] with values in [0,1]
        - interpolation: vertex color interpolation weights, must be "uniform", "area" or "angle"
    Returns:
        - verts_colors: torch tensor of shape [V,C] with values in [0,1]
    '''
    
    assert interpolation in ["uniform", "area", "angle"]
    
    # verts, faces, textures = item["verts"], item["faces"], item["textures"]
    faces = faces.long()
    face_vert_rgb = textures[:,[0,-1,0],[-1,0,0]] # [F,3,C]
    
    V = verts.shape[0]
    F = faces.shape[0]
    C = face_vert_rgb.shape[2]
    
    if interpolation == "uniform":
        weights = torch.ones(F,3, dtype=textures.dtype, device=textures.device)
    
    if interpolation == "area":
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
        face_areas = 0.5 * torch.norm(face_normals, dim=1)  # [F]
        weights = face_areas[:, None].expand(-1, 3)  # [F, 3]

    elif interpolation == "angle":
        v0 = verts[faces[:, 0]]  # [F, 3]
        v1 = verts[faces[:, 1]]  # [F, 3]
        v2 = verts[faces[:, 2]]  # [F, 3]

        e0 = v1 - v2
        e1 = v2 - v0
        e2 = v0 - v1

        e0_len = torch.norm(e0, dim=1)
        e1_len = torch.norm(e1, dim=1)
        e2_len = torch.norm(e2, dim=1)

        angle0 = torch.acos(
            torch.clamp(
                (e1_len ** 2 + e2_len ** 2 - e0_len ** 2) / (2 * e1_len * e2_len), -1.0, 1.0
            )
        )
        angle1 = torch.acos(
            torch.clamp(
                (e0_len ** 2 + e2_len ** 2 - e1_len ** 2) / (2 * e0_len * e2_len), -1.0, 1.0
            )
        )
        angle2 = torch.acos(
            torch.clamp(
                (e0_len ** 2 + e1_len ** 2 - e2_len ** 2) / (2 * e0_len * e1_len), -1.0, 1.0
            )
        )
        weights = torch.stack([angle0, angle1, angle2], dim=1)  # [F, 3]
        
    weights = weights.reshape(-1)  # [F*3]

    weighted_rgb = face_vert_rgb.reshape(F*3,C) * weights.unsqueeze(1)  # [F*3, C]
    sum_weighted_rgb = torch.zeros(V,C, dtype=textures.dtype, device=textures.device)
    sum_weighted_rgb.index_add_(0, faces.flatten(), weighted_rgb)

    sum_weights = torch.zeros(V, dtype=weights.dtype, device=weights.device)
    sum_weights.index_add_(0, faces.flatten(), weights)

    return sum_weighted_rgb / (sum_weights.unsqueeze(1) + 1e-10)

