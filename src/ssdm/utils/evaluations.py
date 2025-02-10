import os
import numpy as np
import torch
import trimesh
import pyvista as pv
# pv.start_xvfb()
import wandb



def qualitative_eval(batch_signals, mesh, output_dir=None, sample_id=None, shapenet=False):
    if shapenet:
        vis_shapenet(batch_signals, mesh, out_obj_path=output_dir, sample_id=sample_id)
    else:
        vis_and_render(batch_signals, mesh, output_dir=None, sample_id=None)



def vis_and_render(batch_signals, mesh, output_dir=None, sample_id=None):
    '''
    aka visualization.
    batch_signal: generated batch signals, np array, detached to cpu.
    mesh: render the view via pyvista.
    '''
    
    bs = batch_signals.shape[0]
    pl = pv.Plotter(off_screen=True)
    for i in range(len(batch_signals)):
        batch_signals[i] = torch.clamp(batch_signals[i], min=-1, max=1)
        unorm_signal = (batch_signals[i].numpy() + 1) / 2
        rgb_signal = (unorm_signal) * 255.0
        mesh["colors"] = np.array(rgb_signal, dtype=int)
        
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(mesh, scalars='colors', rgb=True)
        pl.camera_position = 'xy'
        if not sample_id == None:
            pl.screenshot(os.path.join(output_dir,f"{sample_id}_{i}.jpg"))
        else:
            pl.screenshot(os.path.join(output_dir,f"{i}.jpg"))
        pl.clear()
    pl.close()


def vis_shapenet(vert_colors_list, mesh_list, out_obj_path, sample_id=None):
    '''
    vert_colors: batch torch tensor
    mesh: list of verts and faces
    out_obj_path: output obj file path.
    '''
    for i in range(len(vert_colors_list)):
        verts = mesh_list['verts'][i]
        faces = mesh_list['faces'][i]
        verts_color_padded_mask = mesh_list['verts_color_padded_mask'][i]
        V = verts.shape[0]
        vert_color = np.zeros((V,3))
        color = vert_colors_list[i].cpu().numpy()
        vert_color = (color[:V] + 1) *127.5
        # vert_color = (vert_colors_list[i] + 1)/2
        # vert_color = vert_color.cpu().numpy()*255
        # vert_color = vert_colors_list[i]
        trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy(), vertex_colors=vert_color.astype(int)).export(os.path.join(out_obj_path,f"{sample_id}_{i}_sample.obj"))
