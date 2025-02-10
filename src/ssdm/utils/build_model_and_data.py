import os
import torch.nn as nn
from src.ssdm.models.diffusion_net import DiffusionNet
from src.ssdm.data.diffusion_net.diffusionnet_data import ManifoldDataset
from src.ssdm.data.diffusion_net.shapenetcore_data import Shapenet_Dataset
from src.ssdm.data.diffusion_net.prepare_data import prepare_data
from src.ssdm.utils.tools import check_dir

def get_bunny_path(args):
    if args.precision_level == 'simple':
        # 500 vertices
        return "datasets/objects/manifold_example/bunny_coarse.obj"
    elif args.precision_level == 'median':
        # 2992 vertices
        return "datasets/objects/manifold_example/simplify_bunny.obj"
    elif args.precision_level == 'complex':
        # ~ 30,000 vertices (35947)
        return "datasets/objects/manifold_example/stanford_bunny.obj"
    elif args.precision_level == 'manifold':
        # ~50,000 vertices (52288)
        return "datasets/objects/manifold_example/manifold_processed_bunny.obj"


def build_model_and_data(args, accelerator=None):
    '''
    Return the chosen model and the dataset
    '''
    print(f"Preparing the model {args.model_type} and the dataset {args.object_set}.")
    # args.data.object_path = prepare_data(args)
    if args.object_set == 'bunny':
        args.data.object_path = get_bunny_path(args.data)
        output_path = os.path.join(args.data.preprocessed_data, args.object_set, args.data.precision_level)
    
    shapenet_flag = False
    
    if args.model_type == 'diffusion_net':
        if args.object_set == 'bunny':
            data = ManifoldDataset(args.data.object_path,
                                args.data.image_path,
                                args.dfn_model.k_eig,
                                overfit=args.debug.overfit,
                                overfit_size=args.debug.overfit_size,
                                op_cache_dir=output_path,
                                split_file=args.data.split_file)
            
        elif args.object_set == 'shapenet_core':
            output_path = os.path.join(args.data.preprocessed_data, args.object_set)
            check_dir(output_path)
            data = Shapenet_Dataset(data_dir=args.data.shapenet_path, # for shapenet, the object path is the shapenet directory
                                    version=args.data.shapenet_version,
                                    k_eig=args.dfn_model.k_eig,
                                    overfit=args.debug.overfit,
                                    overfit_size=args.debug.overfit_size,
                                    op_cache_dir=output_path,
                                    synsets=args.data.synsets,
                                    split_file=args.data.split_file,
                                    splits=args.data.splits,
                                    max_verts=args.data.max_verts,
                                    )
            shapenet_flag = True

        
        # if accelerator is not None:
        #     if accelerator.is_main_process:
        #         print("Model Configuration: ")
        #         print("C_in: {}".format(args.dfn_model.C_in))
        #         print("C_out: {}".format(args.dfn_model.C_out))
        #         print("C_width: {}".format(args.dfn_model.C_width))
        #         print("N_block: {}".format(args.dfn_model.N_block))
        #         print("outputs_at: {}".format(args.dfn_model.outputs_at))
        #         print("mlp_hidden_dims: {}".format(args.dfn_model.mlp_hidden_dims))
        #         print("dropout: {}".format(args.dfn_model.dropout))
        #         print("with_gradient_features: {}".format(args.dfn_model.with_gradient_features))
        #         print("with_gradient_rotations: {}".format(args.dfn_model.with_gradient_rotations))
        #         print("diffusion_method: {}".format(args.dfn_model.diffusion_method))
        #         print("num_groups: {}".format(args.dfn_model.num_groups))
        #         print("checkpointing: {}".format(args.dfn_model.checkpointing))
        #         print("normalization_type: {}".format(args.dfn_model.normalization_type))
        #         print("time_embedding_norm: {}".format(args.dfn_model.time_embedding_norm))
        
        if shapenet_flag:
            num_channels = data.max_verts
            print(f"The shapenet dataset subset have been padded to the number of vertices:{num_channels}")
            print(f"Number of group in the model:{args.dfn_model.num_groups}")
        else:
            num_channels = data.num_vertices
        model = DiffusionNet(C_in=args.dfn_model.C_in,
                             C_out=args.dfn_model.C_out,
                             num_channels=num_channels,
                             C_width=args.dfn_model.C_width,
                             N_block=args.dfn_model.N_block,
                             outputs_at=args.dfn_model.outputs_at,
                             mlp_hidden_dims=args.dfn_model.mlp_hidden_dims,
                             dropout=args.dfn_model.dropout,
                             with_gradient_features=args.dfn_model.with_gradient_features,
                             with_gradient_rotations=args.dfn_model.with_gradient_rotations,
                             diffusion_method=args.dfn_model.diffusion_method,
                             num_groups=args.dfn_model.num_groups,
                             checkpointing=args.dfn_model.checkpointing,
                             normalization_type = args.dfn_model.normalization_type,
                             time_embedding_norm=args.dfn_model.time_embedding_norm,
                             shapenet_flag = shapenet_flag) # dfn: DiffusioNet
        
    else: 
        raise NotImplementedError
    
    
    return data, model