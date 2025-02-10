import argparse

class NestedNamespace(argparse.Namespace):
    def __setattr__(self, name, value):
        if '.' in name:
            parts = name.split('.')
            obj = self
            for part in parts[:-1]:
                if not hasattr(obj, part):
                    super(NestedNamespace, obj).__setattr__(part, NestedNamespace())
                obj = getattr(obj, part)
            super(NestedNamespace, obj).__setattr__(parts[-1], value)
        else:
            super(NestedNamespace, self).__setattr__(name, value)

class StoreNestedAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        attr = self.dest
        if '.' in attr:
            parts = attr.split('.')
            obj = namespace
            for part in parts[:-1]:
                if not hasattr(obj, part):
                    setattr(obj, part, NestedNamespace())
                obj = getattr(obj, part)
            setattr(obj, parts[-1], values)
        else:
            setattr(namespace, attr, values)

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 'yes', '1'):
        return True
    elif value.lower() in ('false', 'no', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_train_args(train_parser: argparse._ActionsContainer):
    '''
    Configurations for training.
    '''
    train_parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        dest='train.seed',
        action=StoreNestedAction,
        help="A seed for reproducible training."
    )
    # CUDA_VISIBLE_DEVICES
    train_parser.add_argument(
        "--visible_devices",
        type=str,
        default='2,3,4,5,6,7',
        dest='train.visible_devices',
        action=StoreNestedAction,
        help="Config the visible devices."
    )
    train_parser.add_argument(
        "--train_batch_size",
        type=int,
        default=128,
        dest='train.train_batch_size',
        action=StoreNestedAction,
        help="Batch size (per device) for the training dataloader."
    )
    train_parser.add_argument(
        "--train_num_workers",
        type=int,
        default=4,
        dest='train.train_num_workers',
        action=StoreNestedAction,
        help="-"
    )
    train_parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=40,
        dest='train.num_train_epochs',
        action=StoreNestedAction,
        help="Number of training epochs."
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        dest='train.lr',
        action=StoreNestedAction,
        help="Starting learning rate."
    )
    train_parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        dest='train.lr_warmup_steps',
        action=StoreNestedAction,
        help="Number of steps to warm up the learning rates."
    )
    train_parser.add_argument(
        "--save_epoch_ckpt",
        # action="store_true",
        type=str_to_bool,
        default=True,
        dest='train.save_epoch_ckpt',
        help="Flag to save the checkpoint after each epoch."
    )
    train_parser.add_argument(
        "--vis_epoch",
        type=str_to_bool,
        default=True,
        # action="store_true",
        dest='train.vis_epoch',
        help="Flag for visualization after each epoch."
    )
    train_parser.add_argument(
        "--evaluation_epoch",
        type=str_to_bool,
        default=False,
        # action="store_true",
        dest='train.evaluation_epoch',
        help="Flag for qualitative evaluation after each epoch."
    )
    train_parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        dest='train.checkpointing_steps',
        action=StoreNestedAction,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via '--resume_from_checkpoint'. "
            "In case the checkpoint is better than the final trained model, it can also be used for inference. "
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
        )
    )
    train_parser.add_argument(
        "--visualization_steps",
        type=int,
        default=10000,
        dest='train.visualization_steps',
        action=StoreNestedAction,
        help="Number of steps for rendering and saving the images."
    )
    train_parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=10000,
        dest='train.evaluation_steps',
        action=StoreNestedAction,
        help="Number of steps for evaluation."
    )
    train_parser.add_argument(
        "--pretrain",
        type=str_to_bool,
        default=False,
        # action="store_true",
        dest='train.pretrain',
        action=StoreNestedAction,
        help="Flag indicating if pretrained models are used."
    )
    train_parser.add_argument(
        "--save_interval",
        type=int,
        default=2,
        dest='train.save_interval',
        action=StoreNestedAction,
    )
    return train_parser



def add_data_args(data_parser: argparse._ActionsContainer):
    '''
    Configurations for data and preprocessing.
    '''
    # data_parser.add_argument(
    #     '--object_path',
    #     type=str,
    #     default='datasets/objects/manifold_example/manifold_processed_bunny.obj',
    #     dest='data.object_path',
    #     action=StoreNestedAction,
    #     help="Relative path to the object for dataset preparations."
    # )
    data_parser.add_argument(
        '--image_path',
        type=str,
        default='datasets/images/celeba_hq',
        dest='data.image_path',
        action=StoreNestedAction,
        help='Relative path to the images dataset.'
    )
    data_parser.add_argument(
        '--preprocessed_data',
        type=str,
        default='datasets/preprocessed/op_cache',
        dest='data.preprocessed_data',
        action=StoreNestedAction,
        help='Relative path to the preprocessed data.'
    )
    data_parser.add_argument(
        '--obj_cat',
        type=str,
        default='bunny',
        dest='data.obj_cat',
        action=StoreNestedAction,
        help='Category of the object.'
    )
    data_parser.add_argument(
        '--precision_level',
        choices=['simple', 'median', 'complex', 'manifold'],
        default='simple',
        dest='data.precision_level',
        action=StoreNestedAction,
        help="Select the precision level of the object to be processed."
    )
    
    data_parser.add_argument(
        '--shapenet_version',
        type=int,
        choices=[1,2],
        default=2,
        dest='data.shapenet_version',
        action=StoreNestedAction,
        help="ShapenetCore version, choose from 1 or 2."
    )
    data_parser.add_argument(
        '--shapenet_path',
        type=str,
        default="datasets/objects/shapenetCore",
        dest='data.shapenet_path',
        action=StoreNestedAction,
        help="Path to the ShapenetCore dataset."
    )
    
    data_parser.add_argument(
        '--synsets',
        nargs='*',
        default=None,
        type=str,
        dest='data.synsets',
        action=StoreNestedAction,
        help="Path to the ShapenetCore dataset (can be a list of synsets or None)."
    )
    
    data_parser.add_argument(
        '--split_file',
        default='./splits/train_set.txt',
        dest='data.split_file',
        action=StoreNestedAction,
        help="Select the split file"
    )
    
    data_parser.add_argument(
        '--splits',
        default='train',
        choices=['train', 'val', 'test'],
        dest='data.splits',
        action=StoreNestedAction,
        help="train, val, test"
    )
    
    
    data_parser.add_argument(
        '--max_verts',
        default=60000,
        type=int,
        dest='data.max_verts',
        action=StoreNestedAction,
        help="Predefined maximum number of vertices to be handle by the network."
    )
    
    
    
    return data_parser

def add_dfn_model_args(dfn_model_parser: argparse._ActionsContainer):
    '''
    Configurations for DiffusionNet.
    '''
    dfn_model_parser.add_argument(
        "--input_features",
        choices=['rgb', 'hks', 'xyz'],
        default='rgb',
        dest='dfn_model.input_features',
        action=StoreNestedAction,
        help='Input features for the model.'
    )
    dfn_model_parser.add_argument(
        "--C_in",
        type=int,
        default=3,
        dest='dfn_model.C_in',
        action=StoreNestedAction,
        help='Input feature dimension.'
    )
    dfn_model_parser.add_argument(
        "--C_out",
        type=int,
        default=3,
        dest='dfn_model.C_out',
        action=StoreNestedAction,
        help='Output feature dimension.'
    )
    dfn_model_parser.add_argument(
        '--C_width',
        type=int,
        default=128,
        dest='dfn_model.C_width',
        action=StoreNestedAction,
        help="Dimension of the projected feature."
    )
    dfn_model_parser.add_argument(
        "--N_block",
        type=int,
        default=4,
        dest='dfn_model.N_block',
        action=StoreNestedAction,
        help="Number of DiffusionNet blocks."
    )
    dfn_model_parser.add_argument(
        "--num_groups",
        type=int,
        default=64,
        dest='dfn_model.num_groups',
        action=StoreNestedAction,
        help="Number of groups for the group normalization."
    )
    dfn_model_parser.add_argument(
        '--outputs_at',
        choices=['vertices', 'edges', 'faces'],
        default='vertices',
        dest='dfn_model.outputs_at',
        action=StoreNestedAction,
        help='Return values on either vertices, edges, or faces.'
    )
    dfn_model_parser.add_argument(
        '--mlp_hidden_dims',
        type=int,
        nargs='+',
        default=[256, 256],
        dest='dfn_model.mlp_hidden_dims',
        action=StoreNestedAction,
        help='List of hidden layer sizes for MLPs.'
    )
    dfn_model_parser.add_argument(
        '--with_gradient_features',
        type=str_to_bool,
        default=True,
        # action="store_true",
        dest='dfn_model.with_gradient_features',
        action=StoreNestedAction,
        help='If True, use gradient features.'
    )
    dfn_model_parser.add_argument(
        '--with_gradient_rotations',
        type=str_to_bool,
        default=True,
        # action="store_true",
        dest='dfn_model.with_gradient_rotations',
        action=StoreNestedAction,
        help='If True, learn a rotation of each gradient.'
    )
    dfn_model_parser.add_argument(
        '--diffusion_method',
        choices=['spectral', 'implicit_dense'],
        default='spectral',
        dest='dfn_model.diffusion_method',
        action=StoreNestedAction,
        help='How to evaluate diffusion.'
    )
    dfn_model_parser.add_argument(
        '--dropout',
        type=str_to_bool,
        default=False,
        # action="store_true",
        dest='dfn_model.dropout',
        help="Flag for dropout."
    )
    dfn_model_parser.add_argument(
        '--checkpointing',
        type=str_to_bool,
        default=False,
        # action="store_true",
        dest='dfn_model.checkpointing',
        help="Flag for checkpointing."
    )
    dfn_model_parser.add_argument(
        "--k_eig",
        type=int,
        default=128,
        dest='dfn_model.k_eig',
        action=StoreNestedAction,
        help="Number of eigenvectors for the spectral solver."
    )
    dfn_model_parser.add_argument(
        "--normalization_type",
        type=str,
        choices=['group', 'layer'],
        default='group',
        dest='dfn_model.normalization_type',
        action=StoreNestedAction,
        help=("Using group normalization or batch normalization. "
              "Batch normalization is useful for larger batch."
              "Group normalization performs better when the batch size is small.")
    )
    dfn_model_parser.add_argument(
        '--time_embedding_norm',
        choices=['default', 'scale_shift'],
        default='default',
        dest='dfn_model.time_embedding_norm',
        action=StoreNestedAction,
    )
    
    

    return dfn_model_parser

def add_diffusion_args(diffusion_parser: argparse._ActionsContainer):
    '''
    Adding configurations for diffusion.
    '''
    diffusion_parser.add_argument(
        '--sampling_steps',
        type=int,
        default=1000,
        dest='diffusion.sampling_steps',
        action=StoreNestedAction,
        help="Sampling steps from XT to X0."
    )
    diffusion_parser.add_argument(
        '--training_time_steps',
        type=int,
        default=1000,
        dest='diffusion.training_time_steps',
        action=StoreNestedAction,
        help="Number of timesteps during training."
    )
    diffusion_parser.add_argument(
        "--mixed_precision",
        choices=['no', 'fp16'],
        default='no',
        dest='diffusion.mixed_precision',
        action=StoreNestedAction,
        help="Model precision."
    )
    diffusion_parser.add_argument(
        "--pipe",
        choices=['ddpm', 'edm'],
        default='ddpm',
        dest='diffusion.pipe_type',
        action=StoreNestedAction,
        help="Pipeline Type."
    )
    return diffusion_parser


def add_exp_args(exp_parser: argparse._ActionsContainer):
    '''
    Configuration for the experiments.
    '''
    exp_parser.add_argument(
        "--wandb",
        type=str_to_bool,
        default=False,
        # action="store_true",
        dest='exp.wandb',
        action=StoreNestedAction,
        help="Flag to use the wandb logging."
    )
    exp_parser.add_argument(
        "--project",
        type=str,
        default="ddpm_diffusionNet",
        dest='exp.project',
        action=StoreNestedAction,
        help="Name of the wandb project."
    )
    exp_parser.add_argument(
        "--group",
        choices=['develop', 'debug', 'train', 'eval', 'new_features', 'shapenet_core'],
        default='develop',
        dest='exp.group',
        action=StoreNestedAction,
        help="Type of the experiment."
    )
    exp_parser.add_argument(
        "--job",
        type=str,
        default="test",
        dest='exp.job',
        action=StoreNestedAction,
        help="Specific job or feature being tested."
    )
    exp_parser.add_argument(
        "--name",
        type=str,
        default="simple_bunny",
        dest='exp.name',
        action=StoreNestedAction,
        help="A memorable name for this experiment."
    )
    exp_parser.add_argument(
        "--description",
        type=str,
        dest='exp.description',
        action=StoreNestedAction,
        help="Description of the experiment."
    )
    exp_parser.add_argument(
        "--output_dir",
        type=str,
        default="experiment",
        dest='exp.output_dir',
        action=StoreNestedAction,
        help="Folder to store all outputs for this experiment."
    )
    exp_parser.add_argument(
        "--num_sample",
        type=int,
        default=4,
        dest='exp.num_sample',
        action=StoreNestedAction,
        help="Folder to store all outputs for this experiment."
    )
    
    
    return exp_parser

def add_accelerate_args(accelerate_parser: argparse._ActionsContainer):
    '''
    Configuration for the accelerator.
    '''
    accelerate_parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        dest='accelerate.gradient_accumulation_steps',
        action=StoreNestedAction,
        help="Number of steps for gradient accumulation during training."
    )
    accelerate_parser.add_argument(
        "--save_path",
        type=str,
        dest='accelerate.save_path',
        action=StoreNestedAction,
        help="Path to save the accelerator state."
    )
    accelerate_parser.add_argument(
        "--save_dir",
        type=str,
        default="accelerator",
        dest='accelerate.save_dir',
        action=StoreNestedAction,
        help="Directory to save checkpoints of the accelerator."
    )
    return accelerate_parser


def add_debug_args(debug_parser:argparse._ActionsContainer):
    debug_parser.add_argument(
        "--overfit",
        type=str_to_bool,
        default=False,
        # action="store_true",
        dest='debug.overfit',
        action=StoreNestedAction,
        help="Turn to True if want to overfit on a smaller datasets."
    )
    debug_parser.add_argument(
        "--overfit_size",
        type=int,
        default=1,
        dest='debug.overfit_size',
        action=StoreNestedAction,
        help="Dataset size for overfitting."
    )
    

def parse_args(input_args=None):
    base_parser = argparse.ArgumentParser(description="3D mesh texture generation training script.")
    base_parser.add_argument(
        '--model_type',
        choices=['diffusion_net', 'mdf', 'mesh_gcn'],
        default='diffusion_net',
        dest='model_type',
        action=StoreNestedAction,
        help="Choose the prediction model."
    )

    base_parser.add_argument(
        '--object_set',
        choices=['bunny', 'shapenet_core'],
        default='bunny',
        dest='object_set',
        action=StoreNestedAction,
        help="Choose the prediction model."
    )
    base_parser.add_argument(
        '--infer_ckp_path',
        default='./',
        dest='infer_ckp_path')
    base_parser.add_argument(
        "--show",
        type=str_to_bool,
        default=False,
        dest='infer_show',)
    base_parser.add_argument(
        "--save",
        type=str_to_bool,
        default=False,
        dest='infer_save',)
    
    # Create argument groups instead of subparsers
    train_group = base_parser.add_argument_group('Training Configurations')
    add_train_args(train_group)

    data_group = base_parser.add_argument_group('Data Configurations')
    add_data_args(data_group)

    dfn_model_group = base_parser.add_argument_group('DiffusionNet Model Configurations')
    add_dfn_model_args(dfn_model_group)

    diffusion_group = base_parser.add_argument_group('Diffusion Configurations')
    add_diffusion_args(diffusion_group)

    exp_group = base_parser.add_argument_group('Experiment Configurations')
    add_exp_args(exp_group)

    acc_group = base_parser.add_argument_group('Accelerator Configurations')
    add_accelerate_args(acc_group)
    
    debug_group = base_parser.add_argument_group('Debugging Configurations')
    add_debug_args(debug_group)
    
    

    # Parse the arguments using the custom namespace
    args = base_parser.parse_args(namespace=NestedNamespace())

    return args

