import plotly.graph_objects as go
import plotly.express as px
import requests as req
import pathlib as pb
import pywavefront
import typing as t
import functools
import einops
import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset
from torch import Tensor
import imageio.v3 as iio
import numpy as np
import pyvista as pv
from pytorch3d.io import load_objs_as_meshes
from .ops import ManifoldSampler, PointsOnManifold


class DataManager:
    def __init__(
        self,
        data_dir,
        cache_dir,
        overfit=False,
    ) -> None:
        self.cache_dir = cache_dir if isinstance(cache_dir, pb.Path) else pb.Path(cache_dir)
        self.managers = {
            # 'obj': ObjectManager(data_dir='/ibex/ai/home/liz0l/projects/codes/ddpm_diffusionNet/datasets/objects/manifold_example/manifold_processed_bunny.obj', cache_dir=self.cache_dir),
            'obj': ObjectManager(data_dir='/ibex/ai/home/liz0l/projects/codes/ddpm_diffusionNet/datasets/stanford-bunny.obj', cache_dir=self.cache_dir),
            
            # 'weather': ImageDataset(data_dir=self.data_dir / 'images' / 'weather'),
            # 'mnist': MNISTImageDataset(data_dir=self.data_dir / 'images' / 'mnist'),
            'celeba': ImageDataset(data_dir='/ibex/ai/home/liz0l/projects/codes/ddpm_diffusionNet/datasets/images/celeba_hq_train', overfit=overfit)}

        if not self.cache_dir.exists():
            self.cache_dir.mkdir()

    @property
    def objects(self) -> 'ObjectManager':
        return self.managers['obj']

    @property
    def celeba(self) -> 'ImageDataset':
        return self.managers['celeba']

    def dataset(self, manifold: str, signal: str, n_samples: int, k_evecs: int, device: torch.device):
        return ManifoldDataset(self.objects[manifold], self.managers[signal], n_samples, k_evecs, device)


class ImageEntry(t.TypedDict):
    data: np.ndarray
    label: int
    path: str
    
class MNISTImageDataset(Dataset):
    def __init__(
        self,
        data_dir
    ) -> None:
        self.data_dir = data_dir if isinstance(data_dir, pb.Path) else pb.Path(data_dir)
        self.dataset = MNIST(str(data_dir), download=True, train=True, transform=T.Compose([
            T.ToImage(),
            T.ToDtype(dtype=torch.uint8, scale=True),
            T.Resize(size=(128, 128)),
        ]))

    def __getitem__(self, index: int) -> ImageEntry:
        entry = self.dataset[index]
        return {
            'data': entry[0].numpy(),
            'label': entry[1],
            'path': 'n/a',
        }

    def __len__(self) -> int:
        return len(self.dataset)


class DataEntry(t.TypedDict):
    path: pb.Path
    label: int


class ImageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        overfit=False,
    ) -> None:
        self.overfit = overfit
        
        self.data_dir = data_dir if isinstance(data_dir, pb.Path) else pb.Path(data_dir)

        if not self.data_dir.exists():
            raise RuntimeError('images folder is non-existant: {}'.format(self.data_dir))

        index = -1
        self.label_map: t.Dict[int, str] = {}
        self.image_data: t.Dict[int, DataEntry] = {}
        self.transform = T.Compose([
            # T.ToImage(),
            T.ToImageTensor(),
            # T.ToDtype(dtype=torch.uint8, scale=True),
            T.ConvertDtype(dtype=torch.uint8),
            T.Resize(size=(512, 512), antialias=True),
        ])

        for i, image_dirpath in enumerate(sorted(self.data_dir.iterdir(), key=lambda x: x.stem)):
            # if i not in self.label_map:
            #     self.label_map[i] = image_dirpath.stem
            # for j, image_path in enumerate(sorted(image_dirpath.iterdir(), key=lambda x: x.stem)):
            #     index += 1
            #     self.image_data[index] = {
            #         'path': image_path,
            #         'label': i,
            #     }
            self.image_data[i] = {
                'path': image_dirpath,
                'label': i}

            if self.overfit:
                break

    def __getitem__(self, index: int) -> ImageEntry:
        entry = self.image_data[index]

        try:
            data = torch.from_numpy(iio.imread(entry['path'], mode='RGB'))
            data = einops.rearrange(data, 'H W C -> C H W')
            data = self.transform(data)
            data = einops.rearrange(data, 'C H W -> H W C')
        except BaseException as e:
            raise Exception(entry['path'])

        return {
            'data': data.numpy(),
            'label': entry['label'],
            'path': str(entry['path']),
        }

    def __len__(self) -> int:
        return len(self.image_data)


class ObjectManager:
    def __init__(
            self,
            data_dir,
            cache_dir,
        ) -> None:
            # Source for objects: https://github.com/alecjacobson/common-3d-test-models
            self.common_3d_objects_url = 'https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data'
            self.cache_dir = cache_dir if isinstance(cache_dir, pb.Path) else pb.Path(data_dir)
            self.data_dir = data_dir if isinstance(data_dir, pb.Path) else pb.Path(data_dir)

    def __getitem__(self, name: str) -> 'ObjectInstance':
        return ObjectInstance(self.data_dir)

    def __getattr__(self, name: str) -> 'ObjectInstance':
        return self[name]

    def download(self, name: str) -> None:
        return


class ObjectInstance:
    def __init__(self, path: pb.Path) -> None:
        if not path.exists():
            raise RuntimeError('Attempting to retrieve a non-existing object: {}'.format(path))
        self.path = path
        self.name = self.path.stem
        self.vista = pv.read(self.path)
        self.mesh = load_objs_as_meshes(files=[self.path], load_textures=True, device='cpu')
        self.faces = torch.tensor(self.vista.faces.reshape((-1, 4)))
        self.verts = torch.tensor(self.vista.points)


class ManifoldDataset(Dataset):
    def __init__(
        self,
        manifold: ObjectInstance,
        signal: ImageDataset,
        n_samples: int,
        k_evecs: int,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        super(ManifoldDataset, self).__init__()
        assert n_samples > 0, 'need at least one sample'
        self.n_samples = n_samples

        # Input data source
        self.manifold: ObjectInstance = manifold
        # Output data source
        self.signal: ImageDataset = signal
        # Point sampler over the input data source
        self.sampler: ManifoldSampler = ManifoldSampler(self.manifold.verts, self.manifold.faces, k_evecs, device)

    def __getitem__(self, index: int) -> PointsOnManifold:
        # Retrieve one image as the output domain of the function
        signal: ImageEntry = self.signal[index]

        # Sample points on the given manifold and embed them using eigenfunctions
        return self.sampler.sample(n=self.n_samples, image=torch.tensor(signal['data']))

    def __len__(self) -> int:
        return len(self.signal)
