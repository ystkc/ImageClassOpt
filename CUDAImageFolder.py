'''实验模块：重写了torchvision.datasets.folder.DatasetFolder，用于将数据离线加载到显存中，可能能够加速训练，适用于小数据集'''
from typing import Callable, Optional, Any, Union
from pathlib import Path
from torchvision import datasets
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torchvision.io import read_image
import torch

class CUDAImageFolder(datasets.DatasetFolder):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = read_image,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        pre_transform: Optional[Callable] = None,
        to_cuda: bool = True,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
        self.loader = loader
        self.pre_transform = pre_transform
        self.to_cuda = to_cuda
    
    def preprocess(self, indices=None, memory_format=torch.channels_last):
        data_tensors = []
        target_tensors = []
        self.idx_map = [0] * len(self.samples)
        if indices is None:
            indices = range(len(self.samples))
        for i, idx in enumerate(indices):
            path, target = self.samples[idx]
            sample = self.loader(path)
            if self.pre_transform:
                sample = self.pre_transform(sample)
            data_tensors.append(sample)
            target_tensors.append(target)
            self.idx_map[idx] = i
        
        # move to GPU
        device = "cuda" if self.to_cuda and torch.cuda.is_available() else "cpu"
        self.data = torch.stack(data_tensors).to(device=device, non_blocking=True, memory_format=memory_format)
        self.targets = torch.tensor(target_tensors).to(device=device, non_blocking=True)
    
    def __getitem__(self, idx):
        idx = self.idx_map[idx]
        data, target = self.data[idx], self.targets[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target
    
    def __len__(self):
        return len(self.samples)
