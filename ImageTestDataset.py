'''实验模块：重写了torch.utils.data.Dataset，用于将数据离线加载到显存中，可能能够加速推理，适用于小数据集'''
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import hashlib
import random
import os
import hashlib
print(f"PID: {os.getpid()}, initializing dataset")

def hash_tensor(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    # 包含shape和dtype信息
    h = hashlib.sha256()
    h.update(str(t.shape).encode())
    h.update(str(t.dtype).encode())
    h.update(t.numpy(force=True).tobytes())
    return h.hexdigest()

class ImageTestDataset(Dataset):
    def __init__(self, img_dir, transform=None, to_cuda=False, split_ratio: list=[1.0], seed=42):
        '''to_cuda: 将split_ratio中包括的数据全部读取、离线变换并加载到cuda中。如果False则只split并在线变换'''
        self.transform = transform
        # 获取所有图片路径
        self.img_paths = []
        self.filenames = []
        for f in os.listdir(img_dir):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                self.img_paths.append(os.path.join(img_dir, f))
        random.seed(seed)
        random.shuffle(self.img_paths) # 和允许小于1，也就是舍弃一些数据
        for p in self.img_paths:
            self.filenames.append(os.path.basename(p).split('.')[0])
        self.len = 0
        total_len = len(self.img_paths)
        self.split_sampler = []
        self.images = []
        self.to_cuda = to_cuda
        for ratio in split_ratio:
            subset_len = int(total_len * ratio)
            self.len += subset_len
            self.split_sampler.append((range(self.len - subset_len, self.len)))
            if to_cuda:
                subset_paths = self.img_paths[self.len - subset_len: self.len]
                for subidx, img_path in enumerate(subset_paths):
                    # print(img_path)
                    image = read_image(img_path)
                    # if img_path == 'images/train_exp/ns26_1aa163c9bf524b3b33d48e77018d5efc.jpg':
                        # print(img_path, hash_tensor(image))
                    if transform:
                        image = transform(image)
                    self.images.append(image)
                    # print(self.len - subset_len + subidx, 'was filled', self.img_paths[self.len - subset_len + subidx], self.filenames[self.len - subset_len + subidx])
                    # if img_path == 'images/train_exp/ns26_1aa163c9bf524b3b33d48e77018d5efc.jpg':
                        # print(len(self.images)-1)
                        # print(subset_paths.index(img_path))
                        # for ii in range(len(self.images)):
                        #     img, filename = self[ii]
                        #     if img is None:
                        #         continue
                            # print(ii, hash_tensor(img), filename)
                # for ii in range(len(self.images)):
                #     img, filename = self[ii]
                #     if img is None:
                #         continue
                #     print(ii, hash_tensor(img), filename)
                self.images = torch.stack(self.images, dim=0).to('cuda', non_blocking=True)
        
    def get_split_sampler(self):
        return self.split_sampler
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        if not self.to_cuda:
            img_path = self.img_paths[idx]
            image = read_image(img_path)
            if self.transform:
                image = self.transform(image)
        else:
            image = self.images[idx]
        
        
        # 返回图像 + 文件名（不含路径）
        return image, filename
