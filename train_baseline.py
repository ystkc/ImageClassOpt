'''训练'''
import os
import torch
from torchmetrics import Accuracy
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, models
from torchvision.transforms import v2
from omegaconf import OmegaConf

from dataclasses import dataclass
@dataclass
class Config:
    exp_name: str
    split_ratio: list = None # [discard, train, test]
    batch_size: int = 256
    epoch: int = 5
    lr: float = 9e-4 # 32=1e-4
    
    num_workers: int = 7
    persistent_workers: bool = False

    pin_memory: bool = False
    non_blocking: bool = False

    cuda_if: bool = False

    test_enable: bool = False
    use_profiler: bool = False

cfg: Config = OmegaConf.load("train.yaml")

EPOCH = cfg.epoch
BATCH_SIZE = cfg.batch_size
LR = cfg.lr

TEST_ENABLE = cfg.test_enable
SPLIT_RATIO = cfg.split_ratio

USE_PROFILER = cfg.use_profiler

os.chdir(os.path.dirname(__file__))

if __name__ == '__main__':
    print("load image")
    transform = v2.Compose([
      v2.Resize((224, 224)),
      v2.PILToTensor(),                         # uint8, [0, 255]
      v2.ConvertImageDtype(torch.float32),      # float32, 自动变为 [0, 1]
      v2.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225],
      ),
    ])
    train_ds = datasets.ImageFolder("./data/images/train_sf", transform=transform)

    print('split dataset')
    _, traindf, testdf = random_split(train_ds, SPLIT_RATIO, generator=torch.Generator().manual_seed(42))
    print(len(traindf), len(testdf))
    train_loader = DataLoader(
        traindf,
        batch_size=BATCH_SIZE,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers
    )
    test_loader = DataLoader(
        testdf,
        batch_size=BATCH_SIZE,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers
    )


    print("load model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    classcnt = len(train_ds.classes)
    model.fc = nn.Linear(model.fc.in_features, classcnt)

    model = model.to(device=device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("train model")
    if USE_PROFILER:
        prof_train = torch.profiler.profile(acc_events=True)
        prof_test = torch.profiler.profile(acc_events=True)

    for epoch in range(EPOCH):
        model.train()
        acc_loss = torch.tensor(0.0).to(device)
        print(len(train_loader))
        
        if USE_PROFILER:
            prof_train.start()
        start_time = time.time()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=cfg.non_blocking)
            labels = labels.to(device, non_blocking=cfg.non_blocking)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            acc_loss += loss
        if USE_PROFILER:
            prof_train.stop()
        print(f"Epoch {epoch+1}/{EPOCH}, Loss: {acc_loss.item()/len(train_loader):.4f} Time: {time.time() - start_time:.4f}")

        if not TEST_ENABLE:
            continue

        print("test model...", end='')
        model.eval()
        accuracy = Accuracy(task='multiclass', num_classes=classcnt).to(device)
        if USE_PROFILER:
            prof_test.start()
        start_time = time.time()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=cfg.non_blocking)
                labels = labels.to(device, non_blocking=cfg.non_blocking)
                outputs = model(images)
                accuracy(outputs, labels)
            accuracy = accuracy.compute()
        if USE_PROFILER:
            prof_test.stop()
        print(f"    Test Accuracy: {accuracy:.4f} Time: {time.time() - start_time:.4f}")
        
        
    if USE_PROFILER:
        print("Train Profiler:")
        print(prof_train.key_averages().table(sort_by="cpu_time_total"))
        print(prof_train.key_averages().table(sort_by="cuda_time_total"))
        if TEST_ENABLE:
          print("Test Profiler:")
          print(prof_test.key_averages().table(sort_by="cpu_time_total"))
          print(prof_test.key_averages().table(sort_by="cuda_time_total"))
        
    torch.save(model.state_dict(), f"exp/{cfg.exp_name}/model.pth")
