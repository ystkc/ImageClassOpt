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
    epoch: int = 5
    batch_size: int = 256
    lr: float = 9e-4 # 32=1e-4

    test_enable: bool = False
    split_ratio: list = None # [discard, train, test]

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
        v2.ToImagePIL(),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    train_ds = datasets.ImageFolder("./data/images/train_sf", transform=transform)

    print('split dataset')
    _, traindf, testdf = random_split(train_ds, SPLIT_RATIO, generator=torch.Generator().manual_seed(42))
    print(len(traindf), len(testdf))
    train_loader = DataLoader(
        traindf,
        batch_size=BATCH_SIZE,
    )
    test_loader = DataLoader(
        testdf,
        batch_size=BATCH_SIZE,
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
        acc_loss = 0
        print(len(train_loader))
        
        if USE_PROFILER:
            prof_train.start()
        start_time = time.time()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            acc_loss += loss.item()
        if USE_PROFILER:
            prof_train.stop()
        print(f"Epoch {epoch+1}/{EPOCH}, Loss: {acc_loss/len(train_loader):.4f} Time: {time.time() - start_time:.4f}")

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
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                accuracy(outputs, labels)
            accuracy = accuracy.compute()
        if USE_PROFILER:
            prof_test.stop()
        print(f"    Test Accuracy: {accuracy:.4f} Time: {time.time() - start_time:.4f}")
        
        
    if USE_PROFILER:
        print("Train Profiler:")
        print(prof_train.key_averages().table())
        if TEST_ENABLE:
          print("Test Profiler:")
          print(prof_test.key_averages().table())
        
    torch.save(model.state_dict(), f"{cfg.exp_name}.pth")
