'''训练'''
import json
import os
import torch
from torchmetrics import Accuracy
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, models
from torchvision.transforms import v2
from CUDAImageFolder import CUDAImageFolder
from torch.amp import autocast, GradScaler

EPOCH = 4
# 32=1e-4
BATCH_SIZE = 96
LR = 9e-4
PIN_MEM = True
NUM_WORKERS = 6
AUTOCAST = False

NEED_TEST = False
SPLIT_RATIO = [0.9, 0.08, 0.02] # [discard, train, test]

# 注意！由单进程CUDAImageFolder创建GPU张量时不允许使用多进程、PinMemory
CUDAIF = True
if CUDAIF:
    PIN_MEM = False
    NUM_WORKERS = 0

USE_PROFILER = False

os.chdir(os.path.dirname(__file__))

if __name__ == '__main__':
    print("load image")
    transform = v2.Compose([
        v2.Resize((224, 224), antialias=False),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])  
    train_ds = None
    if CUDAIF:
        train_ds = CUDAImageFolder("./data/images/train_sf", pre_transform=transform, to_cuda=True)
    else:
        train_ds = datasets.ImageFolder("./data/images/train_sf", transform=transform)
    # output train_ds.classes
    # json.dump(train_ds.classes, open("map.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)

    print('split dataset')
    _, traindf, testdf = random_split(train_ds, SPLIT_RATIO, generator=torch.Generator().manual_seed(42))
    print(len(traindf), len(testdf))
    if CUDAIF:
        train_ds.preprocess(traindf.indices + testdf.indices)
    train_loader = DataLoader(
        traindf,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=NUM_WORKERS > 0
    )
    test_loader = DataLoader(
        testdf,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=NUM_WORKERS > 0
    )


    print("load model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    classcnt = len(train_ds.classes)
    model.fc = nn.Linear(model.fc.in_features, classcnt)

    model = model.to(device=device, memory_format=torch.channels_last)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    

    testname = input("测试名：")

    print("train model")
    loss_track = []
    time_track = []
    accuracy_track = []
    if USE_PROFILER:
        prof = torch.profiler.profile(acc_events=True)

    scaler = GradScaler()

    for epoch in range(EPOCH):
        model.train()
        acc_loss = torch.tensor(0.0, device=device)
        print(len(train_loader))
        
        if USE_PROFILER:
            prof.start()
        start_time = time.time()
        for images, labels in train_loader:
            if not CUDAIF:
                images = images.to(device)
                labels = labels.to(device)
            optimizer.zero_grad()
            if AUTOCAST:
                with autocast(device):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            acc_loss += loss
        time_track.append(time.time() - start_time)
        if USE_PROFILER:
            prof.stop()
        print(f"Epoch {epoch+1}/{EPOCH}, Loss: {acc_loss/len(train_loader):.4f}")
        loss_track.append(acc_loss.item()/len(train_loader))

        if not NEED_TEST:
            accuracy_track.append(0.0)
            continue

        print("test model...", end='')
        model.eval()
        accuracy = Accuracy(task='multiclass', num_classes=classcnt).to(device)
        with torch.no_grad():
            for images, labels in test_loader:
                if not CUDAIF:
                    images = images.to(device)
                    labels = labels.to(device)
                if AUTOCAST:
                    with autocast(device):
                        outputs = model(images)
                accuracy(outputs, labels)
            accuracy = accuracy.compute()
            print(f"Test Accuracy: {accuracy:.4f}")
            accuracy_track.append(accuracy.item())
    if USE_PROFILER:
        print(prof.key_averages().table())
        
    torch.save(model.state_dict(), f"{testname}.pth")
    # 向记录文档中追加本次的结果
    with open("record.txt", "a", encoding="utf-8") as f:
        f.write(f"{testname},epoch={EPOCH},batch_size={BATCH_SIZE},num_workers={NUM_WORKERS}\n")
        for i in range(EPOCH):
            f.write(f"Epoch {i+1}/{EPOCH}, Loss: {loss_track[i]:.4f}, Time: {time_track[i]:.4f}, Accuracy: {accuracy_track[i]:.4f}\n")
