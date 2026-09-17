"""Train an image classifier from an OmegaConf YAML configuration."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import datasets, models
from torchvision.transforms import v2

from CUDAImageFolder import CUDAImageFolder




def transform_from(cfg):
    return v2.Compose([
        v2.Resize(tuple(cfg.image_size), antialias=False), v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(cfg.normalize.mean), std=list(cfg.normalize.std)),
    ])


def train(cfg):
    out = root_path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cuda_ds = bool(cfg.data.cuda_image_folder)
    workers = 0 if cuda_ds else int(cfg.data.num_workers)
    pin_memory = False if cuda_ds else bool(cfg.data.pin_memory)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("load image")
    transform = transform_from(cfg)
    data_dir = root_path(cfg.data.train_dir)
    if cuda_ds:
        dataset = CUDAImageFolder(str(data_dir), pre_transform=transform, to_cuda=True)
    else:
        dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    with (out / cfg.output.class_map).open("w", encoding="utf-8") as file:
        json.dump(dataset.classes, file, ensure_ascii=False, indent=4)

    print("split dataset")
    _, train_set, test_set = random_split(
        dataset, list(cfg.data.split_ratio),
        generator=torch.Generator().manual_seed(int(cfg.seed)))
    print(len(train_set), len(test_set))
    if cuda_ds:
        dataset.preprocess(train_set.indices + test_set.indices)
    common = dict(batch_size=int(cfg.train.batch_size), num_workers=workers,
                  pin_memory=pin_memory, persistent_workers=workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)

    print("load model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = models.ResNet50_Weights.DEFAULT if cfg.model.pretrained else None
    model = models.resnet50(weights=weights)
    class_count = len(dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    model = model.to(device=device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(cfg.train.learning_rate))
    scaler = GradScaler(device, enabled=bool(cfg.runtime.autocast))
    profiler = torch.profiler.profile() if cfg.runtime.profiler else None
    losses, times, accuracies = [], [], []

    print("train model")
    for epoch in range(int(cfg.train.epochs)):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        if profiler:
            profiler.start()
        started = time.time()
        for images, labels in train_loader:
            if not cuda_ds:
                images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device, enabled=bool(cfg.runtime.autocast)):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach()
        elapsed = time.time() - started
        if profiler:
            profiler.stop()
        mean_loss = total_loss.item() / len(train_loader)
        print(f"Epoch {epoch + 1}/{cfg.train.epochs}, Loss: {mean_loss:.4f}")
        losses.append(mean_loss)
        times.append(elapsed)
        accuracy_value = 0.0
        if cfg.train.evaluate:
            model.eval()
            metric = Accuracy(task="multiclass", num_classes=class_count).to(device)
            with torch.no_grad():
                for images, labels in test_loader:
                    if not cuda_ds:
                        images, labels = images.to(device), labels.to(device)
                    with autocast(device, enabled=bool(cfg.runtime.autocast)):
                        metric(model(images), labels)
            accuracy_value = metric.compute().item()
            print(f"Test Accuracy: {accuracy_value:.4f}")
        accuracies.append(accuracy_value)

    if profiler:
        print(profiler.key_averages().table())
    torch.save(model.state_dict(), out / cfg.output.model)
    with (out / cfg.output.metrics).open("w", encoding="utf-8") as file:
        file.write(f"epochs={cfg.train.epochs},batch_size={cfg.train.batch_size},num_workers={workers}\n")
        for i, values in enumerate(zip(losses, times, accuracies), 1):
            loss, elapsed, accuracy = values
            file.write(f"Epoch {i}/{cfg.train.epochs}, Loss: {loss:.4f}, Time: {elapsed:.4f}, Accuracy: {accuracy:.4f}\n")


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--output-dir")
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(root_path(args.config)), OmegaConf.from_dotlist(overrides))
    if args.output_dir:
        cfg.output_dir = args.output_dir
    return cfg


if __name__ == "__main__":
    print("start training...")
    with open("stop.txt", "w+") as file:
        file.write("stop training")
    raise ValueError("stop training")
    exit()
    train(load_config())
