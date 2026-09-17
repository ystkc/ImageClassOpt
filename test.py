"""Run inference from an OmegaConf YAML configuration."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import v2

from ImageTestDataset import ImageTestDataset

ROOT = Path(__file__).resolve().parent


def root_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def test(cfg):
    out = root_path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = v2.Compose([
        v2.Resize(tuple(cfg.image_size), antialias=False), v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(cfg.normalize.mean), std=list(cfg.normalize.std)),
    ])
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("load map")
    with root_path(cfg.data.class_map).open("r", encoding="utf-8") as file:
        labels = json.load(file)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    print("load model")
    model.load_state_dict(torch.load(root_path(cfg.model.checkpoint), map_location=device))
    model = model.to(device)
    print("load dataset")
    dataset = ImageTestDataset(str(root_path(cfg.data.test_dir)), transform=transform,
                               to_cuda=bool(cfg.runtime.cuda_dataset),
                               split_ratio=list(cfg.data.split_ratio))
    loader = DataLoader(dataset, batch_size=int(cfg.data.batch_size), shuffle=False,
                        sampler=dataset.get_split_sampler()[int(cfg.data.split_index)])
    print(f"test size: {len(dataset)}")
    ids, results = [], []
    print("test")
    model.eval()
    with torch.no_grad():
        for images, batch_ids in loader:
            if not cfg.runtime.cuda_dataset:
                images = images.to(device)
            with autocast(device, enabled=bool(cfg.runtime.autocast)):
                outputs = model(images)
            ids.extend(list(batch_ids))
            results.extend(labels[i] for i in torch.argmax(outputs, dim=1).cpu().tolist())
    result_path = out / cfg.output.csv
    pd.DataFrame({"id": ids, "label": results}).to_csv(result_path, index=False)
    print(f"saved: {result_path}")


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test.yaml")
    parser.add_argument("--output-dir")
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(root_path(args.config)), OmegaConf.from_dotlist(overrides))
    if args.output_dir:
        cfg.output_dir = args.output_dir
    return cfg


if __name__ == "__main__":
    test(load_config())
