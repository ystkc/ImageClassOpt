'''推理'''
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import json
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision.transforms import v2
import torch.nn as nn
from torch.amp import autocast
from ImageTestDataset import ImageTestDataset
# 输出测试结果
model_name = 'maptest.pth'
CUDA_IF = True
TEST_SIZE = [1] # split from test dataset
AUTOCAST = True
BATCH_SIZE = 128
USE_PROFILER = False
transform = v2.Compose([
    v2.Resize((224, 224), antialias=False),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])  
device = "cuda" if torch.cuda.is_available() else "cpu"

print('load map')
model = models.resnet50()
tag_map = json.load(open("map.json", "r", encoding="utf-8"))
model.fc = nn.Linear(model.fc.in_features, len(tag_map))
print('load model')
model.load_state_dict(torch.load(model_name, map_location=device))
model = model.to(device) # 静态模型


print('load dataset')
ds = ImageTestDataset(f"images/test_sf/unknown", transform=transform, to_cuda=CUDA_IF, split_ratio=TEST_SIZE)
split_sampler = ds.get_split_sampler()
print(f"test size: {len(ds)}")
test_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, sampler=split_sampler[0])

print('test')
outputgather = []
if USE_PROFILER:
    profiler = torch.profiler.profile()
    profiler.start()
with torch.no_grad():
    model.eval()
    for images, id in test_loader:
        if not CUDA_IF:
            images = images.to(device) # CUDA_IF是True时，数据已经离线预处理并放入GPU
        if AUTOCAST:
            with autocast(device):
                outputs = model(images)
        outputgather.append((id, outputs))
        # print(id, tag_map[torch.argmax(outputs)])
if USE_PROFILER:
    profiler.stop()
    print(profiler.key_averages().table())


# save to csv(id, argmax1)
print('save to csv')
ids = []
results = []
for id, outputs in outputgather:
    maxarg = torch.argmax(outputs, dim=1)
    ids.extend(list(id))
    results.extend([tag_map[i] for i in maxarg])
df = pd.DataFrame({"id": ids, "label": results})
df.to_csv("results.csv", index=False)
