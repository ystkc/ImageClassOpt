'''预处理：将data/images中的图片按照csv中的label分类到子文件夹，方便torch的DatasetFolder直接读取'''
import pandas as pd
NAME = 'train'
TEST_NAME = 'test'
# 根据train.csv中的label将图片分类到子文件夹
df = pd.read_csv(f"{NAME}.csv")
labels = set(df["label"])
print(labels)
# create folder
import os
for label in labels:
    os.makedirs(f"data/images{NAME}_sf/{label}", exist_ok=True)
    sub_df = df[df["label"] == label]
    # move files
    cnt = 0
    for idx, row in sub_df.iterrows():
        image_name = row["id"]
        img_path = row["image"]
        try:
            os.rename(img_path, f"data/images{NAME}_sf/{label}/{image_name}.jpg")
        except FileNotFoundError:
            pass
        cnt += 1
        if cnt % 100 == 0:
            print(cnt, len(sub_df))
    print(f"{label} done")

os.makedirs(f"data/images{TEST_NAME}_sf", exist_ok=True)
df = pd.read_csv(f"{TEST_NAME}.csv")
# create folder
os.makedirs(f"data/images{TEST_NAME}_sf/unknown", exist_ok=True)
cnt = 0
for idx, row in df.iterrows():
    image_name = row["id"]
    img_path = row["image"]
    try:
        os.rename(img_path, f"data/images{TEST_NAME}_sf/unknown/{image_name}.jpg")
    except FileNotFoundError:
        pass
    
    cnt += 1
    if cnt % 100 == 0:
        print(cnt, len(df))
print(f"{TEST_NAME} done") 