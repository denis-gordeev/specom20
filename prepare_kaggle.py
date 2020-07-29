from sklearn.model_selection import train_test_split
import pandas as pd
import re
import pickle
import numpy as np
from glob import glob
from shutil import copyfile

df = pd.read_csv("kaggle_train.csv")
df = df[['comment_text', 'severe_toxicity']]
df_0 = df[df["severe_toxicity"] == 0][:100000]
df_1 = df[df["severe_toxicity"] > 0]

df = pd.concat([df_0, df_1])
df.loc[df["severe_toxicity"] > 0, "severe_toxicity"] = 1
df["severe_toxicity"] = df["severe_toxicity"].astype(int)

df.to_csv("kaggle_shortened_binary.csv", index=False)

train, test = train_test_split(df, test_size=0.2)
train, dev = train_test_split(train, test_size=0.2)

datasets = ((train, "train"), (dev, "dev"), (test, "test"))
for dataset, name in datasets:
    dataset.to_csv(f"kaggle_shortened_binary_{name}.csv", index=False)


data = pickle.load(open("images/all_data.pcl", "rb"))
texts = []
for v in data.values():
    for value in v.values():
        texts.append(value["text"])

texts = [re.sub("^>>\d+\n", "", t) for t in texts]
texts = [re.sub(">>\d+", " ", t) for t in texts]
texts = [" ".join(t.split()) for t in texts]
texts = [t for t in texts if t]
df = pd.read_csv(f"kaggle_shortened_binary_test_real.csv")
df = df[: len(texts)]
df["comment_text"] = texts

logits = pd.read_csv("test_predictions_logits.txt", header=None)
logits[0] = logits[0].str[1:-1]
logits[1] = logits[0].str.strip().str.split().str[1].astype(float)
logits[0] = logits[0].str.strip().str.split().str[0].astype(float)

df["logits_0"] = logits[0]
df["logits_1"] = logits[1]
df["margin"] = (df["logits_0"] - df["logits_1"]).abs()
text_indices = [t_i for t_i, t in enumerate(texts) if t]
i = 0
text_for_images = []
for v in data.values():
    for key, value in v.items():
        if i in text_indices:
            text_for_images.append(key)
        i += 1
df["post_name"] = text_for_images
images = [img[7:-4] for img in glob("images/*.jpg")]

images_set = set(text_for_images).intersection(set(images))
text_for_images_indices = [
    t in images_set for t_i, t in enumerate(text_for_images)
]
preds = pd.read_csv("bert_4chan_predictions.txt", header=None)
df["preds"] = preds[0]
df["has_image"] = text_for_images_indices

df = df[df["has_image"]]

df = df.sort_values("margin", ascending=False) 

unlabelled = pd.concat([df[df["preds"] == 1][1000:], df[df["preds"] == 0][1000:]])
df = pd.concat([df[df["preds"] == 1][:1000], df[df["preds"] == 0][:1000]])
# df = df[(df["logits_0"].abs() > 0.9) | (df["logits_1"].abs() > 0.9)]
# df = df[(df["logits_0"] - df["logits_1"]).abs() > 2]

for row_i, row in df.iterrows():
    post_name = row["post_name"]
    class_name = row["preds"]
    print(post_name)
    copyfile(
        f"images/{post_name}.jpg",
        f"images_dataset/class_{class_name}/{post_name}.jpg",
    )

for row_i, row in unlabelled.iterrows():
    post_name = row["post_name"]
    class_name = row["preds"]
    print(post_name)
    copyfile(
        f"images/{post_name}.jpg",
        f"unlabelled/{post_name}.jpg",
    )


df.to_csv("kaggle_shortened_binary_test.csv", index=False)
df.to_csv("kaggle_shortened_binary_test_4chan.csv", index=False)
# 5233
