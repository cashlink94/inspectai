import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ========= CONFIG =========
CSV_PATH = "6. Maintenance Department/train.csv"
IMG_DIR = "6. Maintenance Department/train_images"
MODEL_PATH = "inspectai_model.pth"

BATCH_SIZE = 64
EPOCHS = 1
IMG_SIZE = 128
MAX_SAMPLES = 800   # ⚡ VERY FAST

DEVICE = "cpu"


# ========= DATASET =========
class SteelDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row["ImageId"])
        image = Image.open(img_path).convert("RGB")

        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


# ========= LOAD DATA =========
def load_data():
    df = pd.read_csv(CSV_PATH)

    # FIX labels (0–3)
    df["label"] = df["ClassId"].astype(int) - 1

    df = df.dropna(subset=["EncodedPixels"])
    df = df.groupby("ImageId").first().reset_index()

    # ⚡ LIMIT DATA FOR SPEED
    df = df.sample(min(len(df), MAX_SAMPLES), random_state=42)

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    return df


# ========= MODEL =========
def build_model(num_classes):
    model = models.mobilenet_v2(weights="DEFAULT")

    # 🔥 Freeze everything (BIG speed boost)
    for param in model.parameters():
        param.requires_grad = False

    # Only train last layer
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    return model


# ========= TRAIN =========
def train():
    print("⚡ ULTRA FAST TRAINING (FINAL)\n")

    df = load_data()

    num_classes = df["label"].nunique()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    dataset = SteelDataset(df, IMG_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = build_model(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        total_loss = 0

        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ DONE → {MODEL_PATH}")


if __name__ == "__main__":
    train()