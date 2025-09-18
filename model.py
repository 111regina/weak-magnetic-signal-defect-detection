import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau


# 自动读取多个 CSV 文件
csv_files = glob.glob("image/图像元数据_*.csv")
dfs = [pd.read_csv(f) for f in csv_files]
total_df = pd.concat(dfs, ignore_index=True)

# 参数设置
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 40
LR = 2e-5
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像增强与预处理
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# 自定义数据集
class BladeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "图像路径"]
        label = self.df.loc[idx, "标签"]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# 读取 CSV 并划分训练/验证
df = total_df
train_val_df, test_df = train_test_split(df, test_size=0.15, stratify=df["标签"], random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.1765, stratify=train_val_df["标签"], random_state=42)

train_ds = BladeDataset(train_df, transform)
val_ds = BladeDataset(val_df, transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_ds = BladeDataset(test_df, transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# 定义模型
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()  # 去除初始池化
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# 损失与优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#学习率调度器
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2,
    threshold=1e-3,        # 提升幅度必须超过 0.1% 才算有效
    threshold_mode='rel',  # 相对提升
    min_lr=1e-6
)
# 训练函数
def train_model():
    best_val_acc = 0
    patience = 5
    counter = 0
    best_model_path = "best_model.pth"

    for epoch in range(EPOCHS):
        model.train()
        total, correct = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{EPOCHS} - LR: {current_lr:.6f} - Train Acc: {train_acc:.4f}")

        # 验证评估
        val_acc = evaluate_model()
        #更新学习率
        scheduler.step(val_acc)
        # Early Stopping 判断
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("验证精度提升，保存最佳模型")
        else:
            counter += 1
            print(f"验证精度未提升：{counter}/{patience}")
            if counter >= patience:
                print("提前终止训练（Early Stopping）")
                break

    print(f"最佳验证精度：{best_val_acc:.4f}，模型保存在 {best_model_path}")


# 验证函数
def evaluate_model():
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Validation Acc: {acc:.4f}")
    return acc

#测试函数
def test_model():
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Test Acc: {acc:.4f}")

# 启动训练
train_model()

print("开始在测试集上评估模型性能")
model.load_state_dict(torch.load(r"D:\dataanalysis\best_model.pth"))
test_model()

