import os
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# 参数设置
IMG_SIZE = 128
BATCH_SIZE = 32
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "flashback/ver3best.pth"
CSV_PATTERN = "image/图像元数据_无磁铁_*.csv"
OUTPUT_DIR = "inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# 自定义数据集
class BladeTestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "图像路径"]
        image = Image.open(path).convert("RGB")
        label = self.df.loc[idx, "标签"] if "标签" in self.df.columns else -1
        if self.transform:
            image = self.transform(image)
        return image, label, path

# 加载模型
def load_model():
    model = models.resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# 单文件推理 + 准确率计算
def inference_on_csv(csv_path, model):
    df = pd.read_csv(csv_path)
    dataset = BladeTestDataset(df, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    results = []
    total, correct = 0, 0

    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for i in range(len(paths)):
                true_label = labels[i].item()
                pred_label = preds[i]
                if true_label != -1:
                    total += 1
                    if pred_label == true_label:
                        correct += 1
                results.append((paths[i], true_label, pred_label))

    acc = correct / total if total > 0 else None
    out_df = pd.DataFrame(results, columns=["图像路径", "真实标签", "预测标签"])
    output_csv = os.path.join(OUTPUT_DIR, f"推理结果_{os.path.basename(csv_path)}")
    out_df.to_csv(output_csv, index=False)
    print(f"完成：{csv_path} → 准确率：{acc:.4f}" if acc is not None else "无真实标签，未计算准确率")
    return os.path.basename(csv_path), acc

# 主程序
def batch_inference():
    model = load_model()
    csv_files = glob.glob(CSV_PATTERN)
    acc_summary = []

    for csv_path in csv_files:
        name, acc = inference_on_csv(csv_path, model)
        acc_summary.append({"文件名": name, "准确率": acc if acc is not None else "N/A"})

    # 汇总准确率
    summary_df = pd.DataFrame(acc_summary)
    summary_path = os.path.join(OUTPUT_DIR, "整体准确率统计.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"已保存整体准确率统计：{summary_path}")

if __name__ == "__main__":
    batch_inference()
