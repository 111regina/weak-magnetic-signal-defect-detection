# 基于弱磁信号的刮板机缺陷检测代码安装运行说明

---

## 0. 文件结构说明

项目根目录组织如下：

```bash
.
├── data/                         # 原始数据与标注
│   ├── 有磁铁数据/                # 用于训练的数据集
│   │   ├── *.csv                 # 六通道信号文件
│   │   └── *.csv_biaozhu.xlsx    # 周期缺陷标注（每个csv配套xlsx）
│   └── 无磁铁数据/                # 用于迁移测试的数据集（相同结构）
│       ├── *.csv
│       └── *.csv_biaozhu.xlsx
├── image/                        # 自动生成的图像和元数据
│   ├── 有磁铁/                    # 训练用RGB图像
│   ├── 无磁铁/                    # 推理用RGB图像
│   ├── 图像元数据_有磁铁_*.csv    # 每批图像的路径+标签信息
│   └── 图像元数据_无磁铁_*.csv
├── flashback/
│   └── vernbest.pth              # 已训练好的最佳模型权重
├── inference_results/            # 推理输出结果（CSV + 总结）
├── preprocess.py                 # 有磁铁数据 → 图像预处理脚本
├── preprocess1.py                # 无磁铁数据 → 图像预处理脚本
├── preprocess2.py                # 新数据 → 图像预处理脚本   
├── showchannel.py                # 信号波形可视化与周期尖峰检测
├── model.py                      # 训练+验证+测试主程序
├── inference.py                  # 对图像数据进行推理与评估
```

---

## 1. 环境安装

### 1.1 推荐环境

* Python ≥ 3.8
* 使用 Anaconda 管理虚拟环境

### 1.2 创建环境并安装依赖

```bash
# 创建虚拟环境
conda create -n scraper_mag python=3.8 -y
conda activate scraper_mag

# 安装依赖
pip install numpy pandas matplotlib pywavelets imageio opencv-python tqdm
pip install torch torchvision scikit-learn
```

> 若使用GPU，请替换为支持 CUDA 的 PyTorch 安装命令。

---

## 2. 数据准备

### 2.1 数据命名规范

每组原始数据需包含：

| 文件类型      | 示例文件名                              | 说明               |
| --------- | ---------------------------------- | ---------------- |
| 信号数据 CSV  | `20250523_000001.csv`              | 包含 CH1–CH6 六通道   |
| 标注数据 XLSX | `20250523_000001.csv_biaozhu.xlsx` | 每周期 XYZ 起止位置、链方向 |

### 2.2 文件放置位置

* 训练数据放入：`data/有磁铁数据/`
* 测试数据放入：`data/无磁铁数据/`

---

## 3. 数据预处理流程

### 3.1 查看六通道信号与周期尖峰（可视化）

```bash
python showchannel.py
```

功能：

* 输出周期尖峰位置索引；
* 显示六通道信号图，便于验证周期划分正确性。

### 3.2 将信号转为图像（小波变换）

#### ▶ 处理训练数据（有磁铁）

```bash
python preprocess.py
```

#### ▶ 处理测试数据（无磁铁）

```bash
python preprocess1.py
```

#### ▶ 处理新数据（无磁铁1+无磁铁2）

```bash
python preprocess2.py
```

输出结果包括：

* 每块刮板图像（RGB 128×128）；
* 自动命名 + 标签（0\~3）；
* 输出图像元数据CSV（路径、标签、周期、编号等）。

---

## 4. 模型训练与评估

```bash
python model.py
```

内容包括：

* 从图像元数据中读取训练样本；
* 改进版 ResNet18 架构训练；
* 验证集精度持续提升则保存模型；
* 最终输出 `best_model.pth`；
* 自动评估测试集性能。

模型配置说明：

| 参数             | 设置                |
| -------------- | ----------------- |
| 输入尺寸           | 128 × 128 × 3     |
| 分类数            | 4 类               |
| 学习率            | 2e-5              |
| 优化器            | Adam              |
| Early Stopping | patience=5        |
| 调度器            | ReduceLROnPlateau |

---

## 5. 模型推理

```bash
python inference.py
```

功能：

* 加载 `flashback/vernbest.pth`；
* 对无磁铁图像批量分类；
* 输出预测标签与准确率；
* 保存结果至 `inference_results/` 文件夹：

```bash
inference_results/
├── 推理结果_图像元数据_无磁铁_xxxxx.csv
└── 整体准确率统计.csv
```

---

## 7. 注意事项

* 请确保每组 CSV 和 XLSX 文件配对存在；
* 图像与元数据生成后不建议重复运行 `preprocess.py`；
* 图像路径中请避免中文字符和空格；
* 推理时模型路径需与 `inference.py` 中一致（或修改对应路径）。

---


