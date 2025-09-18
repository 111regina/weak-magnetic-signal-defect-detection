import os
import numpy as np
import pandas as pd
import pywt
import cv2
import imageio
from tqdm import tqdm

# 图像保存设置
IMG_SIZE = (128, 128)
OUTPUT_ROOT = "image"

# 通道映射（左链 CH1-CH3，右链 CH4-CH6）
CHANNEL_MAP = {
    '左': ['CH1', 'CH2', 'CH3'],
    '右': ['CH4', 'CH5', 'CH6']
}
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 小波变换转换为 RGB 图像
def cwt_rgb_image(segment_3ch, size=IMG_SIZE):
    rgb = []
    for i in range(3):
        coef, _ = pywt.cwt(segment_3ch[:, i], scales=np.arange(1, 65), wavelet='morl')
        img = np.abs(coef)
        img = cv2.resize(img, size)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        rgb.append(img)
    return np.stack(rgb, axis=2)

# 单文件处理函数，使用外部传入的周期计数器
def process_file(csv_path, xlsx_path, label_type, cycle_counter):
    df = pd.read_csv(csv_path)
    label_df = pd.read_excel(xlsx_path)

    for col in [
        '周期x轴开始位置', '周期x轴结束位置',
        '周期y轴开始位置', '周期y轴结束位置',
        '周期z轴开始位置', '周期z轴结束位置',
    ]:
        label_df[col] = label_df[col] + 1

    metadata = []

    grouped = label_df.groupby(['运行周期', '是否刮板机右边链环'])
    for (cycle_id, is_right), group in grouped:
        side = '右' if is_right == 1 else '左'
        ch_x, ch_y, ch_z = CHANNEL_MAP[side]
        cyc = cycle_counter[side]

        row = group.iloc[0]
        sx, ex = row['周期x轴开始位置'], row['周期x轴结束位置']
        sy, ey = row['周期y轴开始位置'], row['周期y轴结束位置']
        sz, ez = row['周期z轴开始位置'], row['周期z轴结束位置']

        seg_x = df[ch_x].values[sx:ex]
        seg_y = df[ch_y].values[sy:ey]
        seg_z = df[ch_z].values[sz:ez]
        min_len = min(len(seg_x), len(seg_y), len(seg_z))
        cycle_signal = np.stack([
            seg_x[:min_len], seg_y[:min_len], seg_z[:min_len]
        ], axis=1)

        L = min_len / (1 + 21 * 2)
        lengths = [int(L)] + [int(2 * L)] * 21
        indices = [0]
        for l in lengths:
            indices.append(indices[-1] + l)

        for blade_no in range(1, 23):
            start, end = indices[blade_no - 1], indices[blade_no]
            segment = cycle_signal[start:end]
            img = cwt_rgb_image(segment)

            # 自动标签分类
            if side == '左' and blade_no in [3, 6, 12, 17, 19, 21]:
                label_id = 1
            elif side == '右' and blade_no in [3, 8, 12, 17, 19]:
                label_id = 1
            elif side == '左' and blade_no == 15:
                label_id = 2
            else:
                label_id = 0

            fname = f"{side[0]}_{cyc:02d}_{blade_no:03d}.png"
            out_dir = os.path.join(OUTPUT_ROOT, label_type)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, fname)
            imageio.imwrite(out_path, (img * 255).astype(np.uint8))

            metadata.append({
                "图像路径": out_path,
                "刮板编号": blade_no,
                "周期": cyc,
                "链条方向": side,
                "标签": label_id
            })

        cycle_counter[side] += 1  # 连续编号更新

    # 保存当前文件的元数据
    base = os.path.splitext(os.path.basename(csv_path))[0]
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(os.path.join(OUTPUT_ROOT, f"图像元数据_{label_type}_{base}.csv"), index=False)
    print(f"已处理 {base}，输出图像 {len(metadata)} 张")


# 批处理多个文件，带全局周期编号追踪
def batch_process(csv_dir, xlsx_dir, label_type="无磁铁"):
    cycle_counter = {'左': 1, '右': 1}
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

    for csv_file in tqdm(csv_files, desc="批量处理数据"):
        csv_path = os.path.join(csv_dir, csv_file)
        base = os.path.splitext(csv_file)[0]
        xlsx_file = base + ".csv_biaozhu.xlsx"
        xlsx_path = os.path.join(xlsx_dir, xlsx_file)

        if not os.path.exists(xlsx_path):
            print(f"缺失标注文件，跳过 {csv_file}")
            continue

        process_file(csv_path, xlsx_path, label_type, cycle_counter)

if __name__ == "__main__":
    batch_process(
        csv_dir="data/无磁铁数据",
        xlsx_dir="data/无磁铁数据/缺陷位置标注",
        label_type="无磁铁"
    )

