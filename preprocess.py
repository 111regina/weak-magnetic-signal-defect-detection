import os
import numpy as np
import pandas as pd
import pywt
import cv2
import imageio
from tqdm import tqdm

# 输出图像尺寸与路径
IMG_SIZE = (128, 128)
OUTPUT_ROOT = "image"

# 每个周期 22 个刮板：1 短板 + 21 长板（每长板信号长度约为短板的 2 倍）
def get_blade_indices(total_length):
    L = total_length / (1 + 21 * 2)
    lengths = [int(L)] + [int(2 * L)] * 21
    indices = [0]
    for l in lengths:
        indices.append(indices[-1] + l)
    return indices

# 小波变换生成 RGB 图像
def cwt_rgb_image(segment_3ch, size=IMG_SIZE):
    rgb = []
    for i in range(3):
        coef, _ = pywt.cwt(segment_3ch[:, i], scales=np.arange(1, 65), wavelet='morl')
        img = np.abs(coef)
        img = cv2.resize(img, size)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        rgb.append(img)
    return np.stack(rgb, axis=2)

# 缺陷分类函数
def assign_label(blade_no, side):
    if side == '左' and blade_no in [3, 6, 12, 14, 19, 21]:
        return 1  # 凹陷
    elif side == '右' and blade_no in [3, 8, 12, 17, 19]:
        return 1  # 凹陷
    elif blade_no == 15:
        return 2  # 磨损
    else:
        return 0  # 正常

# 处理单个文件
def process_file(csv_path, xlsx_path, label_type, cycle_counter):
    df = pd.read_csv(csv_path)
    label_df = pd.read_excel(xlsx_path)

    # +1 修正，匹配行索引起始为 1
    for col in ['周期x轴开始位置', '周期x轴结束位置',
                '周期y轴开始位置', '周期y轴结束位置',
                '周期z轴开始位置', '周期z轴结束位置']:
        label_df[col] += 1

    metadata = []

    grouped = label_df.groupby(['运行周期', '是否刮板机右边链环'])
    for (cycle_id, is_right), group in grouped:
        side = '右' if is_right == 1 else '左'
        cyc = cycle_counter[side]
        row = group.iloc[0]

        # 根据文件名判断数据来源（1- 开头是208左、202右；2- 开头是208右、202左）
        fname = os.path.basename(csv_path)
        is_1x = fname.startswith("1-")

        if is_1x and side == '左':
            chs = ['CH3', 'CH4', 'CH5']  # 注意这里大写
        elif is_1x and side == '右':
            chs = ['CH0', 'CH1', 'CH2']
        elif not is_1x and side == '右':
            chs = ['CH3', 'CH4', 'CH5']
        elif not is_1x and side == '左':
            chs = ['CH0', 'CH1', 'CH2']

        seg_x = df[chs[0]].values[row['周期x轴开始位置']:row['周期x轴结束位置']]
        seg_y = df[chs[1]].values[row['周期y轴开始位置']:row['周期y轴结束位置']]
        seg_z = df[chs[2]].values[row['周期z轴开始位置']:row['周期z轴结束位置']]
        min_len = min(len(seg_x), len(seg_y), len(seg_z))

        cycle_signal = np.stack([
            seg_x[:min_len], seg_y[:min_len], seg_z[:min_len]
        ], axis=1)

        indices = get_blade_indices(min_len)

        for blade_no in range(1, 23):
            start, end = indices[blade_no - 1], indices[blade_no]
            segment = cycle_signal[start:end]
            img = cwt_rgb_image(segment)

            label_id = assign_label(blade_no, side)

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

        cycle_counter[side] += 1

    base = os.path.splitext(os.path.basename(csv_path))[0]
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(os.path.join(OUTPUT_ROOT, f"图像元数据_{label_type}_{base}.csv"), index=False)
    print(f"处理完成：{base}，生成图像 {len(metadata)} 张")

# 批处理入口
def batch_process(csv_dir, xlsx_dir, label_type="无数据2"):
    cycle_counter = {'左': 1, '右': 1}
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

    for csv_file in tqdm(csv_files, desc="批量处理数据文件"):
        base = os.path.splitext(csv_file)[0]
        csv_path = os.path.join(csv_dir, csv_file)
        xlsx_path = os.path.join(xlsx_dir, base + ".csv_biaozhu.xlsx")

        if not os.path.exists(xlsx_path):
            print(f"缺失标注文件，跳过：{csv_file}")
            continue

        process_file(csv_path, xlsx_path, label_type, cycle_counter)

if __name__ == "__main__":
    batch_process(
        csv_dir="data/无磁铁数据1",
        xlsx_dir="data/无磁铁数据1/缺陷位置标注",
        label_type="无磁铁1"
    )