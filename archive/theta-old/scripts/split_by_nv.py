import uproot
import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 基本配置 ---
root_path = "/mnt/mydisk/WCDA_simulation/"
output_dir = "/mnt/mydisk/WCDA_split/"
os.makedirs(output_dir, exist_ok=True)

n_workers = 32
branches = [
    "nv", "vx", "vy", "vq", "vt", "vnpe", "vqsamp",
    "theta", "phi", "xc", "yc", "mc_energy"
]
bins = [(60, 150), (150, 500), (500, 3000)]
max_events_per_file = 3800  # 每个文件包含的最大事件数

# --- 单文件提取并分bin ---
def process_file(path):
    try:
        with uproot.open(path) as f:
            if "t_eventout;1" not in f:
                return {}
            tree = f["t_eventout;1"]
            available = [b for b in branches if b in tree.keys()]
            if len(available) < len(branches):
                print(f"⚠️ {os.path.basename(path)} 字段不全，仅提取 {available}")
            df = tree.arrays(available, library="pd")

            # --- 根据 nv 分bin ---
            results = {}
            for (low, high) in bins:
                mask = (df["nv"] >= low) & (df["nv"] < high)
                if mask.any():
                    results[(low, high)] = df[mask]
            return results
    except Exception as e:
        print(f"❌ {os.path.basename(path)} 读取失败: {e}")
        return {}

# --- 并行提取 ---
file_paths = [os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith(".root")]
print(f"📁 共找到 {len(file_paths)} 个 ROOT 文件")

# --- 统计信息 ---
bin_counts = {b: 0 for b in bins}
processed_files = 0

def save_to_root(df_bin, bin_range, file_folder):
    # 按事件数切分成多个文件，每个文件包含约3800个事件
    num_files = int(np.ceil(len(df_bin) / max_events_per_file))
    split_data = np.array_split(df_bin, num_files)

    for i, sub_df in enumerate(split_data):
        # 创建对应的文件夹
        output_folder = os.path.join(output_dir, file_folder)
        os.makedirs(output_folder, exist_ok=True)

        # 每个文件命名为：nv_60_150_0.root, nv_60_150_1.root
        out_path = os.path.join(output_folder, f"nv_{bin_range[0]}_{bin_range[1]}_{i}.root")
        with uproot.recreate(out_path) as fout:
            fout["t_eventout"] = sub_df
        print(f"已保存: {out_path}")

# --- 并行处理数据 ---
with ProcessPoolExecutor(max_workers=n_workers) as executor:
    futures = {executor.submit(process_file, p): p for p in file_paths}
    for i, future in enumerate(as_completed(futures)):
        path = futures[future]
        result = future.result()
        for b, df_bin in result.items():
            bin_counts[b] += len(df_bin)

            # 为每个 bin 创建文件夹并保存数据
            save_to_root(df_bin, b, f"bin_{b[0]}_{b[1]}")

        processed_files += 1
        if processed_files % 50 == 0:
            print(f"进度: {processed_files}/{len(file_paths)} 文件完成")

print("✅ 数据拆分完成！")
for b, n in bin_counts.items():
    print(f"nv ∈ [{b[0]}, {b[1]}): {n} 事件")
