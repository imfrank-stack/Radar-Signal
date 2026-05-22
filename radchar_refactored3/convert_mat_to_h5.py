import os
import h5py
import numpy as np
import scipy.io as sio
from pathlib import Path
from tqdm import tqdm

# 12 种信号类型
SIGNAL_TYPES = ["LFM", "AM", "COMB", "FM", "ISRJ", "MNJ",
              "RGPO", "RMT", "R_VGPO", "SMSP", "VGPO", "VMT"]

TARGET_LEN = 1024

def load_mat(path):
    """加载 .mat 文件并下采样到目标长度"""
    data = sio.loadmat(path)
    key = [k for k in data.keys() if not k.startswith("__")][0]
    iq = data[key].squeeze()  # 时域 IQ，长度 Nfast ≈ 4000
    step = max(1, len(iq) // TARGET_LEN)
    if step > 1:
        iq = iq[::step][:TARGET_LEN]
    return iq.astype(np.complex64)

def collect_all_files(base):
    """收集所有 .mat 文件"""
    files = []
    for idx, sig in enumerate(SIGNAL_TYPES):
        folder = Path(base) / sig
        if folder.exists():
            for f in sorted(folder.glob("*.mat")):
                files.append((str(f), idx))
    return files

def split_dataset(files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """按8:1:1分层划分数据集"""
    np.random.seed(seed)
    train_idx, val_idx, test_idx = [], [], []
    
    # 按类别分层划分
    for label in range(len(SIGNAL_TYPES)):
        indices = [i for i, (_, lbl) in enumerate(files) if lbl == label]
        np.random.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train+n_val])
        test_idx.extend(indices[n_train+n_val:])
    
    # 验证比例
    assert abs(len(train_idx) + len(val_idx) + len(test_idx) - len(files)) <= len(SIGNAL_TYPES), "划分索引和原文件数量不符"
    
    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    test_files = [files[i] for i in test_idx]
    
    return train_files, val_files, test_files

def save_h5(files, output, desc):
    """保存为 HDF5 格式"""
    n = len(files)
    if n == 0:
        print(f"警告: {desc} 没有文件")
        return
    
    print(f"\n{desc}: {n} 个样本")
    iq_data = np.zeros((n, TARGET_LEN), dtype=np.complex64)
    labels = np.zeros(n, dtype=np.int64)
    
    # 加载所有文件
    for i, (path, label) in enumerate(tqdm(files, desc=desc)):
        iq_data[i] = load_mat(path)
        labels[i] = label
    
    # 保存为 HDF5
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with h5py.File(output, "w") as f:
        f.create_dataset("iq", data=iq_data, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
    
    print(f"✓ 保存: {output}")
    
    # 打印类别分布
    for idx, sig in enumerate(SIGNAL_TYPES):
        count = np.sum(labels == idx)
        print(f"  {idx:2d} {sig:8s}: {count:4d} 样本")

def main():
    # 数据集路径
    data_base = "../dataset/Training_data/dataset_seq/All_dB"
    
    print("=" * 60)
    print("雷达信号数据集转换工具")
    print("=" * 60)
    
    # 收集所有文件
    print("\n收集所有文件...")
    all_files = collect_all_files(data_base)
    print(f"找到 {len(all_files)} 个总样本")
    
    # 按8:1:1分层划分数据集
    print("\n按8:1:1比例划分训练/验证/测试集...")
    train_files, val_files, test_files = split_dataset(all_files, 0.8, 0.1, 0.1)
    
    print(f"训练集: {len(train_files)} 样本 ({len(train_files)/len(all_files)*100:.1f}%)")
    print(f"验证集: {len(val_files)} 样本 ({len(val_files)/len(all_files)*100:.1f}%)")
    print(f"测试集: {len(test_files)} 样本 ({len(test_files)/len(all_files)*100:.1f}%)")
    
    # 执行转换
    save_h5(train_files, "Bear_data/RadChar-Train.h5", "训练集")
    save_h5(val_files, "Bear_data/RadChar-Val.h5", "验证集")
    save_h5(test_files, "Bear_data/RadChar-Test.h5", "测试集")
    
    print("\n" + "=" * 60)
    print("✓ 数据集转换完成！")
    print("=" * 60)
    print("\n下一步:")
    print("  python radchar_training.py --no-resume")

def verify_dataset(path):
    print(f"\n验证: {path}")
    with h5py.File(path, 'r') as f:
        iq = f['iq']
        labels = f['labels']
        
        print(f"  IQ 形状: {iq.shape}")
        print(f"  IQ 类型: {iq.dtype}")
        print(f"  标签形状: {labels.shape}")
        print(f"  标签类型: {labels.dtype}")
        
        labels_array = labels[:]
        print(f"  类别分布:")
        for i in range(12):
            count = np.sum(labels_array == i)
            print(f"    类别 {i}: {count} 样本")

if __name__ == "__main__":
    main()
    
    #verify_dataset("Bear_data/RadChar-Train.h5")
    #verify_dataset("Bear_data/RadChar-Val.h5")
    #verify_dataset("Bear_data/RadChar-Test.h5")
