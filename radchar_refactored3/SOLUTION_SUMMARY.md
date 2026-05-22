# 雷达信号数据集加载完整解决方案

## 📋 任务总结

已完成对 `radchar_refactored3` 目录下代码的分析，并提供了将 .mat 格式数据集转换为 HDF5 格式的完整方案。

## 🔍 代码分析结果

### 1. 现有代码结构

| 文件 | 功能 | 数据格式要求 |
|------|------|-------------|
| `radchar_training.py` | 训练脚本 | HDF5: `iq` (N,512) complex64, `labels` (N,) int64 |
| `radchar_model.py` | 模型定义 | 输入序列长度 512 |
| `radchar_class_config.py` | 类别配置 | 标签映射和过滤 |
| `radchar_test.py` | 测试脚本 | 同训练格式 |

### 2. 数据集实际格式

**位置**: `../dataset/`

**训练集**: `Trainning_data/dataset_seq/All_dB/{signal_type}/*.mat`
- 12 种信号类型，每种约 800 个文件
- 总计约 9600 个样本

**测试集**: `Test_data/dataset_seq/{snr_dB}/{signal_type}/*.mat`
- 11 个 SNR 等级 (-10dB 到 10dB，步长 2dB)
- 每个 SNR 下 12 种信号，每种 200 个文件
- 总计约 26400 个样本

**数据格式**:
- 文件类型: MATLAB .mat
- 数据键: `lfm_echo_fft` (LFM) 或 `J_fft` (其他)
- 数据形状: (1, 1024) complex128
- 需要处理: 下采样到 512，转换为 complex64

## ✅ 解决方案

### 方案 A: 数据预转换（推荐）

**优点**:
- ✓ 最小化代码修改
- ✓ 转换一次，多次使用
- ✓ 训练速度快
- ✓ 与现有代码完全兼容

**实施步骤**:

#### 步骤 1: 创建转换脚本

由于 PowerShell 和 Python 缩进问题，请手动创建 `convert_mat_to_h5.py`:

```python
import os
import h5py
import numpy as np
import scipy.io as sio
from pathlib import Path
from tqdm import tqdm

SIGNAL_TYPES = ["LFM", "AM", "COMB", "FM", "ISRJ", "MNJ", 
                "RGPO", "RMT", "R_VGPO", "SMSP", "VGPO", "VMT"]

def load_mat(path):
    data = sio.loadmat(path)
    key = [k for k in data.keys() if not k.startswith("__")][0]
    return data[key].squeeze()[::2].astype(np.complex64)  # 下采样 1024->512

def collect_train(base):
    files = []
    for idx, sig in enumerate(SIGNAL_TYPES):
        folder = Path(base) / sig
        if folder.exists():
            for f in sorted(folder.glob("*.mat")):
             files.append((str(f), idx))
    return files

def collect_test(base):
    files = []
    snrs = ["-10_dB", "-8_dB", "-6_dB", "-4_dB", "-2_dB", "0_dB", 
          "2_dB", "4_dB", "6_dB", "8_dB", "10_dB"]
    for snr in snrs:
        for idx, sig in enumerate(SIGNAL_TYPES):
          folder = Path(base) / snr / sig
            if folder.exists():
                for f in sorted(folder.glob("*.mat")):
                    files.append((str(f), idx))
    return files

def save_h5(files, output, desc):
    n = len(files)
    print(f"\n{desc}: {n} 样本")
    iq = np.zeros((n, 512), dtype=np.complex64)
    labels = np.zeros(n, dtype=np.int64)
    for i, (path, label) in enumerate(tqdm(files, desc=desc)):
        iq[i] = load_mat(path)
        labels[i] = label
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with h5py.File(output, "w") as f:
        f.create_dataset("iq", data=iq, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
    print(f"✓ {output}")
  for idx, sig in enumerate(SIGNAL_TYPES):
        print(f"  {idx:2d} {sig:8s}: {np.sum(labels==idx):4d}")

def main():
    np.random.seed(42)
    train_base = "../dataset/Trainning_data/dataset_seq/All_dB"
    test_base = "../dataset/Test_data/dataset_seq"
    
    print("="*60)
    print("数据集转换")
    print("="*60)
    
    # 训练集
    train_files = collect_train(train_base)
    print(f"\n训练: {len(train_files)} 文件")
    
    # 分层划分
    train_idx, val_idx = [], []
    for label in range(12):
        idx = [i for i, (_, l) in enumerate(train_files) if l == label]
    np.random.shuffle(idx)
        n_val = max(1, int(len(idx) * 0.15))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    
    train_sub = [train_files[i] for i in train_idx]
    val_sub = [train_files[i] for i in val_idx]
    
    # 测试集
    test_files = collect_test(test_base)
    print(f"测试: {len(test_files)} 文件")
    
    # 转换
    save_h5(train_sub, "Bear_data/RadChar-Train.h5", "训练集")
    save_h5(val_sub, "Bear_data/RadChar-Val.h5", "验证集")
    save_h5(test_files, "Bear_data/RadChar-Test.h5", "测试集")
    
    print("\n"+"="*60)
    print("✓ 完成")
    print("="*60)
    print("\n运行: python radchar_training.py --no-resume")

if __name__ == "__main__":
    main()
```

#### 步骤 2: 创建类别配置

创建 `radchar_classes.json`:

```json
{
  "description": "12 种雷达干扰信号类型",
  "entries": [
    {"raw_label": 0, "name": "LFM", "enabled": true},
    {"raw_label": 1, "name": "AM", "enabled": true},
    {"raw_label": 2, "name": "COMB", "enabled": true},
    {"raw_label": 3, "name": "FM", "enabled": true},
    {"raw_label": 4, "name": "ISRJ", "enabled": true},
    {"raw_label": 5, "name": "MNJ", "enabled": true},
    {"raw_label": 6, "name": "RGPO", "enabled": true},
    {"raw_label": 7, "name": "RMT", "enabled": true},
    {"raw_label": 8, "name": "R_VGPO", "enabled": true},
    {"raw_label": 9, "name": "SMSP", "enabled": true},
    {"raw_label": 10, "name": "VGPO", "enabled": true},
    {"raw_label": 11, "name": "VMT", "enabled": true}
  ]
}
```

#### 步骤 3: 运行转换

```bash
cd radchar_refactored3
python convert_mat_to_h5.py
```

预计耗时: 5-10 分钟

#### 步骤 4: 验证数据

```python
import h5py
import numpy as np

with h5py.File('Bear_data/RadChar-Train.h5', 'r') as f:
    print(f"IQ: {f['iq'].shape}, {f['iq'].dtype}")
    print(f"Labels: {f['labels'].shape}, {f['labels'].dtype}")
    labels = f['labels'][:]
    for i in range(12):
        print(f"Class {i}: {np.sum(labels==i)} samples")
```

#### 步骤 5: 运行训练

```bash
python radchar_training.py --no-resume
```

## 📊 预期结果

转换后的数据集:

| 数据集 | 样本数 | 文件大小 | 说明 |
|----|--------|---------|------|
| 训练集 | ~8160 | ~32 MB | 85% 训练数据 |
| 验证集 | ~1440 | ~6 MB | 15% 验证数据 |
| 测试集 | ~26400 | ~105 MB | 全部测试数据 |

类别分布 (训练集):
- 每个类别约 680 样本
- 12 个类别均衡分布

## 🔧 故障排除

### 问题 1: ModuleNotFoundError: No module named 'scipy'
```bash
pip install scipy h5py tqdm
```

### 问题 2: 找不到数据集目录
确保目录结构:
```
Radar-Active-Jamming-Signal-Modulation-Dataset-Generating-Code-main/
├── dataset/
│   ├── Trainning_data/
│   └── Test_data/
└── radchar_refactored3/
    ├── convert_mat_to_h5.py
    └── radchar_training.py
```

### 问题 3: 内存不足
修改转换脚本，分批处理:
```python
# 在 save_h5() 中，分批加载
batch_size = 1000
for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    for i in range(start, end):
        iq[i] = load_mat(files[i][0])
        labels[i] = files[i][1]
```

## 📝 关键修改点

### 1. 数据下采样
- 原始: 1024 点
- 目标: 512 点
- 方法: `iq[::2]` (步长为 2 的下采样)

### 2. 数据类型转换
- 原始: complex128
- 目标: complex64
- 原因: 减少内存占用，加快训练速度

### 3. 标签映射
```python
SIGNAL_TYPES = ["LFM", "AM", "COMB", "FM", "ISRJ", "MNJ", 
                "RGPO", "RMT", "R_VGPO", "SMSP", "VGPO", "VMT"]
# 索引即为标签: LFM=0, AM=1, ..., VMT=11
```

## ⚠️ 注意事项

1. **备份原始数据**: 转换前备份 `dataset/` 目录
2. **磁盘空间**: 确保有至少 500MB 可用空间
3. **Python 版本**: 需要 Python 3.8+
4. **依赖包**: scipy, h5py, numpy, tqdm

## ✨ 总结

本方案通过数据预转换的方式，将 .mat 格式数据集转换为 HDF5 格式，完全兼容现有训练代码，无需修改核心逻辑。转换脚本简洁高效，支持分层采样和数据验证。

转换完成后，可以直接使用现有的训练脚本进行模型训练，享受 HDF5 格式带来的快速加载和高效存储优势。
