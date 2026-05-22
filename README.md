# 雷达有源干扰信号调制数据集 - 完整文档

> 快速生成雷达干扰信号数据集，支持 GPU 加速、CPU 多进程并行，最高 45 倍性能提升

**版本**：v3.1 | **更新日期**：2026-05-18

---

## 目录

1. [快速开始](#1-快速开始)
2. [数据集说明](#2-数据集说明)
3. [使用指南](#3-使用指南)
4. [性能优化](#4-性能优化)
5. [数据格式](#5-数据格式)
6. [常见问题](#6-常见问题)
7. [更新日志](#7-更新日志)

---

## 1. 快速开始

### 1.1 安装依赖

```bash
pip install numpy scipy matplotlib Pillow tqdm cupy-cuda12x
```

### 1.2 生成数据集（2 步完成）

```bash
# 步骤 1：生成 .mat 文件
python generate_data.py --gpu --mat-only --data-num 2
# 或使用 CPU 多进程并行
python generate_data_parallel.py --mat-only
# 可设置随机种子保证可复现性（默认seed=42）
python generate_data.py --gpu --mat-only --seed 123
python generate_data_parallel.py --mat-only --seed 123

# 步骤 2：转换并按 8:1:1 划分数据集
cd radchar_refactored3
python convert_mat_to_h5.py

# 结果位于：radchar_refactored3/Bear_data/
```

### 1.3 加载和使用

```python
import h5py
import numpy as np

# 加载训练集
with h5py.File('radchar_refactored3/Bear_data/RadChar-Train.h5', 'r') as f:
    X = f['iq'][:]           # (N, 1024) 复数时域 IQ
    y = f['labels'][:]       # (N,) 标签 0-11

# 转换为实数特征
X_real = np.hstack([np.real(X), np.imag(X)])  # (N, 2048)

print(f"训练集: {X.shape[0]} 样本")
```

---

## 2. 数据集说明

### 2.1 数据集规模

| 项目 | 值 |
|:----|:---|
| 信号类型 | 12 种（1 个参考信号 + 11 种干扰信号） |
| SNR 范围 | -10 到 10 dB，步长 2 dB（11 个值） |
| 总样本数 | 800 样本/SNR × 11 SNR × 12 信号 = **105,600 样本** |
| 划分比例 | 训练集 80% | 验证集 10% | 测试集 10% |
| 输出格式 | HDF5 文件（复数时域 IQ，1024 点） |
| 磁盘占用 | 约 10-20 GB |

### 2.2 信号类型

| 编号 | 缩写 | 信号名称 | 类别 | 说明 |
|:----:|:--------:|:-----:|:-----:|:-----:|
| 0 | LFM | 线性调频信号 | 参考 | 无干扰的雷达回波（仅含噪声） |
| 1 | AM | 调幅干扰 | 干扰 | 瞄准式调幅噪声干扰 |
| 2 | COMB | 梳状谱干扰 | 干扰 | 多锯齿波频率分量叠加 |
| 3 | FM | 调频干扰 | 干扰 | 噪声调频干扰 |
| 4 | ISRJ | 间歇采样转发干扰 | 干扰 | 采样-切片-转发欺骗 |
| 5 | MNJ | 噪声乘积式灵巧噪声干扰 | 干扰 | 窄带噪声与回波相乘 |
| 6 | RMT | 距离维密集假目标干扰 | 干扰 | 距离维多个延时假目标 |
| 7 | RGPO | 距离拖引干扰 | 干扰 | 目标在距离维被拖引 |
| 8 | R_VGPO | 距离速度联合拖引干扰 | 干扰 | 距离+速度维联合拖引 |
| 9 | SMSP | 频谱弥散干扰 | 干扰 | 频谱展开与脉冲串卷积 |
| 10 | VGPO | 速度拖引干扰 | 干扰 | 多普勒频移速度拖引 |
| 11 | VMT | 速度维密集假目标干扰 | 干扰 | 速度维多个频移假目标 |

### 2.3 雷达参数

| 参数 | 值 | 说明 |
|:-----|:---|:-----|
| 采样频率 (fs) | 100 MHz | 信号采样率 |
| 脉冲重复周期 (PRI) | 100 μs | 雷达脉冲重复间隔 |
| 载频 (fc) | 10 MHz | 中心频率 |
| 信号带宽 (B) | 40 MHz | LFM 信号带宽 |
| 脉宽 (taup) | 40 μs | 脉冲持续时间 |
| 干扰信号比 (JSR) | 5 dB | 干扰与信号功率比 |
| 噪声类型 | AWGN | 加性高斯白噪声 |

---

## 3. 使用指南

### 3.1 命令行参数

#### generate_data.py （基础版）

```bash
python generate_data.py [选项]

参数：
  --gpu              使用 GPU 加速（推荐）
  --cpu              使用 CPU 模式（默认）
  --mat-only         仅生成 .mat 文件，跳过图片（45x 加速）
  --data-num N       每个 SNR 的样本数（默认：800）
  --snr-start N      SNR 起始值（默认：-10）
  --snr-end N        SNR 结束值（默认：10）
  --seed N           随机种子，保证可复现性（默认：42）

示例：
  python generate_data.py --gpu --mat-only            # 生成完整数据集
  python generate_data.py --gpu --mat-only --data-num 100     # 自定义样本数
```

#### generate_data_parallel.py （CPU 多进程并行版，推荐无 GPU 时使用）

```bash
python generate_data_parallel.py [选项]

专为双路/多核 CPU 优化，自动绑定进程到各核心，确保 100% 利用

参数：
  --mat-only         仅生成 .mat 文件，跳过图片（大幅加速）
  --num-workers N    并行进程数（默认：等于 CPU 逻辑核心数）
  --data-num N       每个 SNR 的样本数（默认：800）
  --snr-start N      SNR 起始值（默认：-10）
  --snr-end N        SNR 结束值（默认：10）
  --seed N           随机种子，保证可复现性（默认：42）
  --no-bind-cpu      禁用 CPU 亲和性绑定（推荐用于双路服务器，让操作系统调度）

示例：
  python generate_data_parallel.py --mat-only                  # 使用全部 CPU 核心
  python generate_data_parallel.py --mat-only --num-workers 64 # 64 进程（双路 32 核）
```

#### radchar_refactored3/convert_mat_to_h5.py

```bash
cd radchar_refactored3
python convert_mat_to_h5.py

功能：
  - 从 ../dataset/Training_data/dataset_seq/All_dB 读取所有 .mat 文件
  - 按 8:1:1 比例分层划分训练/验证/测试集
  - 保存为 HDF5 格式到 Bear_data/

输出：
  Bear_data/RadChar-Train.h5  # 训练集
  Bear_data/RadChar-Val.h5    # 验证集
  Bear_data/RadChar-Test.h5   # 测试集
```

### 3.2 目录结构

```
<项目根目录>/
├── radar_jamming_signals.py    # 主信号生成模块
├── generate_data.py             # 基础版批量生成脚本
├── generate_data_parallel.py    # CPU 多进程并行版
├── README.md                    # 本文档
├── matlab/                      # 原始 MATLAB 源文件
│
├── dataset/                     # 数据集输出
│   └── Training_data/
│       └── dataset_seq/
│           └── All_dB/
│               ├── LFM/
│               │   ├── 1.mat
│               │   └── ... (800 个 × 11 SNR)
│               └── ... (12 种信号)
│
└── radchar_refactored3/         # 训练模型代码
    ├── convert_mat_to_h5.py     # 数据集转换脚本
    ├── radchar_training.py      # 模型训练
    ├── radchar_test.py          # 模型测试
    └── Bear_data/               # 转换后的 HDF5 数据集
        ├── RadChar-Train.h5
        ├── RadChar-Val.h5
        └── RadChar-Test.h5
```

---

## 4. 性能优化

### 4.1 性能对比

| 模式 | 耗时（data_num=800） | 加速比 |
|:----|:----------------|:------|
| CPU 单进程 + 图片 | ~3-4 小时 | 1.0x |
| **GPU + MAT-only** | **~40 秒** | **45.5x** |
| **CPU 多进程 (64 核) + MAT-only** | **~2-3 分钟** | **~30x** |

### 4.2 性能选择建议

| 硬件配置 | 推荐方案 | 预计耗时 |
|:----|:----|:----|
| 有 NVIDIA GPU | `generate_data.py --gpu --mat-only` | ~40 秒 |
| 多核 CPU (16 核+) | `generate_data_parallel.py --mat-only` | ~5-10 分钟 |
| 双路 CPU (32 核+) | `generate_data_parallel.py --mat-only --num-workers 64` | ~2-3 分钟 |
| 无 GPU，单核 | `generate_data.py --cpu --mat-only` | ~1-2 小时 |

### 4.5 GPU 要求

- **硬件**：NVIDIA GPU（支持 CUDA）
- **软件**：CUDA 工具包 + CuPy
- **安装**：`pip install cupy-cuda12x`（CUDA 12.x）或 `cupy-cuda11x`（CUDA 11.x）
- **验证**：运行 `nvidia-smi` 查看 GPU 状态

---

## 5. 数据格式

### 5.1 HDF5 数据集文件

```python
import h5py

# 加载数据集
with h5py.File('radchar_refactored3/Bear_data/RadChar-Train.h5', 'r') as f:
    iq_data = f['iq'][:]           # (N, 1024) 复数时域 IQ
    labels = f['labels'][:]       # (N,) 整数标签 0-11

print(f"形状: {iq_data.shape}")
print(f"类型: {iq_data.dtype}")
print(f"标签: {set(labels)}")
```

### 5.2 I/Q 路信号（重要）

**正确理解**：每个复数点同时包含 I 路（实部）和 Q 路（虚部）

```python
iq = data['iq'][:]  # (N, 1024) 复数数组

# 正确的 I/Q 分离
I_channel = np.real(iq)  # (N, 1024) 实部 = I 路
Q_channel = np.imag(iq)  # (N, 1024) 虚部 = Q 路

# 拼接为实数特征用于训练
X = np.hstack([I_channel, Q_channel])  # (N, 2048)
```

### 5.3 训练模型示例

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import h5py
import numpy as np

# 加载数据
with h5py.File('radchar_refactored3/Bear_data/RadChar-Train.h5', 'r') as f:
    X_train = f['iq'][:]
    y_train = f['labels'][:]

with h5py.File('radchar_refactored3/Bear_data/RadChar-Test.h5', 'r') as f:
    X_test = f['iq'][:]
    y_test = f['labels'][:]

# 转换为实数特征
X_train_real = np.hstack([np.real(X_train), np.imag(X_train)])
X_test_real = np.hstack([np.real(X_test), np.imag(X_test)])

# 训练模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_real, y_train)

# 评估
y_pred = clf.predict(X_test_real)
print(classification_report(y_test, y_pred))
```

---

## 6. 常见问题

### 6.1 CuPy 安装失败

```bash
# 查看 CUDA 版本
nvidia-smi

# 安装对应版本
pip install cupy-cuda12x  # CUDA 12.x
pip install cupy-cuda11x  # CUDA 11.x
```

### 6.2 GPU 内存不足

```bash
# 方案 1：减少样本数
python generate_data.py --gpu --mat-only --data-num 100

# 方案 2：使用 CPU 模式
python generate_data.py --cpu --mat-only
```

### 6.3 生成速度慢

**确保使用 `--mat-only` 选项**：

```bash
python generate_data.py --gpu --mat-only  # ✅ 快（45x）
python generate_data.py --gpu           # ❌ 慢（含图片）
```

### 6.4 数据集路径

数据保存在以下位置：

```bash
# 原始 .mat 文件
./dataset/Training_data/dataset_seq/All_dB/

# 转换后的 HDF5 数据集
./radchar_refactored3/Bear_data/
```

---

## 7. 可复现性说明

为了保证每次生成的数据集完全一致，请使用以下方法：

### 7.1 使用相同的随机种子
所有生成脚本都支持`--seed`参数，默认值为42：
```bash
# 使用基础版
python generate_data.py --gpu --mat-only --seed 42

# 使用并行版
python generate_data_parallel.py --mat-only --seed 42
```

### 7.2 并行版的确定性保证
在并行版本中，每个(SNR, 信号类型)组合都会生成独立的确定性随机种子，保证：
- 无论进程数如何变化，数据始终相同
- 无论任务执行顺序如何变化，数据始终相同

### 7.3 convert_mat_to_h5.py 种子设置
转换脚本也使用固定种子（42）来保证划分结果一致。

---

## 8. 更新日志

### v3.1 

**新增功能**：

- ✅ 添加随机种子支持，保证可复现性
- ✅ `generate_data.py` 和 `generate_data_parallel.py` 都支持 `--seed` 参数
- ✅ 并行版为每个(SNR, 信号类型)组合生成确定性独立种子
- ✅ 完善文档中的可复现性说明

### v3.0 

**重大更新**：
- ✅ 简化工作流程：统一生成一个完整数据集
- ✅ 按 8:1:1 比例分层划分训练/验证/测试集
- ✅ 删除测试集单独生成，避免重复
- ✅ 更新 `convert_mat_to_h5.py` 支持新流程

### v2.3 

**新增功能**：
- ✅ 新增 `generate_data_parallel.py` - CPU 多进程并行版
- ✅ 双路 CPU 亲和性绑定，确保所有核心 100% 利用
- ✅ 超细化任务粒度（SNR × 信号类型），最大化并行效率
- ✅ 测试集统一使用 All_dB 连续编号，与训练集逻辑一致

### v2.2 

**修复**：
- ✅ ISRJ 间歇采样转发干扰：修正 range_tar 边界计算，防止数组越界
- ✅ ISRJ 间歇采样转发干扰：修正 add_len 信号长度计算，移除多余 -1
- ✅ MNJ 噪声乘积式灵巧噪声干扰：Butterworth 滤波器阶数 5→10，与 MATLAB 一致
- ✅ 训练集路径从硬编码改为配置变量，提升灵活性

### v2.1 

**新增功能**：
- ✅ 数据集路径修改为当前目录下的 `./dataset/` 文件夹
- ✅ 自动创建输出目录
- ✅ 整合完整文档

### v2.0 

**重大更新**：

- ✅ SNR 范围修改为 -10 到 10 dB（11 个值）
- ✅ 添加 MAT-only 快速生成模式（**45x 加速**）
- ✅ 添加数据集整合脚本（train/val/test 自动划分）
- ✅ GPU 加速支持（CuPy 后端）
- ✅ 完整文档和使用指南

### v1.0 (2024-XX-XX)

- 初始版本（MATLAB 转 Python）
- 12 种信号类型
- SNR 范围 -20 到 10 dB

---

## 附录

### A. 项目文件说明

| 文件 | 说明 |
|:-----|:-----|
| `radar_jamming_signals.py` | 主信号生成模块（1809 行，24 个信号函数） |
| `generate_data.py` | 基础版批量生成脚本（支持 GPU/CPU，MAT-only 模式） |
| `generate_data_parallel.py` | CPU 多进程并行版（双路 CPU 优化，推荐无 GPU 时使用） |
| `README.md` | 完整文档（本文件） |
| `matlab/*.m` | 原始 MATLAB 源文件（参考用） |

### B. 依赖库版本

```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
Pillow>=8.0.0
tqdm>=4.60.0
cupy-cuda12x>=12.0.0  # GPU 加速（可选）
```

### C. 引用格式

如果使用本数据集，请引用：

```bibtex
@dataset{radar_jamming_2024,
  title={Radar Active Jamming Signal Modulation Dataset},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/your-repo}}
}
```

### D. 许可证

本数据集仅供学术研究使用。
