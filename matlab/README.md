# 雷达有源干扰信号调制数据集生成代码（Python 版本）

## 概述

本项目包含从 MATLAB 转换到 Python 的雷达有源干扰信号数据集生成代码。该代码生成并保存 12 种有源干扰信号的 3 种特征数据（时域、频域和时频域），包括 1D 序列模式和 2D 图像模式。

该代码可用于雷达干扰检测、识别或去噪领域。

---

## 目录

1. [环境要求](#环境要求)
2. [文件结构](#文件结构)
3. [函数列表](#函数列表)
4. [生成数据的格式](#生成数据的格式)
5. [各函数的噪声和干扰说明](#各函数的噪声和干扰说明)
6. [使用方法](#使用方法)
7. [参数配置](#参数配置)
8. [注意事项](#注意事项)

---

## 环境要求

运行前请安装以下 Python 包：

```bash
pip install numpy scipy matplotlib Pillow tqdm
```

所需库：
- `numpy`：数值计算和数组操作
- `scipy`：信号处理，包括 `spectrogram`、`butter`、`hilbert`
- `matplotlib`：绘制时频图
- `Pillow (PIL)`：图像缩放和保存
- `tqdm`：显示进度条

---

## 文件结构

```
Radar-Active-Jamming-Signal-Modulation-Dataset-Generating-Code-main/
|
|-- radar_jamming_signals.py     # 主 Python 脚本，包含全部 24 个信号生成函数 + 2 个辅助函数
|-- generate_data.py              # 批量生成数据集脚本（带 tqdm 进度条）
|-- README.md                     # 本文档
|-- *.m                           # 原始 MATLAB 源文件（23 个 .m 文件）
|
-- 输出目录（运行函数时自动生成）：
    D:\Radar_Jamming_Signal_Dataset\Test_data\dataset_img\       # 测试集图像
    D:\Radar_Jamming_Signal_Dataset\Test_data\dataset_seq\       # 测试集序列
    D:\Radar_Jamming_Signal_Dataset\Trainning_data\dataset_img\All_dB\  # 训练集图像
    D:\Radar_Jamming_Signal_Dataset\Trainning_data\dataset_seq\All_dB\  # 训练集序列
```

---

## 函数列表

`radar_jamming_signals.py` 文件共包含 **26 个函数**（2 个辅助 + 24 个信号生成）。

### 辅助函数

| 函数 | 用途 |
|------|------|
| `_save_spectrogram_image()` | 内部辅助：计算时频图，缩放为 224x224，保存为 PNG |
| `_generate_lfm()` | 内部辅助：生成基准 LFM 信号和公共参数 |

### 测试集函数（12 种信号，200 样本/信噪比，按信噪比分文件夹）

| 编号 | 函数 | 中文名称 | 信号类别 |
|:----:|------|---------|---------|
| 1 | `LFM()` | 雷达 LFM 线性调频信号 | 参考信号（无干扰，仅噪声） |
| 2 | `AM()` | 调幅干扰 AM (瞄准式) | 有源干扰 |
| 3 | `COMB()` | 梳状谱干扰 COMB | 有源干扰 |
| 4 | `FM()` | 调频干扰 FM | 有源干扰 |
| 5 | `ISRJ()` | 间歇采样转发干扰 ISRJ | 有源干扰 |
| 6 | `MNJ()` | 噪声乘积式灵巧噪声干扰 MNJ | 有源干扰 |
| 7 | `RMT()` | 距离维密集假目标干扰 RMT | 有源干扰 |
| 8 | `RGPO()` | 距离拖引干扰 RGPO | 有源干扰 |
| 9 | `R_VGPO()` | 距离速度联合拖引干扰 R-VGPO | 有源干扰 |
| 10 | `SMSP()` | 频谱弥散干扰 SMSP | 有源干扰 |
| 11 | `VGPO()` | 速度拖引干扰 VGPO | 有源干扰 |
| 12 | `VMT()` | 速度维密集假目标干扰 VMT | 有源干扰 |

### 训练集函数（12 种信号，800 样本/信噪比，All_dB 合并文件夹）

| 编号 | 函数 | 中文名称 |
|:----:|------|---------|
| 1 | `LFM_alldb()` | 雷达 LFM 线性调频信号 (All_dB) |
| 2 | `AM_alldb()` | 调幅干扰 AM (All_dB) |
| 3 | `COMB_alldb()` | 梳状谱干扰 COMB (All_dB) |
| 4 | `FM_alldb()` | 调频干扰 FM (All_dB) |
| 5 | `ISRJ_alldb()` | 间歇采样转发干扰 ISRJ (All_dB) |
| 6 | `MNJ_alldb()` | 噪声乘积式灵巧噪声干扰 MNJ (All_dB) |
| 7 | `RMT_alldb()` | 距离维密集假目标干扰 RMT (All_dB) |
| 8 | `RGPO_alldb()` | 距离拖引干扰 RGPO (All_dB) |
| 9 | `R_VGPO_alldb()` | 距离速度联合拖引干扰 R-VGPO (All_dB) |
| 10 | `SMSP_alldb()` | 频谱弥散干扰 SMSP (All_dB) |
| 11 | `VGPO_alldb()` | 速度拖引干扰 VGPO (All_dB) |
| 12 | `VMT_alldb()` | 速度维密集假目标干扰 VMT (All_dB) |

---

## 生成数据的格式

### 1. 图像格式（二维时频谱图）

- **分辨率**：224 x 224 像素（通过 PIL 缩放）
- **原始绘图**：由 `matplotlib.imshow()` 从 STFT（短时傅里叶变换）生成
- **STFT 参数**：`nfft=128`，`noverlap=127`，`window=hamming(128)`
- **文件格式**：`.png`
- **色图**：`viridis`
- **坐标轴**：隐藏（`axis('off')`）

### 2. 序列格式（一维频域）

- **数据类型**：复值 FFT 结果（1024 点）
- **数组形状**：`(1024,)`
- **处理过程**：`fftshift(fft(信号范围, n=1024))`
- **范围**：从目标回波区间提取（`range_tar` 到 `Nfast + range_tar`）
- **文件格式**：`.mat`（MATLAB v5 格式）
- **变量名**：
  - `J_fft` 表示干扰信号
  - `lfm_echo_fft` 表示参考 LFM 信号

---

## 关于 I/Q 路信号的重要说明

### 复值数组的 I/Q 路解释

`J_fft` 或 `lfm_echo_fft` 变量是一个**复数数组**，形状为 `(1024,)`，每个元素都是复数形式 `a + bj`。

**注意**：不能将前 512 个点理解为 I 路信号，后 512 个点理解为 Q 路信号。这种理解是**错误的**。

### 正确的理解方式

`J_fft` 是一个包含 1024 个**复数值**的数组。每个复数值本身同时包含了 I 路（实部）和 Q 路（虚部）信息：

- **每个复数点**：`a + bj`
  - `a`（实部）：对应 I 路（In-phase，同相）
  - `b`（虚部）：对应 Q 路（Quadrature，正交）

也就是说，**I 路和 Q 路信息同时存在于每个数据点中**，而不是分布在数组的前半部分和后半部分。

### 分离 I/Q 路信号的代码

```python
import scipy.io
import numpy as np

# 加载 .mat 序列文件
mat_data = scipy.io.loadmat('文件路径.mat')
J_fft = mat_data['J_fft'].flatten()  # 形状 (1024,)，复数数组

# I 路（实部）
I_channel = np.real(J_fft)

# Q 路（虚部）
Q_channel = np.imag(J_fft)
```

### 总结

| 项目 | 说明 |
|:-----|:-----|
| `J_fft` 形状 | `(1024,)` |
| 每个元素类型 | 复数（如 `complex64`） |
| I 路获取 | `np.real(J_fft)` |
| Q 路获取 | `np.imag(J_fft)` |
| 能否按前512后512划分 | 不能。每个点都包含完整的 I+jQ 信息 |

---

## 各函数的噪声和干扰说明

所有生成的信号均包含**加性高斯白噪声 (AWGN)**，除非另有说明。噪声使用 `np.random.randn()` 生成实部和虚部。

| 函数 | 信号类型 | 是否含噪声 | 是否含干扰 | 主要特性 |
|------|---------|---------|---------|---------|
| `LFM()` / `LFM_alldb()` | LFM 参考 | 是 | 否 | 无干扰的基准雷达信号 |
| `AM()` / `AM_alldb()` | AM（瞄准式） | 是 | 是 | 窄带瞄准噪声干扰 |
| `COMB()` / `COMB_alldb()` | 梳状谱 | 是 | 是 | 多锯齿波频率分量叠加 |
| `FM()` / `FM_alldb()` | 调频噪声 | 是 | 是 | 噪声调频干扰 |
| `ISRJ()` / `ISRJ_alldb()` | 间歇采样转发 | 是 | 是 | 采样-切片-转发 |
| `MNJ()` / `MNJ_alldb()` | 噪声乘积 | 是 | 是 | 窄带噪声×回波信号 |
| `RMT()` / `RMT_alldb()` | 距离维密集假目标 | 是 | 是 | 距离维多个延时假目标 |
| `RGPO()` / `RGPO_alldb()` | 距离拖引 | 是 | 是 | 目标在距离维被拖引 |
| `R_VGPO()` / `R_VGPO_alldb()` | 距离速度联合拖引 | 是 | 是 | 距离+速度维联合拖引 |
| `SMSP()` / `SMSP_alldb()` | 频谱弥散 | 是 | 是 | 频谱展开与脉冲串卷积 |
| `VGPO()` / `VGPO_alldb()` | 速度拖引 | 是 | 是 | 多普勒频移速度拖引 |
| `VMT()` / `VMT_alldb()` | 速度维密集假目标 | 是 | 是 | 速度维多个频移假目标 |

---

## 使用方法

### 基本用法

1. 导入模块：

```python
import radar_jamming_signals as rjs
```

2. 运行特定函数生成数据：

```python
# 生成特定信号类型的测试集
rjs.LFM()      # 参考信号
rjs.AM()       # AM 干扰
rjs.COMB()     # 梳状谱干扰
# ... 等等

# 生成训练集（All_dB 文件夹）
rjs.LFM_alldb()
rjs.AM_alldb()
rjs.COMB_alldb()
# ... 等等
```

3. 使用 `generate_data.py` 批量生成：

```python
python generate_data.py
```

运行时会显示 tqdm 进度条，展示当前正在生成哪种信号。

### 读取已保存的文件

在 Python 中：

```python
import scipy.io
from PIL import Image
import numpy as np

# 加载 .mat 序列文件
mat_data = scipy.io.loadmat('文件路径.mat')
fft_sequence = mat_data['J_fft'].flatten()  # 形状 (1024,)

# I 路（实部）
I_channel = np.real(fft_sequence)

# Q 路（虚部）
Q_channel = np.imag(fft_sequence)

# 加载 PNG 图像
img = Image.open('图像路径.png')
img_array = np.array(img)
```

在 MATLAB 中：

```matlab
% 加载序列
load('1.mat');  % 变量 J_fft 或 lfm_echo_fft 进入工作区

% 加载图像
img = imread('1.png');
```

---

## 参数配置

### 公共参数

| 参数 | 值 | 说明 |
|------|------|------|
| `fs` | 100 MHz | 采样频率 |
| `PRI` | 100 us | 脉冲重复周期 |
| `fc` | 10 MHz | 载频 |
| `B` | 40 MHz | 信号带宽 |
| `taup` | 40 us | 脉宽 |
| `JSR` | 5 dB | 干扰信号比 |
| `SNR` 范围 | -20 到 10 dB | 信噪比范围 |

### 样本数量

| 函数集 | 每个信噪比样本数 | 每种信号总样本 | 文件夹结构 |
|--------|----------------|---------------|----------|
| 测试集 | 200 | 3,200 | `信噪比_dB/信号名/` |
| 训练集 | 800 | 12,800 | `All_dB/信号名/` |

---

## 注意事项

1. **磁盘空间**：完整数据集生成需要较大的磁盘空间（数 GB 到数十 GB）。

2. **运行时间**：每个函数可能需要较长时间（数分钟到数小时），代码已集成 `tqdm` 进度条显示进度。

3. **噪声添加**：所有信号（包括 LFM 参考）均包含高斯白噪声。噪声在叠加前已归一化为单位标准差。

4. **归一化**：所有输出信号在保存前都经过 `J = J / max(abs(J))` 归一化处理。

5. **MATLAB 兼容性**：`.mat` 文件以 v5 格式保存，可在 MATLAB 和 Python 中读取。复数格式兼容，可直接用 `np.real()` 和 `np.imag()` 分离 I/Q 路。

6. **原始 MATLAB 文件**：原始 `.m` 代码共 23 个文件。Python 版 `LFM_alldb()` 为额外补充，无对应 `.m` 文件。
