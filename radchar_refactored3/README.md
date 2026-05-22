# RadChar 雷达信号分类模型

## 项目概述

RadChar 是一个用于雷达信号分类的深度学习模型，基于 **CNN + Mamba3 + Temporal Encoder** 架构，可对 IQ 数据序列（长度 1024）进行分类。支持三种时序建模模块：**ProbSparse Self-Attention**、**Informer**、**FEDformer**。

---

## 模型架构

### 架构组成

```
Input: I (B, 1024), Q (B, 1024)
  ↓
ComplexInputAugment: CFO/STO 数据增强
  ↓
IQInputProcessor: (B, 3, 1024) I/Q 两路 + 时频变换特征 (FFT/STFT/GLCT)
  ↓
CNN Stem (3→64→96)
  ↓
DilatedConvBlock (空洞卷积: 1, 2, 4, 8 扩张率) ← 扩大感受野
  ↓
Layer1 (96→128) + MaxPool (可选)
  ↓
Layer2 (128→192) + MaxPool (可选)
  ↓
Layer3 (192→256) + MaxPool (可选)
  ↓
Temporal Projection (256→192)
  ↓
分支:
  ├─ Mamba3 Block (可选) → 时序建模
  └─ Temporal Encoder (可选) → 长期依赖
      ├─ ProbSparse Encoder (默认)
      ├─ Informer Encoder
      └─ FEDformer Encoder
  ↓
LearnableFeatureFusion: 可学习加权融合 (mean + max + mamba_out + attn_out)
  ↓
MLP Head → Classifier
```

### 核心组件

| 组件 | 说明 |
|------|------|
| **ComplexInputAugment** | CFO (载波频率偏移) 和 STO (符号定时偏移) 增强 |
| **IQInputProcessor** | I/Q 两路信号 + 时频变换特征（3 通道），支持 FFT/STFT/GLCT 三种模式 |
| **DilatedConvBlock** | 空洞卷积 (扩张率 1, 2, 4, 8) 扩大感受野不降低分辨率 |
| **CNN Stem** | 基础特征提取，3 通道输入 (I/Q/时频特征) |
| **Residual Blocks** | 可配置池化层数 (默认 2 次 MaxPool) |
| **Mamba3Block** | 状态空间模型，时序建模，使用可学习的 A_log 参数稳定训练 |
| **Temporal Encoder** | 时序建模模块，支持 ProbSparse/Informer/FEDformer 三种选择 |
| **ProbSparse Encoder** | 概率稀疏自注意力，建模长期依赖 |
| **Informer Encoder** | Informer 模型，使用 ProbAttention + ConvLayer distilation |
| **FEDformer Encoder** | FEDformer 模型，使用 AutoCorrelation (FFT-based) + Series Decomp |
| **LearnableFeatureFusion** | 可学习加权融合 + 门控机制 |
| **RMSNorm** | 轻量级归一化，用于稳定深度网络训练 |

### 数据增强 (ComplexInputAugment)

雷达信号对 CFO (载波频率偏移) 和 STO (符号定时偏移) 非常敏感。

| 增强方式 | 参数 | 说明 |
|---------|------|------|
| STO (符号定时偏移) | `sto_max=8` | 随机时间平移 ±8 样本 |
| CFO (载波频率偏移) | `cfo_std=0.02` | 随机相位旋转 (2π × CFO) |

### 空洞卷积 (DilatedConvBlock)

使用扩张率递增的空洞卷积，在不降低分辨率的情况下扩大感受野：

| 扩张率 | 核大小 | 感受野 |
|--------|--------|--------|
| 1 | 5 | 5 |
| 2 | 5 | 9 |
| 4 | 5 | 17 |
| 8 | 5 | 33 |

### 可学习特征融合 (LearnableFeatureFusion)

替代简单的 mean + max 拼接，使用可学习权重和门控机制：

```python
# 加权求和
weighted_sum = Σ(w_i * feature_i)

# 门控机制 (输入所有特征，输出每个特征的重要性)
gate_values = sigmoid(FC(concat(features)))
gated = Σ(gate_i * feature_i)

# 最终输出
Output = 0.5 * weighted_sum + 0.5 * gated
```

---

## 模型架构配置

### ModelArchitectureConfig

```python
@dataclass
class ModelArchitectureConfig:
    # CNN Stem 配置
    use_complex_conv: bool = False       # 复数卷积层（仅用于 8 通道输入）
    use_multiscale_fusion: bool = False  # 多尺度特征融合
    use_enhanced_pooling: bool = False   # 增强特征池化

    # 池化配置
    pool_layers: int = 2                # MaxPool 次数 (2=1024→256, 4=1024→64)
    use_dilated_conv: bool = True     # 使用空洞卷积扩大感受野

    # STFT 参数（保留配置接口）
    stft_n_fft: int = 64
    stft_hop_length: int = 16
    # 时频变换模式: "fft" / "stft" / "glct"
    timefreq_mode: str = "fft"

    # Mamba3 配置
    use_mamba: bool = True               # 启用 Mamba3
    mamba3_layers: int = 2              # Mamba3 层数
    mamba3_d_state: int = 128           # 状态维度
    mamba3_expand: int = 2              # 扩展因子
    mamba3_headdim: int = 64            # 头维度
    mamba3_dropout: float = 0.1         # Dropout
    use_complex_mamba: bool = False      # 复数 Mamba (待完善)

    # ProbSparse / Temporal Encoder 配置
    use_prob_sparse_attention: bool = True  # 启用时序编码器
    prob_sparse_layers: int = 3         # Encoder 层数
    prob_sparse_heads: int = 6           # 注意力头数
    prob_sparse_ffn_dim: int = 512      # 前馈网络维度
    prob_sparse_dropout: float = 0.1     # Dropout
    prob_sparse_top_k: int = 32          # 稀疏采样数

    # Temporal Module Selection: prob_sparse, informer, fedformer
    temporal_module: str = "prob_sparse"

    # Informer 参数
    informer_factor: int = 5             # ProbAttention 因子
    informer_distil: bool = True         # 使用 distilation

    # FEDformer 参数
    fedformer_modes: int = 32            # FFT 模式数
    fedformer_moving_avg: int = 25       # 移动平均窗口大小

    # 可学习特征融合
    use_learnable_fusion: bool = True    # 使用可学习加权融合
```

---

## 训练策略

### 学习率调度

支持三种学习率调度器：

| 调度器 | 说明 | 适用场景 |
|--------|------|---------|
| `plateau` (默认) | 验证集不提升时减半学习率 | 通用 |
| `cosine` | 余弦退火 + 热重启 | 周期性训练 |
| `onecycle` | 单周期策略 | 快速收敛 |

### 数据增强

#### 基础增强 (RadCharDataset)

| 增强方式 | 概率 | 说明 |
|---------|------|------|
| 相位偏移 | 75% | IQ乘以exp(j·phase) |
| 幅度缩放 | 60% | 均匀缩放 [0.88, 1.12] |
| 时间偏移 | 55% | 循环移位 ±24 样本 |
| 高斯噪声 | 60% | SNR自适应噪声 |
| 时间掩码 | 20% | 随机区间置零 |
| 频率偏移 | 25% | 频谱平移 |
| 多径效应 | 15% | 延迟回波叠加 |
| LFM 增强 (chirp) | 60% (LFM 专用) | 加强"干净回波"线性调频特征 |

#### CFO/STO 增强 (ComplexInputAugment)

在 IQInputProcessor 之后应用，模拟载波频率偏移和符号定时偏移：

```python
# CFO: 随机相位旋转
phase_offsets = torch.randn(B) * cfo_std * 2 * π
iq_rotated = iq * exp(j * phase_offsets)

# STO: 时间平移
shifts = torch.randint(-sto_max, sto_max + 1, (B,))
iq_shifted = roll(iq, shifts)
```

### 损失函数

- **CrossEntropy** (默认): 标准交叉熵损失
- Focal Loss 已移除

### 早停机制

训练支持智能早停机制，当验证集性能连续多轮未提升时自动停止训练，防止过拟合。

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--patience` | 30 | 验证集不提升时等待的最大轮数 |
| `--early_stopping_delta` | 0.0 | 被认为是"提升"所需的最小提升值 (百分比) |
| `--early_stopping_cooldown` | 0 | 验证集提升后，等待多少轮再开始计数 |

#### 组合策略

```bash
# 标准早停 (任何提升都重置 patience)
python radchar_training.py --no-resume --patience 10

# 需要显著提升才重置 (delta=0.01 意味着 acc 需提升 1% 才算有效提升)
python radchar_training.py --no-resume --patience 10 --early_stopping_delta 0.01

# 提升后冷却 (避免因短期波动而过早停止)
python radchar_training.py --no-resume --patience 10 --early_stopping_cooldown 3
```

---

## 使用方法

### 基础训练

```bash
python radchar_training.py --no-resume
```

> **注意**：默认 `balance_strategy=none`（训练集已均衡，无需过采样），
> `adaptive_class_weights=False`（关闭自适应类别权重）。
> 如需手动启用：`--balance_strategy both --adaptive_class_weights`

### 池化层配置

```bash
# 2 次 MaxPool (512→128): 适合 Mamba/Attention (默认)
python radchar_training.py --no-resume --pool_layers 2

# 4 次 MaxPool (512→32): 传统配置
python radchar_training.py --no-resume --pool_layers 4
```

### 空洞卷积

```bash
# 启用空洞卷积 (默认)
python radchar_training.py --no-resume --use_dilated_conv

# 禁用空洞卷积
python radchar_training.py --no-resume --no_dilated_conv
```

### 时频变换模式（--timefreq_mode）

第三通道支持三种时频变换，通过 `--timefreq_mode` 参数选择：

| 模式 | 说明 |
|------|------|
| `fft` (默认) | FFT 幅度谱，保留全频段 |
| `stft` | STFT 时频谱，频率维平均后插值到序列长度 |
| `glct` | 双 n_fft 的 STFT 乘积增强，近似 GLCT |

```bash
# 使用 STFT
python radchar_training.py --no-resume --timefreq_mode stft

# 使用 GLCT
python radchar_training.py --no-resume --timefreq_mode glct
```

### CFO/STO 数据增强

CFO/STO 增强在模型内部自动应用（训练时），无需额外参数。

### 可学习特征融合

```bash
# 启用可学习融合 (默认)
python radchar_training.py --no-resume --use_learnable_fusion

# 禁用可学习融合 (使用拼接)
python radchar_training.py --no-resume --no_learnable_fusion
```

### Mamba3 参数

```bash
# 调整 Mamba3 层数
python radchar_training.py --no-resume --mamba3_layers 4

# 调整状态维度
python radchar_training.py --no-resume --mamba3_d_state 256

# 禁用 Mamba3
python radchar_training.py --no-resume --no_mamba
```

### ProbSparse 参数

```bash
# 禁用 ProbSparse Attention
python radchar_training.py --no-resume --no_prob_sparse

# 增加层数
python radchar_training.py --no-resume --prob_sparse_layers 4
```

### 组合架构

```bash
# 仅 Mamba3
python radchar_training.py --no-resume --use_mamba --no_prob_sparse

# 仅 ProbSparse
python radchar_training.py --no-resume --no_mamba --use_prob_sparse_attention

# Mamba3 + 时序编码器 (默认)
python radchar_training.py --no-resume --use_mamba --use_prob_sparse_attention

# 完整配置: 2次池化 + 空洞卷积 + 可学习融合
python radchar_training.py --no-resume --pool_layers 2 --use_dilated_conv --use_learnable_fusion
```

### 时序建模模块选择

```bash
# ProbSparse (默认)
python radchar_training.py --no-resume --temporal_module prob_sparse

# Informer (使用 ProbAttention + ConvLayer distilation)
python radchar_training.py --no-resume --temporal_module informer

# Informer 禁用 distilation
python radchar_training.py --no-resume --temporal_module informer --no_informer_distil

# FEDformer (使用 AutoCorrelation FFT + Series Decomp)
python radchar_training.py --no-resume --temporal_module fedformer

# FEDformer 自定义移动平均窗口
python radchar_training.py --no-resume --temporal_module fedformer --fedformer_moving_avg 31

# 仅 Mamba3 (不使用时序编码器)
python radchar_training.py --no-resume --no_prob_sparse
```

---

## 命令行参数

### 架构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pool_layers` | 2 | MaxPool 层数 (2=1024→256, 4=1024→64) |
| `--use_dilated_conv` | True | 使用空洞卷积扩大感受野 |
| `--use_complex_conv` | False | 启用复数卷积层 |
| `--use_multiscale_fusion` | True | 启用多尺度特征融合 |
| `--use_enhanced_pooling` | True | 启用增强特征池化 |
| `--use_learnable_fusion` | True | 使用可学习特征融合 |
| `--timefreq_mode` | fft | 时频变换模式: fft/stft/glct |

### Mamba3 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_mamba` | True | 启用 Mamba3 层 |
| `--mamba3_layers` | 2 | Mamba3 层数 |
| `--mamba3_d_state` | 128 | 状态维度 |
| `--mamba3_expand` | 2 | 扩展因子 |
| `--mamba3_headdim` | 64 | 头维度 |
| `--mamba3_dropout` | 0.2 | Dropout |
| `--use_complex_mamba` | False | 复数 Mamba (待完善) |

### ProbSparse 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_prob_sparse_attention` | True | 启用时序编码器 |
| `--prob_sparse_layers` | 3 | Encoder 层数 |
| `--prob_sparse_heads` | 6 | 注意力头数 |
| `--prob_sparse_ffn_dim` | 512 | 前馈网络维度 |
| `--prob_sparse_dropout` | 0.2 | Dropout |
| `--prob_sparse_top_k` | 32 | 稀疏采样数 |

### Temporal Module 选择

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--temporal_module` | prob_sparse | 时序建模模块: prob_sparse, informer, fedformer |
| `--informer_factor` | 5 | Informer attention factor |
| `--informer_distil` | True | Informer 使用 distilation |
| `--fedformer_modes` | 32 | FEDformer FFT 模式数 |
| `--fedformer_moving_avg` | 25 | FEDformer 移动平均窗口大小 |

### 早停参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--patience` | 30 | 早停耐心值 |
| `--early_stopping_delta` | 0.0 | 早停最小提升阈值 |
| `--early_stopping_cooldown` | 0 | 早停冷却 epoch 数 |

### 学习率调度

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr_scheduler` | cosine | 学习率调度器: plateau/cosine/onecycle |
| `--max_lr` | 1e-3 | OneCycleLR 最大学习率 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr` | 1e-4 | 学习率 |
| `--weight_decay` | 0.04 | 权重衰减 |
| `--batch_size` | 16 | 批次大小（1024 序列长度建议值） |
| `--epochs` | 60 | 训练轮数 |
| `--label_smoothing` | 0.0 | 标签平滑 |

---

## 架构对比

| 配置 | 序列长度 | 感受野 | 参数量 | 适用场景 |
|------|----------|--------|--------|----------|
| pool=4 | 32 | 大 | ~4M | 传统配置 |
| **pool=2 + dilated** | 256 | **大** | **~3.2M** | **Mamba/Attention 首选（SEQ_LEN=1024）** |
| pool=2 | 128 | 中 | ~2.8M | 标准配置 |

---

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (用于GPU训练)

---

## 输出文件

```
models/
├── radchar_best.pth        # 最佳模型权重 (验证集提升时自动保存)
├── radchar_checkpoint.pth  # 训练检查点

results/
├── radchar_test_stats.json # 测试统计
```

---

## 文件结构

```
radchar_refactored/
├── radchar_model.py         # 模型定义 (CNN+Mamba3+TemporalEncoder)
├── radchar_training.py      # 训练脚本
├── radchar_test.py         # 测试脚本
├── radchar_ensemble.py      # 集成测试脚本
├── radchar_class_config.py  # 类别配置
├── radchar_classes.json     # 类别映射配置
├── requirements.txt         # 依赖列表
├── readme.md               # 本文档
└── models/                  # 模型输出目录
```

## 时序建模模块对比

| 模块 | 机制 | 特点 | 适用场景 |
|------|------|------|----------|
| **ProbSparse** | Top-k 稀疏注意力 | O(n log n) 复杂度，保留最强注意力连接 | 长期依赖，平衡效率与效果 |
| **Informer** | ProbAttention + Distilation | 多层 ConvLayer 下采样，蒸馏加速 | 超长序列，低延迟需求 |
| **FEDformer** | AutoCorrelation (FFT) | FFT 周期发现 + 移动平均分解 | 周期性强的时间序列 |

---

## 调参建议

### Temporal Encoder 模块选择

| 场景 | 推荐模块 | 说明 |
|------|---------|------|
| 默认 / 通用 | prob_sparse | 成熟稳定，平衡效果好 |
| 超长序列 (>1024) | informer | 蒸馏机制降低复杂度 |
| 周期性强信号 | fedformer | FFT 周期发现更适合 |
| 显存受限 | informer (distil=True) | 蒸馏减少计算量 |

### Mamba3 + Temporal Encoder

| 问题 | 解决方案 |
|------|---------|
| 梯度爆炸/消失 | 已修复：使用标准 RMSNorm + 可学习 A 矩阵参数化 |
| 训练不稳定 | 减小 `--lr` 或增大 `--mamba3_dropout` |
| 过拟合 | 增大 `--mamba3_dropout` 或 `--prob_sparse_dropout` |
| 显存不足 | 减小 `--batch_size` 或 `--mamba3_layers` |
| 欠拟合 | 增加 `--mamba3_layers` 或 `--mamba3_d_state` |
| 收敛慢 | 尝试 `--lr_scheduler onecycle` |

### 池化配置

| 场景 | pool_layers | use_dilated_conv | 说明 |
|------|-------------|-------------------|------|
| Mamba/Attention (SEQ_LEN=1024) | 2 | True | 保持 256 序列长度 |
| 传统 CNN | 4 | False | 传统配置 |
| 平衡 | 2 | True | 平衡配置 |

---

## 梯度问题排查

### 症状诊断

| 症状 | 可能原因 |
|------|---------|
| loss 突然变成 NaN | 梯度爆炸，检查学习率是否过高 |
| loss 几乎不下降 | 梯度消失，检查归一化层配置 |
| 训练准确率在 10% 左右 | 可能是梯度完全消失，模型没有学习 |
| 准确率跳动剧烈 | 学习率过高或批次大小不合适 |
| 准确率 30% 左右但某一类 recall=100%，其他类 recall=0% | **类别崩溃**：优化器找到退化解，修复见 v2.1 |

### 快速检查

```python
# 在训练循环中添加梯度监控
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 10:
            print(f"Warning: large gradient in {name}: {grad_norm}")
        elif grad_norm < 1e-6:
            print(f"Warning: vanishing gradient in {name}: {grad_norm}")
```

### 训练建议

1. **学习率**: 默认 `1e-4`，如果梯度问题持续，尝试 `5e-5` 或 `1e-5`
2. **批次大小**: 默认 16（1024 序列长度），如果显存允许可以增大到 32
3. **权重衰减**: 默认 0.04，可以尝试 0.01 或 0.1
4. **梯度裁剪**: 训练脚本已启用 `clip_grad_norm_(model.parameters(), 2.0)`
5. **类别崩溃**: 如果训练准确率远高于随机但多个类别 recall=0%，降低 Dropout (0.2) 并使用更宽的分类头
