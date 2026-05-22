from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from radchar_class_config import (
    DEFAULT_CLASS_CONFIG_PATH,
    LabelSchema,
    build_default_schema,
    infer_num_classes_from_state_dict,
    schema_from_checkpoint_payload,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 1024


@dataclass
class ModelArchitectureConfig:
    # CNN Stem 配置
    use_complex_conv: bool = False
    use_multiscale_fusion: bool = False
    use_enhanced_pooling: bool = False
    # 池化配置: 减少 MaxPool 次数，使用 Dilated Conv 保持感受野
    pool_layers: int = 2  # MaxPool 次数 (2=512→128, 4=512→32)
    use_dilated_conv: bool = True  # 使用空洞卷积扩大感受野
    # STFT 参数
    stft_n_fft: int = 64
    stft_hop_length: int = 16
    # 时频变换模式: fft / stft / glct
    timefreq_mode: str = "fft"
    # Mamba3 配置
    use_mamba: bool = True
    mamba3_layers: int = 2
    mamba3_d_state: int = 128
    mamba3_expand: int = 2
    mamba3_headdim: int = 64
    mamba3_dropout: float = 0.3
    # 复数 Mamba 配置
    use_complex_mamba: bool = False  # 使用复数 Mamba (待完善)
    # ProbSparse Attention 配置 (可选)
    use_prob_sparse_attention: bool = True
    prob_sparse_layers: int = 3
    prob_sparse_heads: int = 6
    prob_sparse_ffn_dim: int = 512
    prob_sparse_dropout: float = 0.3
    prob_sparse_top_k: int = 32
    # 可学习特征融合配置
    use_learnable_fusion: bool = True  # 使用可学习加权融合
    # Temporal Module Selection: prob_sparse, informer, fedformer
    temporal_module: str = "prob_sparse"
    # Informer params
    informer_factor: int = 5
    informer_distil: bool = True
    # FEDformer params
    fedformer_modes: int = 32
    fedformer_moving_avg: int = 25


class ConvNormAct1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        p_drop: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(),
        ]
        if p_drop > 0:
            layers.append(nn.Dropout(p_drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ComplexConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch * 2, out_ch * 2, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_ch * 2)

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_real, x_imag], dim=1)
        out = self.conv(combined)
        out = self.bn(out)
        return out[:, :out.size(1)//2], out[:, out.size(1)//2:]


class ComplexStem(nn.Module):
    def __init__(self, in_ch: int = 8):
        super().__init__()
        self.complex_conv1 = ComplexConv1d(in_ch // 2, 32, kernel_size=9, stride=1)
        self.conv1 = ConvNormAct1d(64, 64, kernel_size=5, stride=1, p_drop=0.05)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.complex_conv2 = ComplexConv1d(32, 48, kernel_size=7, stride=1)
        self.conv2 = ConvNormAct1d(96, 96, kernel_size=5, stride=1, p_drop=0.05)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real = x[:, :4]
        x_imag = x[:, 4:]
        x_real, x_imag = self.complex_conv1(x_real, x_imag)
        x = torch.cat([x_real, x_imag], dim=1)
        x = self.conv1(x)
        x = self.pool1(x)

        x_real, x_imag = self.complex_conv2(x_real, x_imag)
        x = torch.cat([x_real, x_imag], dim=1)
        x = self.conv2(x)
        x = self.pool2(x)
        return x


class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.branch3 = ConvNormAct1d(in_ch, out_ch // 4, kernel_size=3, stride=1, p_drop=0.05)
        self.branch5 = ConvNormAct1d(in_ch, out_ch // 4, kernel_size=5, stride=1, p_drop=0.05)
        self.branch7 = ConvNormAct1d(in_ch, out_ch // 4, kernel_size=7, stride=1, p_drop=0.05)
        self.branch11 = ConvNormAct1d(in_ch, out_ch // 4, kernel_size=11, stride=1, p_drop=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        b11 = self.branch11(x)
        return torch.cat([b3, b5, b7, b11], dim=1)


class ResidualConvBlock1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 5,
        stride: int = 1,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.conv1 = ConvNormAct1d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, p_drop=0.0
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                out_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
        )
        self.dropout = nn.Dropout(p_drop)
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return self.act(x + residual)


class AttentionPool1d(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1)


class RMSNorm(nn.Module):
    """RMSNorm: Root Mean Square Layer Normalization"""

    def __init__(self, d_model: int, eps: float = 1e-5, norm_before_gate: bool = False, group_size: int = 0):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))
        self.norm_before_gate = norm_before_gate
        self.group_size = group_size

    def forward(self, x: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D) or (B, D)
        if self.norm_before_gate and gate is not None:
            # Group RMSNorm for Mamba3 style
            B, T, D = x.shape
            if self.group_size > 0 and D % self.group_size == 0:
                # Reshape to groups: (B, T, num_groups, group_size)
                num_groups = D // self.group_size
                x = x.view(B, T, num_groups, self.group_size)
                rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
                x = x * rms * self.scale.view(1, 1, num_groups, self.group_size)
                x = x.view(B, T, D)
            else:
                rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
                x = x * rms * self.scale
            return x * gate
        else:
            rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            return x * rms * self.scale


import math


class SinusoidalPositionalEncoding1d(nn.Module):
    """Sinusoidal positional encoding for 1D sequences."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class TemporalTransformerEncoder(nn.Module):
    """Transformer encoder replacing the BiGRU for temporal modeling."""

    def __init__(
        self,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 3,
        ffn_dim: int = 768,
        dropout: float = 0.15,
        attn_dropout: float = 0.1,
        use_flash: bool = True,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        if use_flash:
            try:
                from torch.nn.attention import SDPF_BACKEND

                encoder_layer.self_attn.backends[
                    SDPF_BACKEND.CUDNN_ATTENTION
                ] = True
            except Exception:
                pass
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.pos_enc = SinusoidalPositionalEncoding1d(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        return self.encoder(x)


class ProbSparseSelfAttention(nn.Module):
    """ProbSparse Self-Attention (Informer)

    通过 top-k 采样策略，将 O(n²) 降至 O(n log n)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, top_k: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.top_k = top_k
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        H = self.n_heads
        dk = self.head_dim

        Q = self.q_proj(x).view(B, T, H, dk).transpose(1, 2)  # (B, H, T, dk)
        K = self.k_proj(x).view(B, T, H, dk).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, dk).transpose(1, 2)

        # 计算 attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)  # (B, H, T, T)

        # ProbSparse: 选择 top-k 个最相关的 key 位置
        if self.top_k < T:
            # 使用 softmax 后的注意力分布作为采样概率
            attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, T)
            # 选择 top-k 个 key (简化实现：每个 query 只关注 top-k 个 key)
            top_k = min(self.top_k, T)
            topk_scores, topk_indices = torch.topk(scores, k=top_k, dim=-1)  # (B, H, T, top_k)
            # 重构稀疏注意力矩阵
            sparse_mask = torch.zeros_like(scores).scatter_(-1, topk_indices, topk_scores)
            attn = F.softmax(sparse_mask, dim=-1)
        else:
            attn = F.softmax(scores, dim=-1)

        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, H, T, dk)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class ProbSparseEncoderLayer(nn.Module):
    """ProbSparse Encoder Layer"""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = ProbSparseSelfAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class ProbSparseEncoder(nn.Module):
    """ProbSparse Encoder - 多层堆叠"""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ProbSparseEncoderLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.pos_enc = SinusoidalPositionalEncoding1d(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# =============================================================================
# Informer Encoder (replaces ProbSparse attention with Informer's ProbAttention)
# =============================================================================

class InformerProbAttention(nn.Module):
    """Informer-style full attention.

    Uses standard scaled dot-product attention.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Forward pass for self-attention.

        Args:
            queries, keys, values: (B, L, H, E) format (after AttentionLayer projection)
        Returns:
            (B, H, L, E) format
        """
        B, L_Q, H, E = queries.shape
        _, L_K, _, _ = keys.shape

        # Transpose to (B, H, L, E) for standard attention computation
        q = queries.transpose(1, 2)  # (B, H, L, E)
        k = keys.transpose(1, 2)    # (B, H, L, E)
        v = values.transpose(1, 2)  # (B, H, L, E)

        # Compute attention scores: (B, H, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values: (B, H, L, E)
        out = torch.matmul(attn, v)
        return out  # (B, H, L, E)


class InformerAttentionLayer(nn.Module):
    """Informer Attention Layer with Q/K/V projections."""

    def __init__(self, attention: nn.Module, d_model: int, n_heads: int):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        attn_output = self.attention(queries, keys, values)
        # Handle both tuple (output, attn_weights) and single tensor returns
        if isinstance(attn_output, tuple):
            out = attn_output[0]
        else:
            out = attn_output
        # out: (B, H, L, E) -> transpose(H, L) -> (B, L, H, E) -> view -> (B, L, H*E)
        out = out.transpose(2, 1).contiguous().view(B, L, -1)
        return self.out_projection(out)


class InformerConvLayer(nn.Module):
    """Convolution layer for Informer distilation."""

    def __init__(self, c_in: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=3,
                             padding=1, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x.transpose(1, 2)


class InformerEncoderLayer(nn.Module):
    """Informer Encoder Layer with attention + FFN + optional distilation."""

    def __init__(self, attention: nn.Module, d_model: int, d_ff: int = None,
                 dropout: float = 0.1, activation: str = "gelu", use_distil: bool = True):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu
        self.use_distil = use_distil

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.attention(x, x, x)
        # Handle both tuple (output, attn_weights) and single tensor returns
        if isinstance(attn_output, tuple):
            new_x = attn_output[0]
        else:
            new_x = attn_output
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = self.norm2(x + y)

        if self.use_distil:
            y = InformerConvLayer(y.shape[-1]).to(y.device)(y)
        return y


class InformerEncoder(nn.Module):
    """Informer Encoder - multi-layer with optional distilation.

    Interface: (B, T, D) -> (B, T, D)
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, ffn_dim: int = None,
                 dropout: float = 0.1, factor: int = 5, distil: bool = True):
        super().__init__()
        ffn_dim = ffn_dim or 4 * d_model
        self.pos_enc = SinusoidalPositionalEncoding1d(d_model, dropout=dropout)

        attn = InformerProbAttention(d_model, n_heads, dropout=dropout)
        attention_layer = InformerAttentionLayer(attn, d_model, n_heads)

        self.layers = nn.ModuleList([
            InformerEncoderLayer(attention_layer, d_model, ffn_dim, dropout=dropout,
                                  activation="gelu", use_distil=(distil and i < n_layers - 1))
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# =============================================================================
# FEDformer Encoder (replaces ProbSparse with AutoCorrelation - FFT based)
# =============================================================================

class AutoCorrelation(nn.Module):
    """AutoCorrelation Mechanism from FEDformer.

    Period-based dependencies discovery + time delay aggregation via FFT.
    """

    def __init__(self, n_heads: int, factor: int = 1, dropout: float = 0.1,
                 output_attention: bool = False):
        super().__init__()
        self.factor = factor
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.output_attention = output_attention

    def time_delay_agg_training(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """Time delay aggregation during training."""
        B, H, D, L = values.shape
        top_k = int(self.factor * math.log(L))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)

        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, H, D, L))
        return delays_agg

    def time_delay_agg_inference(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """Time delay aggregation during inference."""
        B, H, D, L = values.shape
        top_k = int(self.factor * math.log(L))
        init_index = torch.arange(L).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B, H, D, 1).to(values.device)
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        tmp_corr = torch.softmax(weights, dim=-1)

        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, H, D, L)
            pattern = torch.gather(values.repeat(1, 1, 1, 2), dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, H, D, L))
        return delays_agg

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # Convert to float32 for FFT compatibility
        queries = queries.float()
        keys = keys.float()
        values = values.float()

        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # FFT-based correlation
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    """AutoCorrelation layer with Q/K/V projections."""

    def __init__(self, correlation: nn.Module, d_model: int, n_heads: int):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, _ = self.correlation(queries, keys, values)
        return self.out_projection(out.view(B, L, -1))


class SeriesDecomp(nn.Module):
    """Series decomposition for FEDformer."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        front = x[:, 0:1, :].repeat(1, (self.moving_avg.kernel_size[0] - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.moving_avg.kernel_size[0] - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        moving_avg = self.moving_avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        res = x - moving_avg
        return res, moving_avg


class FEDformerEncoderLayer(nn.Module):
    """FEDformer Encoder Layer with AutoCorrelation + series decomp."""

    def __init__(self, attention: nn.Module, d_model: int, d_ff: int = None,
                 moving_avg: int = 25, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.attention(x, x, x)
        # Handle both tuple (output, attn_weights) and single tensor returns
        if isinstance(attn_output, tuple):
            new_x = attn_output[0]
        else:
            new_x = attn_output
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res


class FEDformerEncoder(nn.Module):
    """FEDformer Encoder - AutoCorrelation based.

    Interface: (B, T, D) -> (B, T, D)
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, ffn_dim: int = None,
                 dropout: float = 0.1, moving_avg: int = 25):
        super().__init__()
        ffn_dim = ffn_dim or 4 * d_model
        self.pos_enc = SinusoidalPositionalEncoding1d(d_model, dropout=dropout)

        attn = AutoCorrelation(n_heads=n_heads, factor=1, dropout=dropout)
        attention_layer = AutoCorrelationLayer(attn, d_model, n_heads)

        self.layers = nn.ModuleList([
            FEDformerEncoderLayer(attention_layer, d_model, ffn_dim,
                                  moving_avg=moving_avg, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class EnhancedPooling(nn.Module):
    def __init__(self, feature_dim: int = 192):
        super().__init__()
        self.feature_dim = feature_dim
        self.spectral_centroid_fc = nn.Linear(feature_dim, 64)
        self.spectral_entropy_fc = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.spectral_spread_fc = nn.Linear(feature_dim, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1).values

        sc_feat = self.spectral_centroid_fc(x_mean)
        se_feat = self.spectral_entropy_fc(x_mean)
        ss_feat = self.spectral_spread_fc(x_max)

        return torch.cat([sc_feat, se_feat, ss_feat], dim=1)


class ComplexInputAugment(nn.Module):
    """复数输入增强: CFO (载波频率偏移) 和 STO (符号定时偏移)

    雷达信号对 CFO 和 STO 非常敏感，在 IQInputProcessor 之后应用。
    """

    def __init__(self, cfo_std: float = 0.02, sto_max: int = 8):
        super().__init__()
        self.cfo_std = cfo_std  # CFO 标准差 (相对载波频率)
        self.sto_max = sto_max  # STO 最大时间偏移

    def forward(self, i_sig: torch.Tensor, q_sig: torch.Tensor, apply_aug: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            i_sig: (B, T) I路信号
            q_sig: (B, T) Q路信号
            apply_aug: 是否应用增强 (训练时 True，推理时 False)
        Returns:
            (i_aug, q_aug): 增强后的 I/Q 信号
        """
        if not apply_aug or not self.training:
            return i_sig, q_sig

        B, T = i_sig.shape

        # STO: 时间平移
        if self.sto_max > 0:
            shifts = torch.randint(-self.sto_max, self.sto_max + 1, (B,), device=i_sig.device)
            # 为每个样本应用不同的时间偏移
            i_aug = torch.zeros_like(i_sig)
            q_aug = torch.zeros_like(q_sig)
            for b in range(B):
                shift = shifts[b].item()
                if shift > 0:
                    i_aug[b, shift:] = i_sig[b, :-shift]
                    q_aug[b, shift:] = q_sig[b, :-shift]
                elif shift < 0:
                    i_aug[b, :shift] = i_sig[b, -shift:]
                    q_aug[b, :shift] = q_sig[b, -shift:]
                else:
                    i_aug[b] = i_sig[b]
                    q_aug[b] = q_sig[b]
        else:
            i_aug = i_sig
            q_aug = q_sig

        # CFO: 随机相位旋转 (复数旋转)
        if self.cfo_std > 0:
            # 为每个样本生成不同的相位偏移
            phase_offsets = torch.randn(B, device=i_sig.device) * self.cfo_std * 2 * np.pi
            # 复数旋转: e^(j*phase)
            cos_phase = torch.cos(phase_offsets)  # (B,)
            sin_phase = torch.sin(phase_offsets)  # (B,)

            # 对每个样本应用相位旋转
            i_rotated = i_aug * cos_phase.unsqueeze(-1) - q_aug * sin_phase.unsqueeze(-1)
            q_rotated = i_aug * sin_phase.unsqueeze(-1) + q_aug * cos_phase.unsqueeze(-1)
            i_aug = i_rotated
            q_aug = q_rotated

        return i_aug, q_aug


import numpy as np


class DilatedConvBlock(nn.Module):
    """空洞卷积块: 扩大感受野而不降低分辨率"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        # 空洞率递增: 1, 2, 4, 8
        self.dilated1 = nn.Conv1d(in_ch, out_ch // 4, kernel_size=kernel_size, dilation=1, padding=(kernel_size - 1) // 2 * 1, bias=False)
        self.dilated2 = nn.Conv1d(in_ch, out_ch // 4, kernel_size=kernel_size, dilation=2, padding=(kernel_size - 1) // 2 * 2, bias=False)
        self.dilated4 = nn.Conv1d(in_ch, out_ch // 4, kernel_size=kernel_size, dilation=4, padding=(kernel_size - 1) // 2 * 4, bias=False)
        self.dilated8 = nn.Conv1d(in_ch, out_ch // 4, kernel_size=kernel_size, dilation=8, padding=(kernel_size - 1) // 2 * 8, bias=False)

        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.dilated1(x)
        d2 = self.dilated2(x)
        d4 = self.dilated4(x)
        d8 = self.dilated8(x)
        out = torch.cat([d1, d2, d4, d8], dim=1)
        out = self.bn(out)
        out = self.act(out)
        return self.dropout(out)


class ComplexMamba3Block(nn.Module):
    """复数 Mamba3 模块: 同时处理 I/Q，保留复数相位关系

    与标准 Mamba3 不同，这里将 I/Q 作为复数的实部和虚部一起处理，
    而不是作为两个独立的实数通道。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim

        self.d_inner = int(self.expand * d_model)
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim

        # 复数投影: 分别对实部和虚部投影，然后合并
        # 实部路径
        self.real_proj = nn.Linear(d_model, d_inner := self.d_inner // 2, bias=False)
        # 虚部路径
        self.imag_proj = nn.Linear(d_model, d_inner // 2, bias=False)

        # dt_bias
        self.dt_bias = nn.Parameter(torch.rand(self.nheads // 2) * 0.1 + 0.001)

        # B 和 C 参数 (对复数)
        self.B = nn.Parameter(torch.randn(self.nheads // 2, d_state))
        self.C = nn.Parameter(torch.randn(self.nheads // 2, d_state))
        self.B._no_weight_decay = True
        self.C._no_weight_decay = True

        # D 跳连接
        self.D = nn.Parameter(torch.ones(self.nheads // 2))
        self.D._no_weight_decay = True

        # RMSNorm for B and C
        self.B_norm = RMSNorm(d_state, eps=1e-5)
        self.C_norm = RMSNorm(d_state, eps=1e-5)

        # 输出投影
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) 复数特征"""
        batch, seqlen, dim = x.shape

        # 分离实部和虚部
        x_real = x[..., 0::2]  # (B, T, D/2)
        x_imag = x[..., 1::2]  # (B, T, D/2)

        # 复数投影
        real_out = self.real_proj(x_real)  # (B, T, d_inner/2)
        imag_out = self.imag_proj(x_imag)   # (B, T, d_inner/2)

        # 合并为复数形式 (real, imag, real, imag, ...)
        z = torch.stack([real_out, imag_out], dim=-1)
        z = z.view(batch, seqlen, self.nheads // 2, self.headdim // 2, 2)
        z = torch.complex(z[..., 0], z[..., 1])  # (B, T, nheads//2, headdim//2) 复数

        # dt: softplus + bias (对复数应用相同变换)
        dt = F.softplus(self.dt_bias)  # (nheads//2,)
        dt = dt.view(1, 1, self.nheads // 2, 1)  # (1, 1, nheads//2, 1)

        # A: softplus to keep negative and stable - use learned A_log
        A = -F.softplus(self.A_log)  # (nheads//2, headdim//2)
        A = torch.clamp(A, max=-1e-4)  # Keep stable
        A = A.unsqueeze(0).unsqueeze(0)  # (1, 1, nheads//2, headdim//2)

        # B and C: apply RMSNorm
        B = self.B_norm(self.B)  # (nheads//2, d_state)
        C = self.C_norm(self.C)  # (nheads//2, d_state)

        # Gate (幅度)
        gate = torch.abs(z)  # (B, T, nheads//2, headdim//2)

        # 简化的状态更新
        h = torch.zeros(batch, self.nheads // 2, self.headdim // 2, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seqlen):
            # h_new = A * dt * h + B * |z[t]|
            h_new = (A * dt).unsqueeze(-1) * h + B.unsqueeze(0).unsqueeze(-1) * gate[:, t, :, :].unsqueeze(-1)

            # y = C @ h + D * |z|
            y_t = torch.matmul(h_new.abs(), C.unsqueeze(0).unsqueeze(-1)).squeeze(-1) + \
                  self.D.view(1, self.nheads // 2, 1) * gate[:, t, :, :].mean(dim=-1, keepdim=True)

            outputs.append(y_t)
            h = h_new

        y = torch.stack(outputs, dim=1)  # (B, T, nheads//2, headdim//2)
        y = y.view(batch, seqlen, self.d_inner // 2)

        y = self.dropout(y)
        y = self.out_proj(y)
        return y


class LearnableFeatureFusion(nn.Module):
    """可学习特征融合: 加权求和 + 门控机制

    替代简单的 mean + max + ... 拼接，让网络学习不同特征的重要性。
    """

    def __init__(self, feat_dim: int = 192, num_features: int = 4, use_gating: bool = True):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_features = num_features
        self.use_gating = use_gating

        # 可学习权重
        self.weights = nn.Parameter(torch.ones(num_features) / num_features)

        # 门控机制: 学习每种特征的重要性
        # FIX: 使用更宽的隐藏层，避免瓶颈
        if use_gating:
            hidden_dim = max(feat_dim, feat_dim * num_features // 2)  # 至少 192
            self.gate_fc = nn.Sequential(
                nn.Linear(feat_dim * num_features, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, num_features),
                nn.Sigmoid(),  # 输出 0-1 之间的门控值
            )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: [(B, D), (B, D), ...] 特征列表
        Returns:
            (B, D) 融合后的特征
        """
        assert len(features) == self.num_features, f"Expected {self.num_features} features, got {len(features)}"

        # 归一化权重
        w = torch.softmax(self.weights, dim=0)  # (num_features,)

        # 加权求和
        weighted_sum = sum(w[i] * features[i] for i in range(self.num_features))

        if self.use_gating:
            # 拼接所有特征用于门控
            concat_feat = torch.cat(features, dim=-1)  # (B, D * num_features)
            gate_values = self.gate_fc(concat_feat)  # (B, num_features)
            # 加权门控
            gated = sum(gate_values[:, i:i+1] * features[i] for i in range(self.num_features))
            # 残差连接
            return 0.5 * weighted_sum + 0.5 * gated
        else:
            return weighted_sum


class IQInputProcessor(nn.Module):
    """I/Q 输入处理：I/Q 原始信号 + 时频变换特征"""

    def __init__(
        self,
        timefreq_mode: str = "fft",
        stft_n_fft: int = 64,
        stft_hop_length: int = 16,
    ):
        """
        Args:
            timefreq_mode: "fft" 使用 FFT 幅度谱, "stft" 使用 STFT 时频谱,
                           "glct" 使用广义线性调频变换 (双 STFT 乘积增强)
            stft_n_fft: STFT 的 FFT 大小
            stft_hop_length: STFT 的跳跃步长
        """
        super().__init__()
        self.timefreq_mode = timefreq_mode
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length

    def forward(
        self, i_sig: torch.Tensor, q_sig: torch.Tensor,
        channel_mean: Optional[torch.Tensor] = None,
        channel_std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            i_sig: (B, T) I路信号
            q_sig: (B, T) Q路信号
            channel_mean: (8,) 全局通道均值，若提供则用于归一化 I/Q
            channel_std: (8,) 全局通道标准差
        Returns:
            (B, 3, T) I/Q 两路 + 时频变换特征
        """
        # 复数信号
        complex_sig = torch.complex(i_sig, q_sig)  # (B, T)

        # === 第三通道：根据 timefreq_mode 生成 ===
        if self.timefreq_mode == "fft":
            # FFT 幅度谱 (复数 IQ 频谱非共轭对称，保留全频段)
            fft_full = torch.fft.fft(complex_sig, dim=-1)  # (B, T)
            third_feat = torch.abs(fft_full)  # (B, T)
            if channel_mean is not None and channel_std is not None:
                third_feat_norm = (third_feat - channel_mean[6]) / channel_std[6].clamp_min(1e-5)
            else:
                third_feat_norm = (third_feat - third_feat.mean(dim=-1, keepdim=True)) / (
                    third_feat.std(dim=-1, keepdim=True) + 1e-8
                )

        elif self.timefreq_mode == "stft":
            # STFT 时频谱 -> 频率维平均 -> 插值到 T
            n_fft = self.stft_n_fft
            hop_length = self.stft_hop_length
            window = torch.hann_window(n_fft, device=i_sig.device)
            stft_out = torch.stft(
                complex_sig, n_fft=n_fft, hop_length=hop_length,
                return_complex=False, window=window,
            )  # (B, n_fft//2+1, T_stft, 2)
            spec = torch.abs(torch.complex(stft_out[..., 0], stft_out[..., 1]))  # (B, F, T_stft)
            spec = spec.mean(dim=1, keepdim=True)  # (B, 1, T_stft)
            spec = F.interpolate(spec, size=i_sig.size(-1), mode='linear', align_corners=False)  # (B, 1, T)
            third_feat = spec.squeeze(1)  # (B, T)
            # STFT 特征尺度与 FFT 不同，使用 per-sample 归一化
            third_feat_norm = (third_feat - third_feat.mean(dim=-1, keepdim=True)) / (
                third_feat.std(dim=-1, keepdim=True) + 1e-8
            )

        elif self.timefreq_mode == "glct":
            # GLCT 近似: 两组不同 n_fft 的 STFT 谱乘积增强
            n_fft = self.stft_n_fft
            hop_length = self.stft_hop_length
            window = torch.hann_window(n_fft, device=i_sig.device)
            stft1 = torch.stft(
                complex_sig, n_fft=n_fft, hop_length=hop_length,
                return_complex=False, window=window,
            )
            spec1 = torch.abs(torch.complex(stft1[..., 0], stft1[..., 1]))  # (B, F1, T1)

            n_fft2 = n_fft * 2
            window2 = torch.hann_window(n_fft2, device=i_sig.device)
            stft2 = torch.stft(
                complex_sig, n_fft=n_fft2, hop_length=hop_length,
                return_complex=False, window=window2,
            )
            spec2 = torch.abs(torch.complex(stft2[..., 0], stft2[..., 1]))  # (B, F2, T2)

            # 将 spec2 的时域和频域同时插值到与 spec1 相同大小，以便逐元素相乘
            # spec2: (B, F2, T2) -> unsqueeze -> (B, 1, F2, T2) -> 双线性插值 -> (B, 1, F1, T1) -> squeeze
            spec2_4d = spec2.unsqueeze(1)  # (B, 1, F2, T2)
            spec2_resized_4d = F.interpolate(
                spec2_4d,
                size=(spec1.shape[-2], spec1.shape[-1]),
                mode='bilinear', align_corners=False,
            )  # (B, 1, F1, T1)
            spec2_resized = spec2_resized_4d.squeeze(1)  # (B, F1, T1)
            glct_feat_raw = spec1 * spec2_resized  # (B, F1, T1) — 增强时频汇聚特征
            glct_feat_raw = glct_feat_raw.mean(dim=1, keepdim=True)  # (B, 1, T_stft)
            glct_feat_raw = F.interpolate(glct_feat_raw, size=i_sig.size(-1), mode='linear', align_corners=False)
            third_feat = glct_feat_raw.squeeze(1)  # (B, T)
            # per-sample 归一化
            third_feat_norm = (third_feat - third_feat.mean(dim=-1, keepdim=True)) / (
                third_feat.std(dim=-1, keepdim=True) + 1e-8
            )

        else:
            raise ValueError(f"Unsupported timefreq_mode: {self.timefreq_mode}")

        # I/Q 路归一化
        if channel_mean is not None and channel_std is not None:
            i_sig_norm = (i_sig - channel_mean[0]) / channel_std[0].clamp_min(1e-5)
            q_sig_norm = (q_sig - channel_mean[1]) / channel_std[1].clamp_min(1e-5)
        else:
            i_sig_norm = i_sig
            q_sig_norm = q_sig

        return torch.stack([i_sig_norm, q_sig_norm, third_feat_norm], dim=1)


class Mamba3Block(nn.Module):
    """Mamba3 风格的 SSM 模块 - 参考 Mamba-3 架构设计"""

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.0,
        is_outproj_norm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim

        self.d_inner = int(self.expand * d_model)
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim

        # Order: [z, x, B, C, dt, A]
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads + self.nheads

        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        # dt_bias
        self.dt_bias = nn.Parameter(torch.rand(self.nheads) * 0.1 + 0.001)

        # B 和 C 参数
        self.B = nn.Parameter(torch.randn(self.nheads, d_state))
        self.C = nn.Parameter(torch.randn(self.nheads, d_state))
        self.B._no_weight_decay = True
        self.C._no_weight_decay = True

        # D 跳连接
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        # RMSNorm for B and C
        self.B_norm = RMSNorm(d_state, eps=1e-5)
        self.C_norm = RMSNorm(d_state, eps=1e-5)

        # Gate norm - FIX: Use standard RMSNorm without norm_before_gate/group_size
        # to prevent gradient explosion/vanishing in deep networks
        if is_outproj_norm:
            self.norm = RMSNorm(self.d_inner, eps=1e-5)

        # Learnable A matrix (log of diagonal SSM parameter) for stability
        # This replaces random initialization each forward pass
        self.A_log = nn.Parameter(torch.randn(self.nheads, self.headdim) * 0.01)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.is_outproj_norm = is_outproj_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        batch, seqlen, dim = x.shape

        # in_proj: (B, T, d_in_proj)
        zxBCdtA = self.in_proj(x)

        # Split: [z, x, B, C, dt, A]
        z = zxBCdtA[..., :self.d_inner]
        x_inner = zxBCdtA[..., self.d_inner:2*self.d_inner]
        B = zxBCdtA[..., 2*self.d_inner:2*self.d_inner + self.d_state]
        C = zxBCdtA[..., 2*self.d_inner + self.d_state:2*self.d_inner + 2*self.d_state]
        dt = zxBCdtA[..., 2*self.d_inner + 2*self.d_state:2*self.d_inner + 2*self.d_state + self.nheads]
        A = zxBCdtA[..., 2*self.d_inner + 2*self.d_state + self.nheads:]

        # Reshape
        z = z.view(batch, seqlen, self.nheads, self.headdim)
        x_inner = x_inner.view(batch, seqlen, self.nheads, self.headdim)

        # dt: softplus + bias
        dt = F.softplus(dt + self.dt_bias)  # (B, T, nheads)
        dt = dt.unsqueeze(-1)  # (B, T, nheads, 1)

        # A: softplus to keep negative and stable
        A = -F.softplus(self.A_log)  # (nheads, headdim)
        A = A.unsqueeze(0).unsqueeze(0).expand(batch, seqlen, -1, -1)  # (B, T, nheads, 1)
        A = torch.clamp(A, max=-1e-4)  # Keep stable

        # B and C: apply RMSNorm and expand
        B = self.B_norm(B)  # (B, T, d_state)
        C = self.C_norm(C)  # (B, T, d_state)
        B = B.unsqueeze(2).expand(-1, -1, self.nheads, -1)  # (B, T, nheads, d_state)
        C = C.unsqueeze(2).expand(-1, -1, self.nheads, -1)

        # Gate
        gate = torch.sigmoid(z)  # (B, T, nheads, headdim)

        # Compute ADT
        ADT = A * dt  # (B, T, nheads, 1)

        # State update (simplified scan)
        h = torch.zeros(batch, self.nheads, self.headdim, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seqlen):
            # h_new = ADT[t] * h + B[t] * x[t]
            # h: (B, nheads, headdim, d_state), x: (B, nheads, headdim)
            x_t = x_inner[:, t, :, :]  # (B, nheads, headdim)

            # Simplified: h = ADT * h + B * x
            # ADT[t]: (B, nheads, 1, 1), h: (B, nheads, headdim, d_state) -> elementwise
            h_new = ADT[:, t, :, :].unsqueeze(-1) * h + \
                    B[:, t, :, :].unsqueeze(2) * x_t.unsqueeze(-1)

            # Gradient clipping
            h_new = torch.clamp(h_new, min=-10, max=10)

            # y = C @ h + D * x
            # C[t]: (B, nheads, d_state), h_new: (B, nheads, headdim, d_state)
            # matmul: (B, nheads, headdim, d_state) @ (B, nheads, d_state, 1) -> (B, nheads, headdim, 1)
            C_t = C[:, t, :, :].unsqueeze(-1)  # (B, nheads, d_state, 1)
            y_t = torch.matmul(h_new, C_t).squeeze(-1) + \
                  self.D.view(1, self.nheads, 1) * x_t

            y_t = y_t * gate[:, t, :, :]
            outputs.append(y_t)
            h = h_new

        y = torch.stack(outputs, dim=1)  # (B, T, nheads, headdim)
        y = y.view(batch, seqlen, self.d_inner)

        if self.is_outproj_norm:
            y = self.norm(y)

        y = self.dropout(y)
        y = self.out_proj(y)
        return y


class TimeFreqRadarNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int = SEQ_LEN,
        arch_config: Optional[ModelArchitectureConfig] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.arch_config = arch_config or ModelArchitectureConfig()
        self.register_buffer("channel_mean", torch.zeros(8, dtype=torch.float32))
        self.register_buffer("channel_std", torch.ones(8, dtype=torch.float32))

        # I/Q Input Processor + CFO/STO 增强
        self.iq_processor = IQInputProcessor(
            timefreq_mode=self.arch_config.timefreq_mode,
            stft_n_fft=self.arch_config.stft_n_fft,
            stft_hop_length=self.arch_config.stft_hop_length,
        )
        self.complex_augment = ComplexInputAugment(cfo_std=0.02, sto_max=8)

        # CNN Stem - 2 通道 I/Q 输入
        use_complex = getattr(self.arch_config, 'use_complex_conv', False)

        if use_complex:
            # ComplexStem 用于复数卷积，需要 8 通道输入
            self.stem = ComplexStem(in_ch=8)
        else:
            # 标准 CNN 处理 3 通道 I/Q/FFT 输入
            self.stem = nn.Sequential(
                ConvNormAct1d(3, 64, kernel_size=9, stride=1, p_drop=0.0),
                ConvNormAct1d(64, 96, kernel_size=5, stride=1, p_drop=0.05),
            )

        # 空洞卷积层 (替代部分 MaxPool)
        self.use_dilated = self.arch_config.use_dilated_conv
        if self.use_dilated:
            self.dilated_conv = DilatedConvBlock(96, 96, kernel_size=5, dropout=0.05)

        # 可配置池化层数
        self.pool_layers = self.arch_config.pool_layers  # 2=512→128, 4=512→32

        if self.arch_config.use_multiscale_fusion:
            self.layer1 = nn.Sequential(
                MultiScaleBlock(96, 128),
                ResidualConvBlock1d(128, 128, kernel_size=3, stride=1, p_drop=0.08),
                nn.MaxPool1d(kernel_size=2),
            )
            self.layer2 = nn.Sequential(
                MultiScaleBlock(128, 192),
                ResidualConvBlock1d(192, 192, kernel_size=3, stride=1, p_drop=0.10),
                nn.MaxPool1d(kernel_size=2) if self.pool_layers >= 2 else nn.Identity(),
            )
            self.layer3 = nn.Sequential(
                MultiScaleBlock(192, 256),
                ResidualConvBlock1d(256, 256, kernel_size=3, stride=1, p_drop=0.12),
                nn.MaxPool1d(kernel_size=2) if self.pool_layers >= 3 else nn.Identity(),
            )
            if self.pool_layers >= 4:
                self.layer4 = nn.Sequential(
                    MultiScaleBlock(256, 256),
                    ResidualConvBlock1d(256, 256, kernel_size=3, stride=1, p_drop=0.12),
                    nn.MaxPool1d(kernel_size=2),
                )
        else:
            self.layer1 = nn.Sequential(
                ResidualConvBlock1d(96, 128, kernel_size=5, stride=1, p_drop=0.08),
                ResidualConvBlock1d(128, 128, kernel_size=5, stride=1, p_drop=0.08),
                nn.MaxPool1d(kernel_size=2),
            )
            self.layer2 = nn.Sequential(
                ResidualConvBlock1d(128, 192, kernel_size=5, stride=1, p_drop=0.10),
                ResidualConvBlock1d(192, 192, kernel_size=3, stride=1, p_drop=0.10),
                nn.MaxPool1d(kernel_size=2) if self.pool_layers >= 2 else nn.Identity(),
            )
            self.layer3 = nn.Sequential(
                ResidualConvBlock1d(192, 256, kernel_size=3, stride=1, p_drop=0.12),
                ResidualConvBlock1d(256, 256, kernel_size=3, stride=1, p_drop=0.12),
                nn.MaxPool1d(kernel_size=2) if self.pool_layers >= 3 else nn.Identity(),
            )
            if self.pool_layers >= 4:
                self.layer4 = nn.Sequential(
                    ResidualConvBlock1d(256, 256, kernel_size=3, stride=1, p_drop=0.12),
                    ResidualConvBlock1d(256, 256, kernel_size=3, stride=1, p_drop=0.12),
                    nn.MaxPool1d(kernel_size=2),
                )

        self.temporal_proj = nn.Sequential(
            nn.Conv1d(256, 192, kernel_size=1, bias=False),
            nn.BatchNorm1d(192),
            nn.SiLU(),
        )

        # Mamba3 blocks (可选) - 根据 use_complex_mamba 选择复数或标准版本
        self.mamba3_blocks = None
        if self.arch_config.use_mamba:
            if self.arch_config.use_complex_mamba:
                self.mamba3_blocks = nn.ModuleList([
                    ComplexMamba3Block(
                        d_model=192,
                        d_state=self.arch_config.mamba3_d_state,
                        expand=self.arch_config.mamba3_expand,
                        headdim=self.arch_config.mamba3_headdim,
                        dropout=self.arch_config.mamba3_dropout,
                    )
                    for _ in range(self.arch_config.mamba3_layers)
                ])
            else:
                self.mamba3_blocks = nn.ModuleList([
                    Mamba3Block(
                        d_model=192,
                        d_state=self.arch_config.mamba3_d_state,
                        expand=self.arch_config.mamba3_expand,
                        headdim=self.arch_config.mamba3_headdim,
                        dropout=self.arch_config.mamba3_dropout,
                        is_outproj_norm=True,
                    )
                    for _ in range(self.arch_config.mamba3_layers)
                ])

        # Temporal Encoder (ProbSparse / Informer / FEDformer)
        self.temporal_encoder = None
        if self.arch_config.use_prob_sparse_attention:
            tm = self.arch_config.temporal_module
            if tm == "informer":
                self.temporal_encoder = InformerEncoder(
                    d_model=192,
                    n_layers=self.arch_config.prob_sparse_layers,
                    n_heads=self.arch_config.prob_sparse_heads,
                    ffn_dim=self.arch_config.prob_sparse_ffn_dim,
                    dropout=self.arch_config.prob_sparse_dropout,
                    factor=self.arch_config.informer_factor,
                    distil=self.arch_config.informer_distil,
                )
            elif tm == "fedformer":
                self.temporal_encoder = FEDformerEncoder(
                    d_model=192,
                    n_layers=self.arch_config.prob_sparse_layers,
                    n_heads=self.arch_config.prob_sparse_heads,
                    ffn_dim=self.arch_config.prob_sparse_ffn_dim,
                    dropout=self.arch_config.prob_sparse_dropout,
                    moving_avg=self.arch_config.fedformer_moving_avg,
                )
            else:  # prob_sparse
                self.temporal_encoder = ProbSparseEncoder(
                    d_model=192,
                    n_layers=self.arch_config.prob_sparse_layers,
                    n_heads=self.arch_config.prob_sparse_heads,
                    ffn_dim=self.arch_config.prob_sparse_ffn_dim,
                    dropout=self.arch_config.prob_sparse_dropout,
                )

        # 可学习特征融合
        self.use_learnable_fusion = self.arch_config.use_learnable_fusion
        if self.use_learnable_fusion:
            # 统计需要的特征数量
            # feat_mean, feat_max, mamba_out, attn_out = 4
            num_features = 2  # mean, max
            if self.arch_config.use_mamba:
                num_features += 1  # mamba
            if self.arch_config.use_prob_sparse_attention:
                num_features += 1  # attn
            self.learnable_fusion = LearnableFeatureFusion(
                feat_dim=192,
                num_features=num_features,
                use_gating=True,
            )
            feat_dim_fused = 192
        else:
            feat_dim_fused = 192

        # 分类头
        # FIX: 使用更宽的隐藏层和更深的结构来处理特征
        if self.use_learnable_fusion:
            self.main_head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(feat_dim_fused, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Dropout(0.15),
                nn.Linear(256, 192),
                nn.BatchNorm1d(192),
                nn.SiLU(),
            )
        else:
            feat_dim = 192
            mamba_dim = 192 if self.arch_config.use_mamba else 0
            attn_dim = 192 if self.arch_config.use_prob_sparse_attention else 0
            total_head_dim = feat_dim * 2 + mamba_dim + attn_dim

            if self.arch_config.use_enhanced_pooling:
                self.attn_pool = EnhancedPooling(feature_dim=feat_dim)
                self.main_head = nn.Sequential(
                    nn.Dropout(0.35),
                    nn.Linear(total_head_dim + 192, 256),
                    nn.BatchNorm1d(256),
                    nn.SiLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 192),
                    nn.BatchNorm1d(192),
                    nn.SiLU(),
                    nn.Dropout(0.15),
                )
            else:
                self.attn_pool = AttentionPool1d(feat_dim)
                self.main_head = nn.Sequential(
                    nn.Dropout(0.35),
                    nn.Linear(total_head_dim, 192),
                    nn.BatchNorm1d(192),
                    nn.SiLU(),
                    nn.Dropout(0.25),
                )

        self.classifier = nn.Linear(192, num_classes)

    def set_global_channel_stats(
        self, mean: np.ndarray | torch.Tensor, std: np.ndarray | torch.Tensor
    ) -> None:
        mean_t = torch.as_tensor(
            mean, dtype=torch.float32, device=self.channel_mean.device
        ).view(-1)
        std_t = (
            torch.as_tensor(std, dtype=torch.float32, device=self.channel_std.device)
            .view(-1)
            .clamp_min(1e-5)
        )
        if mean_t.numel() != 8 or std_t.numel() != 8:
            raise ValueError("global channel stats must contain exactly 8 values")
        self.channel_mean.copy_(mean_t)
        self.channel_std.copy_(std_t)

    def build_input(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        """Build 8-channel input from real/imag (legacy, for compatibility)"""
        c = torch.complex(real, imag)
        mag_t = torch.sqrt(real * real + imag * imag + 1e-8)
        phase_t = torch.angle(c)
        fft = torch.fft.fft(c, dim=-1, norm="ortho")
        fft_real = fft.real
        fft_imag = fft.imag
        fft_mag = torch.log1p(torch.abs(fft))
        fft_phase = torch.angle(fft)
        x = torch.stack(
            [
                real,
                imag,
                mag_t,
                phase_t,
                fft_real,
                fft_imag,
                fft_mag,
                fft_phase,
            ],
            dim=1,
        )
        mean = self.channel_mean.view(1, 8, 1)
        std = self.channel_std.view(1, 8, 1).clamp_min(1e-5)
        return (x - mean) / std

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        # CFO/STO 增强 (训练时应用)
        real_aug, imag_aug = self.complex_augment(real, imag, apply_aug=True)

        # I/Q 处理: (B, T) I路, (B, T) Q路 -> (B, 3, T)
        iq_feat = self.iq_processor(real_aug, imag_aug, self.channel_mean, self.channel_std)

        x = self.stem(iq_feat)

        # 空洞卷积 (扩大感受野)
        if self.use_dilated:
            x = self.dilated_conv(x)

        # 可配置池化层
        x = self.layer1(x)
        if hasattr(self, 'layer2'):
            x = self.layer2(x)
        if hasattr(self, 'layer3'):
            x = self.layer3(x)
        if hasattr(self, 'layer4') and self.pool_layers >= 4:
            x = self.layer4(x)

        x = self.temporal_proj(x)
        seq = x.transpose(1, 2)  # (B, T, 192)

        # 时序建模: Mamba3 和/或 ProbSparse
        features_for_fusion = []
        if self.mamba3_blocks is not None:
            for mamba3 in self.mamba3_blocks:
                seq = mamba3(seq)
            features_for_fusion.append(seq.mean(dim=1))  # pool to (B, 192)
        if self.temporal_encoder is not None:
            seq = self.temporal_encoder(seq)
            features_for_fusion.append(seq.mean(dim=1))  # pool to (B, 192)

        # 特征融合
        feat_mean = seq.mean(dim=1)
        feat_max = seq.max(dim=1).values

        # 可学习融合或拼接
        if self.use_learnable_fusion:
            combined_feats = [feat_mean, feat_max] + features_for_fusion
            combined = self.learnable_fusion(combined_feats)
        else:
            # 拼接 base features + mamba out + attn out
            if features_for_fusion:
                combined = torch.cat([feat_mean, feat_max] + features_for_fusion, dim=-1)
            else:
                combined = torch.cat([feat_mean, feat_max], dim=-1)

            if self.arch_config.use_enhanced_pooling:
                enhanced_feat = self.attn_pool(seq)
                combined = torch.cat([combined, enhanced_feat], dim=-1)

        feat = self.main_head(combined)
        return self.classifier(feat)


def extract_state_dict(loaded: Any) -> dict[str, Any]:
    if not isinstance(loaded, dict):
        return loaded
    if loaded.get("best_state_dict") is not None:
        return loaded["best_state_dict"]
    if "model_state_dict" in loaded:
        return loaded["model_state_dict"]
    return {k: v for k, v in loaded.items() if not str(k).startswith("_")}


def torch_load_any(path: str) -> Any:
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_checkpoint_bundle(
    checkpoint_path: str,
    *,
    config_path: Optional[str] = None,
    strict: bool = False,
) -> tuple[TimeFreqRadarNet, LabelSchema, dict[str, Any], dict[str, Any]]:
    loaded = torch_load_any(checkpoint_path)
    state = extract_state_dict(loaded)
    fallback_num_classes = infer_num_classes_from_state_dict(state)
    schema = schema_from_checkpoint_payload(
        loaded,
        config_path=config_path or DEFAULT_CLASS_CONFIG_PATH,
        fallback_num_classes=fallback_num_classes,
    )
    if schema.num_classes != fallback_num_classes:
        if isinstance(loaded, dict) and (
            "label_schema" in loaded
            or ("config" in loaded and "label_schema" in loaded["config"])
        ):
            pass
        else:
            schema = build_default_schema(range(fallback_num_classes))

    arch_config = ModelArchitectureConfig()
    if isinstance(loaded, dict) and "arch_config" in loaded:
        cfg = loaded["arch_config"]
        arch_config = ModelArchitectureConfig(
            use_complex_conv=cfg.get("use_complex_conv", False),
            use_multiscale_fusion=cfg.get("use_multiscale_fusion", False),
            use_enhanced_pooling=cfg.get("use_enhanced_pooling", False),
            pool_layers=cfg.get("pool_layers", 2),
            use_dilated_conv=cfg.get("use_dilated_conv", True),
            stft_n_fft=cfg.get("stft_n_fft", 64),
            stft_hop_length=cfg.get("stft_hop_length", 16),
            timefreq_mode=cfg.get("timefreq_mode", "fft"),
            use_mamba=cfg.get("use_mamba", True),
            mamba3_layers=cfg.get("mamba3_layers", 2),
            mamba3_d_state=cfg.get("mamba3_d_state", 128),
            mamba3_expand=cfg.get("mamba3_expand", 2),
            mamba3_headdim=cfg.get("mamba3_headdim", 64),
            mamba3_dropout=cfg.get("mamba3_dropout", 0.1),
            use_complex_mamba=cfg.get("use_complex_mamba", False),
            use_prob_sparse_attention=cfg.get("use_prob_sparse_attention", True),
            prob_sparse_layers=cfg.get("prob_sparse_layers", 3),
            prob_sparse_heads=cfg.get("prob_sparse_heads", 6),
            prob_sparse_ffn_dim=cfg.get("prob_sparse_ffn_dim", 512),
            prob_sparse_dropout=cfg.get("prob_sparse_dropout", 0.1),
            prob_sparse_top_k=cfg.get("prob_sparse_top_k", 32),
            use_learnable_fusion=cfg.get("use_learnable_fusion", True),
            temporal_module=cfg.get("temporal_module", "prob_sparse"),
            informer_factor=cfg.get("informer_factor", 5),
            informer_distil=cfg.get("informer_distil", True),
            fedformer_modes=cfg.get("fedformer_modes", 32),
            fedformer_moving_avg=cfg.get("fedformer_moving_avg", 25),
        )

    model = TimeFreqRadarNet(
        num_classes=schema.num_classes,
        seq_len=SEQ_LEN,
        arch_config=arch_config,
    ).to(DEVICE)
    model.load_state_dict(state, strict=strict)
    model.eval()
    
    # 加载全局通道统计信息（如果可用）
    if isinstance(loaded, dict) and "global_channel_stats" in loaded:
        stats = loaded["global_channel_stats"]
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        model.set_global_channel_stats(mean, std)
        print("已加载全局通道统计信息")
    else:
        print("警告：检查点中未找到全局通道统计信息，使用默认值")
    
    return model, schema, loaded, state


def checkpoint_default_paths(base_dir: Optional[str] = None) -> tuple[str, str]:
    root = base_dir or os.path.dirname(os.path.abspath(__file__))
    return (
        os.path.join(root, "models", "radchar_checkpoint.pth"),
        os.path.join(root, "models", "radchar_best.pth"),
    )


def split_iq(iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    iq = np.asarray(iq)
    return iq.real.astype(np.float32), iq.imag.astype(np.float32)
