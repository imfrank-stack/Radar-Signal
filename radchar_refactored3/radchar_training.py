from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from radchar_class_config import LabelSchema, discover_raw_labels, load_label_schema
from radchar_model import DEVICE, SEQ_LEN, TimeFreqRadarNet, ModelArchitectureConfig

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 60
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 0.04


class RadCharDataset(Dataset):
    def __init__(self, iq_data: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.iq_data = iq_data
        self.labels = labels.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.iq_data)

    def augment_signal(self, iq: np.ndarray) -> np.ndarray:
        # 保守优化：删除与数据集/模型内部重复的增强
        # 保留：相位旋转、幅度缩放、局部衰减、多径效应
        
        # 相位旋转：本振相位偏差
        if np.random.random() < 0.75:
            phase = np.random.uniform(-0.35, 0.35)
            iq = iq * np.exp(1j * phase)
        
        # 幅度缩放：AGC增益波动
        if np.random.random() < 0.60:
            iq = iq * np.random.uniform(0.88, 1.12)
        
        # ❌ 删除时间平移（与模型内部 ComplexInputAugment 的 STO 重复）
        # if np.random.random() < 0.55:
        #     iq = np.roll(iq, np.random.randint(-24, 25))
        
        # 加性噪声
        if np.random.random() < 0.60:
            sigma = 0.01 + 0.02 * np.random.random()
            n = np.random.normal(0, sigma, iq.shape) + 1j * np.random.normal(
                0, sigma, iq.shape
            )
            iq = iq + n * (np.abs(iq).mean() + 1e-8)
        
        # 局部衰减：脉冲干扰/深衰落
        if np.random.random() < 0.20:
            start = np.random.randint(0, max(1, iq.shape[0] - 32))
            width = np.random.randint(8, 33)
            iq[start : start + width] *= np.random.uniform(0.2, 0.6)
        
        # ❌ 删除频率偏移（与模型内部 ComplexInputAugment 的 CFO 重复）
        # if np.random.random() < 0.25:
        #     freq_shift = np.random.uniform(-0.03, 0.03)
        #     t = np.arange(iq.shape[0])
        #     iq = iq * np.exp(1j * 2 * np.pi * freq_shift * t)
        
        return iq

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        iq = self.iq_data[idx].copy()
        if self.augment:
            iq = self.augment_signal(iq)
            # LFM 类 (label 0) 额外增强：添加随机 chirp 调制
            # 使"无干扰的干净回波"特征更显著，帮助模型建立 LFM 正向模板
            if self.labels[idx] == 0 and np.random.random() < 0.6:
                t = np.arange(iq.shape[0], dtype=np.float32)
                k_chirp = np.random.uniform(-0.015, 0.015)
                iq = iq * np.exp(1j * np.pi * k_chirp * t ** 2)
        real = torch.from_numpy(iq.real.astype(np.float32))
        imag = torch.from_numpy(iq.imag.astype(np.float32))
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return real, imag, y


@dataclass
class LoadedDataset:
    iq: np.ndarray
    labels: np.ndarray
    raw_labels: np.ndarray


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_h5_raw(path: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        iq = f["iq"][:]
        labels = f["labels"][:]
    if getattr(labels.dtype, "names", None) and "signal_type" in labels.dtype.names:
        raw = labels["signal_type"]
    else:
        raw = labels
    return iq, np.asarray(raw, dtype=np.int64)


def load_encoded_h5(path: str, schema: LabelSchema) -> LoadedDataset:
    iq, raw_labels = load_h5_raw(path)
    if iq.shape[1] != SEQ_LEN:
        raise ValueError(f"{path} 序列长度应为 {SEQ_LEN}，实际为 {iq.shape[1]}")
    iq, labels, keep_mask = schema.filter_iq_and_labels(iq, raw_labels)
    raw_labels = raw_labels[keep_mask]
    if len(iq) == 0:
        raise ValueError(f"{path} 经过类别配置过滤后没有可用样本")
    return LoadedDataset(iq=iq, labels=labels, raw_labels=raw_labels)


def compute_global_channel_stats(
    iq_data: np.ndarray, chunk_size: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    total_count = 0
    # 8 channels: real, imag, mag, phase (time) + FFT real, FFT imag, FFT mag, FFT phase (freq)
    sum_channels = np.zeros(8, dtype=np.float64)
    sumsq_channels = np.zeros(8, dtype=np.float64)
    for start in range(0, len(iq_data), chunk_size):
        chunk = np.asarray(iq_data[start : start + chunk_size])
        real = chunk.real.astype(np.float64, copy=False)
        imag = chunk.imag.astype(np.float64, copy=False)
        mag = np.abs(chunk).astype(np.float64, copy=False)
        phase = np.angle(chunk).astype(np.float64, copy=False)
        # FFT frequency domain (ortho normalization: /sqrt(N) to match time-domain scale)
        fft = np.fft.fft(chunk, axis=-1, norm="ortho")
        fft_real = fft.real.astype(np.float64, copy=False)
        fft_imag = fft.imag.astype(np.float64, copy=False)
        fft_mag = np.log1p(np.abs(fft).astype(np.float64, copy=False))
        fft_phase = np.angle(fft).astype(np.float64, copy=False)
        channels = (real, imag, mag, phase, fft_real, fft_imag, fft_mag, fft_phase)
        n = real.shape[0] * real.shape[1]
        total_count += n
        for idx, channel in enumerate(channels):
            sum_channels[idx] += float(channel.sum())
            sumsq_channels[idx] += float(np.square(channel).sum())
    if total_count <= 0:
        raise ValueError("cannot compute global normalization stats from empty dataset")
    mean = sum_channels / total_count
    var = np.maximum(sumsq_channels / total_count - np.square(mean), 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def stratified_indices(
    labels: np.ndarray, val_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []
    for c in np.unique(labels):
        cls = np.where(labels == c)[0]
        rng.shuffle(cls)
        n_val = max(1, int(len(cls) * val_ratio))
        if n_val >= len(cls):
            n_val = max(1, len(cls) - 1)
        val_idx.append(cls[:n_val])
        train_idx.append(cls[n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx)


def parse_class_weights(s: str, n_cls: int) -> torch.Tensor:
    if str(s).strip().lower() in {"", "auto"}:
        raise ValueError("auto class weights should be built from training labels")
    vals = [v.strip() for v in str(s).split(",") if v.strip()]
    if not vals:
        return torch.ones(n_cls, dtype=torch.float32)
    if len(vals) != n_cls:
        raise ValueError(
            f"--class_weights 需要 {n_cls} 个值，当前得到 {len(vals)} 个: {s}"
        )
    w = torch.tensor([float(v) for v in vals], dtype=torch.float32)
    if torch.any(w <= 0):
        raise ValueError(f"--class_weights 必须全部为正数，当前为 {w.tolist()}")
    return w


def make_scaler(amp_enabled: bool):
    if not (amp_enabled and torch.cuda.is_available()):
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        return torch.cuda.amp.GradScaler()


def setup_cuda_runtime(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if args.allow_tf32 else "medium")


def compute_auto_class_weights(
    labels: np.ndarray,
    n_cls: int,
    *,
    min_weight: float = 0.7,
    max_weight: float = 2.5,
) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_cls).astype(
        np.float64
    )
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, np.maximum(counts, 1.0))
    weights = (1.0 - beta) / np.clip(effective_num, 1e-12, None)
    weights = weights / np.mean(weights)
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def build_initial_class_weights(
    class_weights_arg: str,
    labels: np.ndarray,
    n_cls: int,
    *,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    token = str(class_weights_arg).strip().lower()
    if token in {"", "auto"}:
        return compute_auto_class_weights(
            labels, n_cls, min_weight=min_weight, max_weight=max_weight
        )
    return parse_class_weights(class_weights_arg, n_cls)


def class_recalls_from_preds(
    gts: np.ndarray, preds: np.ndarray, n_cls: int
) -> np.ndarray:
    recalls = np.full((n_cls,), np.nan, dtype=np.float64)
    for c in range(n_cls):
        mask = gts == c
        n = int(np.sum(mask))
        if n > 0:
            recalls[c] = float(np.mean(preds[mask] == c))
    return recalls


def compute_adaptive_class_weights(
    recalls: np.ndarray,
    prev_weights: torch.Tensor,
    *,
    base_weights: torch.Tensor,
    momentum: float = 0.85,
    min_weight: float = 0.7,
    max_weight: float = 2.5,
) -> torch.Tensor:
    valid = np.isfinite(recalls)
    if not np.any(valid):
        return prev_weights
    stable = recalls.copy()
    stable[~valid] = np.nanmean(stable[valid])
    inv = 1.0 / np.clip(stable, 0.2, 1.0)
    inv = inv / np.mean(inv)
    target = base_weights.detach().cpu().numpy() * inv
    target = target / np.mean(target)
    target_t = torch.tensor(target, dtype=torch.float32, device=prev_weights.device)
    blended = momentum * prev_weights + (1.0 - momentum) * target_t
    blended = torch.clamp(blended, min=min_weight, max=max_weight)
    blended = blended / blended.mean()
    return blended


def maybe_update_criterion_class_weights(
    criterion: nn.Module, new_weights: torch.Tensor
) -> bool:
    if isinstance(criterion, nn.CrossEntropyLoss):
        criterion.weight = new_weights.detach().clone()
        return True
    return False


def make_weighted_sampler(
    labels: np.ndarray, n_cls: int, power: float = 1.0
) -> WeightedRandomSampler:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_cls).astype(
        np.float64
    )
    class_weights = np.power(np.clip(counts, 1.0, None), -power)
    sample_weights = class_weights[np.asarray(labels, dtype=np.int64)]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def resolve_loader_worker_counts(args: argparse.Namespace) -> tuple[int, int]:
    train_val_workers = max(0, int(args.num_workers))
    test_workers = max(0, int(args.test_num_workers))
    if (
        os.name == "nt"
        and torch.cuda.is_available()
        and not args.force_windows_loader_workers
    ):
        requested = max(train_val_workers, test_workers)
        if requested > 0:
            print(
                "[提示] 检测到 Windows + CUDA。"
                f"为避免 DataLoader 子进程重复加载 PyTorch/CUDA DLL 导致 WinError 1455，"
                f"已将 num_workers 从 train/val={train_val_workers}, test={test_workers} 自动降为 0。"
            )
        return 0, 0
    return train_val_workers, test_workers


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    sampler=None,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False if sampler is not None else shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler,
    *,
    epoch: int,
    total_epochs: int,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    model.train()
    use_amp = scaler is not None
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} 训练",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.05,
    )
    for real, imag, labels in pbar:
        real = real.to(DEVICE, non_blocking=True)
        imag = imag.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                logits = model(real, imag)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # 梯度诊断
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gnorm = param.grad.norm().item()
                    if gnorm > 10000:
                        print(f"\n[WARNING] Large gradient {name}: {gnorm:.2f}")
                    elif gnorm < 1e-6:
                        print(f"\n[WARNING] Vanishing gradient {name}: {gnorm:.2e}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(real, imag)
            loss = criterion(logits, labels)
            loss.backward()
            # 梯度诊断
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gnorm = param.grad.norm().item()
                    if gnorm > 10000:
                        print(f"\n[WARNING] Large gradient {name}: {gnorm:.2f}")
                    elif gnorm < 1e-6:
                        print(f"\n[WARNING] Vanishing gradient {name}: {gnorm:.2e}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        total_loss += float(loss.item())
        pred = logits.argmax(dim=1)
        total_correct += int(pred.eq(labels).sum().item())
        total_seen += int(labels.size(0))
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * total_correct / max(total_seen, 1):.2f}%",
        )
    return total_loss / max(len(loader), 1), 100.0 * total_correct / max(total_seen, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    pbar_desc: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_correct = 0
    total_seen = 0
    preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    pbar = tqdm(
        loader, desc=pbar_desc, leave=False, dynamic_ncols=True, mininterval=0.05
    )
    for real, imag, labels in pbar:
        real = real.to(DEVICE, non_blocking=True)
        imag = imag.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=amp_enabled and torch.cuda.is_available(),
        ):
            logits = model(real, imag)
        pred = logits.argmax(dim=1)
        total_correct += int(pred.eq(labels).sum().item())
        total_seen += int(labels.size(0))
        preds.append(pred.cpu().numpy())
        gts.append(labels.cpu().numpy())
        pbar.set_postfix(acc=f"{100.0 * total_correct / max(total_seen, 1):.2f}%")
    return (
        100.0 * total_correct / max(total_seen, 1),
        np.concatenate(preds),
        np.concatenate(gts),
    )


def per_class_stats(
    gts: np.ndarray, preds: np.ndarray, schema: LabelSchema
) -> list[dict[str, Any]]:
    rows = []
    for model_idx, raw_label in schema.model_to_raw.items():
        mask_true = gts == model_idx
        n_true = int(np.sum(mask_true))
        n_correct = int(np.sum((preds == gts) & mask_true))
        recall_pct = float("nan") if n_true == 0 else 100.0 * n_correct / n_true
        mask_pred = preds == model_idx
        n_pred = int(np.sum(mask_pred))
        n_tp = int(np.sum((gts == model_idx) & mask_pred))
        precision_pct = float("nan") if n_pred == 0 else 100.0 * n_tp / n_pred
        rows.append(
            {
                "model_label": int(model_idx),
                "raw_label": int(raw_label),
                "class_name": schema.name_for_model_index(model_idx),
                "num_samples_true": n_true,
                "num_correct_within_class": n_correct,
                "accuracy_within_true_class_pct": recall_pct,
                "num_predicted_as_class": n_pred,
                "precision_when_predicted_as_class_pct": precision_pct,
            }
        )
    return rows


def print_schema(schema: LabelSchema) -> None:
    print("当前类别映射:")
    for model_idx, entry in enumerate(schema.entries):
        print(f"  model={model_idx} <- raw={entry.raw_label} | {entry.name}")


def print_dataset_stats(name: str, labels: np.ndarray, schema: LabelSchema) -> None:
    counts = np.bincount(labels, minlength=schema.num_classes)
    print(f"{name} 样本统计:")
    for model_idx, count in enumerate(counts.tolist()):
        print(
            f"  model={model_idx} raw={schema.model_to_raw[model_idx]} {schema.name_for_model_index(model_idx)}: {count}"
        )


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    scaler,
    *,
    epoch: int,
    best_val: float,
    best_state: Optional[dict[str, Any]],
    patience_left: int,
    args: argparse.Namespace,
    schema: LabelSchema,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val": best_val,
        "best_state_dict": best_state,
        "patience_left": patience_left,
        "label_schema": schema.to_dict(),
        "arch_config": {
            "use_complex_conv": args.use_complex_conv,
            "use_multiscale_fusion": args.use_multiscale_fusion,
            "use_enhanced_pooling": args.use_enhanced_pooling,
            "use_mamba": args.use_mamba,
            "mamba3_layers": args.mamba3_layers,
            "mamba3_d_state": args.mamba3_d_state,
            "mamba3_expand": args.mamba3_expand,
            "mamba3_headdim": args.mamba3_headdim,
            "mamba3_dropout": args.mamba3_dropout,
            "use_prob_sparse_attention": args.use_prob_sparse_attention,
            "prob_sparse_layers": args.prob_sparse_layers,
            "prob_sparse_heads": args.prob_sparse_heads,
            "prob_sparse_ffn_dim": args.prob_sparse_ffn_dim,
            "prob_sparse_dropout": args.prob_sparse_dropout,
            "prob_sparse_top_k": args.prob_sparse_top_k,
            "stft_n_fft": args.stft_n_fft,
            "stft_hop_length": args.stft_hop_length,
            "timefreq_mode": args.timefreq_mode,
            "temporal_module": args.temporal_module,
            "informer_factor": args.informer_factor,
            "informer_distil": args.informer_distil,
            "fedformer_modes": args.fedformer_modes,
            "fedformer_moving_avg": args.fedformer_moving_avg,
        },
        "config": {
            "train_h5": args.train_h5,
            "val_h5": args.val_h5,
            "test_h5": args.test_h5,
            "class_config": args.class_config,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "amp": args.amp,
            "balance_strategy": args.balance_strategy,
            "sampler_power": args.sampler_power,
            "label_smoothing": args.label_smoothing,
        },
        "global_channel_stats": {
            "mean": model.channel_mean.cpu().numpy().tolist(),
            "std": model.channel_std.cpu().numpy().tolist(),
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


def load_training_checkpoint(
    path: str, model: nn.Module, optimizer: optim.Optimizer, scheduler: Any, scaler
):
    try:
        payload = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=DEVICE)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    sd = payload.get("scaler_state_dict")
    if scaler is not None and sd is not None:
        scaler.load_state_dict(sd)
    return payload


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    setup_cuda_runtime(args)
    print(f"设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    discovered = discover_raw_labels([args.train_h5, args.val_h5, args.test_h5])
    schema = load_label_schema(
        args.class_config, discovered_labels=discovered, auto_create=True
    )
    filtered_raw_labels = [
        label for label in discovered if label not in schema.raw_to_model
    ]
    print_schema(schema)
    if filtered_raw_labels:
        print(f"已按类别配置过滤原始标签: {filtered_raw_labels}")

    train_data = load_encoded_h5(args.train_h5, schema)
    if args.val_h5:
        val_data = load_encoded_h5(args.val_h5, schema)
    else:
        tr_idx, va_idx = stratified_indices(
            train_data.labels, args.val_ratio, args.seed
        )
        val_data = LoadedDataset(
            iq=train_data.iq[va_idx],
            labels=train_data.labels[va_idx],
            raw_labels=train_data.raw_labels[va_idx],
        )
        train_data = LoadedDataset(
            iq=train_data.iq[tr_idx],
            labels=train_data.labels[tr_idx],
            raw_labels=train_data.raw_labels[tr_idx],
        )
    test_data = load_encoded_h5(args.test_h5, schema) if args.test_h5 else None

    print_dataset_stats("Train", train_data.labels, schema)
    print_dataset_stats("Val", val_data.labels, schema)
    if test_data is not None:
        print_dataset_stats("Test", test_data.labels, schema)
    global_mean, global_std = compute_global_channel_stats(train_data.iq)
    print(f"Global norm mean: {np.round(global_mean, 6).tolist()}")
    print(f"Global norm std: {np.round(global_std, 6).tolist()}")

    train_val_workers, test_workers = resolve_loader_worker_counts(args)
    print(f"DataLoader workers: train/val={train_val_workers}, test={test_workers}")

    train_sampler = None
    if args.balance_strategy in {"sampler", "both"}:
        train_sampler = make_weighted_sampler(
            train_data.labels,
            schema.num_classes,
            power=args.sampler_power,
        )
        print(f"训练采样: WeightedRandomSampler(power={args.sampler_power:.2f})")

    train_loader = build_loader(
        RadCharDataset(train_data.iq, train_data.labels, augment=not args.no_augment),
        args.batch_size,
        True,
        train_val_workers,
        args.prefetch_factor,
        sampler=train_sampler,
    )
    val_loader = build_loader(
        RadCharDataset(val_data.iq, val_data.labels, augment=False),
        args.batch_size,
        False,
        train_val_workers,
        args.prefetch_factor,
    )
    test_loader = None
    if test_data is not None:
        test_loader = build_loader(
            RadCharDataset(test_data.iq, test_data.labels, augment=False),
            args.batch_size,
            False,
            test_workers,
            args.prefetch_factor,
        )

    arch_config = ModelArchitectureConfig(
        use_complex_conv=args.use_complex_conv,
        use_multiscale_fusion=args.use_multiscale_fusion,
        use_enhanced_pooling=args.use_enhanced_pooling,
        pool_layers=args.pool_layers,
        use_dilated_conv=args.use_dilated_conv,
        stft_n_fft=args.stft_n_fft,
        stft_hop_length=args.stft_hop_length,
        timefreq_mode=args.timefreq_mode,
        use_mamba=args.use_mamba,
        mamba3_layers=args.mamba3_layers,
        mamba3_d_state=args.mamba3_d_state,
        mamba3_expand=args.mamba3_expand,
        mamba3_headdim=args.mamba3_headdim,
        mamba3_dropout=args.mamba3_dropout,
        use_complex_mamba=args.use_complex_mamba,
        use_prob_sparse_attention=args.use_prob_sparse_attention,
        prob_sparse_layers=args.prob_sparse_layers,
        prob_sparse_heads=args.prob_sparse_heads,
        prob_sparse_ffn_dim=args.prob_sparse_ffn_dim,
        prob_sparse_dropout=args.prob_sparse_dropout,
        prob_sparse_top_k=args.prob_sparse_top_k,
        use_learnable_fusion=args.use_learnable_fusion,
        temporal_module=args.temporal_module,
        informer_factor=args.informer_factor,
        informer_distil=args.informer_distil,
        fedformer_modes=args.fedformer_modes,
        fedformer_moving_avg=args.fedformer_moving_avg,
    )
    model = TimeFreqRadarNet(
        num_classes=schema.num_classes,
        seq_len=SEQ_LEN,
        arch_config=arch_config,
    ).to(DEVICE)
    model.set_global_channel_stats(global_mean, global_std)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"已启用 torch.compile(mode={args.compile_mode})")
        except Exception as exc:
            print(f"[警告] torch.compile 启用失败，继续使用 eager: {exc}")

    use_class_weights = args.balance_strategy in {"weights", "both"} or str(
        args.class_weights
    ).strip().lower() not in {"", "auto"}
    if use_class_weights:
        class_weights = build_initial_class_weights(
            args.class_weights,
            train_data.labels,
            schema.num_classes,
            min_weight=args.min_class_weight,
            max_weight=args.max_class_weight,
        ).to(DEVICE)
    else:
        class_weights = torch.ones(
            schema.num_classes, dtype=torch.float32, device=DEVICE
        )
    print(
        f"初始 class_weights: {class_weights.detach().cpu().numpy().round(3).tolist()}"
    )
    criterion = nn.CrossEntropyLoss(
    weight=class_weights, label_smoothing=args.label_smoothing
)
    print(f"损失函数: CrossEntropy(label_smoothing={args.label_smoothing:.3f})")
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # 学习率调度器
    if args.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
        )
    elif args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    elif args.lr_scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos',
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
        )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = make_scaler(args.amp)

    start_epoch = 1
    best_val = -1.0
    best_state = None
    best_epoch = 0
    patience_left = args.patience
    cooldown_left = 0
    # 保存初始类别权重作为自适应调整的基准
    base_class_weights = class_weights.detach().clone()
    if (not args.no_resume) and os.path.isfile(args.checkpoint_path):
        payload = load_training_checkpoint(
            args.checkpoint_path, model, optimizer, scheduler, scaler
        )
        start_epoch = int(payload["epoch"]) + 1
        best_val = float(payload.get("best_val", -1.0))
        best_state = payload.get("best_state_dict")
        patience_left = int(payload.get("patience_left", args.patience))
        print(f"检测到 checkpoint，继续从 epoch {start_epoch} 开始")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            epoch=epoch,
            total_epochs=args.epochs,
            amp_dtype=amp_dtype,
        )
        val_acc, val_preds, val_gts = evaluate(
            model,
            val_loader,
            amp_enabled=args.amp,
            amp_dtype=amp_dtype,
            pbar_desc=f"Epoch {epoch}/{args.epochs} 验证",
        )
        # 学习率调度
        if args.lr_scheduler == "plateau":
            scheduler.step(val_acc)
        elif args.lr_scheduler == "onecycle":
            scheduler.step()
        else:
            scheduler.step()

        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% val_acc={val_acc:.2f}% "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        for row in per_class_stats(val_gts, val_preds, schema):
            recall = row["accuracy_within_true_class_pct"]
            precision = row["precision_when_predicted_as_class_pct"]
            recall_s = "nan" if np.isnan(recall) else f"{recall:.2f}"
            precision_s = "nan" if np.isnan(precision) else f"{precision:.2f}"
            print(
                f"  model={row['model_label']} raw={row['raw_label']} {row['class_name']}: "
                f"true={row['num_samples_true']} recall={recall_s}% precision={precision_s}%"
            )

        # 启用自适应类别权重：根据验证集召回率动态调整
        if args.adaptive_class_weights and use_class_weights:
            recalls = class_recalls_from_preds(val_gts, val_preds, schema.num_classes)
            # 保存初始权重作为基准（仅首次）
            if epoch == start_epoch:
                base_class_weights = class_weights.detach().clone()
            new_weights = compute_adaptive_class_weights(
                recalls,
                class_weights,
                base_weights=base_class_weights.to(class_weights.device),
                momentum=args.adapt_momentum,
                min_weight=args.min_class_weight,
                max_weight=args.max_class_weight,
            )
            if maybe_update_criterion_class_weights(criterion, new_weights):
                class_weights = new_weights.detach().clone()
                print(
                    f"  自适应 class_weights: {class_weights.detach().cpu().numpy().round(3).tolist()}"
                )

        if val_acc > best_val + args.early_stopping_delta:
            improvement = val_acc - best_val
            best_val = val_acc
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_left = args.patience
            cooldown_left = args.early_stopping_cooldown
            # 立即保存 best model
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": best_state,
                    "best_state_dict": best_state,
                    "label_schema": schema.to_dict(),
                    "arch_config": {
                        "use_complex_conv": args.use_complex_conv,
                        "use_multiscale_fusion": args.use_multiscale_fusion,
                        "use_enhanced_pooling": args.use_enhanced_pooling,
                        "pool_layers": args.pool_layers,
                        "use_dilated_conv": args.use_dilated_conv,
                        "use_mamba": args.use_mamba,
                        "mamba3_layers": args.mamba3_layers,
                        "mamba3_d_state": args.mamba3_d_state,
                        "mamba3_expand": args.mamba3_expand,
                        "mamba3_headdim": args.mamba3_headdim,
                        "mamba3_dropout": args.mamba3_dropout,
                        "use_complex_mamba": args.use_complex_mamba,
                        "use_prob_sparse_attention": args.use_prob_sparse_attention,
                        "prob_sparse_layers": args.prob_sparse_layers,
                        "prob_sparse_heads": args.prob_sparse_heads,
                        "prob_sparse_ffn_dim": args.prob_sparse_ffn_dim,
                        "prob_sparse_dropout": args.prob_sparse_dropout,
                        "prob_sparse_top_k": args.prob_sparse_top_k,
                        "stft_n_fft": args.stft_n_fft,
                        "stft_hop_length": args.stft_hop_length,
                        "timefreq_mode": args.timefreq_mode,
                        "use_learnable_fusion": args.use_learnable_fusion,
                        "temporal_module": args.temporal_module,
                        "informer_factor": args.informer_factor,
                        "informer_distil": args.informer_distil,
                        "fedformer_modes": args.fedformer_modes,
                        "fedformer_moving_avg": args.fedformer_moving_avg,
                    },
                },
                args.save_path,
            )
        elif cooldown_left > 0:
            cooldown_left -= 1
        elif args.patience > 0:
            patience_left -= 1

        save_checkpoint(
            args.checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch,
            best_val=best_val,
            best_state=best_state,
            patience_left=patience_left,
            args=args,
            schema=schema,
        )
        if args.patience > 0 and patience_left <= 0:
            print(f"早停: 验证集连续 {args.patience} 轮未提升 (阈值={args.early_stopping_delta})，最佳 epoch={best_epoch}, acc={best_val:.2f}%")
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_state_dict": best_state,
            "label_schema": schema.to_dict(),
            "arch_config": {
                "use_complex_conv": args.use_complex_conv,
                "use_multiscale_fusion": args.use_multiscale_fusion,
                "use_enhanced_pooling": args.use_enhanced_pooling,
                "pool_layers": args.pool_layers,
                "use_dilated_conv": args.use_dilated_conv,
                "use_mamba": args.use_mamba,
                "mamba3_layers": args.mamba3_layers,
                "mamba3_d_state": args.mamba3_d_state,
                "mamba3_expand": args.mamba3_expand,
                "mamba3_headdim": args.mamba3_headdim,
                "mamba3_dropout": args.mamba3_dropout,
                "use_complex_mamba": args.use_complex_mamba,
                "use_prob_sparse_attention": args.use_prob_sparse_attention,
                "prob_sparse_layers": args.prob_sparse_layers,
                "prob_sparse_heads": args.prob_sparse_heads,
                "prob_sparse_ffn_dim": args.prob_sparse_ffn_dim,
                "prob_sparse_dropout": args.prob_sparse_dropout,
                "prob_sparse_top_k": args.prob_sparse_top_k,
                "stft_n_fft": args.stft_n_fft,
                "stft_hop_length": args.stft_hop_length,
                "timefreq_mode": args.timefreq_mode,
                "use_learnable_fusion": args.use_learnable_fusion,
                "temporal_module": args.temporal_module,
                "informer_factor": args.informer_factor,
                "informer_distil": args.informer_distil,
                "fedformer_modes": args.fedformer_modes,
                "fedformer_moving_avg": args.fedformer_moving_avg,
            },
        },
        args.save_path,
    )
    print(f"最佳验证准确率: {best_val:.2f}%")
    print(f"最佳模型已保存到: {args.save_path}")

    if test_loader is not None:
        test_acc, test_preds, test_gts = evaluate(
            model,
            test_loader,
            amp_enabled=args.amp,
            amp_dtype=amp_dtype,
            pbar_desc="测试",
        )
        print(f"测试集整体准确率: {test_acc:.2f}%")
        stats_path = args.test_stats_path.strip()
        if stats_path:
            os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(
                    per_class_stats(test_gts, test_preds, schema),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"测试集分类统计已保存到: {stats_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RadChar 动态类别训练脚本")
    p.add_argument("--train_h5", type=str, default="Bear_data/RadChar-Train.h5")
    p.add_argument("--val_h5", type=str, default="Bear_data/RadChar-Val.h5")
    p.add_argument("--test_h5", type=str, default="Bear_data/RadChar-Test.h5")
    p.add_argument("--class_config", type=str, default="radchar_classes.json")
    p.add_argument("--save_path", type=str, default="models/radchar_best.pth")
    p.add_argument(
        "--checkpoint_path", type=str, default="models/radchar_checkpoint.pth"
    )
    p.add_argument(
        "--test_stats_path", type=str, default="results/radchar_test_stats.json"
    )
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--val_ratio", type=float, default=0.12)
    p.add_argument("--patience", type=int, default=30, help="早停耐心值")
    p.add_argument("--early_stopping_delta", type=float, default=0.0, help="早停最小提升阈值")
    p.add_argument("--early_stopping_cooldown", type=int, default=0, help="早停冷却epoch数")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--class_weights", type=str, default="")
    p.add_argument(
        "--balance_strategy",
        type=str,
        choices=["none", "sampler", "weights", "both"],
        default="none",
    )
    p.add_argument("--sampler_power", type=float, default=1.0)
    p.add_argument("--adaptive_class_weights", action="store_true", default=False)
    p.add_argument(
        "--no-adaptive-class-weights",
        action="store_false",
        dest="adaptive_class_weights",
    )
    p.add_argument("--adapt_momentum", type=float, default=0.85)
    p.add_argument("--min_class_weight", type=float, default=0.7)
    p.add_argument("--max_class_weight", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--amp_dtype", type=str, choices=["fp16", "bf16"], default="bf16")
    p.add_argument(
        "--num_workers",
        type=int,
        default=0 if os.name == "nt" else max(2, min(12, (os.cpu_count() or 8) // 2)),
    )
    p.add_argument("--test_num_workers", type=int, default=0)
    p.add_argument(
        "--force-windows-loader-workers",
        action="store_true",
        dest="force_windows_loader_workers",
    )
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--use_complex_conv", action="store_true", default=False, help="启用复数卷积层(仅用于非STFT输入)")
    p.add_argument("--use_multiscale_fusion", action="store_true", default=True, help="启用多尺度特征融合")
    p.add_argument("--use_enhanced_pooling", action="store_true", default=True, help="启用增强特征池化(谱质心、谱熵等)")
    # Mamba3 参数
    p.add_argument("--use_mamba", action="store_true", default=True, help="启用 Mamba3 层")
    p.add_argument("--no_mamba", action="store_false", dest="use_mamba", help="禁用 Mamba3 层")
    p.add_argument("--mamba3_layers", type=int, default=2, help="Mamba3 层数")
    p.add_argument("--mamba3_d_state", type=int, default=128, help="Mamba3 状态维度")
    p.add_argument("--mamba3_expand", type=int, default=2, help="Mamba3 扩展因子")
    p.add_argument("--mamba3_headdim", type=int, default=64, help="Mamba3 头维度")
    p.add_argument("--mamba3_dropout", type=float, default=0.2, help="Mamba3 Dropout")
    p.add_argument("--use_complex_mamba", action="store_true", default=False, help="启用复数 Mamba3 (待完善)")
    p.add_argument("--no_complex_mamba", action="store_false", dest="use_complex_mamba", help="禁用复数 Mamba3")
    # 池化配置
    p.add_argument("--pool_layers", type=int, default=2, help="MaxPool 层数 (2=512→128, 4=512→32)")
    p.add_argument("--use_dilated_conv", action="store_true", default=True, help="使用空洞卷积扩大感受野")
    p.add_argument("--no_dilated_conv", action="store_false", dest="use_dilated_conv", help="禁用空洞卷积")
    # 可学习特征融合
    p.add_argument("--use_learnable_fusion", action="store_true", default=True, help="使用可学习特征融合")
    p.add_argument("--no_learnable_fusion", action="store_false", dest="use_learnable_fusion", help="禁用可学习特征融合")
    # 学习率调度
    p.add_argument("--lr_scheduler", type=str, default="cosine",
                   choices=["plateau", "cosine", "onecycle"],
                   help="学习率调度器: plateau(默认)/cosine/onecycle")
    p.add_argument("--max_lr", type=float, default=1e-4, help="OneCycleLR 最大学习率")
    # ProbSparse Attention 参数
    p.add_argument("--use_prob_sparse_attention", action="store_true", default=True, help="启用 ProbSparse Attention")
    p.add_argument("--no_prob_sparse", action="store_false", dest="use_prob_sparse_attention", help="禁用 ProbSparse Attention")
    p.add_argument("--prob_sparse_layers", type=int, default=3, help="ProbSparse Encoder 层数")
    p.add_argument("--prob_sparse_heads", type=int, default=6, help="注意力头数量")
    p.add_argument("--prob_sparse_ffn_dim", type=int, default=512, help="前馈网络维度")
    p.add_argument("--prob_sparse_dropout", type=float, default=0.2, help="Dropout")
    p.add_argument("--prob_sparse_top_k", type=int, default=32, help="ProbSparse top-k")
    # Temporal Module Selection
    p.add_argument("--temporal_module", type=str, default="prob_sparse",
                   choices=["prob_sparse", "informer", "fedformer"],
                   help="Temporal modeling module: prob_sparse (default), informer, fedformer")
    # Informer params
    p.add_argument("--informer_factor", type=int, default=5, help="Informer attention factor")
    p.add_argument("--informer_distil", action="store_true", default=True, help="Informer use distilation")
    p.add_argument("--no_informer_distil", action="store_false", dest="informer_distil", help="Disable Informer distilation")
    # FEDformer params
    p.add_argument("--fedformer_modes", type=int, default=32, help="FEDformer FFT modes")
    p.add_argument("--fedformer_moving_avg", type=int, default=25, help="FEDformer moving average kernel size")
    # STFT 参数
    p.add_argument("--stft_n_fft", type=int, default=64, help="STFT FFT 大小")
    p.add_argument("--stft_hop_length", type=int, default=16, help="STFT 跳跃长度")
    p.add_argument("--timefreq_mode", type=str, choices=["fft", "stft", "glct"],
                   default="fft", help="时频变换模式: fft/stft/glct")
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument(
        "--compile_mode",
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    p.add_argument("--allow_tf32", action="store_true", default=True)
    p.add_argument("--no-allow_tf32", action="store_false", dest="allow_tf32")
    p.add_argument("--cudnn_benchmark", action="store_true", default=True)
    p.add_argument("--no-cudnn_benchmark", action="store_false", dest="cudnn_benchmark")
    p.add_argument("--no-resume", action="store_true", dest="no_resume")
    p.add_argument("--no-augment", action="store_true", dest="no_augment")
    return p.parse_args()


def main() -> int:
    run_training(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
