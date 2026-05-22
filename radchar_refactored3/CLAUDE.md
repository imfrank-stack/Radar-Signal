# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RadChar is a PyTorch-based radar signal classification model using **CNN + Mamba3 + ProbSparse Self-Attention** architecture for IQ data sequences (length 512). Supports 12 signal classes (configurable via `radchar_classes.json`).

## Commands

```bash
# Training (default config: pool_layers=2, dilated conv, learnable fusion, cosine LR)
python radchar_training.py --no-resume

# Training with specific architecture
python radchar_training.py --no-resume --pool_layers 2 --use_dilated_conv --use_learnable_fusion

# Test
python radchar_test.py

# Ensemble
python radchar_ensemble.py

# Common training options
--no_resume           # Start fresh (don't load checkpoint)
--batch_size 64       # Default 32
--lr 1e-4             # Default 1e-4
--epochs 60           # Default 60
--patience 30         # Early stopping patience
--use_mamba           # Enable Mamba3 (default)
--no_prob_sparse      # Disable ProbSparse attention
--compile             # Enable torch.compile
```

## Architecture

**Input Pipeline**: I (B, 512) + Q (B, 512) → ComplexInputAugment (CFO/STO) → IQInputProcessor → (B, 3, 512) I/Q/FFT

**Backbone**: CNN Stem (3→64→96) → DilatedConvBlock (dilation 1,2,4,8) → Layer1/2/3 (96→128→192→256) with configurable MaxPool → TemporalProj (256→192)

**Temporal Modeling** (parallel branches):
- Mamba3Block: SSM for sequential dependencies (~192 dim output)
- ProbSparseEncoder: Long-range attention with top-k sampling

**Fusion**: LearnableFeatureFusion combines mean/max pooling + Mamba output + Attention output via learnable weights + gating

**Classifier**: MLP Head (feat_dim→256→192) → Linear(192, num_classes)

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `TimeFreqRadarNet` | radchar_model.py:852 | Main model class |
| `Mamba3Block` | radchar_model.py:715 | SSM temporal modeling |
| `ProbSparseEncoder` | radchar_model.py:372 | Sparse attention layer |
| `LearnableFeatureFusion` | radchar_model.py:621 | Feature fusion |
| `ComplexInputAugment` | radchar_model.py:421 | CFO/STO augmentation |
| `LabelSchema` | radchar_class_config.py:22 | Class label mapping |

## Data Format

- Training/Val/Test data: HDF5 files with `iq` (complex64 array) and `labels` datasets
- IQ shape: (num_samples, 512), complex values
- Labels: integer signal type IDs
- Global channel normalization: computed over 8 channels (real, imag, mag, phase, FFT real/imag/mag/phase)

## Output Files

```
models/radchar_best.pth        # Best model weights (auto-saved on val improvement)
models/radchar_checkpoint.pth  # Training checkpoint (resume support)
results/radchar_test_stats.json # Per-class precision/recall
results/radchar_test_detail.csv # Per-sample predictions
```

## Gradient Troubleshooting

If loss becomes NaN → gradient explosion: reduce learning rate or enable `clip_grad_norm_`
If accuracy stays ~10% (random) → gradient vanishing, check RMSNorm configuration
If one class has 100% recall, others 0% → class collapse, reduce dropout
