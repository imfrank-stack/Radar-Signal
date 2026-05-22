# AGENTS.md

Radar jamming signal dataset generator + PyTorch CNN+Mamba3 classifier. 12 signal classes, 105,600 samples.

## Workflow (2 Phases)

1. **Generate dataset** → `python generate_data.py --gpu --mat-only` or `generate_data_parallel.py --mat-only`
2. **Convert & train** → `cd radchar_refactored3 && python convert_mat_to_h5.py && python radchar_training.py --no-resume`

## Key Files

| File | Purpose |
|------|---------|
| `radar_jamming_signals.py` | Core signal generation (1809 lines, 24 functions) |
| `generate_data.py` | Single-process generator (GPU via CuPy or CPU) |
| `generate_data_parallel.py` | NUMA-aware multiprocessing, CPU core binding |
| `gpu_batch_utils.py` | GPU batch processing utilities |
| `radchar_refactored3/` | PyTorch model training (see child AGENTS.md) |
| `matlab/` | Original MATLAB source files (reference only, do not modify) |

## Data Pipeline

```
generate_data.py/parallel.py → .mat files (dataset/Training_data/dataset_seq/All_dB/)
radchar_refactored3/convert_mat_to_h5.py → HDF5 (radchar_refactored3/Bear_data/)
radchar_refactored3/radchar_training.py → model training
```

- 12 signals: LFM, AM, COMB, FM, ISRJ, MNJ, RMT, RGPO, R_VGPO, SMSP, VGPO, VMT
- SNR: -10 to 10 dB (11 values), 800 samples/SNR → 105,600 total
- Split: 80/10/10 train/val/test (stratified, seed=42)
- IQ format: complex64, (N, 1024), real/imag = I/Q channels

## Dependencies

```
numpy scipy matplotlib Pillow tqdm h5py torch
cupy-cuda12x  # optional, GPU acceleration
```

## Performance

| Mode | Time | Notes |
|------|------|-------|
| GPU + MAT-only | ~40s | `--gpu --mat-only` |
| CPU parallel (64-core) | ~2-3min | `--mat-only --num-workers 64` |
| CPU single | ~1-2hr | `--cpu --mat-only` |

## Conventions

- Python 3.8+, NumPy/SciPy vectorized operations
- CuPy drop-in replacement for NumPy on GPU (`cp` vs `np`)
- Random seed propagation for reproducibility (default seed=42)
- MATLAB originals in `matlab/` are reference only; Python is source of truth

## Child AGENTS.md

- `radchar_refactored3/AGENTS.md` → Model architecture, training commands, data format
