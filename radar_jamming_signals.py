"""
Radar Active Jamming Signal Modulation Dataset Generating Code
Converted from MATLAB to Python

This module contains 23 signal generation functions converted from 23 MATLAB (.m) files,
plus 2 helper functions (_save_spectrogram_image, _generate_lfm), for a total of 25 functions.
Each signal generation function generates radar jamming signals with spectrograms and saves
them to image folders and sequence (.mat) files.

Functions mapping (23 signal generation functions):
- Test set (200 samples per SNR, per-SNR folder): LFM, AM, COMB, FM, ISRJ, MNJ, RMT, RGPO, R_VGPO, SMSP, VGPO, VMT (12 functions)
- Train set (800 samples per SNR, All_dB folder): LFM_alldb, AM_alldb, COMB_alldb, FM_alldb, ISRJ_alldb, MNJ_alldb,
RMT_alldb, RGPO_alldb, R_VGPO_alldb, SMSP_alldb, VGPO_alldb, VMT_alldb (12 functions)
Note: LFM_alldb has no corresponding original .m file; it follows the same _alldb structure as other training set functions.

Setup:
pip install numpy scipy matplotlib Pillow
"""

from tqdm import tqdm
from PIL import Image
from scipy.io import savemat
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing

# ===================================================================
# GPU Backend Support (CuPy acceleration)
# ===================================================================

try:
    import cupy as _cp
    _CUPY_AVAILABLE = True
except ImportError:
    _cp = None
    _CUPY_AVAILABLE = False

_g = np  # active backend (np or cp)
_BACKEND = "numpy"


def _to_cpu(x):
    """Convert GPU array to CPU numpy array if needed."""
    if _BACKEND == "cupy" and _CUPY_AVAILABLE:
        return _cp.asnumpy(x)
    return x


def set_backend(backend="numpy"):
    """Set computation backend: "numpy" or "cupy"."""
    global _BACKEND, _g
    if backend == "cupy":
        if not _CUPY_AVAILABLE:
            raise ImportError(
            "CuPy not available. Install: pip install cupy-cuda12x")
        _BACKEND = "cupy"
        _g = _cp
        print("[GPU] CuPy backend active")
    elif backend == "numpy":
        _BACKEND = "numpy"
        _g = np
        print("[CPU] NumPy backend active")
    else:
        raise ValueError("backend must be numpy or cupy")


# ---------------------------------------------------------------------------
# Helper / shared functions
# ---------------------------------------------------------------------------

# ===================================================================
# SNR range configuration
# Set these before calling signal functions to override defaults.
# Example: _SNR_CONFIG["start"] = -10; _SNR_CONFIG["stop"] = 12
# ===================================================================
_SNR_CONFIG = {"start": -10, "stop": 12, "step": 2}

# ======================
# Output configuration
# Set _SKIP_IMAGES = True to only generate .mat files (much faster)
# ==============================
_SKIP_IMAGES = False  # Set to True to skip spectrogram/image generation
_SEED_CONFIG = {"seed": 42, "deterministic": True}  # Seed configuration for reproducibility


def set_seed(seed):
    """Set random seed for reproducibility."""
    _SEED_CONFIG["seed"] = seed
    np.random.seed(seed)
    if _CUPY_AVAILABLE:
        _cp.random.seed(seed)

# =========================
# Dataset output paths (relative to current directory)
# ======================
_DATASET_ROOT = os.path.join(os.getcwd(), 'dataset')
_TEST_IMG_ROOT = os.path.join(_DATASET_ROOT, 'Test_data', 'dataset_img')
_TEST_SEQ_ROOT = os.path.join(_DATASET_ROOT, 'Test_data', 'dataset_seq')
_TEST_IMG_ALLDB_ROOT = os.path.join(
    _DATASET_ROOT,
    'Test_data',
    'dataset_img',
    'All_dB')
_TEST_SEQ_ALLDB_ROOT = os.path.join(
    _DATASET_ROOT,
    'Test_data',
    'dataset_seq',
    'All_dB')
_TRAIN_IMG_ROOT = os.path.join(
_DATASET_ROOT,
'Training_data',
'dataset_img',
'All_dB')
_TRAIN_SEQ_ROOT = os.path.join(
_DATASET_ROOT,
'Training_data',
'dataset_seq',
'All_dB')


def _save_spectrogram_image(signal_in, fs, folder_img, fname, figsize=(5.12, 3.84), dpi=100):
    """
    Plot spectrogram of signal_in, crop to canvas, resize to 224x224, save as PNG.
    Mimics MATLAB: figure(3); hamming(128); spectrogram(...); imagesc(...);
    axis off; set(gca,'Position',[0 0 1 1]); getframe(gcf); imresize(img,[224,224])

    If _SKIP_IMAGES is True, this function returns immediately without generating images.
    """
    if _SKIP_IMAGES:
        return  # Skip image generation for faster dataset creation
    nfft = 128
    noverlap = 127
    window = np.hamming(nfft)
    f, t, Sxx = signal.spectrogram(
    signal_in, fs, window=window, noverlap=noverlap, nfft=nfft, mode='complex')
    # MATLAB: imagesc(T, f, abs(fftshift(S,1))) => S is columns over time, rows over frequency
    # We need to fftshift along frequency axis (axis 0)
    S_shifted = np.fft.fftshift(Sxx, axes=0)
    S_abs = np.abs(S_shifted)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # equivalent to set(gca,'Position',[0 0 1 1])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(S_abs, aspect='auto', origin='lower',
    extent=[t[0], t[-1], f[0], f[-1]], cmap='viridis')
    ax.axis('off')
    fig.savefig(os.path.join(folder_img, fname), dpi=dpi)
    plt.close(fig)

    # Resize to 224x224 using PIL
    img = Image.open(os.path.join(folder_img, fname))
    img = img.resize((224, 224), Image.LANCZOS)
    img.save(os.path.join(folder_img, fname))


def _generate_lfm(fs, PRI, taup, fc):
    """Generate baseband LFM signal parameters and waveform."""
    Nr = int(np.round(fs * PRI))
    Nfast = int(np.round(fs * taup))
    t = np.arange(-Nfast / 2, Nfast / 2) * (1 / fs)
    k = (40e6) / taup  # B = 40e6
    lfm_pulse = np.exp(1j * (2 * np.pi * fc * t + np.pi * k * (t ** 2)))
    lfm = np.concatenate([lfm_pulse, np.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr
    return Nr, Nfast, t, lfm, samp_num, k


# ---------------------------------------------------------------------------
#  Test set functions (data_num=200, per-SNR folder)
# ---------------------------------------------------------------------------

def AM(data_num=None):
    """
    调幅干扰 AM (瞄准式)
    Converted from AM.m
    Creates Test_data folders under per-SFD (flat: SNR_dB/AM)
    """
    if data_num is None:
        data_num = 200
    fs = 100e6
    Ts = 1 / fs
    PRI = 100e-6
    fc = _g.random.uniform(12e6, 14e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * Ts
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    root_folder_img = _TEST_IMG_ROOT
    root_folder_seq = _TEST_SEQ_ROOT

    K_AM_all = [2, 3, 4]

    for SNR in tqdm(
    range(
    _SNR_CONFIG['start'],
    _SNR_CONFIG['stop'],
    _SNR_CONFIG['step']),
    desc="SNR",
    leave=False):
        folder_path_img = os.path.join(root_folder_img, f'{SNR}_dB', 'AM')
        folder_path_seq = os.path.join(root_folder_seq, f'{SNR}_dB', 'AM')
        os.makedirs(folder_path_img, exist_ok=True)
        os.makedirs(folder_path_seq, exist_ok=True)

        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(
            _g.round(_g.random.rand() * (samp_num - Nfast - 1)))
            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]
            echo = sp.copy()

            index1 = int(_g.round(_g.random.rand() * (len(K_AM_all) - 1)))
            K_AM = K_AM_all[index1]

            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            sp0 = sp1.copy()

            sp1_fft = _g.fft.fftshift(_g.fft.fft(sp1))
            range_val = 100 + int(_g.round(_g.random.rand() * 200))
            Bj = 10 + int(_g.round(_g.random.rand() * 30))
            # Original MATLAB: range to samp_num*((1-Bj/(fs/10e5)/2))
            # kept as sideband zeroing
            low = int(_g.round(samp_num * ((1 - Bj / (fs / 10e5) / 2))))
            high = int(_g.round(samp_num * ((1 - Bj / (fs / 10e5) / 2)
            ) + _g.round(Bj / (fs / 10e5) * samp_num)))
            if low > range_val and low < samp_num:
                sp1_fft[range_val: low] = 0
                if high > 0 and high <= samp_num:
                    sp1_fft[high: samp_num] = 0
            sp1 = _g.fft.ifft(_g.fft.ifftshift(sp1_fft))

            J = echo + K_AM * Aj * sp1 + sp0
            J = J / _g.max(_g.abs(J))

            # Plot + save spectrogram
            fname = f'{a}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            # Save sequence
            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(
    os.path.join(
    folder_path_seq, f'{a}.mat'), {
    'iq_slice': iq_slice})


def COMB(data_num=None):
    """
    梳状谱干扰 COMB
    Converted from COMB.m
    """
    if data_num is None:
            data_num = 200
    fs = 100e6
    Ts = 1 / fs
    PRI = 100e-6
    fc = _g.random.uniform(5e6, 7e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * Ts
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    root_folder_img = _TEST_IMG_ROOT
    root_folder_seq = _TEST_SEQ_ROOT

    for SNR in tqdm(
    range(
    _SNR_CONFIG['start'],
    _SNR_CONFIG['stop'],
    _SNR_CONFIG['step']),
    desc="SNR",
    leave=False):
        folder_path_img = os.path.join(root_folder_img, f'{SNR}_dB', 'COMB')
        folder_path_seq = os.path.join(root_folder_seq, f'{SNR}_dB', 'COMB')
        os.makedirs(folder_path_img, exist_ok=True)
        os.makedirs(folder_path_seq, exist_ok=True)

        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            comb_num_opts = [9, 10, 11, 12]
            comb_altitude_opts = [0.5, 0.55, 0.6]
            comb_fi_k_opts = [0.05, 0.06, 0.08]
            comb_altitude_k_opts = [0.5, 0.6, 0.7]

            M = comb_num_opts[int(
            _g.round(_g.random.rand() * (len(comb_num_opts) - 1)))]
            N_alt = comb_altitude_opts[int(
            _g.round(_g.random.rand() * (len(comb_altitude_opts) - 1)))]
            Q = comb_fi_k_opts[int(
            _g.round(_g.random.rand() * (len(comb_fi_k_opts) - 1)))]
            P = comb_altitude_k_opts[int(
            _g.round(_g.random.rand() * (len(comb_altitude_k_opts) - 1)))]

            comb = _g.zeros(samp_num, dtype=complex)
            for i in range(1, M + 1):
                ki = P
                fi = fc + i * Q * fc
                comb_fragment = _g.exp(1j * 2 * _g.pi * fi * t)
                comb += ki * \
                _g.concatenate(
                [comb_fragment, _g.zeros(Nr - Nfast, dtype=complex)])

            As = 10 ** (SNR / 20)
            J0 = As * lfm * comb

            Aj = 10 ** ((SNR + JSR) / 20)
            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(
            _g.round(_g.random.rand() * (samp_num - Nfast - 1)))
            sp[range_tar: range_tar + Nfast] += J0[:Nfast]
            J1 = sp.copy()

            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            J = sp1 + Aj * J1
            J = J / _g.max(_g.abs(J))

            fname = f'{a}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(
    os.path.join(
    folder_path_seq, f'{a}.mat'), {
    'iq_slice': iq_slice})


def FM(data_num=None):
    """
    调频干扰 FM
    Converted from FM.m
    """
    if data_num is None:
            data_num = 200
    fs = 100e6
    Ts = 1 / fs
    PRI = 100e-6
    fc = _g.random.uniform(10e6, 12e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * Ts
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    root_folder_img = _TEST_IMG_ROOT
    root_folder_seq = _TEST_SEQ_ROOT

    for SNR in tqdm(
    range(
    _SNR_CONFIG['start'],
    _SNR_CONFIG['stop'],
    _SNR_CONFIG['step']),
    desc="SNR",
    leave=False):
        folder_path_img = os.path.join(root_folder_img, f'{SNR}_dB', 'FM')
        folder_path_seq = os.path.join(root_folder_seq, f'{SNR}_dB', 'FM')
        os.makedirs(folder_path_img, exist_ok=True)
        os.makedirs(folder_path_seq, exist_ok=True)

        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(
            _g.round(_g.random.rand() * (samp_num - Nfast - 1)))
            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]
            echo = sp.copy()

            # FM parameters
            B_n_all = [6e6, 8e6, 10e6]
            K_FM_all = [1.2, 1.4, 1.6, 1.8]
            m_fe_all = [2, 4, 6, 8]
            A_all = [1, 1.4, 1.7, 2]

            B_n = B_n_all[int(_g.round(_g.random.rand() * (len(B_n_all) - 1)))]
            K_FM = K_FM_all[int(
            _g.round(_g.random.rand() * (len(K_FM_all) - 1)))]
            m_fe = m_fe_all[int(
            _g.round(_g.random.rand() * (len(m_fe_all) - 1)))]
            A = A_all[int(_g.round(_g.random.rand() * (len(A_all) - 1)))]

            if m_fe > 0.75:
                delta_F_n = B_n / (2.35 * m_fe)
                sigma_n = B_n / (2.35 * K_FM)
            else:
                delta_F_n = B_n / (_g.pi * m_fe ** 2)
                sigma_n = B_n / (_g.pi * m_fe * K_FM)

            L = Nr
            Nn = int(_g.round(L * (delta_F_n / fs)))
            x1 = _g.random.randn(1, L)
            x2 = _g.random.randn(1, L)
            if Nn < L - Nn:
                x1[0, Nn: L - Nn] = 0
                x2[0, Nn: L - Nn] = 0
            S = x1 + 1j * x2
            s = _g.fft.ifft(S.squeeze())
            x1Mean = _g.mean(_g.real(s))
            x2Mean = _g.mean(_g.imag(s))
            variance = (_g.sum((_g.real(s) - x1Mean) ** 2) / L +
            _g.sum((_g.imag(s) - x2Mean) ** 2) / L)
            s = s / _g.sqrt(variance + 1e-12)
            s = s * sigma_n

            s = _g.real(s)
            downTriangle = _g.tril(_g.ones((L, L)))
            mathIntegral = (1 / fs) * (downTriangle @ s)
            s_n = A * _g.exp(1j * _g.pi * K_FM * mathIntegral)
            J_sig = s_n * echo

            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            J = sp1 + Aj * J_sig
            J = J / _g.max(_g.abs(J))

            fname = f'{a}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(
    os.path.join(
    folder_path_seq, f'{a}.mat'), {
    'iq_slice': iq_slice})


def ISRJ(data_num=None):
    """
    间歇采样转发干扰 ISRJ
    Converted from ISRJ.m
    """
    if data_num is None:
            data_num = 200
    fs = 100e6
    Ts = 1 / fs
    PRI = 100e-6
    fc = _g.random.uniform(7e6, 9e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5
    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * Ts
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    root_folder_img = _TEST_IMG_ROOT
    root_folder_seq = _TEST_SEQ_ROOT

    for SNR in tqdm(
    range(
    _SNR_CONFIG['start'],
    _SNR_CONFIG['stop'],
    _SNR_CONFIG['step']),
    desc="SNR",
    leave=False):
        folder_path_img = os.path.join(root_folder_img, f'{SNR}_dB', 'ISRJ')
        folder_path_seq = os.path.join(root_folder_seq, f'{SNR}_dB', 'ISRJ')
        os.makedirs(folder_path_img, exist_ok=True)
        os.makedirs(folder_path_seq, exist_ok=True)

        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)
            sp = _g.zeros(samp_num, dtype=complex)
            # MATLAB: range_tar = 1 + round(rand * (samp_num - Nfast - 2000 - 1))
            # Fixed: add -2000 to ensure enough space for ISRJ signal
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 2000 - 1)))
            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]
            echo = sp.copy()

            repetion_times = [4, 2, 1]
            period = [4e-6, 5e-6, 10e-6]
            duty = [20, 25, 33.33, 50]

            index1 = int(_g.round(_g.random.rand() * (len(period) - 1)))
            index2 = int(_g.round(_g.random.rand() * (len(repetion_times) - 1)))

            period1 = period[index1]
            duty1 = duty[index2]
            repetion_times1 = repetion_times[index2]

            m = int(_g.round(taup / period1))
            n = repetion_times1
            len_ = Nfast // (m * n) if (m * n) != 0 else 1
            len_m = len_ * n
            delay_time = period1 * duty1 * 0.01
            delay_num = int(_g.ceil(delay_time / (1 / fs)))
            len_t = delay_num

            J = _g.zeros(samp_num, dtype=complex)
            for i in range(m):
                temp = echo[range_tar + i * len_m: range_tar + i * len_m + len_]
                for j in range(n):
                    start_idx = range_tar + i * len_m + j * len_ + i * len_t
                    end_idx = start_idx + len_
                    if end_idx <= samp_num:
                        J[start_idx: end_idx] = Aj * temp * 2.0

            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            # MATLAB: sp1(1+range_tar:1+range_tar +n*len+(m-1)*len_m-1+(m-1)*len_t)
            # Fixed: remove extra -1 in add_len calculation
            add_len = range_tar + n * len_ + (m - 1) * len_m + (m - 1) * len_t
            if add_len > range_tar and add_len <= samp_num:
                sp1[range_tar: add_len] += J[range_tar: add_len]
            J = sp1

            J = J / _g.max(_g.abs(J))

            fname = f'{a}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{a}.mat'), {'iq_slice': iq_slice})


def LFM(data_num=None):
    """
    雷达LFM线性调频信号
    Converted from LFM.m
    """
    if data_num is None:
        data_num = 200
    fs = 100e6
    Ts = 1 / fs
    PRI = 100e-6
    fc = _g.random.uniform(8e6, 12e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * Ts
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    root_folder_img = _TEST_IMG_ROOT
    root_folder_seq = _TEST_SEQ_ROOT

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        folder_path_img = os.path.join(root_folder_img, f'{SNR}_dB', 'LFM')
        folder_path_seq = os.path.join(root_folder_seq, f'{SNR}_dB', 'LFM')
        os.makedirs(folder_path_img, exist_ok=True)
        os.makedirs(folder_path_seq, exist_ok=True)

        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            sp = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp = sp / _g.std(sp)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 2000 - 1)))
            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]
            lfm_echo = sp / _g.max(_g.abs(sp))

            fname = f'{a}.png'
            lfm_echo_cpu = _to_cpu(lfm_echo)
            _save_spectrogram_image(lfm_echo_cpu, fs, folder_path_img, fname)

            iq_slice = lfm_echo_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{a}.mat'), {'iq_slice': iq_slice})


def LFM_alldb(data_num=None):
    """
    雷达LFM线性调频信号 — 全部dB合一文件夹
    Converted from LFM_alldb.m (follows the same structure as other _alldb functions)
    """
    if data_num is None:
            data_num = 800  # Number of samples per SNR, consistent with other _alldb functions
    fs = 100e6
    Ts = 1 / fs
    PRI = 100e-6
    fc = _g.random.uniform(8e6, 12e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * Ts
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    folder_path_img = os.path.join(_TRAIN_IMG_ROOT, 'LFM')
    folder_path_seq = os.path.join(_TRAIN_SEQ_ROOT, 'LFM')
    os.makedirs(folder_path_img, exist_ok=True)
    os.makedirs(folder_path_seq, exist_ok=True)

    index = 1

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            # Target echo + noise
            sp = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp = sp / _g.std(sp)
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 2000 - 1)))

            # Regenerate LFM pulse for each sample to match MATLAB behavior
            lfm_pulse_local = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
            lfm_local = _g.concatenate([lfm_pulse_local, _g.zeros(Nr - Nfast, dtype=complex)])

            sp[range_tar: range_tar + Nfast] += As * lfm_local[:Nfast]
            lfm_echo = sp / _g.max(_g.abs(sp))

            fname = f'{index}.png'
            lfm_echo_cpu = _to_cpu(lfm_echo)
            _save_spectrogram_image(lfm_echo_cpu, fs, folder_path_img, fname)

            iq_slice = lfm_echo_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{index}.mat'), {'iq_slice': iq_slice})

            index += 1


def MNJ(data_num=None):
    """
    噪声乘积式灵巧噪声干扰 MNJ
    """
    if data_num is None:
            data_num = 200
    fs = 100e6
    PRI = 100e-6
    fc = _g.random.uniform(9.5e6, 10.5e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * (1 / fs)
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    root_folder_img = _TEST_IMG_ROOT
    root_folder_seq = _TEST_SEQ_ROOT

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        folder_path_img = os.path.join(root_folder_img, f'{SNR}_dB', 'MNJ')
        folder_path_seq = os.path.join(root_folder_seq, f'{SNR}_dB', 'MNJ')
        os.makedirs(folder_path_img, exist_ok=True)
        os.makedirs(folder_path_seq, exist_ok=True)

        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 2000 - 1)))
            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]
            echo = sp.copy()

            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            echo_with_noise = sp1 + Aj * echo * 2

            L = Nr
            wgn_noise = _g.random.randn(L)
            bw_filter = [5e6, 7e6, 9e6, 11e6]
            bw = bw_filter[int(_g.round(_g.random.rand() * (len(bw_filter) - 1)))]
            f0 = float(_to_cpu(fc))  # Python float for scipy (CuPy scalar not allowed)
            # Butterworth bandpass in Python (scipy requires numpy arrays)
            # MATLAB uses 10th order Butterworth filter
            import scipy.signal as ss
            wgn_noise_cpu = _to_cpu(wgn_noise)
            b, a_design = ss.butter(10, [f0 - bw / 2, f0 + bw / 2], btype='band', fs=fs)
            narrowband_noise = ss.filtfilt(b, a_design, wgn_noise_cpu)
            analytic_narrowband_noise = ss.hilbert(narrowband_noise)
            J = _g.asarray(analytic_narrowband_noise) * echo_with_noise

            J = J / _g.max(_g.abs(J))

            fname = f'{a}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{a}.mat'), {'iq_slice': iq_slice})


def RMT(data_num=None):
    """
    距离维密集假目标干扰 RMT
    """
    if data_num is None:
            data_num = 200
    fs = 100e6
    PRI = 100e-6
    fc = _g.random.uniform(11e6, 13e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * (1 / fs)
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    root_folder_img = _TEST_IMG_ROOT
    root_folder_seq = _TEST_SEQ_ROOT

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        folder_path_img = os.path.join(root_folder_img, f'{SNR}_dB', 'RMT')
        folder_path_seq = os.path.join(root_folder_seq, f'{SNR}_dB', 'RMT')
        os.makedirs(folder_path_img, exist_ok=True)
        os.makedirs(folder_path_seq, exist_ok=True)

        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_val = 200 + int(_g.round(_g.random.rand() * 1800))
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 1 - range_val - 2000)))
            k_val = 2 + int(_g.round(_g.random.rand() * 3))

            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]
            for i in range(k_val):
                delay_time_num = 100 + int(_g.round(_g.random.rand() * 400))
                start_idx = range_tar + range_val + i * delay_time_num
                end_idx = start_idx + Nfast
                if end_idx <= samp_num:
                    sp[start_idx: end_idx] += Aj * lfm[:Nfast]

            J = sp.copy()
            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            J = sp1 + J
            J = J / _g.max(_g.abs(J))

            fname = f'{a}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{a}.mat'), {'iq_slice': iq_slice})


def RGPO(data_num=None):
    """
    距离拖引干扰 RGPO
    """
    if data_num is None:
            data_num = 200
    fs = 100e6
    PRI = 100e-6
    fc = _g.random.uniform(8e6, 10e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * (1 / fs)
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    root_folder_img = _TEST_IMG_ROOT
    root_folder_seq = _TEST_SEQ_ROOT

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        folder_path_img = os.path.join(root_folder_img, f'{SNR}_dB', 'RGPO')
        folder_path_seq = os.path.join(root_folder_seq, f'{SNR}_dB', 'RGPO')
        os.makedirs(folder_path_img, exist_ok=True)
        os.makedirs(folder_path_seq, exist_ok=True)

        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 1 - 4000)))
            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]

            range_gate_start = 500
            pull_off_rate = 100 + int(_g.round(_g.random.rand() * 400))
            K = 1 + int(_g.round(_g.random.rand() * 5))
            range_jamming = range_gate_start
            for _ in range(K):
                range_jamming += pull_off_rate

            start_idx = range_tar + range_jamming
            end_idx = start_idx + Nfast
            if end_idx <= samp_num:
                sp[start_idx: end_idx] += Aj * lfm[:Nfast]
            J = sp.copy()

            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            J = sp1 + J
            J = J / _g.max(_g.abs(J))

            fname = f'{a}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{a}.mat'), {'iq_slice': iq_slice})


def R_VGPO(data_num=None):
    """
    距离速度联合拖引干扰 R_VGPO
    """
    if data_num is None:
            data_num = 200
    fs = 100e6
    Ts = 1 / fs
    PRI = 100e-6
    fc = _g.random.uniform(6e6, 8e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * Ts
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    root_folder_img = _TEST_IMG_ROOT
    root_folder_seq = _TEST_SEQ_ROOT

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        folder_path_img = os.path.join(root_folder_img, f'{SNR}_dB', 'R_VGPO')
        folder_path_seq = os.path.join(root_folder_seq, f'{SNR}_dB', 'R_VGPO')
        os.makedirs(folder_path_img, exist_ok=True)
        os.makedirs(folder_path_seq, exist_ok=True)

        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 1 - 4000)))
            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]

            range_gate_start = 500
            pull_off_rate = 100 + int(_g.round(_g.random.rand() * 400))
            v_drag = 100 + int(_g.round(_g.random.rand() * 200))
            a_drag = 50 + int(_g.round(_g.random.rand() * 450))
            c = 3e8
            pri_num = 1 + int(_g.round(_g.random.rand() * 6))

            range_jamming = range_gate_start + pri_num * pull_off_rate
            v_drag_current = v_drag + a_drag * pri_num * PRI
            fd = 2 * v_drag_current / c * fc * 1e6
            vgpo_interference = As * lfm[:Nfast] * _g.exp(1j * 2 * _g.pi * fd * _g.arange(Nfast) * Ts)

            start_idx = range_tar + range_jamming
            end_idx = start_idx + Nfast
            if end_idx <= samp_num:
                sp[start_idx: end_idx] += Aj * vgpo_interference

            J = sp.copy()
            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            J = sp1 + J
            J = J / _g.max(_g.abs(J))

            fname = f'{a}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{a}.mat'), {'iq_slice': iq_slice})


def SMSP(data_num=None):
    """
    频谱弥散干扰 SMSP
    """
    if data_num is None:
            data_num = 200
    fs = 100e6
    Ts = 1 / fs
    PRI = 100e-6
    fc = _g.random.uniform(8e6, 12e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * Ts
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    root_folder_img = _TEST_IMG_ROOT
    root_folder_seq = _TEST_SEQ_ROOT

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        folder_path_img = os.path.join(root_folder_img, f'{SNR}_dB', 'SMSP')
        folder_path_seq = os.path.join(root_folder_seq, f'{SNR}_dB', 'SMSP')
        os.makedirs(folder_path_img, exist_ok=True)
        os.makedirs(folder_path_seq, exist_ok=True)

        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 2000 - 1)))

            sample_times = [3, 4, 5]
            N = sample_times[int(_g.round(_g.random.rand() * (len(sample_times) - 1)))]
            t_smsp = (1 / N) * t[::N]
            k_smsp = k * N
            lfm_smsp_pulse = _g.exp(1j * (2 * _g.pi * fc * t_smsp + _g.pi * k_smsp * (t_smsp ** 2)))
            lfm_smsp = _g.concatenate([lfm_smsp_pulse, _g.zeros(Nr - len(lfm_smsp_pulse), dtype=complex)])

            H = _g.zeros(samp_num, dtype=complex)
            for i in range(1, N + 1):
                td = (i - 1) * len(lfm_smsp_pulse) + 1
                if td - 1 < len(H):
                    H[td - 1] = 1

            sp[range_tar: range_tar + Nfast] += As * lfm_smsp[:Nfast]
            echo = sp.copy()

            J = _g.fft.ifft(_g.fft.fft(H) * _g.fft.fft(echo))

            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            J = sp1 + Aj * J * 3
            J = J / _g.max(_g.abs(J))

            fname = f'{a}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{a}.mat'), {'iq_slice': iq_slice})


def VGPO(data_num=None):
    """
    速度拖引干扰 VGPO
    """
    if data_num is None:
            data_num = 200
    fs = 100e6
    Ts = 1 / fs
    PRI = 100e-6
    fc = _g.random.uniform(13e6, 15e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * Ts
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    root_folder_img = _TEST_IMG_ROOT
    root_folder_seq = _TEST_SEQ_ROOT

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        folder_path_img = os.path.join(root_folder_img, f'{SNR}_dB', 'VGPO')
        folder_path_seq = os.path.join(root_folder_seq, f'{SNR}_dB', 'VGPO')
        os.makedirs(folder_path_img, exist_ok=True)
        os.makedirs(folder_path_seq, exist_ok=True)

        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 2000 - 1)))
            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]

            v_drag = 100 + int(_g.round(_g.random.rand() * 200))
            a_drag = 50 + int(_g.round(_g.random.rand() * 450))
            c = 3e8
            pri_num = int(_g.round(_g.random.rand() * 10))
            v_drag_current = v_drag + a_drag * pri_num * PRI
            fd = 2 * v_drag_current / c * fc * 1e6
            vgpo_interference = As * lfm[:Nfast] * _g.exp(1j * 2 * _g.pi * fd * _g.arange(Nfast) * Ts)

            sp[range_tar: range_tar + Nfast] += Aj * vgpo_interference
            J = sp.copy()

            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            J = sp1 + J
            J = J / _g.max(_g.abs(J))

            fname = f'{a}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{a}.mat'), {'iq_slice': iq_slice})


def VMT(data_num=None):
    """
    速度维密集假目标干扰 VMT
    """
    if data_num is None:
            data_num = 200
    fs = 100e6
    Ts = 1 / fs
    PRI = 100e-6
    fc = _g.random.uniform(5e6, 7e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * Ts
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    root_folder_img = _TEST_IMG_ROOT
    root_folder_seq = _TEST_SEQ_ROOT

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        folder_path_img = os.path.join(root_folder_img, f'{SNR}_dB', 'VMT')
        folder_path_seq = os.path.join(root_folder_seq, f'{SNR}_dB', 'VMT')
        os.makedirs(folder_path_img, exist_ok=True)
        os.makedirs(folder_path_seq, exist_ok=True)

        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 2000 - 1)))

            k_val = 2 + int(_g.round(_g.random.rand() * 2))
            fd = (2 + int(_g.round(_g.random.rand() * 13))) * 1e6

            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]

            for i in range(k_val):
                fd_mt = (1 + int(_g.round(_g.random.rand() * 4))) * 1e6
                vgpo_interference = As * lfm[:Nfast] * _g.exp(1j * 2 * _g.pi * (fd + i * fd_mt) * _g.arange(Nfast) * Ts)
                sp[range_tar: range_tar + Nfast] += Aj * vgpo_interference

            J = sp.copy()
            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            J = sp1 + J
            J = J / _g.max(_g.abs(J))

            fname = f'{a}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{a}.mat'), {'iq_slice': iq_slice})


# ---------------------------------------------------------------------------
#  Train set (alldb) functions (data_num=800, single All_dB folder)
# ---------------------------------------------------------------------------

def AM_alldb(data_num=None):
    """
    调幅干扰 AM (瞄准式) — 全部dB合一文件夹
    Converted from AM_alldb.m
    """
    if data_num is None:
            data_num = 800
    fs = 100e6
    Ts = 1 / fs
    PRI = 100e-6
    fc = _g.random.uniform(12e6, 14e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * Ts
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    folder_path_img = os.path.join(_TRAIN_IMG_ROOT, 'AM')
    folder_path_seq = os.path.join(_TRAIN_SEQ_ROOT, 'AM')
    os.makedirs(folder_path_img, exist_ok=True)
    os.makedirs(folder_path_seq, exist_ok=True)

    K_AM_all = [2, 3, 4]
    index = 1

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 2000 - 1)))
            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]
            echo = sp.copy()

            index1 = int(_g.round(_g.random.rand() * (len(K_AM_all) - 1)))
            K_AM = K_AM_all[index1]

            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            sp0 = sp1.copy()

            sp1_fft = _g.fft.fftshift(_g.fft.fft(sp1))
            range_val = 100 + int(_g.round(_g.random.rand() * 200))
            Bj = 10 + int(_g.round(_g.random.rand() * 30))
            low = int(_g.round(samp_num * ((1 - Bj / (fs / 10e5) / 2))))
            high = int(_g.round(samp_num * ((1 - Bj / (fs / 10e5) / 2)) + _g.round(Bj / (fs / 10e5) * samp_num)))
            if low > range_val and low < samp_num:
                sp1_fft[range_val: low] = 0
                if high > 0 and high <= samp_num:
                    sp1_fft[high: samp_num] = 0
            sp1 = _g.fft.ifft(_g.fft.ifftshift(sp1_fft))

            J = echo + K_AM * Aj * sp1 + sp0
            J = J / _g.max(_g.abs(J))

            fname = f'{index}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{index}.mat'), {'iq_slice': iq_slice})

            index += 1


def COMB_alldb(data_num=None):
    """
    梳状谱干扰 COMB — 全部dB合一文件夹
    """
    if data_num is None:
            data_num = 800
    fs = 100e6
    PRI = 100e-6
    fc = _g.random.uniform(5e6, 7e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * (1 / fs)
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    folder_path_img = os.path.join(_TRAIN_IMG_ROOT, 'COMB')
    folder_path_seq = os.path.join(_TRAIN_SEQ_ROOT, 'COMB')
    os.makedirs(folder_path_img, exist_ok=True)
    os.makedirs(folder_path_seq, exist_ok=True)

    index = 1

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            comb_num_opts = [9, 10, 11, 12]
            comb_alt_opts = [0.5, 0.55, 0.6]
            comb_fi_k_opts = [0.05, 0.06, 0.08]
            comb_alt_k_opts = [0.5, 0.6, 0.7]

            M = comb_num_opts[int(_g.round(_g.random.rand() * (len(comb_num_opts) - 1)))]
            N_alt = comb_alt_opts[int(_g.round(_g.random.rand() * (len(comb_alt_opts) - 1)))]
            Q = comb_fi_k_opts[int(_g.round(_g.random.rand() * (len(comb_fi_k_opts) - 1)))]
            P = comb_alt_k_opts[int(_g.round(_g.random.rand() * (len(comb_alt_k_opts) - 1)))]

            comb = _g.zeros(samp_num, dtype=complex)
            for i in range(1, M + 1):
                ki = P
                fi = fc + i * Q * fc
                comb_fragment = _g.exp(1j * 2 * _g.pi * fi * t)
                comb += ki * _g.concatenate([comb_fragment, _g.zeros(Nr - Nfast, dtype=complex)])

            As = 10 ** (SNR / 20)
            J0 = As * lfm * comb

            Aj = 10 ** ((SNR + JSR) / 20)
            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 2000 - 1)))
            sp[range_tar: range_tar + Nfast] += J0[:Nfast]
            J1 = sp.copy()

            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            J = sp1 + Aj * J1
            J = J / _g.max(_g.abs(J))

            fname = f'{index}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{index}.mat'), {'iq_slice': iq_slice})

            index += 1


def FM_alldb(data_num=None):
    """
    调频干扰 FM — 全部dB合一文件夹
    """
    if data_num is None:
            data_num = 800
    fs = 100e6
    PRI = 100e-6
    fc = _g.random.uniform(10e6, 12e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * (1 / fs)
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    folder_path_img = os.path.join(_TRAIN_IMG_ROOT, 'FM')
    folder_path_seq = os.path.join(_TRAIN_SEQ_ROOT, 'FM')
    os.makedirs(folder_path_img, exist_ok=True)
    os.makedirs(folder_path_seq, exist_ok=True)

    index = 1

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 2000 - 1)))
            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]
            echo = sp.copy()

            B_n_all = [6e6, 8e6, 10e6]
            K_FM_all = [1.2, 1.4, 1.6, 1.8]
            m_fe_all = [2, 4, 6, 8]
            A_all = [1, 1.4, 1.7, 2]

            B_n = B_n_all[int(_g.round(_g.random.rand() * (len(B_n_all) - 1)))]
            K_FM = K_FM_all[int(_g.round(_g.random.rand() * (len(K_FM_all) - 1)))]
            m_fe = m_fe_all[int(_g.round(_g.random.rand() * (len(m_fe_all) - 1)))]
            A = A_all[int(_g.round(_g.random.rand() * (len(A_all) - 1)))]

            if m_fe > 0.75:
                delta_F_n = B_n / (2.35 * m_fe)
                sigma_n = B_n / (2.35 * K_FM)
            else:
                delta_F_n = B_n / (_g.pi * m_fe ** 2)
                sigma_n = B_n / (_g.pi * m_fe * K_FM)

            L = Nr
            Nn = int(_g.round(L * (delta_F_n / fs)))
            x1 = _g.random.randn(1, L)
            x2 = _g.random.randn(1, L)
            if Nn < L - Nn:
                x1[0, Nn: L - Nn] = 0
                x2[0, Nn: L - Nn] = 0
            S = x1 + 1j * x2
            s = _g.fft.ifft(S.squeeze())
            x1Mean = _g.mean(_g.real(s))
            x2Mean = _g.mean(_g.imag(s))
            variance = (_g.sum((_g.real(s) - x1Mean) ** 2) / L +
            _g.sum((_g.imag(s) - x2Mean) ** 2) / L)
            s = s / _g.sqrt(variance + 1e-12)
            s = s * sigma_n

            s = _g.real(s)
            downTriangle = _g.tril(_g.ones((L, L)))
            mathIntegral = (1 / fs) * (downTriangle @ s)
            s_n = A * _g.exp(1j * _g.pi * K_FM * mathIntegral)
            J_sig = s_n * echo

            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            J = sp1 + Aj * J_sig
            J = J / _g.max(_g.abs(J))

            fname = f'{index}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{index}.mat'), {'iq_slice': iq_slice})

            index += 1


def ISRJ_alldb(data_num=None):
    """
    间歇采样转发干扰 ISRJ — 全部dB合一文件夹
    """
    if data_num is None:
            data_num = 800
    fs = 100e6
    PRI = 100e-6
    fc = _g.random.uniform(7e6, 9e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * (1 / fs)
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    folder_path_img = os.path.join(_TRAIN_IMG_ROOT, 'ISRJ')
    folder_path_seq = os.path.join(_TRAIN_SEQ_ROOT, 'ISRJ')
    os.makedirs(folder_path_img, exist_ok=True)
    os.makedirs(folder_path_seq, exist_ok=True)

    index = 1

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            # MATLAB: range_tar = 1 + round(rand * (samp_num - Nfast - 2000 - 1))
            # Fixed: add -2000 to ensure enough space for ISRJ signal
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 2000 - 1)))
            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]
            echo = sp.copy()

            repetion_times = [4, 2, 1]
            period = [4e-6, 5e-6, 10e-6]
            duty = [20, 25, 33.33, 50]

            index1 = int(_g.round(_g.random.rand() * (len(period) - 1)))
            index2 = int(_g.round(_g.random.rand() * (len(repetion_times) - 1)))

            period1 = period[index1]
            duty1 = duty[index2]
            repetion_times1 = repetion_times[index2]

            m = int(_g.round(taup / period1))
            n = repetion_times1
            len_ = Nfast // (m * n) if (m * n) != 0 else 1
            len_m = len_ * n
            delay_time = period1 * duty1 * 0.01
            delay_num = int(_g.ceil(delay_time / (1 / fs)))
            len_t = delay_num

            J = _g.zeros(samp_num, dtype=complex)
            for i in range(m):
                temp = echo[range_tar + i * len_m: range_tar + i * len_m + len_]
                for j in range(n):
                    start_idx = range_tar + i * len_m + j * len_ + i * len_t
                end_idx = start_idx + len_
                if end_idx <= samp_num:
                    J[start_idx: end_idx] = Aj * temp * 2.0

            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            # MATLAB: sp1(1+range_tar:1+range_tar +n*len+(m-1)*len_m-1+(m-1)*len_t)
            # Fixed: remove extra -1 in add_len calculation
            add_len = range_tar + n * len_ + (m - 1) * len_m + (m - 1) * len_t
            if add_len > range_tar and add_len <= samp_num:
                sp1[range_tar: add_len] += J[range_tar: add_len]
            J = sp1
            J = J / _g.max(_g.abs(J))

            fname = f'{index}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{index}.mat'), {'iq_slice': iq_slice})

            index += 1


def MNJ_alldb(data_num=None):
    """
    噪声乘积式灵巧噪声干扰 MNJ — 全部dB合一文件夹
    """
    if data_num is None:
            data_num = 800
    fs = 100e6
    PRI = 100e-6
    fc = _g.random.uniform(9.5e6, 10.5e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * (1 / fs)
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    folder_path_img = os.path.join(_TRAIN_IMG_ROOT, 'MNJ')
    folder_path_seq = os.path.join(_TRAIN_SEQ_ROOT, 'MNJ')
    os.makedirs(folder_path_img, exist_ok=True)
    os.makedirs(folder_path_seq, exist_ok=True)

    import scipy.signal as ss
    index = 1

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 2000 - 1)))
            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]
            echo = sp.copy()

            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            echo_with_noise = sp1 + Aj * echo * 2

            L = Nr
            wgn_noise = _g.random.randn(L)
            bw_filter = [5e6, 7e6, 9e6, 11e6]
            bw = bw_filter[int(_g.round(_g.random.rand() * (len(bw_filter) - 1)))]
            f0 = float(_to_cpu(fc))  # Python float for scipy (CuPy scalar not allowed)
            # scipy requires numpy arrays (not CuPy)
            # MATLAB uses 10th order Butterworth filter
            wgn_noise_cpu = _to_cpu(wgn_noise)
            b, a_design = ss.butter(10, [f0 - bw / 2, f0 + bw / 2], btype='band', fs=fs)
            narrowband_noise = ss.filtfilt(b, a_design, wgn_noise_cpu)
            analytic_narrowband_noise = ss.hilbert(narrowband_noise)
            J = _g.asarray(analytic_narrowband_noise) * echo_with_noise

            J = J / _g.max(_g.abs(J))

            fname = f'{index}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{index}.mat'), {'iq_slice': iq_slice})

            index += 1


def RMT_alldb(data_num=None):
    """
    距离维密集假目标干扰 RMT — 全部dB合一文件夹
    """
    if data_num is None:
            data_num = 800
    fs = 100e6
    PRI = 100e-6
    fc = _g.random.uniform(11e6, 13e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * (1 / fs)
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    folder_path_img = os.path.join(_TRAIN_IMG_ROOT, 'RMT')
    folder_path_seq = os.path.join(_TRAIN_SEQ_ROOT, 'RMT')
    os.makedirs(folder_path_img, exist_ok=True)
    os.makedirs(folder_path_seq, exist_ok=True)

    index = 1

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_val = 200 + int(_g.round(_g.random.rand() * 1800))
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 1 - range_val - 2000)))
            k_val = 2 + int(_g.round(_g.random.rand() * 3))

            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]
            for i in range(k_val):
                delay_time_num = 100 + int(_g.round(_g.random.rand() * 400))
                start_idx = range_tar + range_val + i * delay_time_num
                end_idx = start_idx + Nfast
                if end_idx <= samp_num:
                    sp[start_idx: end_idx] += Aj * lfm[:Nfast]

            J = sp.copy()
            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            J = sp1 + J
            J = J / _g.max(_g.abs(J))

            fname = f'{index}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{index}.mat'), {'iq_slice': iq_slice})

            index += 1


def RGPO_alldb(data_num=None):
    """
    距离拖引干扰 RGPO — 全部dB合一文件夹
    """
    if data_num is None:
            data_num = 800
    fs = 100e6
    PRI = 100e-6
    fc = _g.random.uniform(8e6, 10e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * (1 / fs)
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    folder_path_img = os.path.join(_TRAIN_IMG_ROOT, 'RGPO')
    folder_path_seq = os.path.join(_TRAIN_SEQ_ROOT, 'RGPO')
    os.makedirs(folder_path_img, exist_ok=True)
    os.makedirs(folder_path_seq, exist_ok=True)

    index = 1

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 1 - 4000)))
            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]

            range_gate_start = 500
            pull_off_rate = 100 + int(_g.round(_g.random.rand() * 400))
            K = 1 + int(_g.round(_g.random.rand() * 5))
            range_jamming = range_gate_start
            for _ in range(K):
                range_jamming += pull_off_rate

            start_idx = range_tar + range_jamming
            end_idx = start_idx + Nfast
            if end_idx <= samp_num:
                sp[start_idx: end_idx] += Aj * lfm[:Nfast]
            J = sp.copy()

            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            J = sp1 + J
            J = J / _g.max(_g.abs(J))

            fname = f'{index}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{index}.mat'), {'iq_slice': iq_slice})

            index += 1


def R_VGPO_alldb(data_num=None):
    """
    距离速度联合拖引干扰 R_VGPO — 全部dB合一文件夹
    """
    if data_num is None:
            data_num = 800
    fs = 100e6
    Ts = 1 / fs
    PRI = 100e-6
    fc = _g.random.uniform(6e6, 8e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * Ts
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    folder_path_img = os.path.join(_TRAIN_IMG_ROOT, 'R_VGPO')
    folder_path_seq = os.path.join(_TRAIN_SEQ_ROOT, 'R_VGPO')
    os.makedirs(folder_path_img, exist_ok=True)
    os.makedirs(folder_path_seq, exist_ok=True)

    index = 1

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 1 - 4000)))
            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]

            range_gate_start = 500
            pull_off_rate = 100 + int(_g.round(_g.random.rand() * 400))
            v_drag = 100 + int(_g.round(_g.random.rand() * 200))
            a_drag = 50 + int(_g.round(_g.random.rand() * 450))
            c = 3e8
            pri_num = 1 + int(_g.round(_g.random.rand() * 6))

            range_jamming = range_gate_start + pri_num * pull_off_rate
            v_drag_current = v_drag + a_drag * pri_num * PRI
            fd = 2 * v_drag_current / c * fc * 1e6
            vgpo_interference = As * lfm[:Nfast] * _g.exp(1j * 2 * _g.pi * fd * _g.arange(Nfast) * Ts)

            start_idx = range_tar + range_jamming
            end_idx = start_idx + Nfast
            if end_idx <= samp_num:
                sp[start_idx: end_idx] += Aj * vgpo_interference

            J = sp.copy()
            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            J = sp1 + J
            J = J / _g.max(_g.abs(J))

            fname = f'{index}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{index}.mat'), {'iq_slice': iq_slice})

            index += 1


def SMSP_alldb(data_num=None):
    """
    频谱弥散干扰 SMSP — 全部dB合一文件夹
    """
    if data_num is None:
            data_num = 800
    fs = 100e6
    Ts = 1 / fs
    PRI = 100e-6
    fc = _g.random.uniform(8e6, 12e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * Ts
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    folder_path_img = os.path.join(_TRAIN_IMG_ROOT, 'SMSP')
    folder_path_seq = os.path.join(_TRAIN_SEQ_ROOT, 'SMSP')
    os.makedirs(folder_path_img, exist_ok=True)
    os.makedirs(folder_path_seq, exist_ok=True)

    index = 1

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 2000 - 1)))

            sample_times = [3, 4, 5]
            N = sample_times[int(_g.round(_g.random.rand() * (len(sample_times) - 1)))]
            t_smsp = (1 / N) * t[::N]
            k_smsp = k * N
            lfm_smsp_pulse = _g.exp(1j * (2 * _g.pi * fc * t_smsp + _g.pi * k_smsp * (t_smsp ** 2)))
            lfm_smsp = _g.concatenate([lfm_smsp_pulse, _g.zeros(Nr - len(lfm_smsp_pulse), dtype=complex)])

            H = _g.zeros(samp_num, dtype=complex)
            for i in range(1, N + 1):
                td = (i - 1) * len(lfm_smsp_pulse) + 1
                if td - 1 < len(H):
                    H[td - 1] = 1

            sp[range_tar: range_tar + Nfast] += As * lfm_smsp[:Nfast]
            echo = sp.copy()

            J = _g.fft.ifft(_g.fft.fft(H) * _g.fft.fft(echo))

            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            J = sp1 + Aj * J * 3
            J = J / _g.max(_g.abs(J))

            fname = f'{index}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{index}.mat'), {'iq_slice': iq_slice})

            index += 1


def VGPO_alldb(data_num=None):
    """
    速度拖引干扰 VGPO — 全部dB合一文件夹
    """
    if data_num is None:
            data_num = 800
    fs = 100e6
    Ts = 1 / fs
    PRI = 100e-6
    fc = _g.random.uniform(13e6, 15e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * Ts
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    folder_path_img = os.path.join(_TRAIN_IMG_ROOT, 'VGPO')
    folder_path_seq = os.path.join(_TRAIN_SEQ_ROOT, 'VGPO')
    os.makedirs(folder_path_img, exist_ok=True)
    os.makedirs(folder_path_seq, exist_ok=True)

    index = 1

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 2000 - 1)))
            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]

            v_drag = 100 + int(_g.round(_g.random.rand() * 200))
            a_drag = 50 + int(_g.round(_g.random.rand() * 450))
            c = 3e8
            pri_num = int(_g.round(_g.random.rand() * 10))
            v_drag_current = v_drag + a_drag * pri_num * PRI
            fd = 2 * v_drag_current / c * fc * 1e6
            vgpo_interference = As * lfm[:Nfast] * _g.exp(1j * 2 * _g.pi * fd * _g.arange(Nfast) * Ts)

            sp[range_tar: range_tar + Nfast] += Aj * vgpo_interference
            J = sp.copy()

            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            J = sp1 + J
            J = J / _g.max(_g.abs(J))

            fname = f'{index}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{index}.mat'), {'iq_slice': iq_slice})

            index += 1


def VMT_alldb(data_num=None):
    """
    速度维密集假目标干扰 VMT — 全部dB合一文件夹
    """
    if data_num is None:
            data_num = 800
    fs = 100e6
    Ts = 1 / fs
    PRI = 100e-6
    fc = _g.random.uniform(5e6, 7e6)
    taup = 40e-6
    B = 40e6
    k = B / taup
    JSR = 5

    Nr = int(_g.round(fs * PRI))
    Nfast = int(_g.round(fs * taup))
    t = (_g.arange(-Nfast // 2, Nfast // 2)) * Ts
    lfm_pulse = _g.exp(1j * (2 * _g.pi * fc * t + _g.pi * k * (t ** 2)))
    lfm = _g.concatenate([lfm_pulse, _g.zeros(Nr - Nfast, dtype=complex)])
    samp_num = Nr

    folder_path_img = os.path.join(_TRAIN_IMG_ROOT, 'VMT')
    folder_path_seq = os.path.join(_TRAIN_SEQ_ROOT, 'VMT')
    os.makedirs(folder_path_img, exist_ok=True)
    os.makedirs(folder_path_seq, exist_ok=True)

    index = 1

    for SNR in tqdm(range(_SNR_CONFIG['start'], _SNR_CONFIG['stop'], _SNR_CONFIG['step']), desc="SNR", leave=False):
        for a in tqdm(range(1, data_num + 1), desc="Samples", leave=False):
            As = 10 ** (SNR / 20)
            Aj = 10 ** ((SNR + JSR) / 20)

            sp = _g.zeros(samp_num, dtype=complex)
            range_tar = int(_g.round(_g.random.rand() * (samp_num - Nfast - 2000 - 1)))

            k_val = 2 + int(_g.round(_g.random.rand() * 2))
            fd = (2 + int(_g.round(_g.random.rand() * 13))) * 1e6

            sp[range_tar: range_tar + Nfast] += As * lfm[:Nfast]

            for i in range(k_val):
                fd_mt = (1 + int(_g.round(_g.random.rand() * 4))) * 1e6
                vgpo_interference = As * lfm[:Nfast] * _g.exp(1j * 2 * _g.pi * (fd + i * fd_mt) * _g.arange(Nfast) * Ts)
                sp[range_tar: range_tar + Nfast] += Aj * vgpo_interference

            J = sp.copy()
            sp1 = _g.random.randn(samp_num) + 1j * _g.random.randn(samp_num)
            sp1 = sp1 / _g.std(sp1)
            J = sp1 + J
            J = J / _g.max(_g.abs(J))

            fname = f'{index}.png'
            J_cpu = _to_cpu(J)
            _save_spectrogram_image(J_cpu, fs, folder_path_img, fname)

            iq_slice = J_cpu[range_tar: range_tar + Nfast]
            savemat(os.path.join(folder_path_seq, f'{index}.mat'), {'iq_slice': iq_slice})

            index += 1
