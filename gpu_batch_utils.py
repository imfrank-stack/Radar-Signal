"""
GPU-optimized batch signal generation
Reduces CPU-GPU transfer overhead by processing multiple samples at once.
"""

import numpy as np
import cupy as cp
from tqdm import tqdm


def batch_generate_signals_gpu(signal_func, batch_size=32, total_samples=200, **kwargs):
    """
    Generate signals in batches on GPU to reduce transfer overhead.
    
    Args:
        signal_func: Signal generation function (must accept batch_size parameter)
        batch_size: Number of samples to generate per batch
        total_samples: Total number of samples to generate
        **kwargs: Additional arguments for signal_func
        
    Returns:
        List of generated signals (on CPU)
    """
    signals = []
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Batches"):
        current_batch_size = min(batch_size, total_samples - batch_idx * batch_size)
        
        # Generate batch on GPU
        batch_signals = signal_func(batch_size=current_batch_size, **kwargs)
        
        # Transfer to CPU only once per batch
        batch_signals_cpu = cp.asnumpy(batch_signals)
        
        signals.extend(batch_signals_cpu)
    
    return signals


def batch_fft_gpu(signals_gpu, n=1024):
    """
    Batch FFT on GPU - much faster than individual FFTs.
    
    Args:
      signals_gpu: CuPy array of shape (batch_size, signal_length)
        n: FFT size
        
    Returns:
        CuPy array of shape (batch_size, n)
    """
    return cp.fft.fftshift(cp.fft.fft(signals_gpu, n=n, axis=1), axes=1)


def batch_add_noise_gpu(signals_gpu, snr_db):
    """
    Add AWGN noise to batch of signals on GPU.
    
    Args:
        signals_gpu: CuPy array of shape (batch_size, signal_length)
        snr_db: SNR in dB
        
    Returns:
        Noisy signals (CuPy array)
    """
    batch_size, signal_length = signals_gpu.shape
    
    # Calculate noise power
    signal_power = cp.mean(cp.abs(signals_gpu) ** 2, axis=1, keepdims=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate complex noise
    noise_real = cp.random.randn(batch_size, signal_length) * cp.sqrt(noise_power / 2)
    noise_imag = cp.random.randn(batch_size, signal_length) * cp.sqrt(noise_power / 2)
    noise = noise_real + 1j * noise_imag
    
    return signals_gpu + noise

# Example: Batch LFM generation
def generate_lfm_batch_gpu(batch_size, fs=100e6, PRI=100e-6, fc=10e6, taup=40e-6, B=40e6):
    """
    Generate batch of LFM signals on GPU.
    
    Returns:
        CuPy array of shape (batch_size, Nr)
    """
    Ts = 1 / fs
    k = B / taup
    Nr = int(cp.round(fs * PRI))
    Nfast = int(cp.round(fs * taup))
    
    # Generate time vector
    t = (cp.arange(-Nfast // 2, Nfast // 2)) * Ts
    
    # Generate LFM pulse (same for all samples in batch)
    lfm_pulse = cp.exp(1j * (2 * cp.pi * fc * t + cp.pi * k * (t ** 2)))
    
    # Replicate for batch
    lfm_batch = cp.tile(lfm_pulse, (batch_size, 1))
    
    # Pad with zeros
    padding = cp.zeros((batch_size, Nr - Nfast), dtype=complex)
    lfm_batch = cp.concatenate([lfm_batch, padding], axis=1)
    
    return lfm_batch


# Example: Batch COMB generation
def generate_comb_batch_gpu(batch_size, fs=100e6, PRI=100e-6, fc=10e6, taup=40e-6, B=40e6):
    """
    Generate batch of COMB jamming signals on GPU.
    
    Returns:
        CuPy array of shape (batch_size, Nr)
    """
    Ts = 1 / fs
    k = B / taup
    Nr = int(cp.round(fs * PRI))
    Nfast = int(cp.round(fs * taup))
    
    # Generate base LFM
    t = (cp.arange(-Nfast // 2, Nfast // 2)) * Ts
    lfm_pulse = cp.exp(1j * (2 * cp.pi * fc * t + cp.pi * k * (t ** 2)))
    lfm = cp.concatenate([lfm_pulse, cp.zeros(Nr - Nfast, dtype=complex)])
    
    # Generate batch of COMB signals
    comb_batch = cp.zeros((batch_size, Nr), dtype=complex)
    
    for b in range(batch_size):
        # Random parameters for each sample
        M = cp.random.choice([9, 10, 11, 12])
        N_alt = cp.random.choice([0.5, 0.55, 0.6])
        Q = cp.random.choice([0.05, 0.06, 0.08])
        P = cp.random.choice([0.5, 0.6, 0.7])
        
        comb = cp.zeros(Nr, dtype=complex)
        for i in range(1, int(M) + 1):
            fi = Q * i
            altitude = N_alt + P * cp.sin(2 * cp.pi * fi * t)
          sawtooth = cp.exp(1j * 2 * cp.pi * altitude * t)
            comb += sawtooth
     
      comb_batch[b] = comb
    
    return comb_batch


def benchmark_batch_vs_sequential():
    """
    Benchmark batch processing vs sequential processing.
    """
    import time
    
    batch_size = 32
    total_samples = 200
    
    print("Benchmarking GPU batch processing...")
    
    # Sequential processing
    print("\n[1] Sequential processing (200 individual FFTs):")
    start = time.time()
    for _ in range(total_samples):
        signal = cp.random.randn(10000) + 1j * cp.random.randn(10000)
        fft_result = cp.fft.fft(signal)
    cp.cuda.Stream.null.synchronize()
    sequential_time = time.time() - start
    print(f"    Time: {sequential_time:.3f}s")
    
    # Batch processing
    print("\n[2] Batch processing (200 samples in batches of 32):")
    start = time.time()
    num_batches = (total_samples + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, total_samples - batch_idx * batch_size)
        signals = cp.random.randn(current_batch_size, 10000) + 1j * cp.random.randn(current_batch_size, 10000)
        fft_results = cp.fft.fft(signals, axis=1)
    cp.cuda.Stream.null.synchronize()
    batch_time = time.time() - start
    print(f"    Time: {batch_time:.3f}s")
    
    print(f"\n✓ Speedup: {sequential_time / batch_time:.1f}x")


if __name__ == '__main__':
    benchmark_batch_vs_sequential()
