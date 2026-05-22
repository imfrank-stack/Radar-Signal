"""
雷达干扰信号批量生成脚本（GPU/CPU 双模式）

使用方法：
    python generate_data.py              
    python generate_data.py --gpu          
    python generate_data.py --data-num 100 

参数：
    --gpu         : 尝试使用 GPU 加速（需要 NVIDIA GPU 和 CuPy）
    --cpu         : 使用 CPU 模式（默认）
    --data-num    : 每个 SNR 的样本数量
    --snr-start   : SNR 起始值（默认 -20）
    --snr-end     : SNR 结束值（默认 10）

说明：
    GPU 模式使用 CuPy 后端加速信号生成计算（FFT、随机数、
    数组运算），I/O（频谱图、图片、.mat 写入）始终在 CPU 上。
    需 NVIDIA GPU + CUDA 工具包。
"""

import sys
import time
import argparse

# ============================================================
# 参数解析
# ============================================================

parser = argparse.ArgumentParser(
    description='雷达干扰信号批量生成脚本',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
示例:
    python generate_data.py --gpu --data-num 50
    python generate_data.py --gpu
""")

parser.add_argument('--gpu', action='store_true',
                    help='尝试使用 GPU 加速（实验性）')
parser.add_argument('--cpu', action='store_true',
                    help='使用 CPU 模式（默认）')
parser.add_argument('--data-num', type=int, default=None,
                    help='每个 SNR 的样本数量（默认 800）')
parser.add_argument('--snr-start', type=int, default=-10,
                    help='SNR 起始值（默认 -10）')
parser.add_argument('--snr-end', type=int, default=10,
                    help='SNR 上限值（不含，默认 10，即 SNR 10dB 是最大值）')
parser.add_argument('--mat-only', action='store_true',
                    help='仅生成.mat文件，跳过频谱图和图片（大幅加速）')
parser.add_argument('--seed', type=int, default=42,
                    help='随机种子，保证可复现性（默认：42）')

args = parser.parse_args()

# 自动选择后端
gpu_mode = False
if args.gpu:
    try:
        import cupy
        print("[OK] CuPy detected, GPU mode available")
        # 检查是否有 NVIDIA GPU
        if cupy.cuda.runtime.getDeviceCount() > 0:
            print(f"[OK] 检测到 {cupy.cuda.runtime.getDeviceCount()} 个 NVIDIA GPU(s)")
            gpu_mode = True
        else:
            print("[WARN]️ 未检测到 NVIDIA GPU，将使用 CPU 模式")
    except ImportError:
        print("[WARN]️ CuPy 未安装，使用 CPU 模式")
        print("   如需 GPU 加速：pip install cupy-cuda12x")
    except Exception as e:
        print(f"[WARN]️ GPU 检测失败: {e}，使用 CPU 模式")

# ============================================================
# 导入信号生成函数
# ============================================================

print("\n[-] 正在加载雷达干扰信号生成模块...")

try:
    from radar_jamming_signals import (
        set_backend, set_seed, _SNR_CONFIG,
        _TRAIN_IMG_ROOT, _TRAIN_SEQ_ROOT,
        LFM_alldb, AM_alldb, COMB_alldb, FM_alldb,
        ISRJ_alldb, MNJ_alldb, RMT_alldb, RGPO_alldb,
        R_VGPO_alldb, SMSP_alldb, VGPO_alldb, VMT_alldb
    )
    
    # Import _SKIP_IMAGES to control image generation
    import radar_jamming_signals
    if args.mat_only:
        radar_jamming_signals._SKIP_IMAGES = True
        print("[INFO] MAT-only mode: 跳过频谱图和图片生成")

    # 根据 gpu_mode 选择后端
    if gpu_mode:
        set_backend('cupy')
    else:
        set_backend('numpy')
    
    # 设置随机种子，保证可复现性
    set_seed(args.seed)

    # 配置 SNR 范围
    _SNR_CONFIG['start'] = args.snr_start
    _SNR_CONFIG['stop'] = args.snr_end + _SNR_CONFIG['step']
    print(f"[OK] 模块加载成功（{'GPU' if gpu_mode else 'CPU'} 模式，SNR: {args.snr_start}~{args.snr_end}dB）\n")
except ImportError as e:
    print(f"[ERROR] 导入错误: {e}")
    print("请确保 radar_jamming_signals.py 存在于同一目录")
    sys.exit(1)

# ============================================================
# 信号列表
# ============================================================

SIGNAL_FUNCTIONS = [
    ('LFM_alldb', LFM_alldb), ('AM_alldb', AM_alldb), ('COMB_alldb', COMB_alldb),
    ('FM_alldb', FM_alldb), ('ISRJ_alldb', ISRJ_alldb), ('MNJ_alldb', MNJ_alldb),
    ('RMT_alldb', RMT_alldb), ('RGPO_alldb', RGPO_alldb), ('R_VGPO_alldb', R_VGPO_alldb),
    ('SMSP_alldb', SMSP_alldb), ('VGPO_alldb', VGPO_alldb), ('VMT_alldb', VMT_alldb)
]

# ============================================================
# 生成数据集
# ============================================================

def generate_dataset(data_num=None):
    """生成完整数据集（12 种信号）"""
    print("=" * 70)
    print(">>> 正在生成数据集（12 种信号）")
    print("=" * 70)

    for name, func in SIGNAL_FUNCTIONS:
        print(f"\n{'=' * 70}")
        print(f"[GEN] 生成信号: {name:<12}")
        print("-" * 70)
        t0 = time.time()

        try:
            if data_num:
                func(data_num=data_num)
            else:
                func()
            elapsed = time.time() - t0
            print(f"[OK] 完成 {name:<12} 耗时: {elapsed:.2f}s")
        except Exception as e:
            print(f"[ERROR] 错误: {e}")
            import traceback
            traceback.print_exc()

# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    # 显示参数信息
    print("[START] 雷达干扰信号批量生成")
    print("-" * 70)
    print(f"计算模式: {'GPU 加速' if gpu_mode else 'CPU'}")
    print(f"随机种子: {args.seed}（保证可复现性）")
    if args.data_num:
        print(f"样本数量: {args.data_num} / SNR / 信号（自定义）")
    else:
        print(f"样本数量: 800 / SNR / 信号（默认）")
    print(f"SNR 范围: {args.snr_start} 到 {args.snr_end} dB")
    print("-" * 70)

    # 开始计时
    start_time = time.time()

    # 生成数据集
    generate_dataset(args.data_num)

    # 总计时
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("[DONE] 全部完成！")
    print("-" * 70)
    print(f"总耗时: {total_time:.2f}s ({total_time / 60:.2f} 分钟)")
    print("=" * 70)
