# -*- coding: utf-8 -*-
"""
雷达干扰信号批量生成脚本（极致并行版）
按 SNR + 信号类型 拆分任务，确保双路 CPU 所有核心 100% 利用
使用方法:
    python generate_data_parallel.py --help
    python generate_data_parallel.py --data-num 100
    python generate_data_parallel.py --num-workers 64
参数:
    --data-num    : 每个 SNR 的样本数量（默认 800）
    --snr-start   : SNR 起始值（默认 -10）
    --snr-end     : SNR 结束值（默认 10）
    --mat-only    : 仅生成 .mat 文件，跳过频谱图
    --num-workers : CPU 进程数（默认 = CPU 逻辑核心数）
    --no-bind-cpu : 禁用 CPU 亲和性绑定（推荐用于双路服务器，让操作系统自动调度）
"""
import sys
import time
import argparse
import multiprocessing
import os


def get_numa_aware_core_id(worker_idx, total_cores):
    """
    获取NUMA感知的核心ID，确保双路CPU都被利用
    假设第一路CPU是 0..n/2-1，第二路是 n/2..n-1
    交错分配：worker 0→0, worker 1→n/2, worker 2→1, worker 3→n/2+1...
    """
    half = total_cores // 2
    if worker_idx % 2 == 0:
        # 偶数worker分配到第一路
        return worker_idx // 2
    else:
        # 奇数worker分配到第二路
        return half + worker_idx // 2


def pin_process_to_core(worker_idx, total_workers, numa_aware=True):
    """
    强制将进程绑定到指定CPU核心
    如果 numa_aware=True，会智能地跨NUMA节点分配
    """
    cpu_count = os.cpu_count() or 1
    if numa_aware:
        core_id = get_numa_aware_core_id(worker_idx, cpu_count)
    else:
        core_id = worker_idx % cpu_count
    
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity([core_id])
        return core_id
    except ImportError:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            mask = 1 << core_id
            kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask)
            return core_id
        except:
            pass
    return -1


# 使用全局变量传递worker index（初始值）
_worker_idx_counter = None


def init_worker(config):
    """初始化 worker，设置配置"""
    global _worker_idx_counter
    
    # 方式1: 从进程名获取
    proc_name = multiprocessing.current_process().name
    try:
        worker_idx = int(proc_name.split('-')[-1]) - 1
    except:
        worker_idx = -1
    
    # 方式2: 使用计数器（备选方案）
    if worker_idx < 0:
        with _worker_idx_counter.get_lock():
            worker_idx = _worker_idx_counter.value
            _worker_idx_counter.value += 1
    
    # 绑定CPU核心（如果需要）
    core_id = -1
    if not config.get('no_bind_cpu', False):
        num_workers = config.get('num_workers', os.cpu_count() or 1)
        core_id = pin_process_to_core(worker_idx, num_workers)
    
    # 显示绑定信息（调试）
    if core_id >= 0:
        print(f"[Worker-{worker_idx+1}] 绑定到 CPU 核心 {core_id}", flush=True)
    
    # 设置信号生成配置
    import radar_jamming_signals
    radar_jamming_signals._SNR_CONFIG.update(config['snr'])
    if config.get('skip_images'):
        radar_jamming_signals._SKIP_IMAGES = True


def generate_single_snr_signal(args):
    """
    生成单个 SNR 的单个信号类型（最细粒度并行）
    args: (signal_name, func_name, snr_value, snr_index, data_num, img_root, seq_root, base_seed)
    """
    signal_name, func_name, snr_value, snr_idx, data_num, img_root, seq_root, base_seed = args
    import importlib
    import os
    import hashlib
    radar_jamming_signals = importlib.import_module('radar_jamming_signals')
    
    # 为每个(信号类型, SNR)组合生成确定性的随机种子
    seed_str = f"{signal_name}_{snr_value}_{base_seed}"
    task_seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:16], 16) % (2**32)
    radar_jamming_signals.set_seed(task_seed)
    
    func = getattr(radar_jamming_signals, func_name)
    original_img = radar_jamming_signals._TRAIN_IMG_ROOT
    original_seq = radar_jamming_signals._TRAIN_SEQ_ROOT
    try:
        # 设置路径
        radar_jamming_signals._TRAIN_IMG_ROOT = img_root
        radar_jamming_signals._TRAIN_SEQ_ROOT = seq_root
        start_time = time.time()
        # 调用生成函数
        func(data_num=data_num)
        elapsed = time.time() - start_time
        return (True, f"{signal_name}_{snr_value}dB", elapsed, None)
    except Exception as e:
        import traceback
        return (False, f"{signal_name}_{snr_value}dB", 0, f"{str(e)[:100]}")
    finally:
        radar_jamming_signals._TRAIN_IMG_ROOT = original_img
        radar_jamming_signals._TRAIN_SEQ_ROOT = original_seq


def main():
    parser = argparse.ArgumentParser(
        description='雷达干扰信号批量生成脚本（极致并行版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python generate_data_parallel.py --data-num 50
    python generate_data_parallel.py --num-workers 80 --no-bind-cpu
    python generate_data_parallel.py --mat-only
""")
    parser.add_argument('--data-num', type=int, default=None,
                        help='每个 SNR 的样本数量（默认 800）')
    parser.add_argument('--snr-start', type=int, default=-10,
                        help='SNR 起始值（默认 -10）')
    parser.add_argument('--snr-end', type=int, default=10,
                        help='SNR 上限值（不含，默认 10）')
    parser.add_argument('--mat-only', action='store_true',
                        help='仅生成 .mat 文件，跳过频谱图')
parser.add_argument('--num-workers', type=int, default=None,
                    help='并行进程数（默认：等于 CPU 逻辑核心数）')
parser.add_argument('--no-bind-cpu', action='store_true',
                    help='禁用 CPU 亲和性绑定（推荐用于双路服务器，让操作系统调度）')
parser.add_argument('--seed', type=int, default=42,
                    help='随机种子，保证可复现性（默认：42）')
args = parser.parse_args()

    # 加载模块
    print("\n[-] 正在加载雷达干扰信号生成模块...")
    try:
        import radar_jamming_signals
        from radar_jamming_signals import _SNR_CONFIG, set_seed
        from radar_jamming_signals import _TRAIN_IMG_ROOT, _TRAIN_SEQ_ROOT
    except ImportError as e:
        print(f"[ERROR] 导入错误: {e}")
        print("请确保 radar_jamming_signals.py 存在于同一目录")
        sys.exit(1)

    # 配置 SNR
    _SNR_CONFIG['start'] = args.snr_start
    _SNR_CONFIG['stop'] = args.snr_end + _SNR_CONFIG['step']
    snr_list = list(range(args.snr_start, args.snr_end + 1, _SNR_CONFIG['step']))

    # 计算进程数 - 使用所有核心
    cpu_count = os.cpu_count() or 1
    num_workers = args.num_workers or cpu_count
    data_num = args.data_num or 800

    # 显示配置
    print("=" * 70)
    print("[START] 雷达干扰信号批量生成（极致并行版）")
    print("-" * 70)
    print(f"CPU 逻辑核心数: {cpu_count}")
    print(f"使用进程数: {num_workers}")
    print(f"随机种子: {args.seed}（保证可复现性）")
    print(f"SNR 数量: {len(snr_list)} ({args.snr_start} ~ {args.snr_end} dB, 步长 {_SNR_CONFIG['step']})")
    print(f"样本数量: {data_num} / SNR / 信号")
    print(f"跳过图片生成: {'是' if args.mat_only else '否'}")
    if args.no_bind_cpu:
        print(f"CPU 绑定: 禁用（操作系统自动调度，推荐双路服务器）")
    else:
        print(f"CPU 绑定: 启用（NUMA 感知绑定，每个进程绑定独立核心）")
    print("=" * 70)

    # 信号列表
    signals = [
        ('LFM', 'LFM_alldb'),
        ('AM', 'AM_alldb'),
        ('COMB', 'COMB_alldb'),
        ('FM', 'FM_alldb'),
        ('ISRJ', 'ISRJ_alldb'),
        ('MNJ', 'MNJ_alldb'),
        ('RMT', 'RMT_alldb'),
        ('RGPO', 'RGPO_alldb'),
        ('R_VGPO', 'R_VGPO_alldb'),
        ('SMSP', 'SMSP_alldb'),
        ('VGPO', 'VGPO_alldb'),
        ('VMT', 'VMT_alldb'),
    ]

    # 构建超细化任务列表 - 每个 SNR x 每个信号 = 一个任务
    tasks = []
    for snr_idx, snr_val in enumerate(snr_list):
        for name, func_name in signals:
            tasks.append((name, func_name, snr_val, snr_idx, data_num,
                          _TRAIN_IMG_ROOT, _TRAIN_SEQ_ROOT, args.seed))

    start_time = time.time()

    # 设置全局随机种子
    set_seed(args.seed)
    
    # 配置
    config = {
        'snr': dict(_SNR_CONFIG),
        'skip_images': args.mat_only,
        'no_bind_cpu': args.no_bind_cpu,
        'num_workers': num_workers
    }

    # 初始化worker counter
    global _worker_idx_counter
    _worker_idx_counter = multiprocessing.Value('i', 0)

    print(f"\n>>> 共 {len(tasks)} 个并行任务，{num_workers} 个进程执行")
    print(">>> 任务粒度: 每个 SNR x 每个信号 = 1 个独立任务")
    print(f">>> 随机种子: {args.seed}（保证可复现性）")
    if args.no_bind_cpu:
        print(">>> 双路 CPU：由操作系统自动分配到所有核心")
    else:
        print(">>> 双路 CPU：NUMA 感知绑定，确保两路都被利用")
    print("-" * 70)

    # 创建进程池并执行
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(config,)
    ) as pool:
        results = []
        completed = 0
        for result in pool.imap_unordered(generate_single_snr_signal, tasks):
            completed += 1
            success, name, elapsed, error = result
            if success:
                print(f"[{completed}/{len(tasks)}] [OK] {name:<18} 完成, 耗时: {elapsed:.2f}s", flush=True)
            else:
                print(f"[{completed}/{len(tasks)}] [ERROR] {name:<18} 失败: {error}", flush=True)
            results.append(result)

    # 统计
    success_count = sum(1 for r in results if r[0])
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("[FINISHED] 全部任务完成")
    print("-" * 70)
    print(f"成功: {success_count}/{len(tasks)} 个任务")
    print(f"总耗时: {total_time:.2f}s ({total_time / 60:.2f} 分钟)")
    print("=" * 70)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
