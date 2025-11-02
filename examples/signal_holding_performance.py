"""
信号维持功能性能测试
对比不同实现的性能差异
"""
import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.ml_models import SignalGenerator


class MockMLModel:
    """模拟的机器学习模型"""
    def __init__(self):
        self.task = 'classification'
        self.model = None
    
    def predict(self, X):
        return X
    
    def predict_proba(self, X):
        return None


def generate_test_signals(n_samples: int, signal_density: float = 0.1) -> pd.DataFrame:
    """
    生成测试信号数据
    
    Args:
        n_samples: 样本数量
        signal_density: 信号密度（非零信号的比例）
    
    Returns:
        测试信号DataFrame
    """
    np.random.seed(42)
    
    # 生成稀疏的信号（大部分是0）
    signals = np.zeros(n_samples)
    
    # 随机位置放置非零信号
    n_signals = int(n_samples * signal_density)
    signal_indices = np.random.choice(n_samples, n_signals, replace=False)
    
    # 随机分配做多(1)或做空(-1)
    signal_values = np.random.choice([1, -1], n_signals)
    signals[signal_indices] = signal_values
    
    return pd.DataFrame({'signal': signals})


def benchmark_signal_holding(n_samples: int, holding_days: int, n_runs: int = 5):
    """
    性能基准测试
    
    Args:
        n_samples: 样本数量
        holding_days: 信号维持天数
        n_runs: 重复运行次数
    """
    print(f"\n{'='*80}")
    print(f"性能测试: {n_samples:,} 样本, 维持 {holding_days} 天")
    print(f"{'='*80}")
    
    # 生成测试数据
    test_signals = generate_test_signals(n_samples, signal_density=0.1)
    
    mock_model = MockMLModel()
    signal_gen = SignalGenerator(
        mock_model,
        signal_holding_days=holding_days
    )
    
    # 预热（首次运行可能较慢）
    _ = signal_gen._apply_signal_holding(test_signals.copy())
    
    # 性能测试
    times = []
    for i in range(n_runs):
        start_time = time.time()
        result = signal_gen._apply_signal_holding(test_signals.copy())
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    # 统计结果
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\n运行 {n_runs} 次的统计结果:")
    print(f"  平均耗时: {avg_time*1000:.2f} ms")
    print(f"  标准差:   {std_time*1000:.2f} ms")
    print(f"  最快:     {min_time*1000:.2f} ms")
    print(f"  最慢:     {max_time*1000:.2f} ms")
    print(f"  处理速度: {n_samples/avg_time:,.0f} 样本/秒")
    
    # 验证结果正确性
    original_nonzero = (test_signals['signal'] != 0).sum()
    result_nonzero = (result['signal'] != 0).sum()
    print(f"\n信号统计:")
    print(f"  原始非零信号: {original_nonzero}")
    print(f"  维持后非零信号: {result_nonzero}")
    print(f"  增加: {result_nonzero - original_nonzero} ({(result_nonzero/original_nonzero-1)*100:.1f}%)")
    
    return avg_time, result


def test_numba_availability():
    """测试Numba是否可用"""
    print("\n" + "="*80)
    print("检查加速库可用性")
    print("="*80)
    
    # 检查Numba
    try:
        import numba
        print(f"✅ Numba 已安装 (版本 {numba.__version__})")
        print("   性能提升: 10-100x (取决于数据规模)")
        numba_available = True
    except ImportError:
        print("❌ Numba 未安装")
        print("   安装方法: pip install numba")
        print("   建议安装以获得最佳性能")
        numba_available = False
    
    # 检查NumPy版本
    print(f"\n✅ NumPy 版本: {np.__version__}")
    
    return numba_available


def compare_different_scenarios():
    """对比不同场景下的性能"""
    print("\n" + "="*80)
    print("不同场景性能对比")
    print("="*80)
    
    scenarios = [
        # (样本数, 维持天数, 描述)
        (1000, 5, "小规模数据"),
        (10000, 5, "中等规模数据"),
        (50000, 5, "大规模数据（5万）"),
        (10000, 1, "无维持（每日信号）"),
        (10000, 10, "长期维持（10天）"),
        (10000, 30, "超长期维持（30天）"),
    ]
    
    results = []
    
    for n_samples, holding_days, description in scenarios:
        print(f"\n场景: {description}")
        print(f"参数: {n_samples:,} 样本, 维持 {holding_days} 天")
        print("-" * 60)
        
        test_signals = generate_test_signals(n_samples, signal_density=0.1)
        
        mock_model = MockMLModel()
        signal_gen = SignalGenerator(mock_model, signal_holding_days=holding_days)
        
        # 计时
        start = time.time()
        result = signal_gen._apply_signal_holding(test_signals.copy())
        elapsed = time.time() - start
        
        print(f"耗时: {elapsed*1000:.2f} ms")
        print(f"速度: {n_samples/elapsed:,.0f} 样本/秒")
        
        results.append({
            'scenario': description,
            'samples': n_samples,
            'holding_days': holding_days,
            'time_ms': elapsed * 1000,
            'samples_per_sec': n_samples / elapsed
        })
    
    # 汇总表格
    print("\n" + "="*80)
    print("性能汇总")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    return df_results


def test_scalability():
    """测试可扩展性：不同数据规模下的性能"""
    print("\n" + "="*80)
    print("可扩展性测试")
    print("="*80)
    
    sample_sizes = [1000, 5000, 10000, 25000, 50000, 100000]
    holding_days = 5
    
    results = []
    
    for n_samples in sample_sizes:
        print(f"\n测试 {n_samples:,} 样本...")
        
        test_signals = generate_test_signals(n_samples, signal_density=0.1)
        
        mock_model = MockMLModel()
        signal_gen = SignalGenerator(mock_model, signal_holding_days=holding_days)
        
        # 预热
        _ = signal_gen._apply_signal_holding(test_signals.copy())
        
        # 测试3次取平均
        times = []
        for _ in range(3):
            start = time.time()
            _ = signal_gen._apply_signal_holding(test_signals.copy())
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        
        results.append({
            'samples': n_samples,
            'time_ms': avg_time * 1000,
            'samples_per_sec': n_samples / avg_time,
            'time_per_sample_us': (avg_time / n_samples) * 1e6
        })
        
        print(f"  耗时: {avg_time*1000:.2f} ms")
        print(f"  速度: {n_samples/avg_time:,.0f} 样本/秒")
    
    # 分析结果
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("可扩展性分析")
    print("="*80)
    print(df_results.to_string(index=False))
    
    # 计算时间复杂度
    print("\n时间复杂度分析:")
    
    # 简单的线性回归估计
    log_samples = np.log10(df_results['samples'])
    log_time = np.log10(df_results['time_ms'])
    
    coefficients = np.polyfit(log_samples, log_time, 1)
    complexity = coefficients[0]
    
    print(f"  估计复杂度: O(n^{complexity:.2f})")
    
    if complexity < 1.2:
        print("  ✅ 接近线性时间复杂度 O(n)")
    elif complexity < 1.5:
        print("  ⚠️ 略高于线性，但性能良好")
    else:
        print("  ❌ 复杂度较高，可能需要进一步优化")
    
    return df_results


def memory_usage_test():
    """测试内存使用情况"""
    print("\n" + "="*80)
    print("内存使用测试")
    print("="*80)
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # 测试前的内存
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建大规模测试数据
        n_samples = 100000
        test_signals = generate_test_signals(n_samples, signal_density=0.1)
        
        mock_model = MockMLModel()
        signal_gen = SignalGenerator(mock_model, signal_holding_days=5)
        
        # 执行信号维持
        result = signal_gen._apply_signal_holding(test_signals.copy())
        
        # 测试后的内存
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before
        
        print(f"\n测试规模: {n_samples:,} 样本")
        print(f"内存使用前: {mem_before:.2f} MB")
        print(f"内存使用后: {mem_after:.2f} MB")
        print(f"额外内存:   {mem_used:.2f} MB")
        print(f"每样本内存: {(mem_used*1024)/n_samples:.2f} KB")
        
        # 数据大小估计
        data_size = test_signals.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"\n数据帧大小: {data_size:.2f} MB")
        print(f"内存效率:   {(data_size/mem_used)*100:.1f}% (越接近100%越好)")
        
    except ImportError:
        print("⚠️ psutil 未安装，跳过内存测试")
        print("   安装方法: pip install psutil")


def recommend_settings():
    """推荐设置"""
    print("\n" + "="*80)
    print("性能优化建议")
    print("="*80)
    
    print("\n📊 数据规模建议:")
    print("  • < 1万样本:   性能无瓶颈，任意设置")
    print("  • 1-10万样本:  注意信号维持天数，建议 ≤ 10天")
    print("  • > 10万样本:  考虑批处理，或减少信号密度")
    
    print("\n⚡ 加速方法:")
    print("  1. 安装 Numba: pip install numba")
    print("     预期提升: 10-100x (大规模数据)")
    print("  ")
    print("  2. 减少信号密度:")
    print("     使用更高的 threshold 值过滤低质量信号")
    print("  ")
    print("  3. 合理设置 signal_holding_days:")
    print("     不要盲目增大维持天数")
    
    print("\n💾 内存优化:")
    print("  • 使用 float32 代替 float64 (如果精度允许)")
    print("  • 及时删除不需要的中间结果")
    print("  • 考虑使用生成器处理超大数据集")
    
    print("\n🎯 最佳实践:")
    print("  • signal_holding_days = 训练标签的天数")
    print("  • threshold = 0.6-0.8 (根据验证集调优)")
    print("  • 定期监控性能，发现瓶颈")


if __name__ == '__main__':
    print("="*80)
    print("信号维持功能性能测试套件")
    print("="*80)
    
    # 1. 检查加速库
    numba_available = test_numba_availability()
    
    # 2. 基准测试
    benchmark_signal_holding(10000, 5, n_runs=5)
    
    # 3. 不同场景对比
    scenario_results = compare_different_scenarios()
    
    # 4. 可扩展性测试
    scalability_results = test_scalability()
    
    # 5. 内存测试
    memory_usage_test()
    
    # 6. 优化建议
    recommend_settings()
    
    print("\n" + "="*80)
    print("性能测试完成！")
    print("="*80)
    
    # 保存结果
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    if 'scenario_results' in locals():
        scenario_results.to_csv(
            output_dir / f'signal_holding_performance_{timestamp}.csv',
            index=False
        )
        print(f"\n结果已保存到: outputs/signal_holding_performance_{timestamp}.csv")
