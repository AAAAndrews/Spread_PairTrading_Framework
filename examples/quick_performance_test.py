"""
快速验证信号维持功能的性能优化
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
    """模拟模型"""
    def __init__(self):
        self.task = 'classification'


def quick_test():
    """快速测试性能"""
    print("="*60)
    print("信号维持功能 - 快速性能验证")
    print("="*60)
    
    # 检查Numba
    try:
        import numba
        print(f"\n✅ Numba 已安装 (v{numba.__version__})")
        print("   将使用JIT加速，性能最佳")
    except ImportError:
        print("\n⚠️ Numba 未安装")
        print("   将使用NumPy优化版本（仍然很快）")
        print("   安装Numba可获得10-100x加速: pip install numba")
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 10000
    
    # 10%的位置有信号
    signals = pd.DataFrame({
        'signal': np.random.choice([0, 1, -1], n_samples, p=[0.9, 0.05, 0.05])
    })
    
    print(f"\n测试数据: {n_samples:,} 样本")
    print(f"信号数量: {(signals['signal'] != 0).sum()} ({(signals['signal'] != 0).sum()/n_samples*100:.1f}%)")
    
    # 创建信号生成器
    mock_model = MockMLModel()
    signal_gen = SignalGenerator(
        mock_model,
        signal_holding_days=5
    )
    
    # 预热（首次运行可能较慢）
    print("\n预热...")
    _ = signal_gen._apply_signal_holding(signals.copy())
    
    # 性能测试
    print("运行性能测试 (5次)...")
    times = []
    
    for i in range(5):
        start = time.time()
        result = signal_gen._apply_signal_holding(signals.copy())
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  第{i+1}次: {elapsed*1000:.2f} ms")
    
    # 统计
    avg_time = np.mean(times)
    min_time = np.min(times)
    
    print("\n" + "="*60)
    print("性能统计")
    print("="*60)
    print(f"平均耗时: {avg_time*1000:.2f} ms")
    print(f"最快耗时: {min_time*1000:.2f} ms")
    print(f"处理速度: {n_samples/avg_time:,.0f} 样本/秒")
    
    # 验证结果
    original_count = (signals['signal'] != 0).sum()
    result_count = (result['signal'] != 0).sum()
    
    print("\n" + "="*60)
    print("结果验证")
    print("="*60)
    print(f"原始信号数: {original_count}")
    print(f"维持后信号数: {result_count}")
    print(f"增加: {result_count - original_count} ({(result_count/original_count-1)*100:.1f}%)")
    
    # 性能评级
    print("\n" + "="*60)
    print("性能评级")
    print("="*60)
    
    samples_per_sec = n_samples / avg_time
    
    if samples_per_sec > 5000000:
        rating = "⭐⭐⭐⭐⭐ 优秀"
        comment = "Numba加速生效，性能极佳！"
    elif samples_per_sec > 1000000:
        rating = "⭐⭐⭐⭐ 良好"
        comment = "可能使用NumPy优化版本或Numba未完全优化"
    elif samples_per_sec > 500000:
        rating = "⭐⭐⭐ 一般"
        comment = "建议安装Numba以获得更好性能"
    else:
        rating = "⭐⭐ 需要优化"
        comment = "性能较低，请检查环境配置"
    
    print(f"评级: {rating}")
    print(f"说明: {comment}")
    print(f"速度: {samples_per_sec:,.0f} 样本/秒")
    
    print("\n" + "="*60)
    print("✅ 测试完成！")
    print("="*60)


if __name__ == '__main__':
    quick_test()
