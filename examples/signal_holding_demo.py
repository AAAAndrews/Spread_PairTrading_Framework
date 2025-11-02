"""
信号维持功能演示
演示如何使用 signal_holding_days 参数来控制信号的持续性
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.ml_models import SignalGenerator, MLModel


def create_sample_signals():
    """创建示例信号数据"""
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    
    # 原始预测信号：包含一些噪音
    raw_signals = [
        1, 0, 0, 1, 0,      # 前5天：做多信号出现
        0, -1, 0, 0, 0,     # 6-10天：出现做空信号
        0, 0, 1, 0, 0,      # 11-15天：再次做多
        0, 0, 0, -1, 0,     # 16-20天：做空
        1, 0, 0, 0, 0,      # 21-25天：做多
        0, 0, 0, 0, 0       # 26-30天：无信号
    ]
    
    signals_df = pd.DataFrame({
        'date': dates,
        'signal': raw_signals,
        'signal_strength': 1.0,
        'max_proba': 0.8
    })
    signals_df.set_index('date', inplace=True)
    
    return signals_df


def demo_signal_holding():
    """演示不同信号维持天数的效果"""
    
    print("=" * 80)
    print("信号维持功能演示")
    print("=" * 80)
    
    # 创建示例信号
    original_signals = create_sample_signals()
    
    print("\n原始信号（包含噪音）:")
    print("-" * 80)
    print(original_signals[['signal']])
    print(f"\n信号统计:")
    print(original_signals['signal'].value_counts().sort_index())
    
    # 创建一个模拟的MLModel（用于演示）
    class MockMLModel:
        def __init__(self):
            self.task = 'classification'
            self.model = None
        
        def predict(self, X):
            return X  # 直接返回输入作为预测
        
        def predict_proba(self, X):
            return None
    
    mock_model = MockMLModel()
    
    # 测试不同的信号维持天数
    holding_days_list = [1, 3, 5, 7]
    
    for holding_days in holding_days_list:
        print(f"\n{'=' * 80}")
        print(f"信号维持天数 = {holding_days}")
        print("=" * 80)
        
        # 创建信号生成器
        signal_gen = SignalGenerator(
            model=mock_model,
            threshold=0.6,
            signal_holding_days=holding_days
        )
        
        # 应用信号维持逻辑
        held_signals = signal_gen._apply_signal_holding(original_signals.copy())
        
        # 对比显示
        comparison = pd.DataFrame({
            '原始信号': original_signals['signal'],
            f'维持{holding_days}天': held_signals['signal']
        })
        
        print("\n信号对比:")
        print(comparison)
        
        print(f"\n维持后信号统计:")
        print(held_signals['signal'].value_counts().sort_index())
        
        # 统计信号变化
        changes = (comparison['原始信号'] != comparison[f'维持{holding_days}天']).sum()
        print(f"\n信号调整数量: {changes}/{len(comparison)} ({changes/len(comparison)*100:.1f}%)")


def demo_opposite_signal_interruption():
    """演示相反信号打断信号维持的情况"""
    
    print("\n\n" + "=" * 80)
    print("相反信号打断演示")
    print("=" * 80)
    
    # 创建特殊的信号序列：做多信号后很快出现做空信号
    dates = pd.date_range('2024-01-01', periods=15, freq='D')
    
    # 场景：第1天做多，第5天做空（测试信号维持是否会在第5天停止）
    raw_signals = [
        1, 0, 0, 0, -1,     # 做多->相反做空
        0, 0, 0, 0, 1,      # 做空->相反做多
        0, 0, 0, 0, 0
    ]
    
    signals_df = pd.DataFrame({
        'date': dates,
        'signal': raw_signals,
        'signal_strength': 1.0
    })
    signals_df.set_index('date', inplace=True)
    
    print("\n原始信号序列:")
    print("-" * 80)
    print("说明：第1天做多(1)，第5天做空(-1)，第10天再做多(1)")
    print(signals_df[['signal']])
    
    # 创建模拟模型
    class MockMLModel:
        def __init__(self):
            self.task = 'classification'
    
    mock_model = MockMLModel()
    
    # 测试较长的信号维持天数
    holding_days = 7
    
    print(f"\n使用信号维持天数 = {holding_days}")
    print("=" * 80)
    
    signal_gen = SignalGenerator(
        model=mock_model,
        threshold=0.6,
        signal_holding_days=holding_days
    )
    
    held_signals = signal_gen._apply_signal_holding(signals_df.copy())
    
    comparison = pd.DataFrame({
        '原始信号': signals_df['signal'],
        f'维持{holding_days}天': held_signals['signal'],
        '说明': ''
    })
    
    # 添加说明
    comparison.loc[comparison.index[0], '说明'] = '做多信号出现'
    comparison.loc[comparison.index[1:4], '说明'] = '维持做多'
    comparison.loc[comparison.index[4], '说明'] = '做空信号打断，停止维持并反向'
    comparison.loc[comparison.index[5:8], '说明'] = '维持做空'
    comparison.loc[comparison.index[9], '说明'] = '做多信号再次出现'
    comparison.loc[comparison.index[10:], '说明'] = '维持做多'
    
    print("\n信号对比:")
    print(comparison)
    
    print("\n关键点说明:")
    print("1. 第1天出现做多信号(1)，按规则应维持7天至第7天")
    print("2. 但第5天出现做空信号(-1)，与做多完全相反")
    print("3. 因此在第5天停止维持做多，并开始做空")
    print("4. 第10天出现做多信号，再次反转")


def demo_practical_example():
    """实际应用示例：基于n天收益率的交易策略"""
    
    print("\n\n" + "=" * 80)
    print("实际应用示例：基于5天收益率的交易策略")
    print("=" * 80)
    
    # 模拟价格数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    
    # 生成模拟价格（带趋势和噪音）
    trend = np.linspace(100, 110, 60)
    noise = np.random.randn(60) * 2
    prices = trend + noise
    
    # 计算5天收益率
    returns_5d = pd.Series(prices).pct_change(5)
    
    # 生成交易信号：基于5天收益率
    # 如果5天收益率 > 2%，做多；< -2%，做空；否则观望
    raw_signals = np.where(returns_5d > 0.02, 1, 
                          np.where(returns_5d < -0.02, -1, 0))
    
    signals_df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'returns_5d': returns_5d,
        'signal': raw_signals
    })
    signals_df.set_index('date', inplace=True)
    
    print("\n数据概览:")
    print(signals_df.head(10))
    
    # 创建模拟模型
    class MockMLModel:
        def __init__(self):
            self.task = 'classification'
    
    mock_model = MockMLModel()
    
    # 不维持信号 vs 维持5天
    print("\n" + "=" * 80)
    print("对比：不维持信号 vs 维持5天信号")
    print("=" * 80)
    
    # 场景1：不维持（holding_days=1）
    signal_gen_1 = SignalGenerator(mock_model, signal_holding_days=1)
    signals_no_hold = signal_gen_1._apply_signal_holding(signals_df[['signal']].copy())
    
    # 场景2：维持5天（holding_days=5）
    signal_gen_5 = SignalGenerator(mock_model, signal_holding_days=5)
    signals_hold_5 = signal_gen_5._apply_signal_holding(signals_df[['signal']].copy())
    
    # 统计对比
    comparison = pd.DataFrame({
        '价格': signals_df['price'],
        '5日收益率': signals_df['returns_5d'],
        '原始信号': signals_df['signal'],
        '维持5天': signals_hold_5['signal']
    })
    
    print("\n信号对比（前20天）:")
    print(comparison.head(20))
    
    # 统计信号频率
    print("\n信号统计:")
    print(f"原始信号 - 做多: {(signals_df['signal'] == 1).sum()}, "
          f"做空: {(signals_df['signal'] == -1).sum()}, "
          f"观望: {(signals_df['signal'] == 0).sum()}")
    print(f"维持5天 - 做多: {(signals_hold_5['signal'] == 1).sum()}, "
          f"做空: {(signals_hold_5['signal'] == -1).sum()}, "
          f"观望: {(signals_hold_5['signal'] == 0).sum()}")
    
    # 计算信号变化次数（交易频率）
    signal_changes_original = (signals_df['signal'].diff() != 0).sum()
    signal_changes_held = (signals_hold_5['signal'].diff() != 0).sum()
    
    print(f"\n交易频率（信号变化次数）:")
    print(f"原始信号: {signal_changes_original} 次")
    print(f"维持5天: {signal_changes_held} 次")
    print(f"减少交易次数: {signal_changes_original - signal_changes_held} 次 "
          f"({(1 - signal_changes_held/signal_changes_original)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("总结：")
    print("1. 使用与标签制作相同的天数（5天）来维持信号")
    print("2. 可以显著减少交易频率，降低交易成本")
    print("3. 保持了与模型训练时一致的时间跨度假设")
    print("4. 相反信号出现时会及时反转，不会错过重要的趋势变化")
    print("=" * 80)


if __name__ == '__main__':
    # 运行所有演示
    demo_signal_holding()
    demo_opposite_signal_interruption()
    demo_practical_example()
    
    print("\n\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)
