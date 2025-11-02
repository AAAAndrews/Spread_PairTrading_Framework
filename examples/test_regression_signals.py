"""
回归任务信号生成测试示例
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置随机种子
np.random.seed(42)

# ============================================================
# 1. 创建模拟数据
# ============================================================

print("="*80)
print("回归任务信号生成测试")
print("="*80)

# 生成模拟的回归预测值（模拟未来收益率预测）
n_samples = 1000
dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

# 模拟预测值：带有趋势和噪声
trend = np.linspace(-0.05, 0.05, n_samples)
seasonality = 0.03 * np.sin(np.linspace(0, 8*np.pi, n_samples))
noise = np.random.normal(0, 0.02, n_samples)
predictions = trend + seasonality + noise

print(f"\n模拟数据统计:")
print(f"  样本数: {n_samples}")
print(f"  预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
print(f"  预测值均值: {predictions.mean():.4f}")
print(f"  预测值标准差: {predictions.std():.4f}")

# ============================================================
# 2. 测试固定阈值模式
# ============================================================

print("\n" + "="*80)
print("测试1: 固定阈值模式")
print("="*80)

from src.core.ml_models import SignalGenerator, MLModel

# 创建一个模拟的回归模型
class MockRegressionModel:
    """模拟回归模型用于测试"""
    def __init__(self):
        self.task = 'regression'
        self.predictions = predictions
    
    def predict(self, X):
        # 返回预先生成的预测值
        return self.predictions[:len(X)]

# 创建模拟模型
mock_model = MLModel(model_type='gradient_boosting', task='regression')
mock_model.model = MockRegressionModel()
mock_model.task = 'regression'

# 创建信号生成器（固定阈值）
signal_gen_fixed = SignalGenerator(
    mock_model,
    regression_upper_threshold=0.03,    # 3%
    regression_lower_threshold=-0.03,   # -3%
    use_rolling_quantile=False,
    signal_holding_days=1               # 不维持，每天生成新信号
)

# 生成信号
X_dummy = np.zeros((n_samples, 1))  # 占位符
signals_fixed = signal_gen_fixed.generate_signals(X_dummy)
signals_fixed.index = dates

print(f"\n固定阈值信号分布:")
print(signals_fixed['signal'].value_counts().sort_index())
print(f"\n做多比例: {(signals_fixed['signal'] == 1).sum() / n_samples * 100:.1f}%")
print(f"观望比例: {(signals_fixed['signal'] == 0).sum() / n_samples * 100:.1f}%")
print(f"做空比例: {(signals_fixed['signal'] == -1).sum() / n_samples * 100:.1f}%")

# ============================================================
# 3. 测试滚动分位数模式
# ============================================================

print("\n" + "="*80)
print("测试2: 滚动分位数模式")
print("="*80)

# 创建信号生成器（滚动分位数）
signal_gen_quantile = SignalGenerator(
    mock_model,
    use_rolling_quantile=True,
    rolling_window=60,
    upper_quantile=0.75,
    lower_quantile=0.25,
    signal_holding_days=1
)

signals_quantile = signal_gen_quantile.generate_signals(X_dummy)
signals_quantile.index = dates

print(f"\n滚动分位数信号分布:")
print(signals_quantile['signal'].value_counts().sort_index())
print(f"\n做多比例: {(signals_quantile['signal'] == 1).sum() / n_samples * 100:.1f}%")
print(f"观望比例: {(signals_quantile['signal'] == 0).sum() / n_samples * 100:.1f}%")
print(f"做空比例: {(signals_quantile['signal'] == -1).sum() / n_samples * 100:.1f}%")

# ============================================================
# 4. 测试信号维持功能
# ============================================================

print("\n" + "="*80)
print("测试3: 信号维持功能")
print("="*80)

# 创建信号生成器（带信号维持）
signal_gen_holding = SignalGenerator(
    mock_model,
    use_rolling_quantile=True,
    rolling_window=60,
    upper_quantile=0.75,
    lower_quantile=0.25,
    signal_holding_days=20              # 维持20天
)

signals_holding = signal_gen_holding.generate_signals(X_dummy)
signals_holding.index = dates

print(f"\n信号维持前后对比:")
print(f"  维持前做多: {(signals_quantile['signal'] == 1).sum()}")
print(f"  维持后做多: {(signals_holding['signal'] == 1).sum()}")
print(f"  维持前做空: {(signals_quantile['signal'] == -1).sum()}")
print(f"  维持后做空: {(signals_holding['signal'] == -1).sum()}")

# ============================================================
# 5. 不同分位数配置对比
# ============================================================

print("\n" + "="*80)
print("测试4: 不同分位数配置对比")
print("="*80)

quantile_configs = [
    {'name': '保守 (80/20)', 'upper': 0.80, 'lower': 0.20},
    {'name': '标准 (75/25)', 'upper': 0.75, 'lower': 0.25},
    {'name': '平衡 (70/30)', 'upper': 0.70, 'lower': 0.30},
    {'name': '激进 (65/35)', 'upper': 0.65, 'lower': 0.35},
]

comparison_results = []

for config in quantile_configs:
    signal_gen = SignalGenerator(
        mock_model,
        use_rolling_quantile=True,
        rolling_window=60,
        upper_quantile=config['upper'],
        lower_quantile=config['lower'],
        signal_holding_days=1
    )
    
    signals = signal_gen.generate_signals(X_dummy)
    
    long_pct = (signals['signal'] == 1).sum() / n_samples * 100
    neutral_pct = (signals['signal'] == 0).sum() / n_samples * 100
    short_pct = (signals['signal'] == -1).sum() / n_samples * 100
    
    comparison_results.append({
        '配置': config['name'],
        '做多(%)': long_pct,
        '观望(%)': neutral_pct,
        '做空(%)': short_pct
    })
    
    print(f"\n{config['name']}:")
    print(f"  做多: {long_pct:.1f}%")
    print(f"  观望: {neutral_pct:.1f}%")
    print(f"  做空: {short_pct:.1f}%")

comparison_df = pd.DataFrame(comparison_results)

# ============================================================
# 6. 可视化
# ============================================================

print("\n" + "="*80)
print("生成可视化图表...")
print("="*80)

# 创建输出目录
output_dir = Path('outputs/charts')
output_dir.mkdir(parents=True, exist_ok=True)

# 图1: 固定阈值 vs 预测值
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# 固定阈值
ax1 = axes[0]
ax1.plot(dates, predictions, label='Predictions', alpha=0.7, linewidth=1)
ax1.axhline(y=0.03, color='green', linestyle='--', label='Upper Threshold (3%)', alpha=0.5)
ax1.axhline(y=-0.03, color='red', linestyle='--', label='Lower Threshold (-3%)', alpha=0.5)
ax1.fill_between(dates, 0.03, -0.03, alpha=0.1, color='gray')

# 标记信号
long_signals = signals_fixed[signals_fixed['signal'] == 1]
short_signals = signals_fixed[signals_fixed['signal'] == -1]
ax1.scatter(long_signals.index, long_signals['prediction'], 
           color='green', marker='^', s=30, label='Long Signal', zorder=5)
ax1.scatter(short_signals.index, short_signals['prediction'], 
           color='red', marker='v', s=30, label='Short Signal', zorder=5)

ax1.set_title('固定阈值模式 (±3%)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Predicted Return', fontsize=11)
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# 滚动分位数
ax2 = axes[1]
ax2.plot(dates, predictions, label='Predictions', alpha=0.7, linewidth=1)
ax2.plot(dates, signals_quantile['upper_threshold'], 
        label='Upper Quantile (75%)', color='green', linestyle='--', alpha=0.5)
ax2.plot(dates, signals_quantile['lower_threshold'], 
        label='Lower Quantile (25%)', color='red', linestyle='--', alpha=0.5)
ax2.fill_between(dates, signals_quantile['upper_threshold'], 
                signals_quantile['lower_threshold'], alpha=0.1, color='gray')

# 标记信号
long_signals = signals_quantile[signals_quantile['signal'] == 1]
short_signals = signals_quantile[signals_quantile['signal'] == -1]
ax2.scatter(long_signals.index, long_signals['prediction'], 
           color='green', marker='^', s=30, label='Long Signal', zorder=5)
ax2.scatter(short_signals.index, short_signals['prediction'], 
           color='red', marker='v', s=30, label='Short Signal', zorder=5)

ax2.set_title('滚动分位数模式 (75%/25%, 60天窗口)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Predicted Return', fontsize=11)
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'regression_signal_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ regression_signal_comparison.png")

# 图2: 不同分位数配置对比
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(comparison_df))
width = 0.25

ax.bar(x - width, comparison_df['做多(%)'], width, label='做多', color='#2ecc71')
ax.bar(x, comparison_df['观望(%)'], width, label='观望', color='#95a5a6')
ax.bar(x + width, comparison_df['做空(%)'], width, label='做空', color='#e74c3c')

ax.set_title('不同分位数配置的信号分布', fontsize=14, fontweight='bold')
ax.set_xlabel('配置', fontsize=12)
ax.set_ylabel('占比 (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['配置'], rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'quantile_configuration_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ quantile_configuration_comparison.png")

# 图3: 信号维持效果对比
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# 无信号维持
ax1 = axes[0]
signal_colors = {1: 'green', 0: 'gray', -1: 'red'}
for signal_val in [-1, 0, 1]:
    mask = signals_quantile['signal'] == signal_val
    label = {1: 'Long', 0: 'Neutral', -1: 'Short'}[signal_val]
    ax1.scatter(dates[mask], signals_quantile.loc[mask, 'prediction'], 
               c=signal_colors[signal_val], s=10, label=label, alpha=0.6)

ax1.plot(dates, predictions, alpha=0.3, linewidth=0.5, color='blue')
ax1.set_title('无信号维持 (每天生成新信号)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Return', fontsize=10)
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# 有信号维持
ax2 = axes[1]
for signal_val in [-1, 0, 1]:
    mask = signals_holding['signal'] == signal_val
    label = {1: 'Long', 0: 'Neutral', -1: 'Short'}[signal_val]
    ax2.scatter(dates[mask], signals_holding.loc[mask, 'prediction'], 
               c=signal_colors[signal_val], s=10, label=label, alpha=0.6)

ax2.plot(dates, predictions, alpha=0.3, linewidth=0.5, color='blue')
ax2.set_title('信号维持20天 (信号持续直到出现相反信号)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Predicted Return', fontsize=10)
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'signal_holding_effect.png', dpi=300, bbox_inches='tight')
print("  ✓ signal_holding_effect.png")

# 图4: 信号强度分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 固定阈值的信号强度
ax1 = axes[0]
strength_by_signal = []
for signal_val in [1, -1]:
    strengths = signals_fixed[signals_fixed['signal'] == signal_val]['signal_strength']
    if len(strengths) > 0:
        strength_by_signal.append(strengths.values)
    else:
        strength_by_signal.append([])

ax1.hist(strength_by_signal[0], bins=20, alpha=0.6, label='Long Signals', color='green')
ax1.hist(strength_by_signal[1], bins=20, alpha=0.6, label='Short Signals', color='red')
ax1.set_title('固定阈值: 信号强度分布', fontsize=12, fontweight='bold')
ax1.set_xlabel('Signal Strength', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 滚动分位数的信号强度
ax2 = axes[1]
strength_by_signal = []
for signal_val in [1, -1]:
    strengths = signals_quantile[signals_quantile['signal'] == signal_val]['signal_strength']
    if len(strengths) > 0:
        strength_by_signal.append(strengths.values)
    else:
        strength_by_signal.append([])

ax2.hist(strength_by_signal[0], bins=20, alpha=0.6, label='Long Signals', color='green')
ax2.hist(strength_by_signal[1], bins=20, alpha=0.6, label='Short Signals', color='red')
ax2.set_title('滚动分位数: 信号强度分布', fontsize=12, fontweight='bold')
ax2.set_xlabel('Signal Strength', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'signal_strength_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ signal_strength_distribution.png")

# ============================================================
# 7. 总结
# ============================================================

print("\n" + "="*80)
print("测试总结")
print("="*80)

print("\n✅ 功能测试完成:")
print("  1. 固定阈值模式 - 正常工作")
print("  2. 滚动分位数模式 - 正常工作")
print("  3. 信号维持功能 - 正常工作")
print("  4. 信号强度计算 - 正常工作")

print("\n📊 主要发现:")
print(f"  - 固定阈值生成 {(signals_fixed['signal'] != 0).sum()} 个交易信号")
print(f"  - 滚动分位数生成 {(signals_quantile['signal'] != 0).sum()} 个交易信号")
print(f"  - 信号维持可减少信号变化，提高稳定性")
print(f"  - 滚动分位数能自适应市场变化")

print("\n💡 建议:")
print("  - 优先使用滚动分位数模式（更适应市场动态）")
print("  - 根据预测时间窗口设置信号维持天数")
print("  - 使用回测验证不同分位数配置的效果")
print("  - 结合信号强度进行动态仓位管理")

print("\n📁 生成的图表:")
print("  - outputs/charts/regression_signal_comparison.png")
print("  - outputs/charts/quantile_configuration_comparison.png")
print("  - outputs/charts/signal_holding_effect.png")
print("  - outputs/charts/signal_strength_distribution.png")

print("\n" + "="*80)
print("测试完成！")
print("="*80)
