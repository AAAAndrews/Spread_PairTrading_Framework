"""
EIA数据在原油裂解价差套利中的应用示例
展示如何将EIA库存数据作为基本面因子整合到交易策略中
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data_fetcher import DataFetcher
import pandas as pd
import matplotlib.pyplot as plt
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """主函数"""
    
    print("=" * 70)
    print("EIA数据在原油裂解价差套利中的应用")
    print("=" * 70)
    
    # 创建数据获取器
    fetcher = DataFetcher()
    
    # 1. 获取库存数据
    print("\n【步骤1】获取EIA库存数据...")
    print("-" * 70)
    
    # 原油库存
    print("  ➤ 获取美国原油库存...")
    crude_stock = fetcher.fetch_eia_data('PET.WCRSTUS1.W')
    
    # 汽油库存
    print("  ➤ 获取美国汽油库存...")
    gasoline_stock = fetcher.fetch_eia_data('PET.WGTSTUS1.W')
    
    # 馏分油库存
    print("  ➤ 获取美国馏分油库存...")
    distillate_stock = fetcher.fetch_eia_data('PET.WDISTUS1.W')
    
    if crude_stock.empty or gasoline_stock.empty or distillate_stock.empty:
        print("\n✗ 数据获取失败，请检查网络连接和API密钥")
        return
    
    print(f"\n✓ 数据获取成功!")
    print(f"  原油库存: {len(crude_stock)} 条数据")
    print(f"  汽油库存: {len(gasoline_stock)} 条数据")
    print(f"  馏分油库存: {len(distillate_stock)} 条数据")
    
    # 2. 合并数据
    print("\n【步骤2】数据合并与预处理...")
    print("-" * 70)
    
    df = pd.DataFrame({
        'crude_stock': crude_stock['value'],
        'gasoline_stock': gasoline_stock['value'],
        'distillate_stock': distillate_stock['value']
    })
    
    # 删除缺失值
    df.dropna(inplace=True)
    
    print(f"  合并后数据: {len(df)} 条")
    print(f"  时间范围: {df.index.min()} 至 {df.index.max()}")
    
    # 3. 计算派生指标
    print("\n【步骤3】计算基本面指标...")
    print("-" * 70)
    
    # 计算库存变化率
    df['crude_stock_chg'] = df['crude_stock'].pct_change()
    df['gasoline_stock_chg'] = df['gasoline_stock'].pct_change()
    df['distillate_stock_chg'] = df['distillate_stock'].pct_change()
    
    # 计算库存Z分数（标准化）
    df['crude_stock_zscore'] = (df['crude_stock'] - df['crude_stock'].rolling(52).mean()) / df['crude_stock'].rolling(52).std()
    df['gasoline_stock_zscore'] = (df['gasoline_stock'] - df['gasoline_stock'].rolling(52).mean()) / df['gasoline_stock'].rolling(52).std()
    
    # 计算裂解库存比率
    df['crack_stock_ratio'] = (df['gasoline_stock'] + df['distillate_stock']) / df['crude_stock']
    
    print("  ✓ 计算了以下指标:")
    print("    - 库存变化率")
    print("    - 库存Z分数（标准化）")
    print("    - 裂解库存比率")
    
    # 4. 显示最新数据
    print("\n【步骤4】最新基本面数据...")
    print("-" * 70)
    
    latest = df.iloc[-1]
    print(f"\n  日期: {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"\n  库存水平:")
    print(f"    原油库存: {latest['crude_stock']:,.0f} 千桶")
    print(f"    汽油库存: {latest['gasoline_stock']:,.0f} 千桶")
    print(f"    馏分油库存: {latest['distillate_stock']:,.0f} 千桶")
    print(f"\n  库存Z分数（相对1年均值）:")
    print(f"    原油: {latest['crude_stock_zscore']:.2f}")
    print(f"    汽油: {latest['gasoline_stock_zscore']:.2f}")
    print(f"\n  裂解库存比率: {latest['crack_stock_ratio']:.3f}")
    
    # 5. 生成交易信号逻辑示例
    print("\n【步骤5】基本面交易信号示例...")
    print("-" * 70)
    
    # 简单信号规则
    signal = None
    if latest['crude_stock_zscore'] > 1.5:
        signal = "做空裂解价差（原油库存高，成品油相对强）"
    elif latest['crude_stock_zscore'] < -1.5:
        signal = "做多裂解价差（原油库存低，成品油相对弱）"
    elif latest['crack_stock_ratio'] > df['crack_stock_ratio'].quantile(0.8):
        signal = "做空裂解价差（成品油库存相对原油过高）"
    elif latest['crack_stock_ratio'] < df['crack_stock_ratio'].quantile(0.2):
        signal = "做多裂解价差（成品油库存相对原油过低）"
    else:
        signal = "观望（基本面中性）"
    
    print(f"\n  基于当前基本面的建议: {signal}")
    
    # 6. 可视化
    print("\n【步骤6】生成可视化图表...")
    print("-" * 70)
    
    # 获取最近2年的数据用于可视化
    df_recent = df.last('730D')
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 图1：库存水平
    ax1 = axes[0]
    ax1.plot(df_recent.index, df_recent['crude_stock'], label='原油库存', linewidth=2)
    ax1.plot(df_recent.index, df_recent['gasoline_stock'], label='汽油库存', linewidth=2)
    ax1.plot(df_recent.index, df_recent['distillate_stock'], label='馏分油库存', linewidth=2)
    ax1.set_title('美国石油库存水平', fontsize=14, fontweight='bold')
    ax1.set_ylabel('库存（千桶）', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 图2：库存Z分数
    ax2 = axes[1]
    ax2.plot(df_recent.index, df_recent['crude_stock_zscore'], label='原油库存Z分数', linewidth=2)
    ax2.plot(df_recent.index, df_recent['gasoline_stock_zscore'], label='汽油库存Z分数', linewidth=2)
    ax2.axhline(y=1.5, color='r', linestyle='--', alpha=0.5, label='超买/超卖阈值')
    ax2.axhline(y=-1.5, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title('库存Z分数（标准化）', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Z分数', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 图3：裂解库存比率
    ax3 = axes[2]
    ax3.plot(df_recent.index, df_recent['crack_stock_ratio'], label='裂解库存比率', linewidth=2, color='purple')
    ax3.axhline(y=df_recent['crack_stock_ratio'].quantile(0.8), color='r', linestyle='--', alpha=0.5, label='80%分位')
    ax3.axhline(y=df_recent['crack_stock_ratio'].quantile(0.2), color='g', linestyle='--', alpha=0.5, label='20%分位')
    ax3.set_title('裂解库存比率', fontsize=14, fontweight='bold')
    ax3.set_xlabel('日期', fontsize=12)
    ax3.set_ylabel('比率', fontsize=12)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'eia_fundamental_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ 图表已保存至: {output_path}")
    
    # 显示图表
    plt.show()
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    print("\n💡 使用建议:")
    print("  1. 将EIA库存数据作为基本面因子添加到特征工程中")
    print("  2. 库存Z分数可以作为价差强度的验证信号")
    print("  3. 裂解库存比率可以预示供需关系变化")
    print("  4. 结合价格数据和库存数据可以提高策略胜率")
    print("=" * 70)


if __name__ == "__main__":
    main()
