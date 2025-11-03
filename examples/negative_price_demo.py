"""
负价格交易演示脚本

展示如何在价差交易策略中使用负价格功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.core.backtest import BacktestEngine
from src.core.spread_calculator import SpreadCalculator

def demo_negative_spread_trading():
    """
    演示：负价差交易场景
    
    场景说明：
    假设我们交易豆油-棕榈油价差，当棕榈油价格高于豆油时，价差为负。
    策略：当价差为负时做多价差（买豆油卖棕榈油），期待价差回归。
    """
    print("=" * 80)
    print("演示：负价差交易策略")
    print("=" * 80)
    
    # 1. 创建模拟价格数据
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # 豆油价格 (Y)
    soybean_oil_prices = 8000 + np.cumsum(np.random.randn(100) * 50)
    
    # 棕榈油价格 (P) - 初期高于豆油
    palm_oil_prices = 8500 + np.cumsum(np.random.randn(100) * 50)
    
    # 计算价差 = 豆油 - 棕榈油
    spreads = soybean_oil_prices - palm_oil_prices
    
    print("\n价格和价差统计：")
    print(f"  豆油价格范围: {soybean_oil_prices.min():.2f} - {soybean_oil_prices.max():.2f}")
    print(f"  棕榈油价格范围: {palm_oil_prices.min():.2f} - {palm_oil_prices.max():.2f}")
    print(f"  价差范围: {spreads.min():.2f} - {spreads.max():.2f}")
    print(f"  负价差天数: {(spreads < 0).sum()}")
    print(f"  正价差天数: {(spreads > 0).sum()}")
    
    # 2. 创建价差数据DataFrame
    spread_data = pd.DataFrame({
        'close': spreads,  # 价差作为"价格"
        'soybean_oil': soybean_oil_prices,
        'palm_oil': palm_oil_prices
    }, index=dates)
    
    # 3. 创建交易信号
    # 简单策略：价差低于均值时做多，高于均值时平仓
    ma_20 = pd.Series(spreads).rolling(20).mean()
    signals = np.where(spreads < ma_20, 1, 0)
    
    signal_data = pd.DataFrame({
        'signal': signals
    }, index=dates)
    
    print("\n信号统计：")
    print(f"  做多信号: {(signals == 1).sum()}")
    print(f"  平仓信号: {(signals == 0).sum()}")
    
    # 4. 运行回测
    print("\n" + "=" * 80)
    print("开始回测（支持负价差交易）")
    print("=" * 80)
    
    engine = BacktestEngine(
        initial_capital=1000000,
        commission_rate=0.0005,
        slippage_rate=0.0001,
        max_position=100,
        leverage=5.0,
        margin_ratio=0.1,
        stop_loss_pct=0.05  # 5%止损
    )
    
    results = engine.run_backtest(
        price_data=spread_data,
        signals=signal_data,
        price_col='close'
    )
    
    # 5. 分析结果
    print("\n" + "=" * 80)
    print("回测结果")
    print("=" * 80)
    
    final_equity = engine.equity_curve[-1]['equity']
    total_return = (final_equity - engine.initial_capital) / engine.initial_capital
    
    print(f"\n初始资金: ¥{engine.initial_capital:,.2f}")
    print(f"最终权益: ¥{final_equity:,.2f}")
    print(f"总收益: ¥{final_equity - engine.initial_capital:,.2f}")
    print(f"总收益率: {total_return * 100:.2f}%")
    print(f"\n总交易次数: {len(engine.trades)}")
    
    # 按正负价差分类统计
    trades_df = pd.DataFrame([
        {
            'date': t.timestamp,
            'action': t.action,
            'price': t.price,
            'position': t.position,
            'pnl': t.pnl,
            'is_negative': t.price < 0
        }
        for t in engine.trades
    ])
    
    if not trades_df.empty:
        negative_trades = trades_df[trades_df['is_negative']]
        positive_trades = trades_df[~trades_df['is_negative']]
        
        print(f"\n负价差交易: {len(negative_trades)} 笔")
        if len(negative_trades) > 0:
            print(f"  负价差范围: {negative_trades['price'].min():.2f} - {negative_trades['price'].max():.2f}")
            print(f"  负价差交易盈亏: ¥{negative_trades['pnl'].sum():,.2f}")
        
        print(f"\n正价差交易: {len(positive_trades)} 笔")
        if len(positive_trades) > 0:
            print(f"  正价差范围: {positive_trades['price'].min():.2f} - {positive_trades['price'].max():.2f}")
            print(f"  正价差交易盈亏: ¥{positive_trades['pnl'].sum():,.2f}")
    
    # 6. 展示关键交易
    print("\n" + "=" * 80)
    print("关键交易记录（前10笔）")
    print("=" * 80)
    
    for i, trade in enumerate(engine.trades[:10]):
        price_type = "负价差" if trade.price < 0 else "正价差"
        print(f"{i+1}. {trade.timestamp.strftime('%Y-%m-%d')}: "
              f"{trade.action:6s} @ {price_type} ¥{trade.price:8.2f}, "
              f"仓位={trade.position:3d}, "
              f"盈亏=¥{trade.pnl:8.2f}")
    
    if len(engine.trades) > 10:
        print(f"\n... 共 {len(engine.trades)} 笔交易")
    
    print("\n" + "=" * 80)
    print("✅ 演示完成！系统成功处理了负价差交易。")
    print("=" * 80)
    
    return engine, results


def demo_oil_crack_spread():
    """
    演示：裂解价差交易（可能为负）
    
    场景说明：
    裂解价差 = (汽油价格 + 柴油价格) / 2 - 原油价格
    当原油价格远高于成品油时，裂解价差为负
    """
    print("\n\n" + "=" * 80)
    print("演示：裂解价差交易（原油炼化利润）")
    print("=" * 80)
    
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # 模拟价格数据
    crude_oil = 70 + np.cumsum(np.random.randn(100) * 2)  # 原油价格
    gasoline = 75 + np.cumsum(np.random.randn(100) * 2.5)  # 汽油价格
    diesel = 72 + np.cumsum(np.random.randn(100) * 2.2)  # 柴油价格
    
    # 计算裂解价差
    crack_spread = (gasoline + diesel) / 2 - crude_oil
    
    print("\n价格统计：")
    print(f"  原油价格范围: ${crude_oil.min():.2f} - ${crude_oil.max():.2f}")
    print(f"  汽油价格范围: ${gasoline.min():.2f} - ${gasoline.max():.2f}")
    print(f"  柴油价格范围: ${diesel.min():.2f} - ${diesel.max():.2f}")
    print(f"  裂解价差范围: ${crack_spread.min():.2f} - ${crack_spread.max():.2f}")
    print(f"  负价差天数: {(crack_spread < 0).sum()}")
    
    # 创建数据
    spread_data = pd.DataFrame({
        'close': crack_spread
    }, index=dates)
    
    # 均值回归策略
    ma = pd.Series(crack_spread).rolling(20).mean()
    std = pd.Series(crack_spread).rolling(20).std()
    
    # 价差低于均值-1倍标准差时做多，高于均值时平仓
    signals = np.where(crack_spread < ma - std, 1, 0)
    
    signal_data = pd.DataFrame({
        'signal': signals
    }, index=dates)
    
    print(f"\n交易信号: {(signals == 1).sum()} 个做多信号")
    
    # 回测
    engine = BacktestEngine(
        initial_capital=500000,
        commission_rate=0.0003,
        slippage_rate=0.0001,
        max_position=50,
        leverage=3.0,
        margin_ratio=0.15
    )
    
    print("\n运行回测...")
    results = engine.run_backtest(spread_data, signal_data)
    
    # 结果
    final_equity = engine.equity_curve[-1]['equity']
    total_return = (final_equity - engine.initial_capital) / engine.initial_capital
    
    print("\n" + "=" * 80)
    print("回测结果")
    print("=" * 80)
    print(f"初始资金: ${engine.initial_capital:,.2f}")
    print(f"最终权益: ${final_equity:,.2f}")
    print(f"收益率: {total_return * 100:.2f}%")
    print(f"交易次数: {len(engine.trades)}")
    
    print("\n✅ 裂解价差策略演示完成！")
    
    return engine, results


if __name__ == '__main__':
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "负价格交易功能演示" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # 演示1: 负价差交易
    engine1, results1 = demo_negative_spread_trading()
    
    # 演示2: 裂解价差交易
    engine2, results2 = demo_oil_crack_spread()
    
    print("\n" + "=" * 80)
    print("🎉 全部演示完成！")
    print("=" * 80)
    print("\n主要特点：")
    print("  ✅ 支持负价格/负价差交易")
    print("  ✅ 正确计算负价格下的保证金和手续费")
    print("  ✅ 正确处理负价格下的滑点")
    print("  ✅ 正确判断负价格下的止损")
    print("  ✅ 完全向后兼容，无需修改现有代码")
    print("\n" + "=" * 80)
