"""
可视化模块
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style('whitegrid')


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, figsize=(12, 6), output_dir='outputs/charts'):
        """
        初始化可视化器
        
        Args:
            figsize: 默认图表大小
            output_dir: 图表输出目录
        """
        self.figsize = figsize
        self.output_dir = output_dir
        
        # 创建输出目录
        from pathlib import Path
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def plot_price_and_spread(self, price_data: Dict[str, pd.DataFrame],
                             spread_data: pd.DataFrame,
                             title: str = "Price and Spread"):
        """
        绘制价格和价差走势图
        
        Args:
            price_data: 价格数据字典
            spread_data: 价差数据
            title: 图表标题
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 绘制价格
        ax1 = axes[0]
        for symbol, df in price_data.items():
            if 'close' in df.columns:
                ax1.plot(df.index, df['close'], label=symbol, alpha=0.7)
        
        ax1.set_title(f'{title} - Price')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制价差
        ax2 = axes[1]
        if 'spread' in spread_data.columns:
            ax2.plot(spread_data.index, spread_data['spread'], color='purple', linewidth=1.5)
            
            # 添加均值线
            if 'spread_ma_20' in spread_data.columns:
                ax2.plot(spread_data.index, spread_data['spread_ma_20'], 
                        color='orange', linestyle='--', alpha=0.7, label='MA20')
            
            # 添加标准差带
            if 'spread_ma_20' in spread_data.columns and 'spread_std_20' in spread_data.columns:
                upper = spread_data['spread_ma_20'] + 2 * spread_data['spread_std_20']
                lower = spread_data['spread_ma_20'] - 2 * spread_data['spread_std_20']
                ax2.fill_between(spread_data.index, upper, lower, alpha=0.2, color='gray')
        
        ax2.set_title(f'{title} - Spread')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Spread Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/price_spread_chart.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"价格和价差图表已保存: {output_path}")
        plt.close()
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, top_n: int = 20):
        """
        绘制特征重要性图
        
        Args:
            feature_importance: 特征重要性DataFrame
            top_n: 显示前N个特征
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = feature_importance.head(top_n)
        
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/feature_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"特征重要性图表已保存: {output_path}")
        plt.close()
    
    def plot_prediction_vs_actual(self, y_true, y_pred, 
                                  timestamps: Optional[pd.DatetimeIndex] = None):
        """
        绘制预测值vs实际值
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            timestamps: 时间戳
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if timestamps is not None:
            ax.plot(timestamps, y_true, label='Actual', alpha=0.7)
            ax.plot(timestamps, y_pred, label='Predicted', alpha=0.7)
            ax.set_xlabel('Date')
        else:
            ax.plot(y_true, label='Actual', alpha=0.7)
            ax.plot(y_pred, label='Predicted', alpha=0.7)
            ax.set_xlabel('Sample')
        
        ax.set_ylabel('Value')
        ax.set_title('Prediction vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/prediction_vs_actual.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"预测对比图表已保存: {output_path}")
        plt.close()
    
    def plot_equity_curve(self, equity_curve: pd.DataFrame):
        """
        绘制权益曲线
        
        Args:
            equity_curve: 权益曲线DataFrame
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 权益曲线
        ax1 = axes[0]
        ax1.plot(equity_curve.index, equity_curve['equity'], 
                linewidth=2, color='blue', label='Portfolio Value')
        ax1.axhline(y=equity_curve['equity'].iloc[0], 
                   color='red', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.fill_between(equity_curve.index, equity_curve['equity'].iloc[0], 
                        equity_curve['equity'], alpha=0.3)
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 回撤
        ax2 = axes[1]
        cummax = equity_curve['equity'].cummax()
        drawdown = (equity_curve['equity'] - cummax) / cummax * 100
        ax2.fill_between(equity_curve.index, 0, drawdown, 
                        color='red', alpha=0.3)
        ax2.plot(equity_curve.index, drawdown, color='red', linewidth=1)
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/equity_curve.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"权益曲线图表已保存: {output_path}")
        plt.close()
    
    def plot_returns_distribution(self, returns: pd.Series):
        """
        绘制收益率分布
        
        Args:
            returns: 收益率序列
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图
        ax1 = axes[0]
        returns.hist(bins=50, ax=ax1, edgecolor='black', alpha=0.7)
        ax1.axvline(returns.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {returns.mean():.4f}')
        ax1.axvline(returns.median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {returns.median():.4f}')
        ax1.set_title('Returns Distribution')
        ax1.set_xlabel('Returns')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # QQ图（检验正态性）
        ax2 = axes[1]
        from scipy import stats
        stats.probplot(returns.dropna(), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/returns_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"收益率分布图表已保存: {output_path}")
        plt.close()
    
    def plot_monthly_returns_heatmap(self, equity_curve: pd.DataFrame):
        """
        绘制月度收益热力图
        
        Args:
            equity_curve: 权益曲线DataFrame
        """
        # 计算月度收益率
        monthly = equity_curve['equity'].resample('M').last()
        monthly_returns = monthly.pct_change() * 100
        
        # 重塑为年-月矩阵
        monthly_returns_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        pivot = monthly_returns_df.pivot(index='year', columns='month', values='return')
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': 'Return (%)'})
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/monthly_returns_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"月度收益热力图已保存: {output_path}")
        plt.close()
    
    def plot_rolling_metrics(self, equity_curve: pd.DataFrame, window: int = 252):
        """
        绘制滚动指标
        
        Args:
            equity_curve: 权益曲线DataFrame
            window: 滚动窗口（天数）
        """
        returns = equity_curve['equity'].pct_change()
        
        # 计算滚动指标
        rolling_return = returns.rolling(window).mean() * 252 * 100
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
        rolling_sharpe = rolling_return / rolling_vol
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 滚动收益率
        ax1 = axes[0]
        ax1.plot(rolling_return.index, rolling_return, linewidth=2)
        ax1.set_title(f'{window}-Day Rolling Annual Return (%)')
        ax1.set_ylabel('Return (%)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 滚动波动率
        ax2 = axes[1]
        ax2.plot(rolling_vol.index, rolling_vol, linewidth=2, color='orange')
        ax2.set_title(f'{window}-Day Rolling Volatility (%)')
        ax2.set_ylabel('Volatility (%)')
        ax2.grid(True, alpha=0.3)
        
        # 滚动夏普比率
        ax3 = axes[2]
        ax3.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2, color='green')
        ax3.set_title(f'{window}-Day Rolling Sharpe Ratio')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/rolling_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"滚动指标图表已保存: {output_path}")
        plt.close()
    
    def plot_trade_analysis(self, trade_log: pd.DataFrame):
        """
        绘制交易分析图
        
        Args:
            trade_log: 交易日志DataFrame
        """
        if trade_log.empty:
            logger.warning("交易日志为空，无法绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 累计PnL
        ax1 = axes[0, 0]
        cumulative_pnl = trade_log['pnl'].cumsum()
        ax1.plot(trade_log['timestamp'], cumulative_pnl, linewidth=2)
        ax1.set_title('Cumulative PnL')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('PnL ($)')
        ax1.grid(True, alpha=0.3)
        
        # PnL分布
        ax2 = axes[0, 1]
        trade_log['pnl'].hist(bins=30, ax=ax2, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('PnL Distribution')
        ax2.set_xlabel('PnL ($)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 持仓变化
        ax3 = axes[1, 0]
        ax3.plot(trade_log['timestamp'], trade_log['position'], 
                linewidth=2, marker='o', markersize=3)
        ax3.set_title('Position Over Time')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Position')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 累计手续费
        ax4 = axes[1, 1]
        cumulative_commission = trade_log['commission'].cumsum()
        ax4.plot(trade_log['timestamp'], cumulative_commission, 
                linewidth=2, color='red')
        ax4.set_title('Cumulative Commission')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Commission ($)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/trade_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"交易分析图表已保存: {output_path}")
        plt.close()
