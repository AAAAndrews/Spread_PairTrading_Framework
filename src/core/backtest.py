"""
回测系统模块
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """交易记录"""
    timestamp: datetime
    action: str  # 'open' or 'close'
    position: int  # 1: long, -1: short, 0: flat
    price: float
    quantity: int
    pnl: float = 0.0
    commission: float = 0.0


class BacktestEngine:
    """回测引擎（支持期货杠杆）"""
    
    def __init__(self, initial_capital: float = 1000000,
                 commission_rate: float = 0.0005,
                 slippage_rate: float = 0.0001,
                 max_position: int = 100,
                 leverage: float = 10.0,
                 margin_ratio: float = 0.1,
                 max_capital_usage: float = 0.8,
                 stop_loss_pct: float = 0.0,
                 stop_loss_amount: float = 0.0):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率（双边）
            slippage_rate: 滑点率
            max_position: 最大持仓（合约数量限制）
            leverage: 杠杆倍数（如10表示10倍杠杆）
            margin_ratio: 保证金比例（如0.1表示10%保证金）
            max_capital_usage: 最大资金使用率（如0.8表示最多使用80%资金开仓）
            stop_loss_pct: 单笔交易止损百分比（如0.05表示5%，0表示不启用）
            stop_loss_amount: 单笔交易止损金额（如5000表示亏损5000元止损，0表示不启用）
        
        杠杆说明：
            - leverage=1: 无杠杆，相当于股票交易
            - leverage=10: 10倍杠杆，10万资金可控制100万价值的合约
            - 实际占用保证金 = 合约价值 × margin_ratio
            - 可开仓价值 = 可用资金 × leverage
        
        止损说明：
            - 同时设置百分比和金额时，先触发哪个就执行哪个
            - stop_loss_pct: 基于开仓价的跌幅，如5%表示价格下跌5%触发止损
            - stop_loss_amount: 基于绝对金额的亏损，如5000表示亏损5000元触发止损
            - 止损仅对持仓生效，开仓时重置止损计数
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position = max_position
        
        # ✅ 杠杆相关参数
        self.leverage = leverage
        self.margin_ratio = margin_ratio
        self.max_capital_usage = max_capital_usage
        
        # ✅ 止损相关参数
        self.stop_loss_pct = stop_loss_pct
        self.stop_loss_amount = stop_loss_amount
        self.stop_loss_triggered_count = 0  # 止损触发次数统计
        
        # 回测状态
        self.capital = initial_capital
        self.available_capital = initial_capital  # ✅ 可用资金（未占用的保证金）
        self.margin_used = 0.0  # ✅ 已占用保证金
        self.position = 0  # 当前持仓
        self.entry_price = 0.0  # ✅ 开仓价格（用于计算浮动盈亏）
        self.trades = []  # 交易记录
        self.equity_curve = []  # 权益曲线
        self.returns = []  # 收益率
        
    def reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.available_capital = self.initial_capital
        self.margin_used = 0.0
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss_triggered_count = 0
        self.trades = []
        self.equity_curve = []
        self.returns = []
    
    def calculate_slippage(self, price: float, action: str) -> float:
        """
        计算滑点
        
        Args:
            price: 价格
            action: 动作（buy/sell）
        
        Returns:
            考虑滑点后的价格
        """
        if action == 'buy':
            return price * (1 + self.slippage_rate)
        else:  # sell
            return price * (1 - self.slippage_rate)
    
    def calculate_commission(self, price: float, quantity: int) -> float:
        """计算手续费"""
        return abs(price * quantity * self.commission_rate)
    
    def calculate_margin_required(self, price: float, quantity: int) -> float:
        """
        计算所需保证金
        
        Args:
            price: 合约价格
            quantity: 合约数量（绝对值）
        
        Returns:
            所需保证金金额
        
        公式：
            保证金 = |合约价值| × 保证金比例
            合约价值 = 价格 × 数量
        """
        contract_value = abs(price * quantity)
        margin = contract_value * self.margin_ratio
        return margin
    
    def check_stop_loss(self, current_price: float) -> bool:
        """
        检查是否触发止损
        
        Args:
            current_price: 当前价格
        
        Returns:
            是否触发止损
        
        止损逻辑：
            1. 如果没有持仓，返回False
            2. 计算当前浮动盈亏
            3. 检查百分比止损（如果设置）
            4. 检查金额止损（如果设置）
            5. 任一条件触发则返回True
        """
        # 没有持仓，无需止损
        if self.position == 0 or self.entry_price == 0:
            return False
        
        # 没有设置止损参数，不启用止损
        if self.stop_loss_pct <= 0 and self.stop_loss_amount <= 0:
            return False
        
        # 计算当前浮动盈亏
        if self.position > 0:  # 多头
            unrealized_pnl = (current_price - self.entry_price) * self.position
            price_change_pct = (current_price - self.entry_price) / self.entry_price
        else:  # 空头
            unrealized_pnl = (self.entry_price - current_price) * abs(self.position)
            price_change_pct = (self.entry_price - current_price) / self.entry_price
        
        # 检查百分比止损
        if self.stop_loss_pct > 0:
            if price_change_pct <= -self.stop_loss_pct:
                logger.warning(
                    f"触发百分比止损！当前亏损: {price_change_pct*100:.2f}% "
                    f"(止损线: {self.stop_loss_pct*100:.2f}%), "
                    f"浮亏: ${unrealized_pnl:,.2f}"
                )
                return True
        
        # 检查金额止损
        if self.stop_loss_amount > 0:
            if unrealized_pnl <= -self.stop_loss_amount:
                logger.warning(
                    f"触发金额止损！当前亏损: ${unrealized_pnl:,.2f} "
                    f"(止损线: ${self.stop_loss_amount:,.2f})"
                )
                return True
        
        return False
    
    def calculate_max_position_size(self, price: float) -> int:
        """
        根据杠杆和保证金计算最大可开仓位
        
        Args:
            price: 当前价格
        
        Returns:
            最大可开仓位（合约数）
        
        计算逻辑：
            1. 可用于开仓资金 = 可用资金 × 最大资金使用率
            2. 杠杆放大后可控制价值 = 可用资金 × 杠杆倍数
            3. 最大合约数 = 可控制价值 / 单价
            4. 同时受 max_position 限制
        
        示例：
            可用资金100万，杠杆10倍，使用率80%，原油价格70美元/桶
            → 可用于开仓 = 100万 × 0.8 = 80万
            → 杠杆放大 = 80万 × 10 = 800万
            → 最大合约 = 800万 / 70 = 114,285手
            → 如果max_position=100，则实际最大100手
            → 所需保证金 = 100 × 70 × 0.1 = 700美元
        """
        if price <= 0:
            return 0
        
        # 可用于开仓的资金
        available_for_trading = self.available_capital * self.max_capital_usage
        
        if available_for_trading <= 0:
            return 0
        
        # 考虑杠杆：实际可控制价值 = 可用资金 × 杠杆倍数
        max_contract_value = available_for_trading * self.leverage
        
        # 转换为合约数量
        max_contracts = int(max_contract_value / price)
        
        # 不超过持仓限制
        return min(max_contracts, self.max_position)
    
    def execute_trade(self, timestamp: datetime, signal: int, 
                     price: float, volatility: float = 0.01):
        """
        执行交易（支持杠杆）
        
        Args:
            timestamp: 时间戳
            signal: 信号（1: 做多, -1: 做空, 0: 平仓）
            price: 价格
            volatility: 波动率（用于计算仓位）
        
        杠杆交易流程：
            1. 根据信号和杠杆计算目标仓位
            2. 检查保证金是否充足
            3. 执行交易并更新保证金占用
            4. 记录盈亏
        """
        # ✅ 1. 根据杠杆计算目标仓位
        if signal != 0:
            max_size = self.calculate_max_position_size(price)
            target_position = signal * max_size
        else:
            target_position = 0
        
        # 根据波动率调整仓位（波动率越大，仓位越小）
        if target_position != 0:
            risk_adjusted_position = int(target_position / (1 + volatility * 10))
            risk_adjusted_position = np.clip(risk_adjusted_position, -self.max_position, self.max_position)
        else:
            risk_adjusted_position = 0
        
        # 计算需要调整的仓位
        position_change = risk_adjusted_position - self.position
        
        if position_change == 0:
            return
        
        # 执行交易
        action = 'buy' if position_change > 0 else 'sell'
        execution_price = self.calculate_slippage(price, action)
        commission = self.calculate_commission(execution_price, abs(position_change))
        
        # ✅ 2. 计算所需保证金
        margin_for_change = self.calculate_margin_required(execution_price, abs(position_change))
        
        # ✅ 3. 检查保证金是否充足（开仓或加仓时）
        if position_change != 0 and np.sign(position_change) == np.sign(risk_adjusted_position):
            # 开仓或加仓
            if margin_for_change + commission > self.available_capital:
                logger.warning(
                    f"{timestamp}: 保证金不足！"
                    f"需要 ${margin_for_change + commission:,.2f}，"
                    f"可用 ${self.available_capital:,.2f}"
                )
                return  # ❌ 保证金不足，放弃交易
        
        # ✅ 4. 计算盈亏并更新保证金
        pnl = 0.0
        
        if self.position == 0:
            # 开新仓
            self.margin_used += margin_for_change
            self.available_capital -= (margin_for_change + commission)
            self.entry_price = execution_price
            
        elif risk_adjusted_position == 0:
            # 全部平仓
            if self.position > 0:  # 平多
                pnl = (execution_price - self.entry_price) * abs(self.position)
            else:  # 平空
                pnl = (self.entry_price - execution_price) * abs(self.position)
            
            # 释放所有保证金
            self.available_capital += (self.margin_used + pnl - commission)
            self.margin_used = 0.0
            self.entry_price = 0.0
            
        elif np.sign(position_change) != np.sign(self.position):
            # 部分平仓或反向开仓
            closed_position = min(abs(position_change), abs(self.position))
            
            if self.position > 0:  # 平多
                pnl = (execution_price - self.entry_price) * closed_position
            else:  # 平空
                pnl = (self.entry_price - execution_price) * closed_position
            
            # 释放部分保证金
            released_margin = self.calculate_margin_required(self.entry_price, closed_position)
            self.margin_used -= released_margin
            self.available_capital += (released_margin + pnl - commission)
            
            # 如果是反向开仓
            if abs(risk_adjusted_position) > 0:
                remaining_change = abs(position_change) - closed_position
                if remaining_change > 0:
                    new_margin = self.calculate_margin_required(execution_price, remaining_change)
                    self.margin_used += new_margin
                    self.available_capital -= new_margin
                    self.entry_price = execution_price
        else:
            # 加仓
            # 重新计算平均开仓价
            total_value = self.entry_price * abs(self.position) + execution_price * abs(position_change)
            self.entry_price = total_value / abs(risk_adjusted_position)
            
            self.margin_used += margin_for_change
            self.available_capital -= (margin_for_change + commission)
        
        # 记录交易
        trade = TradeRecord(
            timestamp=timestamp,
            action='open' if self.position == 0 else ('close' if risk_adjusted_position == 0 else 'adjust'),
            position=risk_adjusted_position,
            price=execution_price,
            quantity=abs(position_change),
            pnl=pnl,
            commission=commission
        )
        self.trades.append(trade)
        
        # 更新持仓
        self.position = risk_adjusted_position
        
        # 日志记录
        logger.debug(
            f"{timestamp}: {trade.action} {abs(position_change)}手 @ ${execution_price:.2f}, "
            f"持仓: {self.position}, 保证金占用: ${self.margin_used:,.2f}, "
            f"可用资金: ${self.available_capital:,.2f}, PnL: ${pnl:,.2f}"
        )
    
    def update_equity(self, timestamp: datetime, current_price: float):
        """
        更新权益（考虑杠杆浮动盈亏）
        
        Args:
            timestamp: 时间戳
            current_price: 当前价格
        
        杠杆权益计算：
            总权益 = 可用资金 + 占用保证金 + 浮动盈亏
            浮动盈亏 = (当前价 - 开仓价) × 持仓数量（多头）
                      或 (开仓价 - 当前价) × |持仓数量|（空头）
        """
        # ✅ 计算持仓浮动盈亏（杠杆模式）
        if self.position != 0 and self.entry_price > 0:
            if self.position > 0:  # 多头
                unrealized_pnl = (current_price - self.entry_price) * self.position
            else:  # 空头
                unrealized_pnl = (self.entry_price - current_price) * abs(self.position)
        else:
            unrealized_pnl = 0.0
        
        # ✅ 总权益 = 可用资金 + 占用保证金 + 浮动盈亏
        total_equity = self.available_capital + self.margin_used + unrealized_pnl
        
        # 保证金占用率
        margin_usage_rate = self.margin_used / self.initial_capital if self.initial_capital > 0 else 0.0
        
        # 记录权益
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'available_capital': self.available_capital,
            'margin_used': self.margin_used,
            'unrealized_pnl': unrealized_pnl,
            'margin_usage_rate': margin_usage_rate,
            'position': self.position,
            'leverage_ratio': abs(self.position * current_price) / total_equity if total_equity > 0 else 0.0
        })
        
        # 计算收益率
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]['equity']
            ret = (total_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            self.returns.append(ret)
        else:
            self.returns.append(0.0)
    
    def run_backtest(self, price_data: pd.DataFrame, signals: pd.DataFrame,
                    price_col: str = 'close', volatility_col: Optional[str] = None) -> pd.DataFrame:
        """
        运行回测
        
        Args:
            price_data: 价格数据DataFrame
            signals: 信号DataFrame（必须包含'signal'列）
            price_col: 价格列名
            volatility_col: 波动率列名（可选）
        
        Returns:
            回测结果DataFrame
        """
        self.reset()
        
        logger.info("开始运行回测...")
        logger.info(f"初始资金: ${self.initial_capital:,.2f}")
        logger.info(f"手续费率: {self.commission_rate * 100:.3f}%")
        logger.info(f"滑点率: {self.slippage_rate * 100:.3f}%")
        logger.info(f"杠杆倍数: {self.leverage}x")
        logger.info(f"保证金比例: {self.margin_ratio * 100:.1f}%")
        logger.info(f"最大资金使用率: {self.max_capital_usage * 100:.0f}%")
        
        # ✅ 止损参数日志
        if self.stop_loss_pct > 0:
            logger.info(f"百分比止损: {self.stop_loss_pct * 100:.1f}%")
        if self.stop_loss_amount > 0:
            logger.info(f"金额止损: ${self.stop_loss_amount:,.2f}")
        
        # 合并价格和信号
        data = price_data[[price_col]].join(signals[['signal']], how='inner')
        
        if volatility_col and volatility_col in price_data.columns:
            data['volatility'] = price_data[volatility_col]
        else:
            # 使用滚动标准差估计波动率
            data['volatility'] = data[price_col].pct_change().rolling(20).std()
        
        data = data.dropna()
        
        # 逐行回测
        for timestamp, row in data.iterrows():
            price = row[price_col]
            signal = int(row['signal'])
            volatility = row['volatility'] if not pd.isna(row['volatility']) else 0.01
            
            # ✅ 检查止损（在执行交易前）
            if self.check_stop_loss(price):
                logger.warning(f"{timestamp}: 触发止损，强制平仓")
                self.execute_trade(timestamp, 0, price, volatility)
                self.stop_loss_triggered_count += 1
                # 更新权益
                self.update_equity(timestamp, price)
                continue  # 跳过本次信号执行
            
            # 执行交易
            self.execute_trade(timestamp, signal, price, volatility)
            
            # 更新权益
            self.update_equity(timestamp, price)
        
        # 平仓所有持仓
        if self.position != 0:
            last_price = data[price_col].iloc[-1]
            last_timestamp = data.index[-1]
            self.execute_trade(last_timestamp, 0, last_price, 0.01)
            self.update_equity(last_timestamp, last_price)
        
        # 生成回测结果DataFrame
        results_df = pd.DataFrame(self.equity_curve)
        results_df.set_index('timestamp', inplace=True)
        
        logger.info(f"回测完成，共执行 {len(self.trades)} 笔交易")
        logger.info(f"最终权益: ${results_df['equity'].iloc[-1]:,.2f}")
        
        # ✅ 止损统计
        if self.stop_loss_triggered_count > 0:
            logger.info(f"⚠️  止损触发次数: {self.stop_loss_triggered_count}")
        
        return results_df
    
    def get_trade_log(self) -> pd.DataFrame:
        """获取交易日志"""
        if not self.trades:
            return pd.DataFrame()
        
        trades_dict = {
            'timestamp': [t.timestamp for t in self.trades],
            'action': [t.action for t in self.trades],
            'position': [t.position for t in self.trades],
            'price': [t.price for t in self.trades],
            'quantity': [t.quantity for t in self.trades],
            'pnl': [t.pnl for t in self.trades],
            'commission': [t.commission for t in self.trades]
        }
        
        return pd.DataFrame(trades_dict)
    
    def get_margin_stats(self) -> Dict:
        """
        获取保证金使用统计
        
        Returns:
            保证金统计字典
        """
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        return {
            'leverage': self.leverage,
            'margin_ratio': self.margin_ratio,
            'max_capital_usage': self.max_capital_usage,
            'current_margin_used': self.margin_used,
            'current_available_capital': self.available_capital,
            'peak_margin_used': equity_df['margin_used'].max(),
            'avg_margin_used': equity_df['margin_used'].mean(),
            'peak_margin_usage_rate': equity_df['margin_usage_rate'].max(),
            'avg_margin_usage_rate': equity_df['margin_usage_rate'].mean(),
            'peak_leverage_ratio': equity_df['leverage_ratio'].max() if 'leverage_ratio' in equity_df else 0.0
        }
    
    def get_stop_loss_stats(self) -> Dict:
        """
        获取止损统计
        
        Returns:
            止损统计字典
        """
        return {
            'stop_loss_pct': self.stop_loss_pct,
            'stop_loss_amount': self.stop_loss_amount,
            'stop_loss_triggered_count': self.stop_loss_triggered_count,
            'stop_loss_enabled': self.stop_loss_pct > 0 or self.stop_loss_amount > 0
        }


class PerformanceAnalyzer:
    """绩效分析器"""
    
    def __init__(self, equity_curve: pd.DataFrame, 
                 initial_capital: float = 1000000,
                 risk_free_rate: float = 0.02):
        """
        初始化绩效分析器
        
        Args:
            equity_curve: 权益曲线DataFrame
            initial_capital: 初始资金
            risk_free_rate: 无风险利率（年化）
        """
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        
        # 计算收益率
        self.returns = equity_curve['equity'].pct_change().dropna()
    
    def calculate_total_return(self) -> float:
        """计算总收益率"""
        final_equity = self.equity_curve['equity'].iloc[-1]
        return (final_equity - self.initial_capital) / self.initial_capital
    
    def calculate_annual_return(self) -> float:
        """计算年化收益率"""
        total_return = self.calculate_total_return()
        
        # 计算天数
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        years = days / 365.25
        
        if years > 0:
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = 0.0
        
        return annual_return
    
    def calculate_volatility(self) -> float:
        """计算年化波动率"""
        return self.returns.std() * np.sqrt(252)
    
    def calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        annual_return = self.calculate_annual_return()
        volatility = self.calculate_volatility()
        
        if volatility > 0:
            return (annual_return - self.risk_free_rate) / volatility
        return 0.0
    
    def calculate_max_drawdown(self) -> Dict[str, float]:
        """
        计算最大回撤
        
        Returns:
            包含最大回撤、开始和结束日期的字典
        """
        equity = self.equity_curve['equity']
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        
        max_dd = drawdown.min()
        max_dd_end = drawdown.idxmin()
        
        # 找到回撤开始点
        max_dd_start = equity[:max_dd_end].idxmax()
        
        return {
            'max_drawdown': max_dd,
            'drawdown_start': max_dd_start,
            'drawdown_end': max_dd_end,
            'recovery_date': None  # 可以添加恢复日期计算
        }
    
    def calculate_sortino_ratio(self) -> float:
        """计算索提诺比率（只考虑下行波动）"""
        annual_return = self.calculate_annual_return()
        
        # 下行偏差
        downside_returns = self.returns[self.returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        
        if downside_std > 0:
            return (annual_return - self.risk_free_rate) / downside_std
        return 0.0
    
    def calculate_calmar_ratio(self) -> float:
        """计算卡玛比率（年化收益/最大回撤）"""
        annual_return = self.calculate_annual_return()
        max_dd = abs(self.calculate_max_drawdown()['max_drawdown'])
        
        if max_dd > 0:
            return annual_return / max_dd
        return 0.0
    
    def calculate_win_rate(self, trade_log: pd.DataFrame) -> float:
        """计算胜率"""
        if trade_log.empty or 'pnl' not in trade_log.columns:
            return 0.0
        
        winning_trades = (trade_log['pnl'] > 0).sum()
        total_trades = len(trade_log[trade_log['pnl'] != 0])
        
        if total_trades > 0:
            return winning_trades / total_trades
        return 0.0
    
    def calculate_profit_factor(self, trade_log: pd.DataFrame) -> float:
        """计算盈亏比"""
        if trade_log.empty or 'pnl' not in trade_log.columns:
            return 0.0
        
        gross_profit = trade_log[trade_log['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trade_log[trade_log['pnl'] < 0]['pnl'].sum())
        
        if gross_loss > 0:
            return gross_profit / gross_loss
        return 0.0
    
    def calculate_var(self, confidence: float = 0.95) -> float:
        """
        计算VaR（在险价值）
        
        Args:
            confidence: 置信水平
        
        Returns:
            VaR值
        """
        return self.returns.quantile(1 - confidence)
    
    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """
        计算CVaR（条件VaR）
        
        Args:
            confidence: 置信水平
        
        Returns:
            CVaR值
        """
        var = self.calculate_var(confidence)
        return self.returns[self.returns <= var].mean()
    
    def generate_performance_report(self, trade_log: Optional[pd.DataFrame] = None) -> Dict:
        """
        生成完整的绩效报告
        
        Args:
            trade_log: 交易日志DataFrame
        
        Returns:
            绩效指标字典
        """
        report = {
            'total_return': self.calculate_total_return(),
            'annual_return': self.calculate_annual_return(),
            'volatility': self.calculate_volatility(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'calmar_ratio': self.calculate_calmar_ratio(),
            'max_drawdown': self.calculate_max_drawdown()['max_drawdown'],
            'var_95': self.calculate_var(0.95),
            'cvar_95': self.calculate_cvar(0.95)
        }
        
        if trade_log is not None and not trade_log.empty:
            report['win_rate'] = self.calculate_win_rate(trade_log)
            report['profit_factor'] = self.calculate_profit_factor(trade_log)
            report['total_trades'] = len(trade_log)
            report['total_commission'] = trade_log['commission'].sum()
        
        logger.info("\n" + "="*60)
        logger.info("绩效分析报告")
        logger.info("="*60)
        logger.info(f"总收益率: {report['total_return']*100:.2f}%")
        logger.info(f"年化收益率: {report['annual_return']*100:.2f}%")
        logger.info(f"年化波动率: {report['volatility']*100:.2f}%")
        logger.info(f"夏普比率: {report['sharpe_ratio']:.4f}")
        logger.info(f"索提诺比率: {report['sortino_ratio']:.4f}")
        logger.info(f"卡玛比率: {report['calmar_ratio']:.4f}")
        logger.info(f"最大回撤: {report['max_drawdown']*100:.2f}%")
        logger.info(f"VaR (95%): {report['var_95']*100:.2f}%")
        logger.info(f"CVaR (95%): {report['cvar_95']*100:.2f}%")
        
        if 'win_rate' in report:
            logger.info(f"胜率: {report['win_rate']*100:.2f}%")
            logger.info(f"盈亏比: {report['profit_factor']:.2f}")
            logger.info(f"总交易次数: {report['total_trades']}")
            logger.info(f"总手续费: ${report['total_commission']:,.2f}")
        
        logger.info("="*60 + "\n")
        
        return report
    
    def get_monthly_returns(self) -> pd.DataFrame:
        """获取月度收益率"""
        equity = self.equity_curve['equity']
        monthly = equity.resample('M').last()
        monthly_returns = monthly.pct_change()
        
        return monthly_returns
    
    def get_drawdown_series(self) -> pd.Series:
        """获取回撤序列"""
        equity = self.equity_curve['equity']
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        
        return drawdown
