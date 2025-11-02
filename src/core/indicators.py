"""
指标构建模块：计算技术指标和自定义特征
"""
import pandas as pd
import numpy as np
from typing import Optional, Callable, Dict, Any
from statsmodels.tsa.stattools import adfuller
import logging

logger = logging.getLogger(__name__)


class IndicatorBuilder:
    """指标构建器"""
    
    def __init__(self, db_manager=None):
        """
        初始化指标构建器
        
        Args:
            db_manager: 数据库管理器实例
        """
        self.db_manager = db_manager
    
    def apply_custom_rule(self, df: pd.DataFrame, rule_func: Callable,
                         column: str = 'close', **kwargs) -> pd.DataFrame:
        """
        应用自定义规则处理数据
        
        Args:
            df: 输入DataFrame
            rule_func: 处理函数
            column: 要处理的列名
            **kwargs: 传递给rule_func的参数
        
        Returns:
            处理后的DataFrame
        """
        result = df.copy()
        result[f'{column}_processed'] = rule_func(df[column], **kwargs)
        return result
    
    def calculate_moving_average(self, df: pd.DataFrame, column: str = 'close',
                                windows: list = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """
        计算移动平均线
        
        Args:
            df: 输入DataFrame
            column: 价格列名
            windows: 窗口大小列表
        
        Returns:
            包含MA指标的DataFrame
        """
        result = df.copy()
        
        for window in windows:
            result[f'MA_{window}'] = result[column].rolling(window=window).mean()
        
        logger.info(f"计算移动平均线完成，窗口: {windows}")
        return result
    
    def calculate_ema(self, df: pd.DataFrame, column: str = 'close',
                     spans: list = [12, 26, 50]) -> pd.DataFrame:
        """
        计算指数移动平均线
        
        Args:
            df: 输入DataFrame
            column: 价格列名
            spans: span参数列表
        
        Returns:
            包含EMA指标的DataFrame
        """
        result = df.copy()
        
        for span in spans:
            result[f'EMA_{span}'] = result[column].ewm(span=span, adjust=False).mean()
        
        logger.info(f"计算EMA完成，span: {spans}")
        return result
    
    def calculate_rsi(self, df: pd.DataFrame, column: str = 'close',
                     period: int = 14) -> pd.DataFrame:
        """
        计算相对强弱指数(RSI)
        
        Args:
            df: 输入DataFrame
            column: 价格列名
            period: 周期
        
        Returns:
            包含RSI指标的DataFrame
        """
        result = df.copy()
        
        # 计算价格变化
        delta = result[column].diff()
        
        # 分离涨跌
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # 计算RS和RSI
        rs = gain / loss
        result[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        logger.info(f"计算RSI完成，周期: {period}")
        return result
    
    def calculate_macd(self, df: pd.DataFrame, column: str = 'close',
                      fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        计算MACD指标
        
        Args:
            df: 输入DataFrame
            column: 价格列名
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
        
        Returns:
            包含MACD指标的DataFrame
        """
        result = df.copy()
        
        # 计算EMA
        ema_fast = result[column].ewm(span=fast, adjust=False).mean()
        ema_slow = result[column].ewm(span=slow, adjust=False).mean()
        
        # MACD线
        result['MACD'] = ema_fast - ema_slow
        
        # 信号线
        result['MACD_signal'] = result['MACD'].ewm(span=signal, adjust=False).mean()
        
        # MACD柱
        result['MACD_hist'] = result['MACD'] - result['MACD_signal']
        
        logger.info(f"计算MACD完成，参数: {fast}/{slow}/{signal}")
        return result
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, column: str = 'close',
                                 window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        计算布林带
        
        Args:
            df: 输入DataFrame
            column: 价格列名
            window: 窗口大小
            num_std: 标准差倍数
        
        Returns:
            包含布林带指标的DataFrame
        """
        result = df.copy()
        
        # 中轨（移动平均）
        result['BB_middle'] = result[column].rolling(window=window).mean()
        
        # 标准差
        std = result[column].rolling(window=window).std()
        
        # 上下轨
        result['BB_upper'] = result['BB_middle'] + (std * num_std)
        result['BB_lower'] = result['BB_middle'] - (std * num_std)
        
        # 带宽
        result['BB_width'] = (result['BB_upper'] - result['BB_lower']) / result['BB_middle']
        
        # 价格位置（%B）
        result['BB_percent'] = (result[column] - result['BB_lower']) / (result['BB_upper'] - result['BB_lower'])
        
        logger.info(f"计算布林带完成，窗口: {window}, 标准差: {num_std}")
        return result
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算平均真实范围(ATR)
        
        Args:
            df: 输入DataFrame（需包含high, low, close）
            period: 周期
        
        Returns:
            包含ATR指标的DataFrame
        """
        result = df.copy()
        
        # 真实范围
        high_low = result['high'] - result['low']
        high_close = np.abs(result['high'] - result['close'].shift())
        low_close = np.abs(result['low'] - result['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # ATR
        result[f'ATR_{period}'] = true_range.rolling(window=period).mean()
        
        logger.info(f"计算ATR完成，周期: {period}")
        return result
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量指标
        
        Args:
            df: 输入DataFrame（需包含volume）
        
        Returns:
            包含成交量指标的DataFrame
        """
        result = df.copy()
        
        if 'volume' not in result.columns:
            logger.warning("缺少volume列，跳过成交量指标计算")
            return result
        
        # 成交量移动平均
        result['volume_MA_20'] = result['volume'].rolling(window=20).mean()
        
        # 成交量比率
        result['volume_ratio'] = result['volume'] / result['volume_MA_20']
        
        # OBV (On Balance Volume)
        result['price_change'] = result['close'].diff()
        result['obv'] = (np.sign(result['price_change']) * result['volume']).fillna(0).cumsum()
        result.drop('price_change', axis=1, inplace=True)
        
        logger.info("计算成交量指标完成")
        return result
    
    def calculate_momentum_indicators(self, df: pd.DataFrame, column: str = 'close',
                                    periods: list = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        计算动量指标
        
        Args:
            df: 输入DataFrame
            column: 价格列名
            periods: 周期列表
        
        Returns:
            包含动量指标的DataFrame
        """
        result = df.copy()
        
        for period in periods:
            # 收益率
            result[f'return_{period}d'] = result[column].pct_change(period)
            
            # 动量
            result[f'momentum_{period}d'] = result[column] - result[column].shift(period)
        
        logger.info(f"计算动量指标完成，周期: {periods}")
        return result
    
    def calculate_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算季节性特征
        
        Args:
            df: 输入DataFrame（索引必须是DatetimeIndex）
        
        Returns:
            包含季节性特征的DataFrame
        """
        result = df.copy()
        
        if not isinstance(result.index, pd.DatetimeIndex):
            logger.warning("索引不是DatetimeIndex，跳过季节性特征计算")
            return result
        
        # 月份
        result['month'] = result.index.month
        
        # 季度
        result['quarter'] = result.index.quarter
        
        # # 星期几
        # result['day_of_week'] = result.index.dayofweek
        
        # # 年份中的第几天
        # result['day_of_year'] = result.index.dayofyear
        
        # 年份
        result['year'] = result.index.year
        
        # 季节性虚拟变量（冬季需求旺季）
        result['is_winter'] = result['month'].isin([12, 1, 2]).astype(int)
        result['is_summer'] = result['month'].isin([6, 7, 8]).astype(int)
        
        logger.info("计算季节性特征完成")
        return result
    
    def calculate_statistical_features(self, df: pd.DataFrame, column: str = 'close',
                                      windows: list = [5, 10, 20]) -> pd.DataFrame:
        """
        计算统计特征
        
        Args:
            df: 输入DataFrame
            column: 数据列名
            windows: 窗口大小列表
        
        Returns:
            包含统计特征的DataFrame
        """
        result = df.copy()
        
        for window in windows:
            # 均值
            result[f'mean_{window}d'] = result[column].rolling(window=window).mean()
            
            # 标准差
            result[f'std_{window}d'] = result[column].rolling(window=window).std()
            
            # 偏度
            result[f'skew_{window}d'] = result[column].rolling(window=window).skew()
            
            # 峰度
            result[f'kurt_{window}d'] = result[column].rolling(window=window).kurt()
            
            # Z-score
            result[f'zscore_{window}d'] = (result[column] - result[f'mean_{window}d']) / result[f'std_{window}d']
        
        logger.info(f"计算统计特征完成，窗口: {windows}")
        return result
    
    def align_with_trading_time(self, source_df: pd.DataFrame, 
                               target_trading_times: pd.DatetimeIndex,
                               method: str = 'asof') -> pd.DataFrame:
        """
        将数据与目标交易时间对齐（避免未来数据泄露）
        
        Args:
            source_df: 源数据DataFrame
            target_trading_times: 目标交易时间索引
            method: 对齐方法（asof-向后查找，ffill-前向填充）
        
        Returns:
            对齐后的DataFrame
        """
        # 简化版本：直接使用reindex with ffill，这样更稳定
        try:
            # 移除时区信息进行对齐
            if isinstance(source_df.index, pd.DatetimeIndex) and source_df.index.tz is not None:
                source_df_no_tz = source_df.copy()
                source_df_no_tz.index = source_df_no_tz.index.tz_localize(None)
            else:
                source_df_no_tz = source_df
            
            if isinstance(target_trading_times, pd.DatetimeIndex) and target_trading_times.tz is not None:
                target_times_no_tz = target_trading_times.tz_localize(None)
            else:
                target_times_no_tz = target_trading_times
            
            # 使用reindex with method='ffill'（前向填充，避免未来数据）
            result = source_df_no_tz.reindex(target_times_no_tz, method='ffill')
            
            logger.info(f"数据时间对齐完成，方法: {method}")
            return result
            
        except Exception as e:
            logger.error(f"时间对齐失败: {e}")
            # 返回空DataFrame作为fallback
            return pd.DataFrame(index=target_trading_times, columns=source_df.columns)
    
    def test_stationarity(self, series: pd.Series, name: str = "Series") -> Dict[str, Any]:
        """
        使用ADF检验测试序列平稳性
        
        Args:
            series: 时间序列
            name: 序列名称
        
        Returns:
            检验结果字典
        """
        # 移除NaN值
        series_clean = series.dropna()
        
        if len(series_clean) < 10:
            logger.warning(f"{name}: 数据点太少，无法进行ADF检验")
            return {'error': '数据点不足'}
        
        # 执行ADF检验
        adf_result = adfuller(series_clean, autolag='AIC')
        
        result = {
            'name': name,
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'used_lag': adf_result[2],
            'n_obs': adf_result[3],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05  # p值小于0.05认为是平稳的
        }
        
        # 打印结论
        logger.info(f"\n{'='*50}")
        logger.info(f"ADF平稳性检验结果 - {name}")
        logger.info(f"{'='*50}")
        logger.info(f"ADF统计量: {result['adf_statistic']:.6f}")
        logger.info(f"P值: {result['p_value']:.6f}")
        logger.info(f"使用滞后阶数: {result['used_lag']}")
        logger.info(f"观测值数量: {result['n_obs']}")
        logger.info(f"临界值:")
        for key, value in result['critical_values'].items():
            logger.info(f"  {key}: {value:.6f}")
        
        if result['is_stationary']:
            logger.info(f"结论: {name} 是平稳序列 (p < 0.05)")
        else:
            logger.info(f"结论: {name} 不是平稳序列 (p >= 0.05)")
            logger.info(f"建议: 考虑差分或其他变换来使序列平稳")
        logger.info(f"{'='*50}\n")
        
        return result
    
    def save_indicator(self, df: pd.DataFrame, indicator_name: str, 
                      symbol: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        保存指标到数据库
        
        Args:
            df: 指标数据DataFrame
            indicator_name: 指标名称
            symbol: 关联品种
            metadata: 元数据
        """
        if not self.db_manager:
            logger.warning("未设置数据库管理器，无法保存指标")
            return
        
        self.db_manager.insert_indicator_data(indicator_name, df, symbol, metadata)
        logger.info(f"指标已保存: {indicator_name}")
