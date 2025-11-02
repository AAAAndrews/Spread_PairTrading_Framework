"""
数据抓取模块：从各种数据源获取行情和基本面数据
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
import logging
from functools import wraps
import pytz

logger = logging.getLogger(__name__)


def data_normalizer(func: Callable) -> Callable:
    """
    装饰器：标准化数据格式
    确保返回的DataFrame具有统一的列名和格式
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        
        if df is None or df.empty:
            return df
        
        # 标准化列名（小写）
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'Volume': 'volume', 'Adj Close': 'adjusted_close',
            'Open Interest': 'open_interest'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        # 确保索引是日期时间类型
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
        
        # 确保索引有时区信息
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        return df
    
    return wrapper


class DataFetcher:
    """数据抓取类"""
    
    def __init__(self):
        self.timezone = pytz.UTC
    
    @data_normalizer
    def fetch_yfinance_data(self, symbol: str, start_date: Optional[str] = None,
                           end_date: Optional[str] = None, interval: str = '1d') -> pd.DataFrame:
        """
        使用yfinance获取数据
        
        Args:
            symbol: 品种代码
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据频率
        
        Returns:
            DataFrame
        """
        try:
            import yfinance as yf
            
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"正在从yfinance获取 {symbol} 的数据...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                logger.warning(f"未获取到 {symbol} 的数据")
                return pd.DataFrame()
            
            logger.info(f"成功获取 {len(df)} 条 {symbol} 的数据")
            return df
            
        except Exception as e:
            logger.error(f"获取yfinance数据失败: {e}")
            return pd.DataFrame()
    
    @data_normalizer
    def fetch_akshare_futures_data(self, symbol: str, market: str = 'CN') -> pd.DataFrame:
        """
        使用akshare获取期货数据
        
        Args:
            symbol: 品种代码
            market: 市场（CN-中国，US-美国）
        
        Returns:
            DataFrame
        """
        try:
            import akshare as ak
            
            logger.info(f"正在从akshare获取 {symbol} 的期货数据...")
            
            if market == 'CN':
                # 获取国内期货主力合约数据
                df = ak.futures_main_sina(symbol=symbol)
            else:
                logger.warning("akshare主要支持中国市场数据")
                return pd.DataFrame()
            
            if df.empty:
                logger.warning(f"未获取到 {symbol} 的数据")
                return pd.DataFrame()
            
            logger.info(f"成功获取 {len(df)} 条 {symbol} 的数据")
            return df
            
        except Exception as e:
            logger.error(f"获取akshare数据失败: {e}")
            return pd.DataFrame()
    
    def fetch_eia_data(self, series_id: str, api_key: Optional[str] = None) -> pd.DataFrame:
        """
        获取EIA数据
        
        Args:
            series_id: EIA序列ID
            api_key: API密钥（可选）
        
        Returns:
            DataFrame
        """
        try:
            # 注意：实际使用时需要EIA API密钥
            logger.warning("EIA数据获取需要API密钥，这里返回示例数据")
            
            # 示例：创建模拟EIA数据
            dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='W')
            data = {
                'crude_oil_stock': np.random.normal(420, 20, len(dates)),
                'gasoline_stock': np.random.normal(230, 15, len(dates)),
                'distillate_stock': np.random.normal(130, 10, len(dates))
            }
            df = pd.DataFrame(data, index=dates)
            df.index = df.index.tz_localize('UTC')
            
            return df
            
        except Exception as e:
            logger.error(f"获取EIA数据失败: {e}")
            return pd.DataFrame()
    
    def fetch_cot_data(self) -> pd.DataFrame:
        """
        获取COT持仓报告数据
        
        Returns:
            DataFrame
        """
        try:
            logger.warning("COT数据获取需要特定API，这里返回示例数据")
            
            # 示例：创建模拟COT数据
            dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='W-TUE')
            data = {
                'commercial_long': np.random.randint(200000, 300000, len(dates)),
                'commercial_short': np.random.randint(250000, 350000, len(dates)),
                'noncommercial_long': np.random.randint(150000, 250000, len(dates)),
                'noncommercial_short': np.random.randint(100000, 200000, len(dates))
            }
            df = pd.DataFrame(data, index=dates)
            df.index = df.index.tz_localize('UTC')
            
            return df
            
        except Exception as e:
            logger.error(f"获取COT数据失败: {e}")
            return pd.DataFrame()
    
    @data_normalizer
    def fetch_index_data(self, symbol: str, start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取指数数据（如VIX、DXY等）
        
        Args:
            symbol: 指数代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame
        """
        # 使用yfinance获取指数数据
        symbol_map = {
            'VIX': '^VIX',
            'DXY': 'DX-Y.NYB',
            'SPX': '^GSPC',
            'TNX': '^TNX'  # 10年期国债收益率
        }
        
        yf_symbol = symbol_map.get(symbol, symbol)
        return self.fetch_yfinance_data(yf_symbol, start_date, end_date)
    
    def fetch_csv_data(self, file_path: str, date_column: str = 'date') -> pd.DataFrame:
        """
        从CSV文件读取数据
        
        Args:
            file_path: CSV文件路径
            date_column: 日期列名
        
        Returns:
            DataFrame
        """
        try:
            df = pd.read_csv(file_path, parse_dates=[date_column])
            df.set_index(date_column, inplace=True)
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            logger.info(f"成功从CSV读取 {len(df)} 条数据")
            return df
            
        except Exception as e:
            logger.error(f"读取CSV文件失败: {e}")
            return pd.DataFrame()


# class FuturesAdjuster:
#     """期货复权处理类"""
    
#     @staticmethod
#     def adjust_futures_roll(df: pd.DataFrame, roll_dates: Optional[list] = None,
#                            method: str = 'ratio') -> pd.DataFrame:
#         """
#         期货主力合约复权处理
        
#         Args:
#             df: 原始价格数据
#             roll_dates: 换月日期列表（如果为None则自动检测）
#             method: 复权方法（ratio-比例复权，diff-差值复权）
        
#         Returns:
#             复权后的DataFrame
#         """
#         df_adjusted = df.copy()
        
#         # 自动检测换月点（价格跳变超过5%）
#         if roll_dates is None:
#             price_change = df['close'].pct_change()
#             roll_indices = price_change[abs(price_change) > 0.05].index
#             logger.info(f"自动检测到 {len(roll_indices)} 个换月点")
#         else:
#             roll_indices = pd.to_datetime(roll_dates)
        
#         if len(roll_indices) == 0:
#             return df_adjusted
        
#         # 比例复权
#         if method == 'ratio':
#             adjustment_factor = 1.0
#             for i, roll_date in enumerate(roll_indices[::-1]):  # 从最后一个换月点开始
#                 mask = df_adjusted.index < roll_date
#                 if mask.any():
#                     # 计算复权因子
#                     idx = df_adjusted.index.get_loc(roll_date)
#                     if idx > 0:
#                         ratio = df_adjusted['close'].iloc[idx] / df_adjusted['close'].iloc[idx - 1]
#                         adjustment_factor *= ratio
                        
#                         # 应用复权
#                         for col in ['open', 'high', 'low', 'close']:
#                             if col in df_adjusted.columns:
#                                 df_adjusted.loc[mask, col] *= adjustment_factor
        
#         # 差值复权
#         elif method == 'diff':
#             adjustment_diff = 0.0
#             for roll_date in roll_indices[::-1]:
#                 mask = df_adjusted.index < roll_date
#                 if mask.any():
#                     idx = df_adjusted.index.get_loc(roll_date)
#                     if idx > 0:
#                         diff = df_adjusted['close'].iloc[idx] - df_adjusted['close'].iloc[idx - 1]
#                         adjustment_diff += diff
                        
#                         for col in ['open', 'high', 'low', 'close']:
#                             if col in df_adjusted.columns:
#                                 df_adjusted.loc[mask, col] += adjustment_diff
        
#         df_adjusted['adjusted_close'] = df_adjusted['close']
#         logger.info(f"期货复权完成，使用方法: {method}")
        
#         return df_adjusted
    
#     @staticmethod
#     def create_continuous_contract(main_contract_data: Dict[str, pd.DataFrame],
#                                   roll_method: str = 'volume') -> pd.DataFrame:
#         """
#         创建连续合约
        
#         Args:
#             main_contract_data: 各合约数据字典 {合约代码: DataFrame}
#             roll_method: 换月方法（volume-成交量，open_interest-持仓量）
        
#         Returns:
#             连续合约DataFrame
#         """
#         # 合并所有合约数据
#         all_data = []
#         for contract, df in main_contract_data.items():
#             df = df.copy()
#             df['contract'] = contract
#             all_data.append(df)
        
#         combined = pd.concat(all_data).sort_index()
        
#         # 根据成交量或持仓量选择主力合约
#         if roll_method == 'volume':
#             idx = combined.groupby(combined.index)['volume'].idxmax()
#         else:
#             idx = combined.groupby(combined.index)['open_interest'].idxmax()
        
#         continuous = combined.loc[idx].copy()
        
#         # 检测换月点并复权
#         roll_dates = continuous['contract'].ne(continuous['contract'].shift()).index[1:]
        
#         return FuturesAdjuster.adjust_futures_roll(continuous, list(roll_dates))
