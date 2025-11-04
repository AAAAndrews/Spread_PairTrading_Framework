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
            'Open Interest': 'open_interest',
            "日期":"date",
            "开盘价":"open",
            "收盘价":"close",
            "最高价":"high",
            "最低价":"low",
            "成交量":"volume",
            "持仓量":"holding_volume"
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
                df["日期"] = pd.to_datetime(df["日期"])
                if hasattr(df["日期"].dt, 'tz') and df["日期"].dt.tz is None:
                    df["日期"] = df["日期"].dt.tz_localize('Asia/Shanghai')
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
    
    def fetch_eia_data(self, series_id: str, api_key: str = "O8N0mhrHyhUV3pg6qus3cSZ62EMYteG4Ar2SvgAc") -> pd.DataFrame:
        """
        使用EIA Python Library获取EIA数据
        
        Args:
            series_id: EIA序列ID，例如:
                - 'PET.WCRSTUS1.W': 美国原油库存（周度）
                - 'PET.WGTSTUS1.W': 美国汽油库存（周度）
                - 'PET.WDISTUS1.W': 美国馏分油库存（周度）
                - 'PET.RWTC.D': WTI原油现货价格（日度）
            api_key: EIA API密钥
        
        Returns:
            DataFrame，包含时间序列数据
        """
        try:
            import requests
            
            logger.info(f"正在从EIA获取数据，序列ID: {series_id}")
            
            # 使用EIA v2 API（更稳定）
            url = f"https://api.eia.gov/v2/seriesid/{series_id}"
            params = {
                'api_key': api_key
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"EIA API请求失败，状态码: {response.status_code}")
                logger.error(f"响应内容: {response.text}")
                return pd.DataFrame()
            
            data_json = response.json()
            
            # 解析响应数据
            if 'response' not in data_json:
                logger.warning(f"未找到数据，尝试使用v1 API")
                # 尝试使用v1 API
                return self._fetch_eia_data_v1(series_id, api_key)
            
            if 'data' not in data_json['response']:
                logger.warning(f"序列 {series_id} 没有可用数据")
                return pd.DataFrame()
            
            data_list = data_json['response']['data']
            
            if not data_list:
                logger.warning(f"序列 {series_id} 返回空数据")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(data_list)
            
            # 查找日期列和数值列
            date_col = None
            value_col = None
            
            for col in df.columns:
                if 'period' in col.lower() or 'date' in col.lower():
                    date_col = col
                if 'value' in col.lower():
                    value_col = col
            
            if date_col is None or value_col is None:
                logger.error(f"无法识别数据列，可用列: {df.columns.tolist()}")
                return pd.DataFrame()
            
            # 处理日期和数值
            df['date'] = pd.to_datetime(df[date_col])
            df['value'] = pd.to_numeric(df[value_col], errors='coerce')
            
            # 设置索引并排序
            df = df[['date', 'value']].copy()
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # 添加时区信息
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            logger.info(f"成功获取 {len(df)} 条EIA数据，时间范围: {df.index.min()} 至 {df.index.max()}")
            
            return df
            
        except ImportError as e:
            logger.error(f"缺少必要的库: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取EIA数据失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
    
    def _fetch_eia_data_v1(self, series_id: str, api_key: str) -> pd.DataFrame:
        """
        使用EIA v1 API获取数据（备用方法）
        """
        try:
            import requests
            
            url = "https://api.eia.gov/series/"
            params = {
                'api_key': api_key,
                'series_id': series_id
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"EIA v1 API请求失败，状态码: {response.status_code}")
                return pd.DataFrame()
            
            data_json = response.json()
            
            if 'series' not in data_json or not data_json['series']:
                logger.warning(f"v1 API未返回数据")
                return pd.DataFrame()
            
            series_data = data_json['series'][0]
            data_list = series_data.get('data', [])
            
            if not data_list:
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(data_list, columns=['date', 'value'])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            logger.info(f"通过v1 API成功获取 {len(df)} 条数据")
            
            return df
            
        except Exception as e:
            logger.error(f"v1 API获取失败: {e}")
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
