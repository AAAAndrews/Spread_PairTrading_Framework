"""
价差配比组件：构造不同比例的套利价差
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import json

logger = logging.getLogger(__name__)

# 单位转换常量
GALLONS_PER_BARREL = 42  # 1桶 = 42加仑


class SpreadCalculator:
    """价差计算器"""
    
    def __init__(self, db_manager=None):
        """
        初始化价差计算器
        
        Args:
            db_manager: 数据库管理器实例
        """
        self.db_manager = db_manager
        self.spread_configs = {}
    
    @staticmethod
    def convert_units(price: pd.Series, product: str) -> pd.Series:
        """
        将所有产品价格统一转换为美元/桶
        
        Args:
            price: 价格序列
            product: 产品代码
        
        Returns:
            转换后的价格序列（美元/桶）
        
        单位说明：
            - CL (WTI原油): 美元/桶 → 无需转换
            - RBOB (汽油): 美元/加仑 → 乘以42转换为美元/桶
            - HO (取暖油/柴油): 美元/加仑 → 乘以42转换为美元/桶
        
        Example:
            >>> price_cl = pd.Series([70.0, 71.0])  # WTI原油，美元/桶
            >>> price_rbob = pd.Series([2.0, 2.1])  # RBOB汽油，美元/加仑
            >>> 
            >>> convert_units(price_cl, 'CL')  # 70.0, 71.0 (不变)
            >>> convert_units(price_rbob, 'RBOB')  # 84.0, 88.2 (2.0*42, 2.1*42)
        """
        if product == 'CL':
            # WTI原油已经是美元/桶，无需转换
            logger.debug(f"{product}: 价格已是美元/桶，无需转换")
            return price
        
        elif product in ['RBOB', 'HO']:
            # RBOB汽油和取暖油从美元/加仑转换为美元/桶
            converted_price = price * GALLONS_PER_BARREL
            logger.info(f"{product}: 价格从美元/加仑转换为美元/桶 (乘以{GALLONS_PER_BARREL})")
            logger.debug(f"  原始价格范围: {price.min():.4f} - {price.max():.4f} 美元/加仑")
            logger.debug(f"  转换后范围: {converted_price.min():.4f} - {converted_price.max():.4f} 美元/桶")
            return converted_price
        
        else:
            # 未知产品，不转换但记录警告
            logger.warning(f"未知产品代码 {product}，价格不进行单位转换")
            return price
    
    def create_spread(self, spread_name: str, components: List[Tuple[str, float]],
                     save_config: bool = True) -> Dict:
        """
        创建价差配置
        
        Args:
            spread_name: 价差名称
            components: 组成部分列表 [(品种代码, 权重), ...]
                       正权重表示多头，负权重表示空头
            save_config: 是否保存配置到数据库
        
        Returns:
            配置字典
        
        Example:
            # 1:1价差（1份汽油多头，1份原油空头）
            create_spread("RBOB_CL_1_1", [("RBOB", 1), ("CL", -1)])
            
            # 3:2:1裂解价差（3份原油空头，2份汽油多头，1份柴油多头）
            create_spread("CRACK_3_2_1", [("CL", -3), ("RBOB", 2), ("HO", 1)])
        """
        config = {
            'spread_name': spread_name,
            'components': components,
            'description': self._generate_description(components)
        }
        
        self.spread_configs[spread_name] = config
        
        if save_config and self.db_manager:
            self._save_config_to_db(spread_name, config)
        
        logger.info(f"创建价差配置: {spread_name} - {config['description']}")
        
        return config
    
    def _generate_description(self, components: List[Tuple[str, float]]) -> str:
        """生成价差描述"""
        long_parts = []
        short_parts = []
        
        for symbol, weight in components:
            if weight > 0:
                long_parts.append(f"{abs(weight):.1f}x{symbol}")
            else:
                short_parts.append(f"{abs(weight):.1f}x{symbol}")
        
        long_str = " + ".join(long_parts) if long_parts else ""
        short_str = " + ".join(short_parts) if short_parts else ""
        
        if long_str and short_str:
            return f"多头({long_str}) - 空头({short_str})"
        elif long_str:
            return f"多头({long_str})"
        else:
            return f"空头({short_str})"
    
    def _save_config_to_db(self, spread_name: str, config: Dict):
        """保存配置到数据库"""
        if not self.db_manager:
            return
        
        try:
            cursor = self.db_manager.conn.cursor()
            from datetime import datetime
            import pytz
            
            config_json = json.dumps(config, ensure_ascii=False)
            created_at = datetime.now(pytz.UTC).isoformat()
            
            cursor.execute('''
            INSERT OR REPLACE INTO spread_config (spread_name, config_json, created_at)
            VALUES (?, ?, ?)
            ''', (spread_name, config_json, created_at))
            
            self.db_manager.conn.commit()
            logger.info(f"价差配置已保存到数据库: {spread_name}")
            
        except Exception as e:
            logger.error(f"保存价差配置失败: {e}")
    
    def calculate_spread(self, spread_name: str, price_data: Dict[str, pd.DataFrame],
                        price_column: str = 'close', convert_to_barrels: bool = True) -> pd.DataFrame:
        """
        计算价差序列
        
        Args:
            spread_name: 价差名称
            price_data: 价格数据字典 {品种代码: DataFrame}
            price_column: 使用的价格列（close/open等）
            convert_to_barrels: 是否将所有价格转换为美元/桶单位（默认True）
        
        Returns:
            包含价差值的DataFrame
        
        Note:
            当 convert_to_barrels=True 时，会自动将不同单位的价格统一为美元/桶：
            - CL (WTI原油): 美元/桶 → 不变
            - RBOB (汽油): 美元/加仑 → 美元/桶 (×42)
            - HO (取暖油): 美元/加仑 → 美元/桶 (×42)
        """
        if spread_name not in self.spread_configs:
            raise ValueError(f"未找到价差配置: {spread_name}")
        
        config = self.spread_configs[spread_name]
        components = config['components']
        
        # 收集所有组件的价格数据
        component_prices = []
        
        for symbol, weight in components:
            if symbol not in price_data:
                raise ValueError(f"缺少品种数据: {symbol}")
            
            df = price_data[symbol]
            if price_column not in df.columns:
                raise ValueError(f"品种 {symbol} 缺少价格列: {price_column}")
            
            # 获取价格序列
            price = df[price_column].copy()
            
            # 单位转换：统一为美元/桶
            if convert_to_barrels:
                price = self.convert_units(price, symbol)
            
            # 创建加权价格序列
            weighted_price = price * weight
            weighted_price.name = f"{symbol}_weighted"
            component_prices.append(weighted_price)
        
        # 合并所有组件（使用内连接确保时间对齐）
        spread_df = pd.concat(component_prices, axis=1, join='inner')
        
        # 计算价差
        spread_df['spread'] = spread_df.sum(axis=1)
        
        # 添加元数据
        spread_df['spread_name'] = spread_name
        
        logger.info(f"计算价差 {spread_name}: {len(spread_df)} 个数据点")
        
        return spread_df[['spread', 'spread_name']]
    
    def calculate_multiple_spreads(self, price_data: Dict[str, pd.DataFrame],
                                  spread_names: List[str] = None,
                                  price_column: str = 'close') -> Dict[str, pd.DataFrame]:
        """
        计算多个价差
        
        Args:
            price_data: 价格数据字典
            spread_names: 要计算的价差名称列表（None表示全部）
            price_column: 使用的价格列
        
        Returns:
            价差数据字典 {价差名称: DataFrame}
        """
        if spread_names is None:
            spread_names = list(self.spread_configs.keys())
        
        results = {}
        
        for spread_name in spread_names:
            try:
                spread_df = self.calculate_spread(spread_name, price_data, price_column)
                results[spread_name] = spread_df
            except Exception as e:
                logger.error(f"计算价差 {spread_name} 失败: {e}")
        
        return results
    
    def get_spread_statistics(self, spread_df: pd.DataFrame, 
                            window: int = 20) -> pd.DataFrame:
        """
        计算价差统计特征
        
        Args:
            spread_df: 价差DataFrame
            window: 滚动窗口大小
        
        Returns:
            包含统计特征的DataFrame
        """
        result = spread_df.copy()
        spread_values = result['spread']
        
        # 基本统计量
        result['spread_mean'] = spread_values.rolling(window).mean()
        result['spread_std'] = spread_values.rolling(window).std()
        result['spread_min'] = spread_values.rolling(window).min()
        result['spread_max'] = spread_values.rolling(window).max()
        
        # Z-score（标准化）
        result['spread_zscore'] = (spread_values - result['spread_mean']) / result['spread_std']
        
        # 百分位数
        result['spread_percentile'] = spread_values.rolling(window).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / len(x) * 100
        )
        
        # 变化率
        result['spread_pct_change'] = spread_values.pct_change()
        result['spread_pct_change_5d'] = spread_values.pct_change(5)
        
        # 高低点距离
        result['distance_to_high'] = result['spread_max'] - spread_values
        result['distance_to_low'] = spread_values - result['spread_min']
        
        logger.info(f"价差统计特征计算完成，窗口: {window}")
        
        return result
    
    def load_config_from_db(self, spread_name: str) -> Dict:
        """从数据库加载配置"""
        if not self.db_manager:
            raise ValueError("未设置数据库管理器")
        
        cursor = self.db_manager.conn.cursor()
        cursor.execute(
            "SELECT config_json FROM spread_config WHERE spread_name = ?",
            (spread_name,)
        )
        
        row = cursor.fetchone()
        if row:
            config = json.loads(row[0])
            self.spread_configs[spread_name] = config
            logger.info(f"从数据库加载价差配置: {spread_name}")
            return config
        else:
            raise ValueError(f"数据库中未找到价差配置: {spread_name}")
    
    def load_all_configs_from_db(self):
        """从数据库加载所有配置"""
        if not self.db_manager:
            raise ValueError("未设置数据库管理器")
        
        cursor = self.db_manager.conn.cursor()
        cursor.execute("SELECT spread_name, config_json FROM spread_config")
        
        count = 0
        for row in cursor.fetchall():
            spread_name, config_json = row
            config = json.loads(config_json)
            self.spread_configs[spread_name] = config
            count += 1
        
        logger.info(f"从数据库加载了 {count} 个价差配置")


def create_standard_crack_spreads(calculator: SpreadCalculator):
    """
    创建标准裂解价差配置
    
    Args:
        calculator: SpreadCalculator实例
    """
    # 1:1裂解价差
    calculator.create_spread("RBOB_CL_1_1", [("RBOB", 1), ("CL", -1)])
    calculator.create_spread("HO_CL_1_1", [("HO", 1), ("CL", -1)])
    
    # 3:2:1裂解价差（最常见的配比）
    calculator.create_spread("CRACK_3_2_1", [("CL", -3), ("RBOB", 2), ("HO", 1)])
    
    # 2:1:1裂解价差
    calculator.create_spread("CRACK_2_1_1", [("CL", -2), ("RBOB", 1), ("HO", 1)])
    
    # 5:3:2裂解价差
    calculator.create_spread("CRACK_5_3_2", [("CL", -5), ("RBOB", 3), ("HO", 2)])
    
    logger.info("标准裂解价差配置创建完成")
