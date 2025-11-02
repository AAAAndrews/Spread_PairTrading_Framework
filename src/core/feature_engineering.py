"""
特征工程模块：为机器学习模型准备特征
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self, indicator_builder=None):
        """
        初始化特征工程器
        
        Args:
            indicator_builder: IndicatorBuilder实例
        """
        self.indicator_builder = indicator_builder
        self.feature_names = []
    
    def create_spread_features(self, spread_df: pd.DataFrame) -> pd.DataFrame:
        """
        创建价差相关特征
        
        Args:
            spread_df: 价差DataFrame
        
        Returns:
            包含特征的DataFrame
        """
        features = spread_df.copy()
        
        if 'spread' not in features.columns:
            logger.error("DataFrame中缺少'spread'列")
            return features
        
        spread = features['spread']
        
        # 滚动统计特征
        for window in [5, 10, 20, 60]:
            features[f'spread_ma_{window}'] = spread.rolling(window).mean()
            features[f'spread_std_{window}'] = spread.rolling(window).std()
            features[f'spread_min_{window}'] = spread.rolling(window).min()
            features[f'spread_max_{window}'] = spread.rolling(window).max()
            
            # 归一化特征
            ma = features[f'spread_ma_{window}']
            std = features[f'spread_std_{window}']
            features[f'spread_zscore_{window}'] = (spread - ma) / std
            
            # 相对位置
            range_val = features[f'spread_max_{window}'] - features[f'spread_min_{window}']
            features[f'spread_position_{window}'] = (spread - features[f'spread_min_{window}']) / range_val
        
        # 变化率特征
        for period in [1, 5, 10, 20]:
            features[f'spread_return_{period}d'] = spread.pct_change(period)
            features[f'spread_diff_{period}d'] = spread.diff(period)
        
        # 高阶统计特征
        for window in [20, 60]:
            features[f'spread_skew_{window}'] = spread.rolling(window).skew()
            features[f'spread_kurt_{window}'] = spread.rolling(window).kurt()
        
        # 趋势特征
        features['spread_ema_12'] = spread.ewm(span=12).mean()
        features['spread_ema_26'] = spread.ewm(span=26).mean()
        features['spread_macd'] = features['spread_ema_12'] - features['spread_ema_26']
        
        logger.info(f"创建价差特征完成，特征数: {len(features.columns) - len(spread_df.columns)}")
        
        return features
    
    def create_price_features(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        创建价格相关特征
        
        Args:
            price_data: 价格数据字典 {品种代码: DataFrame}
        
        Returns:
            包含特征的DataFrame
        """
        all_features = []
        
        for symbol, df in price_data.items():
            if df.empty:
                continue
            
            features = pd.DataFrame(index=df.index)
            
            # 基本价格特征
            if 'close' in df.columns:
                close = df['close']
                
                # 收益率
                for period in [1, 5, 10, 20]:
                    features[f'{symbol}_return_{period}d'] = close.pct_change(period)
                
                # 移动平均
                for window in [5, 10, 20, 50]:
                    ma = close.rolling(window).mean()
                    features[f'{symbol}_ma_{window}'] = ma
                    features[f'{symbol}_price_to_ma_{window}'] = close / ma - 1
                
                # 波动率
                for window in [10, 20]:
                    features[f'{symbol}_volatility_{window}'] = close.pct_change().rolling(window).std()
            
            # 成交量特征
            if 'volume' in df.columns:
                volume = df['volume']
                features[f'{symbol}_volume_ma_20'] = volume.rolling(20).mean()
                features[f'{symbol}_volume_ratio'] = volume / features[f'{symbol}_volume_ma_20']
            
            all_features.append(features)
        
        # 合并所有品种的特征
        if all_features:
            combined = pd.concat(all_features, axis=1)
            logger.info(f"创建价格特征完成，特征数: {len(combined.columns)}")
            return combined
        else:
            return pd.DataFrame()
    
    def create_technical_features(self, df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        """
        创建技术指标特征
        
        Args:
            df: 价格DataFrame
            symbol: 品种代码前缀
        
        Returns:
            包含技术指标的DataFrame
        """
        if not self.indicator_builder:
            logger.warning("未设置IndicatorBuilder，跳过技术指标计算")
            return df
        
        result = df.copy()
        prefix = f"{symbol}_" if symbol else ""
        
        # RSI
        result = self.indicator_builder.calculate_rsi(result, period=14)
        if 'RSI_14' in result.columns:
            result.rename(columns={'RSI_14': f'{prefix}RSI_14'}, inplace=True)
        
        # MACD
        result = self.indicator_builder.calculate_macd(result)
        for col in ['MACD', 'MACD_signal', 'MACD_hist']:
            if col in result.columns:
                result.rename(columns={col: f'{prefix}{col}'}, inplace=True)
        
        # 布林带
        result = self.indicator_builder.calculate_bollinger_bands(result)
        for col in ['BB_middle', 'BB_upper', 'BB_lower', 'BB_width', 'BB_percent']:
            if col in result.columns:
                result.rename(columns={col: f'{prefix}{col}'}, inplace=True)
        
        logger.info(f"创建技术指标特征完成: {prefix}")
        
        return result
    
    def create_fundamental_features(self, fundamental_data: Dict[str, pd.DataFrame],
                                   target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        创建基本面特征
        
        Args:
            fundamental_data: 基本面数据字典
            target_index: 目标时间索引
        
        Returns:
            包含基本面特征的DataFrame
        """
        features = pd.DataFrame(index=target_index)
        
        for source, df in fundamental_data.items():
            if df.empty:
                continue
            
            # 提取数据字段
            if 'data' in df.columns:
                # 展开JSON数据
                data_expanded = pd.json_normalize(df['data'])
                data_expanded.index = df['report_date']
                
                # 对齐到目标时间（向后查找）
                for col in data_expanded.columns:
                    aligned = pd.merge_asof(
                        pd.DataFrame(index=target_index).reset_index(),
                        data_expanded[[col]].reset_index(),
                        left_on='index',
                        right_on='report_date',
                        direction='backward'
                    )
                    features[f'{source}_{col}'] = aligned[col].values
        
        logger.info(f"创建基本面特征完成，特征数: {len(features.columns)}")
        
        return features
    
    def create_macro_features(self, macro_data: Dict[str, pd.DataFrame],
                            target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        创建宏观经济特征
        
        Args:
            macro_data: 宏观数据字典 {指标名: DataFrame}
            target_index: 目标时间索引
        
        Returns:
            包含宏观特征的DataFrame
        """
        features = pd.DataFrame(index=target_index)
        
        for indicator, df in macro_data.items():
            if df.empty:
                continue
            
            # 对齐数据
            if 'close' in df.columns:
                aligned = self.indicator_builder.align_with_trading_time(
                    df[['close']], target_index
                )
                features[f'{indicator}'] = aligned['close']
                
                # 添加变化率
                features[f'{indicator}_return_1d'] = features[f'{indicator}'].pct_change()
                features[f'{indicator}_return_5d'] = features[f'{indicator}'].pct_change(5)
        
        logger.info(f"创建宏观特征完成，特征数: {len(features.columns)}")
        
        return features
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建季节性特征
        
        Args:
            df: DataFrame
        
        Returns:
            包含季节性特征的DataFrame
        """
        if not self.indicator_builder:
            logger.warning("未设置IndicatorBuilder")
            return df
        
        return self.indicator_builder.calculate_seasonal_features(df)
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str],
                          lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """
        创建滞后特征
        
        Args:
            df: DataFrame
            columns: 要创建滞后的列名
            lags: 滞后期数
        
        Returns:
            包含滞后特征的DataFrame
        """
        result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for lag in lags:
                result[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        logger.info(f"创建滞后特征完成，列数: {len(columns)}, 滞后期: {lags}")
        
        return result
    
    def create_target_variable(self, spread_df: pd.DataFrame, 
                             method: str = 'classification',
                             forward_period: int = 10,
                             threshold: float = 0.03) -> pd.DataFrame:
        """
        创建目标变量（标签）
        
        Args:
            spread_df: 价差DataFrame
            method: 方法（classification-分类，regression-回归）
            forward_period: 前瞻期（天数）
            threshold: 分类阈值（百分比）
        
        Returns:
            包含目标变量的DataFrame
        """
        result = spread_df.copy()
        
        if 'spread' not in result.columns:
            logger.error("DataFrame中缺少'spread'列")
            return result
        
        # 计算未来收益率
        result['forward_return'] = result['spread'].shift(-forward_period) / result['spread'] - 1

        # 计算未来窗口的收益率序列
        future_returns = pd.DataFrame({
            f'fr_{i}': result['spread'].shift(-i) / result['spread'] - 1
            for i in range(1, forward_period + 1)
        })

        # 未来窗口收益率的标准差
        result['forward_sigma'] = future_returns.std(axis=1)

        # 未来窗口“夏普比率”
        result['forward_sharpe'] = result['forward_return'] / (result['forward_sigma'] + 1e-8)
        
        if method == 'classification':
            # 三分类：做多(1)、观望(0)、做空(-1)
            result['target'] = 0
            result.loc[result['forward_return'] > threshold, 'target'] = 1  # 做多
            result.loc[result['forward_return'] < -threshold, 'target'] = -1  # 做空
            
            logger.info(f"创建分类目标变量，阈值: {threshold}")
            logger.info(f"标签分布:\n{result['target'].value_counts()}")

        elif method == 'revenue_regress':
            result['target'] = result['forward_return']
            logger.info(f"创建回归目标变量")

        elif method == 'sharpe_regress':
            result['target'] = result['forward_sharpe']
            logger.info(f"创建夏普比率回归目标变量")

        result = result.drop(columns=['forward_return'])

        return result
    
    def merge_all_features(self, feature_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        合并所有特征
        
        Args:
            feature_dfs: 特征DataFrame列表
        
        Returns:
            合并后的DataFrame
        """
        if not feature_dfs:
            return pd.DataFrame()
        
        # 使用内连接合并，确保时间对齐
        result = feature_dfs[0]
        
        for df in feature_dfs[1:]:
            result = result.join(df, how='inner')
        
        logger.info(f"特征合并完成，总特征数: {len(result.columns)}, 样本数: {len(result)}")
        
        return result
    
    def clean_features(self, df: pd.DataFrame, 
                      fill_method: str = 'drop',
                      inf_replace: Optional[float] = None) -> pd.DataFrame:
        """
        清理特征数据
        
        Args:
            df: 特征DataFrame
            fill_method: 缺失值处理方法（drop, ffill, mean）
            inf_replace: 无穷值替换值
        
        Returns:
            清理后的DataFrame
        """
        result = df.copy()
        
        # 替换无穷值
        if inf_replace is not None:
            result.replace([np.inf, -np.inf], inf_replace, inplace=True)
        else:
            result.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 处理缺失值
        if fill_method == 'drop':
            result = result.dropna()
            logger.info(f"删除缺失值后剩余样本数: {len(result)}")
        elif fill_method == 'ffill':
            result = result.fillna(method='ffill')
        elif fill_method == 'mean':
            result = result.fillna(result.mean())
        
        return result
    
    def select_features(self, df: pd.DataFrame, target_col: str = 'target',
                       method: str = 'correlation', top_k: int = 50) -> List[str]:
        """
        特征选择
        
        Args:
            df: 包含特征和目标的DataFrame
            target_col: 目标变量列名
            method: 选择方法（correlation, variance）
            top_k: 选择特征数量
        
        Returns:
            选择的特征列表
        """
        if target_col not in df.columns:
            logger.error(f"目标变量 {target_col} 不存在")
            return []
        
        feature_cols = [col for col in df.columns if col != target_col]
        
        if method == 'correlation':
            # 计算与目标的相关性
            correlations = df[feature_cols].corrwith(df[target_col]).abs()
            selected = correlations.nlargest(top_k).index.tolist()
            
        elif method == 'variance':
            # 选择方差最大的特征
            variances = df[feature_cols].var()
            selected = variances.nlargest(top_k).index.tolist()
        
        else:
            selected = feature_cols[:top_k]
        
        logger.info(f"特征选择完成，选择了 {len(selected)} 个特征")
        
        return selected
