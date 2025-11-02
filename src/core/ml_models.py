"""
机器学习模型模块
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class MLModel:
    """机器学习模型基类"""
    
    def __init__(self, model_type: str = 'xgboost', task: str = 'classification'):
        """
        初始化模型
        
        Args:
            model_type: 模型类型（xgboost, lightgbm, random_forest, gradient_boosting）
            task: 任务类型（classification, regression）
        """
        self.model_type = model_type
        self.task = task
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importance = None
    
    def _create_model(self, **kwargs):
        """创建模型实例"""
        if self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                if self.task == 'classification':
                    self.model = xgb.XGBClassifier(**kwargs)
                else:
                    self.model = xgb.XGBRegressor(**kwargs)
            except ImportError:
                logger.warning("XGBoost未安装，使用GradientBoosting代替")
                self.model_type = 'gradient_boosting'
                self._create_model(**kwargs)
        
        elif self.model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                if self.task == 'classification':
                    self.model = lgb.LGBMClassifier(**kwargs)
                else:
                    self.model = lgb.LGBMRegressor(**kwargs)
            except ImportError:
                logger.warning("LightGBM未安装，使用GradientBoosting代替")
                self.model_type = 'gradient_boosting'
                self._create_model(**kwargs)
        
        elif self.model_type == 'random_forest':
            if self.task == 'classification':
                self.model = RandomForestClassifier(**kwargs)
            else:
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(**kwargs)
        
        elif self.model_type == 'gradient_boosting':
            if self.task == 'classification':
                self.model = GradientBoostingClassifier(**kwargs)
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                self.model = GradientBoostingRegressor(**kwargs)
        
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'target',
                    feature_cols: Optional[list] = None,
                    test_size: float = 0.2, 
                    scale: bool = True) -> Tuple:
        """
        准备训练数据
        
        Args:
            df: 完整数据DataFrame
            target_col: 目标变量列名
            feature_cols: 特征列名列表（None表示除目标外的所有列）
            test_size: 测试集比例
            scale: 是否标准化
        
        Returns:
            (X_train, X_test, y_train, y_test, train_index, test_index)
        """
        # 移除缺失值
        df_clean = df.dropna()
        
        # 选择特征
        if feature_cols is None:
            feature_cols = [col for col in df_clean.columns if col != target_col]
        
        self.feature_names = feature_cols
        
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values
        
        # 时间序列分割（按顺序分割，不打乱）
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        train_index = df_clean.index[:split_idx]
        test_index = df_clean.index[split_idx:]
        
        # 标准化
        if scale:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        logger.info(f"数据准备完成:")
        logger.info(f"  特征数: {len(feature_cols)}")
        logger.info(f"  训练集样本数: {len(X_train)}")
        logger.info(f"  测试集样本数: {len(X_test)}")
        
        if self.task == 'classification':
            unique, counts = np.unique(y_train, return_counts=True)
            logger.info(f"  训练集标签分布: {dict(zip(unique, counts))}")
        
        return X_train, X_test, y_train, y_test, train_index, test_index
    
    def train(self, X_train, y_train, **model_params):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            **model_params: 模型参数
        """
        # 创建模型
        self._create_model(**model_params)
        
        logger.info(f"开始训练 {self.model_type} 模型...")
        
        # 训练
        self.model.fit(X_train, y_train)
        
        # 获取特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 10 重要特征:")
            logger.info(self.feature_importance.head(10).to_string())
        
        logger.info("模型训练完成")
    
    def predict(self, X):
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            logger.warning("模型不支持概率预测")
            return None
    
    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
        
        Returns:
            评估指标字典
        """
        y_pred = self.predict(X_test)
        
        if self.task == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            logger.info("\n模型评估结果:")
            logger.info(f"准确率: {metrics['accuracy']:.4f}")
            logger.info(f"精确率: {metrics['precision']:.4f}")
            logger.info(f"召回率: {metrics['recall']:.4f}")
            logger.info(f"F1分数: {metrics['f1']:.4f}")
            
            logger.info("\n分类报告:")
            logger.info(classification_report(y_test, y_pred))
            
            logger.info("\n混淆矩阵:")
            logger.info(confusion_matrix(y_test, y_pred))
            
        else:  # regression
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            logger.info("\n模型评估结果:")
            logger.info(f"MSE: {metrics['mse']:.6f}")
            logger.info(f"RMSE: {metrics['rmse']:.6f}")
            logger.info(f"MAE: {metrics['mae']:.6f}")
            logger.info(f"R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def cross_validate(self, X, y, n_splits: int = 5) -> Dict[str, float]:
        """
        时间序列交叉验证
        
        Args:
            X: 特征
            y: 标签
            n_splits: 分割数
        
        Returns:
            交叉验证结果
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            self.model.fit(X_train_cv, y_train_cv)
            y_pred = self.model.predict(X_val_cv)
            
            if self.task == 'classification':
                score = accuracy_score(y_val_cv, y_pred)
            else:
                from sklearn.metrics import r2_score
                score = r2_score(y_val_cv, y_pred)
            
            scores.append(score)
            logger.info(f"Fold {fold + 1}: {score:.4f}")
        
        result = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores
        }
        
        logger.info(f"\n交叉验证结果: {result['mean_score']:.4f} (+/- {result['std_score']:.4f})")
        
        return result
    
    def hyperparameter_tuning(self, X_train, y_train, param_grid: Dict,
                            cv: int = 3) -> Dict[str, Any]:
        """
        超参数调优
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            param_grid: 参数网格
            cv: 交叉验证折数
        
        Returns:
            最佳参数
        """
        logger.info("开始超参数调优...")
        
        # 使用时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=cv)
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=tscv,
            scoring='accuracy' if self.task == 'classification' else 'r2',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"最佳参数: {grid_search.best_params_}")
        logger.info(f"最佳得分: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'task': self.task
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"模型已保存: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.task = model_data['task']
        
        logger.info(f"模型已加载: {filepath}")


class SignalGenerator:
    """交易信号生成器"""
    
    def __init__(self, model: MLModel, 
                 threshold: float = 0.6, 
                 signal_holding_days: int = 5,
                 regression_upper_threshold: float = 0.05,
                 regression_lower_threshold: float = -0.05,
                 use_rolling_quantile: bool = False,
                 rolling_window: int = 60,
                 upper_quantile: float = 0.75,
                 lower_quantile: float = 0.25):
        """
        初始化信号生成器
        
        Args:
            model: 训练好的MLModel实例
            threshold: 概率阈值（用于分类任务）
            signal_holding_days: 信号维持天数，默认为1天（即每天重新生成信号）
                                当设置>1时，信号会向后填充n天，除非出现相反方向的信号
            
            回归任务参数：
            regression_upper_threshold: 回归预测值上阈值，超过则信号=1（做多）
            regression_lower_threshold: 回归预测值下阈值，低于则信号=-1（做空）
            use_rolling_quantile: 是否使用滚动分位数作为动态阈值
            rolling_window: 滚动窗口大小（用于计算分位数）
            upper_quantile: 上分位数（如0.75表示75%分位数）
            lower_quantile: 下分位数（如0.25表示25%分位数）
            
        回归任务信号生成逻辑：
            - 固定阈值模式（use_rolling_quantile=False）:
              - prediction > regression_upper_threshold → signal = 1
              - prediction < regression_lower_threshold → signal = -1
              - 其他 → signal = 0
            
            - 滚动分位数模式（use_rolling_quantile=True）:
              - prediction > rolling_upper_quantile → signal = 1
              - prediction < rolling_lower_quantile → signal = -1
              - 其他 → signal = 0
              
        示例：
            # 固定阈值：预测涨跌幅超过±5%时开仓
            SignalGenerator(model, regression_upper_threshold=0.05, 
                          regression_lower_threshold=-0.05)
            
            # 动态分位数：预测值在75%和25%分位数之外时开仓
            SignalGenerator(model, use_rolling_quantile=True, 
                          rolling_window=60, upper_quantile=0.75, 
                          lower_quantile=0.25)
        """
        self.model = model
        self.threshold = threshold
        self.signal_holding_days = signal_holding_days
        
        # 回归任务参数
        self.regression_upper_threshold = regression_upper_threshold
        self.regression_lower_threshold = regression_lower_threshold
        self.use_rolling_quantile = use_rolling_quantile
        self.rolling_window = rolling_window
        self.upper_quantile = upper_quantile
        self.lower_quantile = lower_quantile
    
    def generate_signals(self, X: np.ndarray, 
                        use_probability: bool = True) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            X: 特征数据
            use_probability: 是否使用概率阈值（仅用于分类任务）
        
        Returns:
            包含信号的DataFrame
        """
        signals = pd.DataFrame()
        
        # 预测
        predictions = self.model.predict(X)
        signals['prediction'] = predictions
        
        # 根据任务类型生成信号
        if self.model.task == 'classification':
            # 分类任务：使用概率阈值
            signals = self._generate_classification_signals(signals, X, use_probability)
        
        elif self.model.task == 'regression':
            # 回归任务：使用阈值或滚动分位数
            signals = self._generate_regression_signals(signals)
        
        else:
            # 未知任务类型，直接使用预测值作为信号
            logger.warning(f"未知任务类型: {self.model.task}, 直接使用预测值作为信号")
            signals['signal'] = predictions
            signals['signal_strength'] = 1.0
        
        logger.info(f"生成交易信号完成，信号分布:")
        logger.info(signals['signal'].value_counts())
        
        # 应用信号维持逻辑
        if self.signal_holding_days > 1:
            signals = self._apply_signal_holding(signals)
            logger.info(f"应用{self.signal_holding_days}天信号维持后，信号分布:")
            logger.info(signals['signal'].value_counts())
        
        return signals
    
    def _generate_classification_signals(self, signals: pd.DataFrame, 
                                        X: np.ndarray,
                                        use_probability: bool) -> pd.DataFrame:
        """
        生成分类任务的交易信号
        
        Args:
            signals: 包含预测值的DataFrame
            X: 特征数据
            use_probability: 是否使用概率阈值
        
        Returns:
            包含信号的DataFrame
        """
        predictions = signals['prediction'].values
        
        if use_probability:
            proba = self.model.predict_proba(X)
            
            if proba is not None:
                # 获取每个类别的概率
                for i in range(proba.shape[1]):
                    signals[f'proba_class_{i}'] = proba[:, i]
                
                # 最大概率
                signals['max_proba'] = proba.max(axis=1)
                
                # 应用阈值过滤
                signals['signal'] = predictions
                signals.loc[signals['max_proba'] < self.threshold, 'signal'] = 0  # 观望
                
                # 信号强度
                signals['signal_strength'] = signals['max_proba']
            else:
                signals['signal'] = predictions
                signals['signal_strength'] = 1.0
        else:
            signals['signal'] = predictions
            signals['signal_strength'] = 1.0
        
        return signals
    
    def _generate_regression_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        生成回归任务的交易信号（支持固定阈值和滚动分位数）
        
        Args:
            signals: 包含预测值的DataFrame
        
        Returns:
            包含信号的DataFrame
        
        信号规则：
            - 固定阈值模式：
              - prediction > upper_threshold → signal = 1
              - prediction < lower_threshold → signal = -1
              - 其他 → signal = 0
            
            - 滚动分位数模式：
              - prediction > rolling_upper_quantile → signal = 1
              - prediction < rolling_lower_quantile → signal = -1
              - 其他 → signal = 0
        """
        predictions = signals['prediction'].values
        n = len(predictions)
        signal_values = np.zeros(n, dtype=int)
        signal_strength = np.zeros(n)
        
        if self.use_rolling_quantile:
            # 使用滚动分位数作为动态阈值
            logger.info(f"使用滚动分位数模式: window={self.rolling_window}, "
                       f"上分位数={self.upper_quantile}, 下分位数={self.lower_quantile}")
            
            # 计算滚动分位数
            pred_series = pd.Series(predictions)
            upper_thresholds = pred_series.rolling(
                window=self.rolling_window, 
                min_periods=max(1, self.rolling_window // 2)
            ).quantile(self.upper_quantile)
            
            lower_thresholds = pred_series.rolling(
                window=self.rolling_window,
                min_periods=max(1, self.rolling_window // 2)
            ).quantile(self.lower_quantile)
            
            # 前向填充（确保初始窗口也有阈值）
            upper_thresholds = upper_thresholds.fillna(method='bfill')
            lower_thresholds = lower_thresholds.fillna(method='bfill')
            
            # 保存阈值到DataFrame（用于调试和可视化）
            signals['upper_threshold'] = upper_thresholds.values
            signals['lower_threshold'] = lower_thresholds.values
            
            # 生成信号（向量化）
            signal_values[predictions > upper_thresholds.values] = 1
            signal_values[predictions < lower_thresholds.values] = -1
            
            # 计算信号强度（基于预测值偏离阈值的程度）
            # 信号强度 = |prediction - threshold| / |threshold|
            threshold_range = upper_thresholds.values - lower_thresholds.values
            threshold_range = np.where(threshold_range > 0, threshold_range, 1e-6)  # 避免除零
            
            for i in range(n):
                if signal_values[i] == 1:
                    signal_strength[i] = min(1.0, abs(predictions[i] - upper_thresholds.iloc[i]) / 
                                           threshold_range[i])
                elif signal_values[i] == -1:
                    signal_strength[i] = min(1.0, abs(predictions[i] - lower_thresholds.iloc[i]) / 
                                           threshold_range[i])
                else:
                    signal_strength[i] = 0.0
            
            logger.info(f"滚动分位数统计:")
            logger.info(f"  上阈值范围: [{upper_thresholds.min():.4f}, {upper_thresholds.max():.4f}], "
                       f"均值: {upper_thresholds.mean():.4f}")
            logger.info(f"  下阈值范围: [{lower_thresholds.min():.4f}, {lower_thresholds.max():.4f}], "
                       f"均值: {lower_thresholds.mean():.4f}")
        
        else:
            # 使用固定阈值
            logger.info(f"使用固定阈值模式: 上阈值={self.regression_upper_threshold}, "
                       f"下阈值={self.regression_lower_threshold}")
            
            upper_threshold = self.regression_upper_threshold
            lower_threshold = self.regression_lower_threshold
            
            # 生成信号（向量化）
            signal_values[predictions > upper_threshold] = 1
            signal_values[predictions < lower_threshold] = -1
            
            # 计算信号强度
            threshold_range = upper_threshold - lower_threshold
            threshold_range = max(threshold_range, 1e-6)  # 避免除零
            
            for i in range(n):
                if signal_values[i] == 1:
                    signal_strength[i] = min(1.0, abs(predictions[i] - upper_threshold) / 
                                           threshold_range)
                elif signal_values[i] == -1:
                    signal_strength[i] = min(1.0, abs(predictions[i] - lower_threshold) / 
                                           threshold_range)
                else:
                    signal_strength[i] = 0.0
        
        # 保存信号和强度
        signals['signal'] = signal_values
        signals['signal_strength'] = signal_strength
        
        # 统计信息
        logger.info(f"回归信号统计:")
        logger.info(f"  预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
        logger.info(f"  做多信号(1): {(signal_values == 1).sum()} ({(signal_values == 1).sum() / n * 100:.1f}%)")
        logger.info(f"  观望信号(0): {(signal_values == 0).sum()} ({(signal_values == 0).sum() / n * 100:.1f}%)")
        logger.info(f"  做空信号(-1): {(signal_values == -1).sum()} ({(signal_values == -1).sum() / n * 100:.1f}%)")
        
        return signals
    
    def _apply_signal_holding(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        应用信号维持逻辑（向量化优化版本）
        
        当信号产生后，后续n天该信号应当继续维持，除非n天内出现了完全相反方向的信号
        例如：出了信号1，向后填充n天，但如果n天内出现信号-1，则在-1处停止填充并反向开仓
        
        算法优化：
        1. 使用NumPy向量化操作代替Python循环
        2. 预先计算信号变化点，减少重复判断
        3. 使用滚动窗口快速检测相反信号
        
        Args:
            signals: 原始信号DataFrame
        
        Returns:
            应用维持逻辑后的信号DataFrame
        """
        # 创建副本避免修改原始数据
        signals_held = signals.copy()
        signal_col = 'signal'
        
        # 获取信号值（NumPy数组，性能更好）
        signal_values = signals_held[signal_col].values.copy()
        n = len(signal_values)
        
        # 特殊情况：信号维持天数为1或数据为空
        if self.signal_holding_days <= 1 or n == 0:
            return signals_held
        
        # 方法1：优化的循环（减少条件判断）
        # 找出所有非零信号的位置（向量化）
        nonzero_indices = np.where(signal_values != 0)[0]
        
        if len(nonzero_indices) == 0:
            return signals_held
        
        # 使用numba加速（如果可用）
        try:
            signal_values = self._apply_holding_numba(
                signal_values, nonzero_indices, self.signal_holding_days
            )
        except (ImportError, AttributeError):
            # numba不可用，使用优化的NumPy实现
            signal_values = self._apply_holding_numpy(
                signal_values, nonzero_indices, self.signal_holding_days
            )
        
        # 更新信号列
        signals_held[signal_col] = signal_values
        
        return signals_held
    
    @staticmethod
    def _apply_holding_numpy(signal_values: np.ndarray, 
                            nonzero_indices: np.ndarray,
                            holding_days: int) -> np.ndarray:
        """
        使用NumPy优化的信号维持实现
        
        Args:
            signal_values: 信号数组
            nonzero_indices: 非零信号的索引位置
            holding_days: 维持天数
        
        Returns:
            应用维持逻辑后的信号数组
        """
        n = len(signal_values)
        result = signal_values.copy()
        
        # 遍历每个非零信号点
        i = 0
        while i < len(nonzero_indices):
            idx = nonzero_indices[i]
            current_signal = result[idx]
            
            # 如果当前位置已被之前的信号覆盖，跳过
            if current_signal != signal_values[idx]:
                i += 1
                continue
            
            # 计算维持的结束位置
            hold_end = min(idx + holding_days, n)
            
            # 检查维持区间内是否有相反信号
            # 向量化操作：找出区间内的原始非零信号
            range_slice = slice(idx + 1, hold_end)
            range_signals = signal_values[range_slice]
            
            # 找到相反信号的位置（向量化）
            opposite_mask = (range_signals != 0) & (range_signals != current_signal)
            
            if np.any(opposite_mask):
                # 找到第一个相反信号的位置
                opposite_positions = np.where(opposite_mask)[0]
                first_opposite = opposite_positions[0]
                actual_end = idx + 1 + first_opposite
            else:
                actual_end = hold_end
            
            # 向量化填充信号
            result[idx + 1:actual_end] = current_signal
            
            i += 1
        
        return result
    
    @staticmethod
    def _apply_holding_numba(signal_values: np.ndarray,
                            nonzero_indices: np.ndarray,
                            holding_days: int) -> np.ndarray:
        """
        使用Numba JIT编译加速的信号维持实现
        
        Numba可以将Python代码编译成机器码，大幅提升性能
        如果numba未安装，会fallback到NumPy版本
        
        Args:
            signal_values: 信号数组
            nonzero_indices: 非零信号的索引位置
            holding_days: 维持天数
        
        Returns:
            应用维持逻辑后的信号数组
        """
        try:
            from numba import jit
            
            @jit(nopython=True)
            def _holding_core(signals, indices, hold_days):
                """Numba加速的核心算法"""
                n = len(signals)
                result = signals.copy()
                
                i = 0
                while i < len(indices):
                    idx = indices[i]
                    current_signal = result[idx]
                    
                    # 计算维持结束位置
                    hold_end = min(idx + hold_days, n)
                    
                    # 向后填充，检查相反信号
                    for j in range(idx + 1, hold_end):
                        original_signal = signals[j]
                        
                        # 遇到相反信号，停止填充
                        if original_signal != 0 and original_signal != current_signal:
                            break
                        
                        # 填充信号
                        result[j] = current_signal
                    
                    i += 1
                
                return result
            
            return _holding_core(signal_values, nonzero_indices, holding_days)
            
        except ImportError:
            # Numba未安装，使用NumPy版本
            raise ImportError("Numba not available")
    
    def backtest_signals(self, signals: pd.DataFrame, returns: pd.Series) -> Dict[str, float]:
        """
        回测信号表现
        
        Args:
            signals: 信号DataFrame
            returns: 实际收益率Series
        
        Returns:
            回测结果字典
        """
        # 计算策略收益
        strategy_returns = signals['signal'] * returns
        
        # 计算累计收益
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # 计算指标
        results = {
            'total_return': cumulative_returns.iloc[-1] - 1,
            'annual_return': strategy_returns.mean() * 252,
            'volatility': strategy_returns.std() * np.sqrt(252),
            'sharpe_ratio': (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252),
            'max_drawdown': (cumulative_returns / cumulative_returns.cummax() - 1).min(),
            'win_rate': (strategy_returns > 0).sum() / len(strategy_returns)
        }
        
        logger.info("\n信号回测结果:")
        for key, value in results.items():
            logger.info(f"{key}: {value:.4f}")
        
        return results
