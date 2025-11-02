"""
超参数优化模块
支持网格搜索、随机搜索和贝叶斯优化
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from pathlib import Path
import time

logger = logging.getLogger(__name__)

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    logger.warning("scikit-optimize未安装，贝叶斯优化不可用。安装: pip install scikit-optimize")


class HyperparameterTuner:
    """超参数优化器"""
    
    # 预定义的参数空间
    PARAM_GRIDS = {
        'random_forest': {
            'grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'random': {
                'n_estimators': [50, 100, 150, 200, 300],
                'max_depth': [3, 5, 7, 10, 15, None],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None]
            },
            'bayesian': {
                'n_estimators': Integer(50, 300),
                'max_depth': Integer(3, 15),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
                'max_features': Categorical(['sqrt', 'log2', None])
            }
        },
        'gradient_boosting': {
            'grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10]
            },
            'random': {
                'n_estimators': [50, 100, 150, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10, 15]
            },
            'bayesian': {
                'n_estimators': Integer(50, 300),
                'max_depth': Integer(3, 10),
                'learning_rate': Real(0.001, 0.3, prior='log-uniform'),
                'subsample': Real(0.6, 1.0),
                'min_samples_split': Integer(2, 20)
            }
        },
        'xgboost': {
            'grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'random': {
                'n_estimators': [50, 100, 150, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2, 0.3],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0, 0.1, 0.5, 1.0]
            },
            'bayesian': {
                'n_estimators': Integer(50, 300),
                'max_depth': Integer(3, 10),
                'learning_rate': Real(0.001, 0.3, prior='log-uniform'),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'gamma': Real(0, 0.5),
                'reg_alpha': Real(0, 2.0),
                'reg_lambda': Real(0, 2.0)
            }
        },
        'svm': {
            'grid': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            },
            'random': {
                'C': [0.01, 0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly']
            },
            'bayesian': {
                'C': Real(0.01, 100, prior='log-uniform'),
                'gamma': Categorical(['scale', 'auto']),
                'kernel': Categorical(['rbf', 'linear'])
            }
        },
        'logistic': {
            'grid': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'bayesian': {
                'C': Real(0.01, 100, prior='log-uniform'),
                'penalty': Categorical(['l1', 'l2'])
            }
        }
    }
    
    def __init__(self, model_type: str = 'gradient_boosting', 
                 task: str = 'classification',
                 scoring: Optional[str] = None,
                 cv: int = 5,
                 n_jobs: int = -1,
                 verbose: int = 1):
        """
        初始化超参数优化器
        
        Args:
            model_type: 模型类型
            task: 任务类型 ('classification' 或 'regression')
            scoring: 评分指标
            cv: 交叉验证折数
            n_jobs: 并行任务数
            verbose: 详细程度
        """
        self.model_type = model_type
        self.task = task
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # 设置评分指标
        if scoring is None:
            self.scoring = 'accuracy' if task == 'classification' else 'neg_mean_squared_error'
        else:
            self.scoring = scoring
        
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.search_results_ = None
        self.search_history_ = []
    
    def grid_search(self, model, X_train, y_train, 
                   param_grid: Optional[Dict] = None) -> Dict:
        """
        网格搜索
        
        Args:
            model: 模型实例
            X_train: 训练特征
            y_train: 训练标签
            param_grid: 参数网格（None则使用预定义）
        
        Returns:
            最佳参数字典
        """
        logger.info(f"开始网格搜索优化 {self.model_type}...")
        start_time = time.time()
        
        # 使用预定义参数网格
        if param_grid is None:
            if self.model_type not in self.PARAM_GRIDS:
                raise ValueError(f"未定义模型 {self.model_type} 的参数网格")
            param_grid = self.PARAM_GRIDS[self.model_type]['grid']
        
        # 创建网格搜索
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            return_train_score=True
        )
        
        # 执行搜索
        grid_search.fit(X_train, y_train)
        
        # 保存结果
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.best_estimator_ = grid_search.best_estimator_
        self.search_results_ = pd.DataFrame(grid_search.cv_results_)
        
        elapsed_time = time.time() - start_time
        
        # 记录历史
        self.search_history_.append({
            'method': 'grid_search',
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'time': elapsed_time,
            'n_combinations': len(self.search_results_)
        })
        
        logger.info(f"网格搜索完成，耗时: {elapsed_time:.2f}秒")
        logger.info(f"最佳参数: {self.best_params_}")
        logger.info(f"最佳得分: {self.best_score_:.4f}")
        
        return self.best_params_
    
    def random_search(self, model, X_train, y_train,
                     param_distributions: Optional[Dict] = None,
                     n_iter: int = 50,
                     random_state: int = 42) -> Dict:
        """
        随机搜索
        
        Args:
            model: 模型实例
            X_train: 训练特征
            y_train: 训练标签
            param_distributions: 参数分布（None则使用预定义）
            n_iter: 迭代次数
            random_state: 随机种子
        
        Returns:
            最佳参数字典
        """
        logger.info(f"开始随机搜索优化 {self.model_type}...")
        start_time = time.time()
        
        # 使用预定义参数分布
        if param_distributions is None:
            if self.model_type not in self.PARAM_GRIDS:
                raise ValueError(f"未定义模型 {self.model_type} 的参数分布")
            param_distributions = self.PARAM_GRIDS[self.model_type]['random']
        
        # 创建随机搜索
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=random_state,
            return_train_score=True
        )
        
        # 执行搜索
        random_search.fit(X_train, y_train)
        
        # 保存结果
        self.best_params_ = random_search.best_params_
        self.best_score_ = random_search.best_score_
        self.best_estimator_ = random_search.best_estimator_
        self.search_results_ = pd.DataFrame(random_search.cv_results_)
        
        elapsed_time = time.time() - start_time
        
        # 记录历史
        self.search_history_.append({
            'method': 'random_search',
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'time': elapsed_time,
            'n_iterations': n_iter
        })
        
        logger.info(f"随机搜索完成，耗时: {elapsed_time:.2f}秒")
        logger.info(f"最佳参数: {self.best_params_}")
        logger.info(f"最佳得分: {self.best_score_:.4f}")
        
        return self.best_params_
    
    def bayesian_search(self, model, X_train, y_train,
                       search_spaces: Optional[Dict] = None,
                       n_iter: int = 50,
                       random_state: int = 42) -> Dict:
        """
        贝叶斯优化搜索
        
        Args:
            model: 模型实例
            X_train: 训练特征
            y_train: 训练标签
            search_spaces: 搜索空间（None则使用预定义）
            n_iter: 迭代次数
            random_state: 随机种子
        
        Returns:
            最佳参数字典
        """
        if not BAYESIAN_AVAILABLE:
            raise ImportError("贝叶斯优化需要安装 scikit-optimize: pip install scikit-optimize")
        
        logger.info(f"开始贝叶斯优化搜索 {self.model_type}...")
        start_time = time.time()
        
        # 使用预定义搜索空间
        if search_spaces is None:
            if self.model_type not in self.PARAM_GRIDS:
                raise ValueError(f"未定义模型 {self.model_type} 的搜索空间")
            search_spaces = self.PARAM_GRIDS[self.model_type]['bayesian']
        
        # 创建贝叶斯搜索
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=search_spaces,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=random_state,
            return_train_score=True
        )
        
        # 执行搜索
        bayes_search.fit(X_train, y_train)
        
        # 保存结果
        self.best_params_ = bayes_search.best_params_
        self.best_score_ = bayes_search.best_score_
        self.best_estimator_ = bayes_search.best_estimator_
        self.search_results_ = pd.DataFrame(bayes_search.cv_results_)
        
        elapsed_time = time.time() - start_time
        
        # 记录历史
        self.search_history_.append({
            'method': 'bayesian_search',
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'time': elapsed_time,
            'n_iterations': n_iter
        })
        
        logger.info(f"贝叶斯搜索完成，耗时: {elapsed_time:.2f}秒")
        logger.info(f"最佳参数: {self.best_params_}")
        logger.info(f"最佳得分: {self.best_score_:.4f}")
        
        return self.best_params_
    
    def get_search_results_summary(self) -> pd.DataFrame:
        """
        获取搜索结果摘要
        
        Returns:
            结果摘要DataFrame
        """
        if self.search_results_ is None:
            logger.warning("尚未执行搜索")
            return pd.DataFrame()
        
        # 提取关键列
        summary_cols = [
            'mean_test_score', 'std_test_score',
            'mean_train_score', 'std_train_score',
            'mean_fit_time', 'rank_test_score'
        ]
        
        # 添加参数列
        param_cols = [col for col in self.search_results_.columns if col.startswith('param_')]
        
        summary = self.search_results_[param_cols + summary_cols].copy()
        summary = summary.sort_values('rank_test_score')
        
        return summary
    
    def plot_search_results(self, param_name: str, save_path: Optional[str] = None):
        """
        绘制参数搜索结果
        
        Args:
            param_name: 参数名称
            save_path: 保存路径
        """
        if self.search_results_ is None:
            logger.warning("尚未执行搜索")
            return
        
        import matplotlib.pyplot as plt
        
        param_col = f'param_{param_name}'
        if param_col not in self.search_results_.columns:
            logger.warning(f"参数 {param_name} 不存在")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 按参数值分组并计算平均得分
        grouped = self.search_results_.groupby(param_col).agg({
            'mean_test_score': ['mean', 'std'],
            'mean_train_score': ['mean', 'std']
        })
        
        x = grouped.index
        y_test = grouped['mean_test_score']['mean']
        y_test_std = grouped['mean_test_score']['std']
        y_train = grouped['mean_train_score']['mean']
        
        ax.plot(x, y_test, 'o-', label='Test Score', linewidth=2)
        ax.fill_between(x, y_test - y_test_std, y_test + y_test_std, alpha=0.3)
        ax.plot(x, y_train, 's--', label='Train Score', linewidth=2, alpha=0.7)
        
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Search Results: {param_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"搜索结果图已保存: {save_path}")
        
        plt.close()
    
    def save_results(self, save_dir: str = 'models/tuning_results'):
        """
        保存优化结果
        
        Args:
            save_dir: 保存目录
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存最佳参数
        params_file = save_path / f'{self.model_type}_best_params.json'
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump({
                'best_params': self.best_params_,
                'best_score': float(self.best_score_),
                'model_type': self.model_type,
                'task': self.task,
                'scoring': self.scoring
            }, f, indent=2)
        logger.info(f"最佳参数已保存: {params_file}")
        
        # 保存搜索历史
        if self.search_history_:
            history_file = save_path / f'{self.model_type}_search_history.json'
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.search_history_, f, indent=2)
            logger.info(f"搜索历史已保存: {history_file}")
        
        # 保存详细结果
        if self.search_results_ is not None:
            results_file = save_path / f'{self.model_type}_search_results.csv'
            self.search_results_.to_csv(results_file, index=False)
            logger.info(f"搜索结果已保存: {results_file}")
    
    def compare_methods(self, methods_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        比较不同搜索方法的结果
        
        Args:
            methods_results: 方法结果字典 {方法名: {'best_score': x, 'time': y}}
        
        Returns:
            比较结果DataFrame
        """
        comparison = pd.DataFrame(methods_results).T
        comparison = comparison.sort_values('best_score', ascending=False)
        
        logger.info("\n搜索方法比较:")
        logger.info(comparison.to_string())
        
        return comparison
