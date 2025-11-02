"""
主程序：裂解价差交易策略
整合所有模块，执行完整的策略研究流程
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from src.core.database import DatabaseManager
from src.core.data_fetcher import DataFetcher, FuturesAdjuster
from src.core.spread_calculator import SpreadCalculator, create_standard_crack_spreads
from src.core.indicators import IndicatorBuilder
from src.core.feature_engineering import FeatureEngineer
from src.core.ml_models import MLModel, SignalGenerator
from src.core.backtest import BacktestEngine, PerformanceAnalyzer
from src.core.visualization import Visualizer

# 配置日志
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'trading_strategy.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CrackSpreadStrategy:
    """裂解价差交易策略"""
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        """
        初始化策略
        
        Args:
            db_path: 数据库路径
        """
        logger.info("初始化裂解价差交易策略...")
        
        # 初始化各模块
        self.db = DatabaseManager(db_path)
        self.fetcher = DataFetcher()
        self.spread_calc = SpreadCalculator(self.db)
        self.indicator_builder = IndicatorBuilder(self.db)
        self.feature_engineer = FeatureEngineer(self.indicator_builder)
        self.visualizer = Visualizer()
        
        # 数据存储
        self.price_data = {}
        self.spread_data = {}
        self.features_df = None
        self.model = None
        
        logger.info("策略初始化完成")
    
    def fetch_all_data(self):
        """步骤1：获取所有需要的数据"""
        logger.info("\n" + "="*60)
        logger.info("步骤1：数据获取")
        logger.info("="*60)
        
        # 获取期货价格数据（使用yfinance）
        symbols = {
            'CL': 'CL=F',    # WTI原油
            'RBOB': 'RB=F',  # RBOB汽油
            'HO': 'HO=F'     # 取暖油/柴油
        }
        
        for name, symbol in symbols.items():
            logger.info(f"获取 {name} 数据...")
            df = self.fetcher.fetch_yfinance_data(
                symbol, 
                start_date='2014-01-01',
                interval='1d'
            )
            
            if not df.empty:
                # 期货复权处理
                df_adjusted = FuturesAdjuster.adjust_futures_roll(df, method='ratio')
                
                # 保存到数据库
                self.db.insert_price_data(df_adjusted, name, 'adjusted')
                self.price_data[name] = df_adjusted
                logger.info(f"{name} 数据获取成功: {len(df_adjusted)} 条记录")
            else:
                logger.warning(f"{name} 数据获取失败")
        
        # 获取宏观数据
        logger.info("获取宏观经济数据...")
        
        # VIX波动率指数
        vix_data = self.fetcher.fetch_index_data('VIX', start_date='2014-01-01')
        if not vix_data.empty:
            self.db.insert_price_data(vix_data, 'VIX', 'index')
            logger.info(f"VIX数据获取成功: {len(vix_data)} 条记录")
        
        # 美元指数
        dxy_data = self.fetcher.fetch_index_data('DXY', start_date='2014-01-01')
        if not dxy_data.empty:
            self.db.insert_price_data(dxy_data, 'DXY', 'index')
            logger.info(f"DXY数据获取成功: {len(dxy_data)} 条记录")
        
        # 获取基本面数据（示例数据）
        logger.info("获取基本面数据...")
        eia_data = self.fetcher.fetch_eia_data('PET.WCRSTUS1.W')
        if not eia_data.empty:
            for date, row in eia_data.iterrows():
                self.db.insert_fundamental_data(
                    'EIA',
                    date.strftime('%Y-%m-%d'),
                    date.strftime('%Y-%m-%d'),
                    row.to_dict()
                )
            logger.info(f"EIA数据获取成功: {len(eia_data)} 条记录")
        
        logger.info("数据获取完成\n")
    
    def calculate_spreads(self):
        """步骤2：计算价差"""
        logger.info("\n" + "="*60)
        logger.info("步骤2：计算价差")
        logger.info("="*60)
        
        # 创建标准裂解价差配置
        create_standard_crack_spreads(self.spread_calc)
        
        # 计算3:2:1裂解价差
        if all(symbol in self.price_data for symbol in ['CL', 'RBOB', 'HO']):
            spread_df = self.spread_calc.calculate_spread(
                'CRACK_3_2_1',
                self.price_data,
                price_column='close'
            )
            
            # 添加统计特征
            spread_df = self.spread_calc.get_spread_statistics(spread_df, window=20)
            
            self.spread_data['CRACK_3_2_1'] = spread_df
            
            # 保存到数据库
            self.db.insert_indicator_data(
                'CRACK_3_2_1',
                spread_df[['spread']],
                metadata={'type': 'crack_spread', 'ratio': '3:2:1'}
            )
            
            logger.info(f"3:2:1裂解价差计算完成: {len(spread_df)} 个数据点")
            
            # 平稳性检验
            self.indicator_builder.test_stationarity(
                spread_df['spread'],
                name='CRACK_3_2_1 Spread'
            )
        
        logger.info("价差计算完成\n")
    
    def build_features(self):
        """步骤3：构建特征"""
        logger.info("\n" + "="*60)
        logger.info("步骤3：特征工程")
        logger.info("="*60)
        
        # 获取主价差数据
        spread_df = self.spread_data.get('CRACK_3_2_1')
        if spread_df is None or spread_df.empty:
            logger.error("价差数据不可用")
            return
        
        # 1. 价差特征
        logger.info("创建价差特征...")
        spread_features = self.feature_engineer.create_spread_features(spread_df)
        
        # 2. 价格特征
        logger.info("创建价格特征...")
        price_features = self.feature_engineer.create_price_features(self.price_data)
        
        # 3. 技术指标特征
        logger.info("创建技术指标特征...")
        technical_features = pd.DataFrame(index=spread_df.index)
        for symbol, df in self.price_data.items():
            if symbol in ['CL', 'RBOB', 'HO']:
                tech_df = self.feature_engineer.create_technical_features(df, symbol)
                # 选择关键列
                key_cols = [col for col in tech_df.columns 
                           if any(x in col for x in ['RSI', 'MACD', 'BB_percent'])]
                if key_cols:
                    technical_features = technical_features.join(tech_df[key_cols], how='outer')
        
        # 4. 季节性特征
        logger.info("创建季节性特征...")
        seasonal_features = self.feature_engineer.create_seasonal_features(
            pd.DataFrame(index=spread_df.index)
        )
        
        # 5. 创建宏观特征（使用merge_asof对齐时间戳）
        logger.info("创建宏观特征...")
        # 创建基准DataFrame - 使用reset_index()确保正确的datetime类型
        temp_macro = spread_df.reset_index()
        temp_macro.columns = ['date'] + list(spread_df.columns)
        
        # 确保date列是datetime类型且无时区
        temp_macro['date'] = pd.to_datetime(temp_macro['date'])
        if hasattr(temp_macro['date'].dtype, 'tz') and temp_macro['date'].dtype.tz is not None:
            temp_macro['date'] = temp_macro['date'].dt.tz_localize(None)
        
        for symbol in ['VIX', 'DXY']:
            df = self.db.get_price_data(symbol)
            if not df.empty and 'close' in df.columns:
                # 准备宏观数据：重置索引，计算收益率
                macro_df = df[['close']].copy()
                macro_df[f'{symbol}_return_1d'] = macro_df['close'].pct_change()
                macro_df = macro_df.reset_index()
                macro_df.columns = ['date', f'{symbol}_close', f'{symbol}_return_1d']
                
                # 确保有干净的datetime列
                macro_df['date'] = pd.to_datetime(macro_df['date'])
                if hasattr(macro_df['date'].dtype, 'tz') and macro_df['date'].dtype.tz is not None:
                    macro_df['date'] = macro_df['date'].dt.tz_localize(None)
                
                # 使用merge_asof进行时间对齐（向后填充）
                temp_macro = pd.merge_asof(
                    temp_macro.sort_values('date'),
                    macro_df[['date', f'{symbol}_close', f'{symbol}_return_1d']].sort_values('date'),
                    on='date',
                    direction='backward'  # 使用最近的历史数据
                )
                
                null_count = temp_macro[f'{symbol}_close'].isnull().sum()
                logger.info(f"{symbol} 时间对齐完成，缺失值: {null_count} 个")
        
        # 恢复为索引格式（确保索引无时区）
        macro_features = temp_macro.set_index('date')
        # 再次确认索引无时区
        if hasattr(macro_features.index.dtype, 'tz') and macro_features.index.tz is not None:
            macro_features.index = macro_features.index.tz_localize(None)
        
        # 只保留宏观数据列，去除来自spread_df的列（避免与spread_features重复）
        macro_cols = [col for col in macro_features.columns if any(x in col for x in ['VIX', 'DXY'])]
        macro_features = macro_features[macro_cols]
        
        logger.info(f"宏观特征创建完成: {len(macro_features.columns)} 个特征（已时间对齐）")
        
        # 6. 创建目标变量
        logger.info("创建目标变量...")
        target_df = self.feature_engineer.create_target_variable(
            spread_df,
            method='classification',
            forward_period=5,
            threshold=0.01
        )
        
        # 合并所有特征
        logger.info("合并特征...")
        all_features = [
            spread_features,
            price_features,
            technical_features,
            seasonal_features,
            macro_features,
            target_df[['target', 'forward_return']]
        ]
        
        self.features_df = self.feature_engineer.merge_all_features(all_features)
        
        # 诊断：打印合并前的状态
        logger.info("="*60)
        logger.info("数据合并前诊断:")
        logger.info("="*60)
        
        df_names = ['价差特征', '价格特征', '技术指标', '季节性特征', '宏观特征', '目标变量']
        df_list = [spread_features, price_features, technical_features, seasonal_features, macro_features, target_df[['target', 'forward_return']]]
        
        for i, (name, df) in enumerate(zip(df_names, df_list)):
            null_count = df.isnull().sum().sum()
            logger.info(f"{name}: {len(df)} 样本, {len(df.columns)} 列, {null_count} 个缺失值")
            if null_count > 0:
                null_cols = df.isnull().sum()
                null_cols = null_cols[null_cols > 0]
                logger.warning(f"  {name}中有缺失值的列: {dict(list(null_cols.items())[:5])}")
        
        logger.info("="*60)
        
        # 诊断合并后的状态
        logger.info(f"\n合并后DataFrame状态:")
        logger.info(f"  总样本数: {len(self.features_df)}")
        logger.info(f"  总特征数: {len(self.features_df.columns)}")
        
        total_nulls = self.features_df.isnull().sum().sum()
        logger.info(f"  总缺失值: {total_nulls}")
        
        if total_nulls > 0:
            null_counts = self.features_df.isnull().sum()
            cols_with_nulls = null_counts[null_counts > 0].sort_values(ascending=False)
            
            logger.warning(f"\n缺失值详细分析 (main.py - build_features方法):")
            logger.warning(f"  共有 {len(cols_with_nulls)} 列包含缺失值")
            logger.warning(f"  缺失值最多的前10列:")
            for col, count in cols_with_nulls.head(10).items():
                pct = count / len(self.features_df) * 100
                logger.warning(f"    {col}: {count} ({pct:.2f}%)")
        
        # 清理特征
        logger.info("\n开始清理特征...")
        
        # 步骤1：使用前向填充处理时间序列缺失值
        logger.info("步骤1: 使用ffill填充缺失值...")
        self.features_df = self.features_df.fillna(method='ffill')
        
        remaining_nulls = self.features_df.isnull().sum().sum()
        logger.info(f"  前向填充后剩余缺失值: {remaining_nulls}")
        
        # 步骤2: 对前向填充无法处理的（开头的NaN），使用后向填充
        if remaining_nulls > 0:
            logger.info("步骤2: 使用bfill填充开头的缺失值...")
            self.features_df = self.features_df.fillna(method='bfill')
            remaining_nulls = self.features_df.isnull().sum().sum()
            logger.info(f"  后向填充后剩余缺失值: {remaining_nulls}")
        
        # 步骤3: 对仍然存在的缺失值使用列均值填充（针对数值列）
        if remaining_nulls > 0:
            logger.info("步骤3: 使用列均值填充剩余缺失值...")
            numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
            self.features_df[numeric_cols] = self.features_df[numeric_cols].fillna(
                self.features_df[numeric_cols].mean()
            )
            remaining_nulls = self.features_df.isnull().sum().sum()
            logger.info(f"  均值填充后剩余缺失值: {remaining_nulls}")
        
        # 步骤4: 删除仍有缺失值的列（如果有的话）
        if remaining_nulls > 0:
            null_cols = self.features_df.columns[self.features_df.isnull().any()].tolist()
            logger.warning(f"步骤4: 删除仍有缺失值的 {len(null_cols)} 列: {null_cols[:10]}")
            self.features_df = self.features_df.dropna(axis=1)
        
        logger.info(f"\n最终清理结果:")
        logger.info(f"  剩余样本数: {len(self.features_df)}")
        logger.info(f"  剩余特征数: {len(self.features_df.columns) - 2}")  # 减去target和forward_return
        logger.info("="*60)
        
        logger.info(f"特征构建完成，总特征数: {len(self.features_df.columns) - 2}")
        logger.info(f"样本数: {len(self.features_df)}")
        logger.info("特征工程完成\n")
    
    def train_model(self):
        """步骤4：训练机器学习模型"""
        logger.info("\n" + "="*60)
        logger.info("步骤4：模型训练")
        logger.info("="*60)
        
        if self.features_df is None or self.features_df.empty:
            logger.error("特征数据不可用")
            return
        
        # 特征选择
        logger.info("进行特征选择...")
        selected_features = self.feature_engineer.select_features(
            self.features_df,
            target_col='target',
            method='correlation',
            top_k=30
        )
        
        # 创建模型
        self.model = MLModel(model_type='gradient_boosting', task='classification')
        
        # 准备数据
        X_train, X_test, y_train, y_test, train_idx, test_idx = self.model.prepare_data(
            self.features_df,
            target_col='target',
            feature_cols=selected_features,
            test_size=0.2,
            scale=True
        )
        
        # 训练模型
        model_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        self.model.train(X_train, y_train, **model_params)
        
        # 评估模型
        metrics = self.model.evaluate(X_test, y_test)
        
        # 可视化特征重要性
        if self.model.feature_importance is not None:
            self.visualizer.plot_feature_importance(self.model.feature_importance, top_n=15)
        
        # 保存模型
        model_path = Path('models/crack_spread_model.pkl')
        model_path.parent.mkdir(exist_ok=True)
        self.model.save_model(str(model_path))
        
        logger.info("模型训练完成\n")
        
        return X_train, X_test, y_train, y_test, train_idx, test_idx
    
    def run_backtest(self, X_test, y_test, test_idx):
        """步骤5：运行回测"""
        logger.info("\n" + "="*60)
        logger.info("步骤5：回测")
        logger.info("="*60)
        
        # 生成信号
        signal_generator = SignalGenerator(self.model, threshold=0.5)
        signals = signal_generator.generate_signals(X_test, use_probability=True)
        signals.index = test_idx
        
        # 获取价差价格数据
        # 确保spread_data索引与test_idx时区一致
        spread_df_for_backtest = self.spread_data['CRACK_3_2_1'].copy()
        if hasattr(spread_df_for_backtest.index, 'tz') and spread_df_for_backtest.index.tz is not None:
            spread_df_for_backtest.index = spread_df_for_backtest.index.tz_localize(None)
        
        spread_prices = spread_df_for_backtest.loc[test_idx, ['spread']]
        spread_prices.columns = ['close']
        
        # 计算波动率
        spread_prices['volatility'] = spread_prices['close'].pct_change().rolling(20).std()
        
        # 运行回测
        backtest_engine = BacktestEngine(
            initial_capital=1000000,
            commission_rate=0.0005,
            slippage_rate=0.0001,
            max_position=100
        )
        
        equity_curve = backtest_engine.run_backtest(
            spread_prices,
            signals,
            price_col='close',
            volatility_col='volatility'
        )
        
        # 获取交易日志
        trade_log = backtest_engine.get_trade_log()
        
        # 绩效分析
        analyzer = PerformanceAnalyzer(
            equity_curve,
            initial_capital=1000000,
            risk_free_rate=0.02
        )
        
        performance_report = analyzer.generate_performance_report(trade_log)
        
        logger.info("回测完成\n")
        
        return equity_curve, trade_log, performance_report
    
    def visualize_results(self, equity_curve, trade_log):
        """步骤6：可视化结果"""
        logger.info("\n" + "="*60)
        logger.info("步骤6：结果可视化")
        logger.info("="*60)
        
        # 1. 价格和价差图
        logger.info("生成价格和价差图...")
        self.visualizer.plot_price_and_spread(
            self.price_data,
            self.spread_data['CRACK_3_2_1'],
            title='Crack Spread 3:2:1'
        )
        
        # 2. 权益曲线
        logger.info("生成权益曲线图...")
        self.visualizer.plot_equity_curve(equity_curve)
        
        # 3. 收益率分布
        logger.info("生成收益率分布图...")
        returns = equity_curve['equity'].pct_change().dropna()
        self.visualizer.plot_returns_distribution(returns)
        
        # 4. 月度收益热力图
        logger.info("生成月度收益热力图...")
        self.visualizer.plot_monthly_returns_heatmap(equity_curve)
        
        # 5. 滚动指标
        logger.info("生成滚动指标图...")
        self.visualizer.plot_rolling_metrics(equity_curve, window=60)
        
        # 6. 交易分析
        logger.info("生成交易分析图...")
        self.visualizer.plot_trade_analysis(trade_log)
        
        logger.info("可视化完成\n")
    
    def run_complete_strategy(self):
        """运行完整策略流程"""
        logger.info("\n" + "="*60)
        logger.info("开始运行完整策略流程")
        logger.info("="*60 + "\n")
        
        try:
            # 1. 数据获取
            self.fetch_all_data()
            
            # 2. 计算价差
            self.calculate_spreads()
            
            # 3. 特征工程
            self.build_features()
            
            # 4. 模型训练
            X_train, X_test, y_train, y_test, train_idx, test_idx = self.train_model()
            
            # 5. 回测
            equity_curve, trade_log, performance_report = self.run_backtest(
                X_test, y_test, test_idx
            )
            
            # 6. 可视化
            self.visualize_results(equity_curve, trade_log)
            
            logger.info("\n" + "="*60)
            logger.info("策略流程执行完成！")
            logger.info("="*60)
            
            return {
                'equity_curve': equity_curve,
                'trade_log': trade_log,
                'performance_report': performance_report,
                'model': self.model
            }
            
        except Exception as e:
            logger.error(f"策略执行出错: {e}", exc_info=True)
            raise


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("裂解价差交易策略研究系统")
    logger.info("="*60)
    
    # 创建策略实例
    strategy = CrackSpreadStrategy()
    
    # 运行完整策略
    results = strategy.run_complete_strategy()
    
    logger.info("\n所有任务完成！")
    logger.info("生成的文件:")
    logger.info("  - data/trading_data.db (数据库)")
    logger.info("  - models/crack_spread_model.pkl (模型文件)")
    logger.info("  - outputs/charts/*.png (各类图表)")
    logger.info("  - logs/trading_strategy.log (日志文件)")
    
    return results


if __name__ == "__main__":
    results = main()
