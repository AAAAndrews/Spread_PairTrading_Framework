"""
结果保存模块
用于保存策略参数、绩效报告、模型配置等信息
"""
import json
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ResultSaver:
    """结果保存器"""
    
    def __init__(self, output_dir: str = "outputs/strategy_results"):
        """
        初始化结果保存器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建时间戳文件夹
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / self.timestamp
        self.run_dir.mkdir(exist_ok=True)
        
        logger.info(f"结果保存器初始化完成，输出目录: {self.run_dir}")
    
    def save_strategy_config(self, 
                            model_params: Dict[str, Any],
                            signal_params: Dict[str, Any],
                            backtest_params: Dict[str, Any],
                            data_info: Dict[str, Any],
                            filename: str = "strategy_config.json") -> str:
        """
        保存策略配置参数
        
        Args:
            model_params: 模型参数
            signal_params: 信号生成参数
            backtest_params: 回测参数
            data_info: 数据信息
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        config = {
            "run_timestamp": self.timestamp,
            "model_parameters": model_params,
            "signal_parameters": signal_params,
            "backtest_parameters": backtest_params,
            "data_information": data_info
        }
        
        filepath = self.run_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False, default=str)
        
        logger.info(f"策略配置已保存: {filepath}")
        return str(filepath)
    
    def save_performance_report(self,
                               performance_report: Dict[str, Any],
                               trade_log: pd.DataFrame,
                               equity_curve: pd.DataFrame,
                               filename_prefix: str = "performance") -> Dict[str, str]:
        """
        保存绩效报告
        
        Args:
            performance_report: 绩效报告字典
            trade_log: 交易日志DataFrame
            equity_curve: 权益曲线DataFrame
            filename_prefix: 文件名前缀
            
        Returns:
            保存的文件路径字典
        """
        saved_files = {}
        
        # 1. 保存绩效报告JSON
        json_file = self.run_dir / f"{filename_prefix}_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(performance_report, f, indent=4, ensure_ascii=False, default=str)
        saved_files['report_json'] = str(json_file)
        logger.info(f"绩效报告JSON已保存: {json_file}")
        
        # 2. 保存绩效报告CSV（便于Excel查看）
        report_df = pd.DataFrame([performance_report]).T
        report_df.columns = ['Value']
        csv_file = self.run_dir / f"{filename_prefix}_report.csv"
        report_df.to_csv(csv_file, encoding='utf-8-sig')
        saved_files['report_csv'] = str(csv_file)
        logger.info(f"绩效报告CSV已保存: {csv_file}")
        
        # 3. 保存交易日志
        if not trade_log.empty:
            trade_file = self.run_dir / f"{filename_prefix}_trades.csv"
            trade_log.to_csv(trade_file, index=False, encoding='utf-8-sig')
            saved_files['trades'] = str(trade_file)
            logger.info(f"交易日志已保存: {trade_file}")
        
        # 4. 保存权益曲线
        if not equity_curve.empty:
            equity_file = self.run_dir / f"{filename_prefix}_equity_curve.csv"
            equity_curve.to_csv(equity_file, encoding='utf-8-sig')
            saved_files['equity_curve'] = str(equity_file)
            logger.info(f"权益曲线已保存: {equity_file}")
        
        # 5. 生成汇总文本报告
        txt_file = self.run_dir / f"{filename_prefix}_summary.txt"
        self._generate_text_summary(performance_report, trade_log, txt_file)
        saved_files['summary_txt'] = str(txt_file)
        
        return saved_files
    
    def save_feature_info(self,
                         selected_features: list,
                         feature_importance: Optional[pd.DataFrame] = None,
                         filename: str = "feature_info.json") -> str:
        """
        保存特征信息
        
        Args:
            selected_features: 选择的特征列表
            feature_importance: 特征重要性DataFrame
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        feature_info = {
            "selected_features": selected_features,
            "feature_count": len(selected_features)
        }
        
        if feature_importance is not None:
            feature_info["feature_importance"] = feature_importance.to_dict('records')
        
        filepath = self.run_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, indent=4, ensure_ascii=False)
        
        # 同时保存CSV格式的特征重要性
        if feature_importance is not None:
            csv_file = self.run_dir / "feature_importance.csv"
            feature_importance.to_csv(csv_file, index=False, encoding='utf-8-sig')
            logger.info(f"特征重要性CSV已保存: {csv_file}")
        
        logger.info(f"特征信息已保存: {filepath}")
        return str(filepath)
    
    def save_model_metrics(self,
                          train_metrics: Dict[str, Any],
                          test_metrics: Dict[str, Any],
                          filename: str = "model_metrics.json") -> str:
        """
        保存模型评估指标
        
        Args:
            train_metrics: 训练集指标
            test_metrics: 测试集指标
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        metrics = {
            "timestamp": self.timestamp,
            "training_metrics": train_metrics,
            "testing_metrics": test_metrics
        }
        
        filepath = self.run_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False, default=str)
        
        logger.info(f"模型指标已保存: {filepath}")
        return str(filepath)
    
    def save_signals(self,
                    signals,
                    probabilities: Optional[pd.DataFrame] = None,
                    filename: str = "signals.csv") -> str:
        """
        保存交易信号
        
        Args:
            signals: 信号Series或DataFrame
            probabilities: 概率DataFrame（可选）
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        # 判断signals是Series还是DataFrame
        if isinstance(signals, pd.Series):
            signals_df = pd.DataFrame({
                'date': signals.index,
                'signal': signals.values
            })
        elif isinstance(signals, pd.DataFrame):
            # 如果已经是DataFrame，直接使用
            signals_df = signals.copy()
            if 'date' not in signals_df.columns and signals_df.index.name != 'date':
                signals_df = signals_df.reset_index()
                if 'index' in signals_df.columns:
                    signals_df.rename(columns={'index': 'date'}, inplace=True)
        else:
            raise ValueError(f"signals必须是Series或DataFrame，当前类型: {type(signals)}")
        
        if probabilities is not None and 'date' in signals_df.columns:
            # 重置probabilities的索引以便合并
            prob_df = probabilities.reset_index()
            if 'index' in prob_df.columns:
                prob_df.rename(columns={'index': 'date'}, inplace=True)
            signals_df = pd.merge(signals_df, prob_df, on='date', how='left')
        
        filepath = self.run_dir / filename
        signals_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        logger.info(f"交易信号已保存: {filepath}")
        return str(filepath)
    
    def _generate_text_summary(self,
                              performance_report: Dict[str, Any],
                              trade_log: pd.DataFrame,
                              filepath: Path):
        """
        生成文本格式的汇总报告
        
        Args:
            performance_report: 绩效报告
            trade_log: 交易日志
            filepath: 保存路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("策略绩效汇总报告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"运行ID: {self.timestamp}\n\n")
            
            # 核心指标
            f.write("-"*80 + "\n")
            f.write("核心绩效指标\n")
            f.write("-"*80 + "\n")
            f.write(f"总收益率:           {performance_report.get('total_return', 0)*100:>12.2f}%\n")
            f.write(f"年化收益率:         {performance_report.get('annual_return', 0)*100:>12.2f}%\n")
            f.write(f"夏普比率:           {performance_report.get('sharpe_ratio', 0):>12.2f}\n")
            f.write(f"最大回撤:           {performance_report.get('max_drawdown', 0)*100:>12.2f}%\n")
            f.write(f"卡尔玛比率:         {performance_report.get('calmar_ratio', 0):>12.2f}\n\n")
            
            # 交易统计
            f.write("-"*80 + "\n")
            f.write("交易统计\n")
            f.write("-"*80 + "\n")
            f.write(f"总交易次数:         {performance_report.get('total_trades', 0):>12}\n")
            f.write(f"盈利交易:           {performance_report.get('winning_trades', 0):>12}\n")
            f.write(f"亏损交易:           {performance_report.get('losing_trades', 0):>12}\n")
            f.write(f"胜率:               {performance_report.get('win_rate', 0)*100:>12.2f}%\n")
            f.write(f"平均盈利:           ${performance_report.get('avg_win', 0):>12,.2f}\n")
            f.write(f"平均亏损:           ${performance_report.get('avg_loss', 0):>12,.2f}\n")
            f.write(f"盈亏比:             {performance_report.get('profit_factor', 0):>12.2f}\n\n")
            
            # 最近10笔交易
            if not trade_log.empty and len(trade_log) > 0:
                f.write("-"*80 + "\n")
                f.write("最近10笔交易\n")
                f.write("-"*80 + "\n")
                # 根据实际存在的列选择要显示的列
                available_cols = []
                desired_cols = ['entry_date', 'exit_date', 'entry_time', 'exit_time', 'direction', 'pnl', 'return', 'profit']
                for col in desired_cols:
                    if col in trade_log.columns:
                        available_cols.append(col)
                
                if available_cols:
                    recent_trades = trade_log.tail(10)[available_cols]
                    f.write(recent_trades.to_string(index=False))
                else:
                    # 如果没有匹配的列，显示所有列
                    f.write(trade_log.tail(10).to_string(index=False))
                f.write("\n\n")
            
            f.write("="*80 + "\n")
            f.write("报告结束\n")
            f.write("="*80 + "\n")
        
        logger.info(f"文本汇总报告已保存: {filepath}")
    
    def create_readme(self):
        """创建README文件说明结果目录结构"""
        readme_path = self.run_dir / "README.md"
        
        content = f"""# 策略运行结果

**运行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**运行ID**: {self.timestamp}

## 目录结构

```
{self.timestamp}/
├── strategy_config.json         # 策略配置参数
├── performance_report.json      # 绩效报告（JSON格式）
├── performance_report.csv       # 绩效报告（CSV格式）
├── performance_summary.txt      # 绩效汇总（文本格式）
├── performance_trades.csv       # 交易日志
├── performance_equity_curve.csv # 权益曲线
├── feature_info.json            # 特征信息
├── feature_importance.csv       # 特征重要性
├── model_metrics.json           # 模型评估指标
├── signals.csv                  # 交易信号
└── README.md                    # 本文件
```

## 文件说明

### 1. strategy_config.json
包含完整的策略配置：
- 模型参数（算法、超参数等）
- 信号生成参数（阈值、持有期等）
- 回测参数（初始资金、手续费等）
- 数据信息（数据范围、品种等）

### 2. performance_*.* 
绩效相关文件：
- `performance_report.json`: 详细的绩效指标（JSON格式）
- `performance_report.csv`: 绩效指标表格（便于Excel打开）
- `performance_summary.txt`: 易读的文本汇总
- `performance_trades.csv`: 每笔交易的详细记录
- `performance_equity_curve.csv`: 权益曲线数据

### 3. feature_*.* 
特征相关文件：
- `feature_info.json`: 选择的特征列表
- `feature_importance.csv`: 特征重要性排序

### 4. model_metrics.json
模型在训练集和测试集上的表现指标

### 5. signals.csv
生成的交易信号记录

## 使用建议

1. **查看绩效**: 打开 `performance_summary.txt` 快速了解策略表现
2. **分析交易**: 使用Excel打开 `performance_trades.csv` 分析每笔交易
3. **比较策略**: 不同运行的结果可以通过时间戳文件夹进行对比
4. **导入数据**: 所有CSV文件均为UTF-8编码，可直接导入数据库或分析工具

## 注意事项

- 所有日期时间均为本地时区
- 货币单位为美元（USD）
- 收益率为百分比形式（已乘以100）
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"README已创建: {readme_path}")
    
    def get_run_directory(self) -> str:
        """获取当前运行的输出目录"""
        return str(self.run_dir)