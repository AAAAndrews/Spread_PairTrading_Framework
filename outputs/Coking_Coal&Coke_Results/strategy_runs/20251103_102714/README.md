# 策略运行结果

**运行时间**: 2025-11-03 10:27:14  
**运行ID**: 20251103_102714

## 目录结构

```
20251103_102714/
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
