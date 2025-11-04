# 价差套利交易框架 📈

一个基于Python的完整价差套利交易系统，专注于能源期货裂解价差策略

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 项目概述

本项目实现了一个**工业级别的价差套利交易框架**，专注于**能源期货裂解价差**（Crack Spread）策略。系统包含完整的数据获取、特征工程、机器学习、回测和可视化模块。

### 核心功能
- ✅ 多数据源接入（yfinance, akshare）
- ✅ 期货自动换月调整
- ✅ 灵活的价差组合配置
- ✅ 127维特征工程（价差/价格/技术/季节性/宏观）
- ✅ 4种ML模型（XGBoost, LightGBM, RandomForest, GradientBoosting）
- ✅ 完整回测引擎（含交易成本）
- ✅ 专业可视化（7种图表）



## 🚀 快速开始

### 1. 环境安装
```bash
# 创建虚拟环境（推荐）
conda create -n trading python=3.9
conda activate trading

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行策略
```bash
python main.py
```

### 3. 查看结果
运行完成后会生成：
- `trading_data.db` - SQLite数据库（约15MB）
- `crack_spread_model.pkl` - 训练好的模型
- `*.png` - 7张可视化图表
- `trading_strategy.log` - 详细日志

## 📁 项目结构

```
价差套利框架/
├── database.py              # 数据库管理
├── data_fetcher.py          # 数据获取（yfinance/akshare）
├── spread_calculator.py     # 价差计算引擎
├── indicators.py            # 技术指标库
├── feature_engineering.py   # 特征工程
├── ml_models.py             # 机器学习模型
├── backtest.py              # 回测引擎
├── visualization.py         # 可视化系统
├── main.py                  # 主控流程
├── requirements.txt         # Python依赖
├── README_NEW.md            # 本文件
├── 任务完成报告.md           # 详细技术报告
└── task_doc.md              # 任务文档
```

## 🔧 模块说明

### 1. 数据管理 (`database.py`)
- SQLite数据库，4张表设计
- 支持价格数据、基本面、指标、价差配置
- 自动建表、索引优化

### 2. 数据获取 (`data_fetcher.py`)
- **yfinance**: 美股期货（CL, RBOB, HO）
- **akshare**: 中国市场（预留）
- **期货换月**: 比率法调整，自动检测换月点
- **数据标准化**: 统一列名和时区

### 3. 价差计算 (`spread_calculator.py`)
- 灵活的多腿价差配置
- 标准裂解价差模板（3:2:1, 2:1:1, 5:3:2）
- 滚动统计特征（20/60周期）
- Z-score标准化

### 4. 技术指标 (`indicators.py`)
- **趋势**: MA, EMA
- **动量**: RSI, MACD
- **波动**: Bollinger Bands, ATR
- **统计**: ADF平稳性检验
- **季节性**: 月份、季度、星期特征

### 5. 特征工程 (`feature_engineering.py`)
- **价差特征** : 收益率、动量、波动率、统计
- **价格特征** : 各品种的技术特征
- **技术指标** : RSI/MACD/BB
- **季节性** : 时间特征
- **宏观特征** : VIX, DXY
- **缺失值处理**: 2步填充策略（ffill→drop）

### 6. 机器学习 (`ml_models.py`)
- **支持算法**: XGBoost, LightGBM, RandomForest, GradientBoosting
- **超参数调优**: GridSearch
- **特征选择**: 基于重要性
- **模型评估**: 14项指标
- **模型持久化**: pickle保存/加载

### 7. 回测引擎 (`backtest.py`)
- **完整交易成本**: 手续费+滑点
- **绩效指标** (14项):
  - 收益: 总收益率、年化收益率
  - 风险: 波动率、最大回撤
  - 风险调整: 夏普、索提诺、卡玛比率
  - 风险价值: VaR, CVaR
  - 交易: 胜率、盈亏比、交易次数
- **详细交易记录**: 每笔交易的进出场、盈亏、持仓周期

### 8. 可视化 (`visualization.py`)
- **特征重要性**: Top 10特征排序
- **价格价差图**: 多品种价格+价差+信号
- **权益曲线**: 策略vs基准+回撤
- **收益分布**: 直方图+正态拟合
- **月度热力图**: 按年月展示收益
- **滚动指标**: 夏普/回撤/胜率

### 9. 主控流程 (`main.py`)
完整的6步策略执行：
1. 数据获取（10年历史）
2. 价差计算（3:2:1裂解价差）
3. 特征工程（123个特征）
4. 模型训练（GradientBoosting）
5. 回测验证（596个交易日）
6. 结果可视化（7张图表）



## 📚 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| 语言 | Python | 3.9+ |
| 数据处理 | pandas | 1.5.3 |
| 数值计算 | numpy | 1.24.3 |
| 机器学习 | scikit-learn | 1.2.1 |
| 梯度提升 | XGBoost | 1.7.3 |
| 梯度提升 | LightGBM | 4.6.0 |
| 可视化 | matplotlib | 3.9.4 |
| 可视化 | seaborn | 0.13.2 |
| 数据获取 | yfinance | 0.2.66 |
| 统计分析 | statsmodels | 0.14.5 |
| 数据库 | SQLite3 | 内置 |

## ⚠️ 免责声明

本项目仅供**学习和研究**使用，不构成任何投资建议。期货交易存在高风险，可能导致本金全部损失。使用本框架进行实盘交易的任何损失，作者不承担任何责任。

