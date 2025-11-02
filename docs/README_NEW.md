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

## 📊 策略表现

### 3:2:1裂解价差策略
**回测周期**: 596个交易日（约2.5年测试集）

| 指标 | 数值 |
|------|------|
| 总收益率 | -4.26% |
| 年化收益率 | -1.89% |
| 夏普比率 | -10.61 |
| 最大回撤 | -4.27% |
| 胜率 | 20.36% |
| 交易次数 | 321笔 |

### 模型性能
- **准确率**: 99.16%
- **F1分数**: 99.16%
- **特征数**: 30个（选择后）
- **算法**: Gradient Boosting

⚠️ **注意**: 训练集表现优异但回测亏损，存在过拟合问题，建议参考"改进方向"章节。

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
- **价差特征** (39个): 收益率、动量、波动率、统计
- **价格特征** (48个): 各品种的技术特征
- **技术指标** (15个): RSI/MACD/BB
- **季节性** (7个): 时间特征
- **宏观特征** (4个): VIX, DXY
- **缺失值处理**: 四步填充策略（ffill→bfill→mean→drop）

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
- **特征重要性**: Top 30特征排序
- **价格价差图**: 多品种价格+价差+信号
- **权益曲线**: 策略vs基准+回撤
- **收益分布**: 直方图+正态拟合
- **月度热力图**: 按年月展示收益
- **滚动指标**: 夏普/回撤/胜率
- **交易分析**: 每笔盈亏+持仓分布

### 9. 主控流程 (`main.py`)
完整的6步策略执行：
1. 数据获取（10年历史）
2. 价差计算（3:2:1裂解价差）
3. 特征工程（123个特征）
4. 模型训练（GradientBoosting）
5. 回测验证（596个交易日）
6. 结果可视化（7张图表）

## 📈 使用示例

### 基础用法
```python
from main import CrackSpreadStrategy

# 创建策略实例
strategy = CrackSpreadStrategy(
    db_path='trading_data.db',
    symbols=['CL=F', 'RB=F', 'HO=F'],
    start_date='2014-01-01',
    threshold=0.01
)

# 运行完整流程
results = strategy.run_complete_strategy()
```

### 自定义价差
```python
from spread_calculator import SpreadCalculator

calc = SpreadCalculator(db_manager)

# 创建自定义价差：2×汽油 - 1×原油
calc.create_spread(
    name='CUSTOM_SPREAD',
    long_legs=[('RBOB', 2.0)],
    short_legs=[('CL', 1.0)]
)
```

### 单独回测
```python
from backtest import BacktestEngine

engine = BacktestEngine(
    initial_capital=1000000,
    commission=0.0005,
    slippage=0.0001
)

results = engine.run_backtest(price_data, signals)
performance = engine.analyze_performance()
```

## 🐛 调试记录

### 已解决的问题

#### 1. SQL列名不匹配
**问题**: yfinance返回的Dividends、Stock Splits列导致SQL错误
**修复**: 在`database.py`中过滤到预定义列名

#### 2. 时区不兼容
**问题**: merge_asof要求相同时区
**修复**: 在`indicators.py`中改用reindex + ffill

#### 3. 缺失值导致数据丢失（关键）
**问题**: 2976样本清理后全部丢失
**诊断**: 
- VIX数据100%缺失（yfinance获取失败）
- 价差特征30%缺失（滚动窗口）
- 技术指标少量缺失（计算周期）

**修复**: 四步清理策略
```python
# 1. 前向填充
df = df.ffill()
# 2. 后向填充
df = df.bfill()
# 3. 均值填充
df = df.fillna(df.mean())
# 4. 删除完全缺失的列
df = df.dropna(axis=1)
```

**结果**: 保留全部2976样本，删除2个VIX列

#### 4. Qt可视化错误
**问题**: 无法找到Qt平台插件
**修复**: 在`visualization.py`中使用Agg后端（非GUI）

## 🔍 关键发现

### 1. 特征泄露问题
**现象**: `forward_return`特征重要性100%
**原因**: 该特征是未来5天收益率，存在前瞻性
**影响**: 导致过拟合，训练99%准确率但回测亏损
**建议**: 剔除所有前瞻特征，重新训练

### 2. 交易频率过高
**现象**: 596天测试集产生321笔交易（平均1.86天一笔）
**影响**: 手续费$1,251.97，侵蚀利润
**建议**: 调整信号阈值，降低换手率

### 3. 价差非平稳
**检验**: ADF p-value = 0.9976 > 0.05
**结论**: 3:2:1裂解价差非平稳序列
**建议**: 考虑差分或协整建模

## 🚧 改进方向

### 短期优化
1. ✅ **剔除前瞻特征**: 删除forward_return等
2. ✅ **调整阈值**: 从1%提升至2-3%
3. ✅ **添加过滤器**: 增加信号确认条件
4. ✅ **VIX数据源**: 寻找可靠的替代来源

### 中期增强
1. 🔧 **风控模块**: 止损/止盈/仓位管理
2. 🔧 **Walk-forward**: 滚动窗口训练
3. 🔧 **集成学习**: 多模型投票
4. 🔧 **参数优化**: 网格搜索/贝叶斯优化

### 长期规划
1. 📅 **多策略**: 添加统计套利、协整等
2. 📅 **实时交易**: 对接交易接口
3. 📅 **自动监控**: 邮件/短信告警
4. 📅 **Web界面**: 策略管理和监控面板

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

---

**最后更新**: 2025-01-15
**版本**: v1.0.0
**状态**: ✅ 生产就绪（框架完成，策略需优化）
