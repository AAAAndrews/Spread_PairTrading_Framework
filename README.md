# 价差套利框架项目详细介绍 | Spread Arbitrage Framework - Detailed Introduction

## 重要提示 | Important Notice

**中文：**
如需通过 Git 拉取（pull）或克隆（clone）本项目，请联系 **e1538433@u.nus.edu** 获取私有仓库的访问权限。

**English:**
To pull or clone this project via Git, please contact **e1538433@u.nus.edu** to obtain access permissions for the private repository.

---

## 项目背景与意义 | Project Background and Significance

**中文：**
价差套利是一种基于商品期货市场的交易策略,通过捕捉相关商品之间的价格关系变化来获利。本项目旨在构建一个完整的价差套利框架,涵盖数据处理、特征工程、模型训练、策略回测等全流程,帮助用户快速实现并优化套利策略。

本项目的意义在于：
1. **提高交易效率**：通过自动化流程减少人工干预。
2. **优化策略表现**：利用机器学习模型挖掘潜在的价差交易机会。
3. **降低交易风险**：通过回测和评估,验证策略的稳定性和可靠性。

**English:**
Spread arbitrage is a trading strategy based on commodity futures markets that profits by capturing changes in price relationships between related commodities. This project aims to build a complete spread arbitrage framework covering the entire workflow including data processing, feature engineering, model training, and strategy backtesting, helping users quickly implement and optimize arbitrage strategies.

The significance of this project lies in:
1. **Improving Trading Efficiency**: Reducing manual intervention through automated processes.
2. **Optimizing Strategy Performance**: Using machine learning models to uncover potential spread trading opportunities.
3. **Reducing Trading Risk**: Validating strategy stability and reliability through backtesting and evaluation.

---

## 项目结构 | Project Structure

### 1. **核心模块 | Core Modules**

**中文：** 项目的核心代码位于 `src/core/` 目录下，主要包括以下模块：

**English:** The core code of the project is located in the `src/core/` directory and includes the following modules:

#### **1.1 数据处理模块 | Data Processing Module (data_fetcher.py, database.py)**
- **功能 | Function**: 负责从外部数据源获取原始数据并存储到本地数据库 | Responsible for fetching raw data from external sources and storing it in a local database
- **优势 | Advantages**:
  - 支持多种数据源（API、CSV 文件等） | Supports multiple data sources (API, CSV files, etc.)
  - 数据存储采用高效的数据库结构，便于后续查询和处理 | Data storage uses efficient database structures for easy querying and processing

#### **1.2 特征工程模块 | Feature Engineering Module (feature_engineering.py)**
- **功能 | Function**: 对原始数据进行清洗、转换和特征提取 | Performs data cleaning, transformation, and feature extraction on raw data
- **优势 | Advantages**:
  - 提供多种特征选择方法（如方差选择、相关性分析） | Provides various feature selection methods (e.g., variance selection, correlation analysis)
  - 支持自定义特征生成，增强模型的表达能力 | Supports custom feature generation to enhance model expressiveness

#### **1.3 模型模块 | Model Module (ml_models.py)**
- **功能 | Function**: 封装了常用的机器学习模型（如 XGBoost、LightGBM、随机森林等） | Encapsulates common machine learning models (e.g., XGBoost, LightGBM, Random Forest)
- **优势 | Advantages**:
  - 支持分类和回归任务 | Supports both classification and regression tasks
  - 提供数据预处理、模型训练、评估和保存的完整流程 | Provides complete workflow for data preprocessing, model training, evaluation, and saving

#### **1.4 回测模块 | Backtesting Module (backtest.py)**
- **功能 | Function**: 对策略进行历史数据回测，评估其表现 | Conducts historical data backtesting on strategies to evaluate their performance
- **优势 | Advantages**:
  - 支持多种回测指标（如收益率、最大回撤） | Supports various backtesting metrics (e.g., returns, maximum drawdown)
  - 提供可视化功能，直观展示策略效果 | Provides visualization features for intuitive display of strategy performance

#### **1.5 可视化模块 | Visualization Module (visualization.py)**
- **功能 | Function**: 生成数据分析和模型评估的图表 | Generates charts for data analysis and model evaluation
- **优势 | Advantages**:
  - 图表美观易读，便于快速理解数据和结果 | Charts are aesthetically pleasing and easy to read for quick data and result comprehension
  - 支持多种图表类型（如特征重要性图、回测收益曲线） | Supports multiple chart types (e.g., feature importance plots, backtesting equity curves)

---

### 2. **工具模块 | Utility Modules**
**中文：** `tools/` 目录下包含一些辅助脚本，如数据清洗脚本（`clean.py`）、爬虫脚本（`crawler.py`）等。

**English:** The `tools/` directory contains auxiliary scripts such as data cleaning scripts (`clean.py`), crawler scripts (`crawler.py`), etc.

### 3. **示例模块 | Example Modules**
**中文：** `examples/` 目录下提供了一些使用示例，帮助用户快速上手。

**English:** The `examples/` directory provides usage examples to help users get started quickly.

### 4. **文档模块 | Documentation Modules**
**中文：** `docs/` 目录下包含详细的功能说明文档和使用指南，如：

**English:** The `docs/` directory contains detailed functional documentation and user guides, such as:

- **快速开始指南 | Quick Start Guide**: 帮助用户快速搭建环境并运行项目 | Helps users quickly set up the environment and run the project
- **超参数优化指南 | Hyperparameter Optimization Guide**: 介绍如何调整模型参数以提升性能 | Introduces how to adjust model parameters to improve performance
- **调试指南 | Debugging Guide**: 提供常见问题的解决方案 | Provides solutions to common issues

---

## 建模步骤与流程 | Modeling Steps and Workflow

### **1. 数据准备 | Data Preparation**
- **数据获取 | Data Acquisition**: 通过 `data_fetcher.py` 从外部数据源获取原始数据 | Fetch raw data from external sources through `data_fetcher.py`
- **数据存储 | Data Storage**: 使用 `database.py` 将数据存储到本地数据库 | Store data to local database using `database.py`

### **2. 特征工程 | Feature Engineering**
- **数据清洗 | Data Cleaning**: 去除缺失值和异常值 | Remove missing values and outliers
- **特征生成 | Feature Generation**: 利用 `feature_engineering.py` 提取关键特征 | Extract key features using `feature_engineering.py`
- **特征选择 | Feature Selection**: 根据方差或相关性选择最优特征 | Select optimal features based on variance or correlation

### **3. 模型训练 | Model Training**
- **模型选择 | Model Selection**: 通过 `ml_models.py` 创建机器学习模型（如 Gradient Boosting） | Create machine learning models (e.g., Gradient Boosting) through `ml_models.py`
- **数据划分 | Data Splitting**: 将数据分为训练集和测试集，避免数据泄露 | Split data into training and testing sets to avoid data leakage
- **模型训练 | Model Training**: 在训练集上训练模型，并保存训练结果 | Train the model on the training set and save the results
- **模型评估 | Model Evaluation**: 在测试集上评估模型性能，输出准确率、F1 分数等指标 | Evaluate model performance on the test set, outputting metrics like accuracy and F1 score

### **4. 策略回测 | Strategy Backtesting**
- **回测设置 | Backtest Setup**: 定义回测参数（如初始资金、交易成本） | Define backtesting parameters (e.g., initial capital, trading costs)
- **回测执行 | Backtest Execution**: 利用 `backtest.py` 运行回测，生成收益曲线 | Run backtests using `backtest.py` and generate equity curves
- **结果分析 | Result Analysis**: 通过可视化模块生成回测报告 | Generate backtest reports through the visualization module

---

## Notebook 文件说明 | Notebook File Description

**中文：** 项目中包含多个 Jupyter Notebook 文件，用于演示不同的建模流程：

**English:** The project contains multiple Jupyter Notebook files to demonstrate different modeling workflows:

### **1. trainmodel_coking_spread.ipynb**
- **目标 | Objective**: 预测焦煤和焦炭的价差 | Predict the spread between coking coal and coke
- **流程 | Workflow**:
  1. 数据加载与清洗 | Data loading and cleaning
  2. 特征工程与选择 | Feature engineering and selection
  3. 模型训练与评估 | Model training and evaluation
  4. 策略回测与优化 | Strategy backtesting and optimization

### **2. trainmodel_crack_spread.ipynb**
- **目标 | Objective**: 预测原油裂解价差 | Predict crude oil crack spread
- **流程 | Workflow**:
  1. 数据预处理 | Data preprocessing
  2. 特征选择与模型训练 | Feature selection and model training
  3. 回测与结果分析 | Backtesting and result analysis

### **3. trainmodel_sp_spread.ipynb**
- **目标 | Objective**: 预测大豆和棕榈油的价差 | Predict the spread between soybean oil and palm oil
- **流程 | Workflow**:
  1. 数据加载与清洗 | Data loading and cleaning
  2. 特征工程与选择 | Feature engineering and selection
  3. 模型训练与评估 | Model training and evaluation

---

## 项目优势 | Project Advantages

1. **模块化设计 | Modular Design**:
   - 每个功能模块独立，便于维护和扩展 | Each functional module is independent, facilitating maintenance and extension

2. **灵活性强 | High Flexibility**:
   - 支持多种数据源、模型和回测参数 | Supports multiple data sources, models, and backtesting parameters

3. **高效性 | High Efficiency**:
   - 自动化流程减少了人工干预，提高了开发效率 | Automated processes reduce manual intervention and improve development efficiency

4. **可视化能力 | Visualization Capabilities**:
   - 提供丰富的图表，帮助用户快速理解数据和结果 | Provides rich charts to help users quickly understand data and results

---

## 已知问题与注意事项 | Known Issues and Important Notes

### **数据类型错误问题 | Data Type Error Issue**

**中文说明：**

在使用特征工程模块的 `select_features` 函数时，当选择方法为 `'variance'`（方差选择）时，可能会遇到 `TypeError: Cannot use method 'nlargest' with dtype object` 错误。

**问题根源：**
- 该错误由数据框（DataFrame）中包含的字符串列引起
- 具体来说，`spread_calculator.py` 文件中的 `calculate_spread` 函数在计算完价差后，会额外添加一个名为 `'spread_name'` 的字符串列，用于标识价差类型（例如 `'COKING_PROFIT'`）
- 当特征工程模块尝试对所有特征列计算方差（variance）时，该字符串列无法进行数值计算，导致报错

**特殊之处：**
- 即使在包版本完全相同的情况下，该错误在不同计算机上的表现可能不一致
- 某些环境可能会报错，而另一些环境则可以正常运行
- 这可能与底层 NumPy 或 pandas 的行为细节有关，但具体原因尚未完全确定

**临时解决方案：**
- 在特征选择之前，从数据框中移除 `'spread_name'` 列或其他字符串类型的列
- 修改 `calculate_spread` 函数，不在返回的数据框中包含 `'spread_name'` 列
- 如果需要保留该信息，可以将其作为单独的变量返回，而不是混入特征数据框

**English Explanation:**

When using the `select_features` function in the feature engineering module with the method parameter set to `'variance'` (variance selection), you may encounter a `TypeError: Cannot use method 'nlargest' with dtype object` error.

**Root Cause:**
- This error is caused by string columns present in the DataFrame
- Specifically, the `calculate_spread` function in `spread_calculator.py` adds an extra column named `'spread_name'` after calculating the spread, which contains string identifiers for the spread type (e.g., `'COKING_PROFIT'`)
- When the feature engineering module attempts to calculate variance on all feature columns, this string column cannot be numerically processed, causing the error

**Unusual Behavior:**
- Even with identical package versions, this error may manifest inconsistently across different computers
- Some environments may throw the error while others run normally
- This is likely related to subtle behavioral differences in underlying NumPy or pandas implementations, though the exact cause remains undetermined

**Temporary Solution:**
- Remove the `'spread_name'` column or other string-type columns from the DataFrame before feature selection
- Modify the `calculate_spread` function to exclude the `'spread_name'` column from the returned DataFrame
- If this information needs to be preserved, return it as a separate variable rather than mixing it into the feature DataFrame

**如有疑问，请联系 | For Questions, Please Contact:**
e1538433@u.nus.edu

---

## 总结 | Summary

**中文：**
本项目为价差套利策略的开发和优化提供了完整的解决方案。通过模块化设计和自动化流程，用户可以快速实现从数据处理到策略回测的全流程操作，为实际交易提供有力支持。

**English:**
This project provides a complete solution for the development and optimization of spread arbitrage strategies. Through modular design and automated workflows, users can quickly implement the entire process from data processing to strategy backtesting, providing strong support for actual trading.
