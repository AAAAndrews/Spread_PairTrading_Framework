# SignalGenerator 更新日志

## 🎉 版本更新：回归任务信号生成

### 新增功能

#### 1. 回归任务阈值逻辑
- **功能**：将回归模型的连续预测值转换为离散交易信号（1, 0, -1）
- **支持两种模式**：
  - 固定阈值模式
  - 滚动分位数模式（推荐）

#### 2. 滚动分位数阈值
- **动态阈值**：根据历史窗口数据自动计算阈值
- **自适应性**：随市场环境变化自动调整
- **可配置**：窗口大小、分位数位置可调

### 参数说明

#### 新增参数

```python
SignalGenerator(
    model,                              # 原有参数
    threshold=0.6,                      # 原有参数
    signal_holding_days=5,              # 原有参数
    
    # ✅ 新增：回归任务参数
    regression_upper_threshold=0.05,    # 固定上阈值
    regression_lower_threshold=-0.05,   # 固定下阈值
    use_rolling_quantile=False,         # 是否使用滚动分位数
    rolling_window=60,                  # 滚动窗口大小
    upper_quantile=0.75,                # 上分位数
    lower_quantile=0.25                 # 下分位数
)
```

### 使用场景

#### 适用于回归任务
- ✅ 预测未来收益率
- ✅ 预测价差变化
- ✅ 预测对数收益

#### 不适用于
- ❌ 直接预测价格
- ❌ 预测非方向性指标

### 核心优势

1. **统一接口**：分类和回归任务使用相同的 `SignalGenerator`
2. **灵活配置**：支持固定阈值和动态分位数
3. **自适应性**：滚动分位数自动适应市场变化
4. **信号强度**：自动计算信号强度，可用于仓位管理
5. **兼容性**：与现有的信号维持功能完全兼容

### 代码示例

#### 分类任务（原有功能）
```python
# 使用概率阈值
signal_gen = SignalGenerator(
    model,                  # 分类模型
    threshold=0.6,
    signal_holding_days=5
)
signals = signal_gen.generate_signals(X_test, use_probability=True)
```

#### 回归任务 - 固定阈值
```python
# 固定±3%阈值
signal_gen = SignalGenerator(
    model,                              # 回归模型
    regression_upper_threshold=0.03,
    regression_lower_threshold=-0.03,
    use_rolling_quantile=False,
    signal_holding_days=10
)
signals = signal_gen.generate_signals(X_test)
```

#### 回归任务 - 滚动分位数（推荐）
```python
# 动态75%/25%分位数
signal_gen = SignalGenerator(
    model,                          # 回归模型
    use_rolling_quantile=True,      # ✅ 使用滚动分位数
    rolling_window=60,
    upper_quantile=0.75,
    lower_quantile=0.25,
    signal_holding_days=20
)
signals = signal_gen.generate_signals(X_test)
```

### 输出格式

#### 信号DataFrame包含：

| 列名 | 说明 | 分类任务 | 回归任务 |
|-----|------|---------|---------|
| `prediction` | 原始预测值 | ✅ | ✅ |
| `signal` | 交易信号（1/0/-1） | ✅ | ✅ |
| `signal_strength` | 信号强度（0-1） | ✅ | ✅ |
| `proba_class_*` | 类别概率 | ✅ | ❌ |
| `max_proba` | 最大概率 | ✅ | ❌ |
| `upper_threshold` | 上阈值 | ❌ | ✅（滚动分位数时） |
| `lower_threshold` | 下阈值 | ❌ | ✅（滚动分位数时） |

### 性能优化

- ✅ 向量化操作：使用NumPy加速信号生成
- ✅ 高效滚动计算：pandas rolling API
- ✅ 内存优化：避免不必要的复制

### 测试和文档

#### 文档
- `docs/回归信号生成说明.md` - 详细说明文档
- `docs/回归信号快速指南.md` - 快速入门指南

#### 示例代码
- `examples/test_regression_signals.py` - 完整测试示例

#### 可视化
测试脚本会生成以下图表：
- `regression_signal_comparison.png` - 固定阈值 vs 滚动分位数
- `quantile_configuration_comparison.png` - 不同分位数配置对比
- `signal_holding_effect.png` - 信号维持效果
- `signal_strength_distribution.png` - 信号强度分布

### 向后兼容性

✅ **完全向后兼容**
- 原有的分类任务代码无需修改
- 新参数都有默认值
- 不影响现有功能

### 建议和最佳实践

1. **优先使用滚动分位数**
   - 更适应市场动态变化
   - 自动平衡信号频率
   
2. **合理设置窗口大小**
   - 日线数据：40-80天
   - 周线数据：12-26周
   
3. **分位数选择**
   - 保守：0.80/0.20
   - 标准：0.75/0.25
   - 激进：0.70/0.30
   
4. **信号维持匹配预测窗口**
   - 预测20天收益 → signal_holding_days=20
   
5. **利用信号强度**
   - 动态仓位管理
   - 过滤弱信号
   - 风险控制

### 未来计划

- [ ] 添加自适应分位数（根据波动率调整）
- [ ] 支持多阈值分级信号（强买/买/中性/卖/强卖）
- [ ] 添加信号置信区间
- [ ] 集成更多阈值选择方法（ATR、历史波动率等）

### 变更历史

**2025-11-01**
- ✅ 新增回归任务阈值逻辑
- ✅ 新增滚动分位数功能
- ✅ 新增信号强度计算
- ✅ 添加完整文档和示例
- ✅ 创建测试脚本和可视化

---

## 📞 使用反馈

如有问题或建议，请查看：
- 详细文档：`docs/回归信号生成说明.md`
- 快速指南：`docs/回归信号快速指南.md`
- 测试示例：`examples/test_regression_signals.py`
