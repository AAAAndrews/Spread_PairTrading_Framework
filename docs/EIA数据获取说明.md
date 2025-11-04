# EIA数据获取功能使用说明

## 概述

本项目已成功集成EIA数据获取功能，通过EIA官方API直接从美国能源信息署（EIA）获取权威的能源数据，包括原油库存、成品油库存、价格、产量等数据。

**✅ 已验证功能：**
- ✓ 美国原油库存数据（周度）
- ✓ 美国汽油库存数据（周度）
- ✓ 美国馏分油库存数据（周度）
- ✓ 自动数据解析和时间序列转换
- ✓ v1和v2 API双重兼容

## 安装依赖

```bash
pip install eia-python
```

或者使用项目的requirements.txt：

```bash
pip install -r requirements.txt
```

## 获取API密钥

1. 访问 [EIA Open Data](https://www.eia.gov/opendata/) 
2. 点击 "Register" 注册账号
3. 登录后在个人资料页面获取API密钥
4. 默认密钥已包含在代码中，但建议使用自己的密钥以避免速率限制

## 使用方法

### 基本用法

```python
from src.core.data_fetcher import DataFetcher

# 创建数据获取器
fetcher = DataFetcher()

# 获取美国原油库存数据（周度）
crude_stock = fetcher.fetch_eia_data(
    series_id='PET.WCRSTUS1.W',
    api_key='YOUR_API_KEY'
)

print(crude_stock.tail())
```

### 运行示例脚本

```bash
python examples/eia_data_example.py
```

## 常用序列ID

### 原油库存相关
- `PET.WCRSTUS1.W` - 美国原油库存（周度）
- `PET.WCESTUS1.W` - 美国原油库欣地区库存（周度）
- `PET.W_EPC0_SAX_YCUOK_MBBL.W` - 库欣地区原油库存（周度）

### 成品油库存相关
- `PET.WGTSTUS1.W` - 美国汽油库存（周度）
- `PET.WDISTUS1.W` - 美国馏分油库存（周度）
- `PET.WKJSTUS1.W` - 美国煤油库存（周度）

### 价格相关
- `PET.RWTC.D` - WTI原油现货价格（日度，美元/桶）
- `PET.RBRTE.D` - 布伦特原油现货价格（日度，美元/桶）
- `PET.EER_EPMRU_PF4_Y35NY_DPG.D` - 纽约港汽油价格（日度）

### 产量相关
- `PET.WCRFPUS2.W` - 美国原油产量（周度，千桶/日）
- `PET.WGFRPUS2.W` - 美国汽油产量（周度，千桶/日）
- `PET.WDIUPUS2.W` - 美国馏分油产量（周度，千桶/日）

### 炼厂相关
- `PET.WPULEUS3.W` - 美国炼厂开工率（周度，%）
- `PET.WCRRIUS2.W` - 美国炼厂原油加工量（周度，千桶/日）

### 进出口相关
- `PET.WCRIMUS2.W` - 美国原油进口（周度，千桶/日）
- `PET.WCREXUS2.W` - 美国原油出口（周度，千桶/日）

## 在价差套利策略中使用

### 原油裂解价差示例

```python
from src.core.data_fetcher import DataFetcher
from src.core.spread_calculator import SpreadCalculator

# 创建数据获取器
fetcher = DataFetcher()

# 获取原油价格
wti_price = fetcher.fetch_eia_data('PET.RWTC.D')

# 获取汽油价格
gasoline_price = fetcher.fetch_yfinance_data('RB=F')  # RBOB汽油期货

# 获取库存数据作为基本面因子
crude_stock = fetcher.fetch_eia_data('PET.WCRSTUS1.W')
gasoline_stock = fetcher.fetch_eia_data('PET.WGTSTUS1.W')

# 计算裂解价差
spread_calc = SpreadCalculator()
crack_spread = spread_calc.calculate_crack_spread(
    crude_price=wti_price,
    product_prices={'gasoline': gasoline_price},
    ratio='3:2:1'  # 3桶原油 -> 2桶汽油 + 1桶馏分油
)
```

## 数据频率说明

- **日度（D）**: 每个工作日更新
- **周度（W）**: 每周三更新（库存数据）
- **月度（M）**: 每月更新
- **年度（A）**: 每年更新

## API限制

- 免费API每小时最多请求1000次
- 建议在代码中添加缓存机制，避免重复请求
- 可以使用`time.sleep()`在批量请求间添加延迟

## 错误处理

如果遇到以下错误：

1. **ImportError**: 请安装eia-python库
   ```bash
   pip install eia-python
   ```

2. **API Key Invalid**: 请检查API密钥是否正确

3. **Series ID Not Found**: 请检查序列ID是否正确，可在[EIA数据浏览器](https://www.eia.gov/opendata/browser/)中查找

4. **Rate Limit Exceeded**: 请求过于频繁，请等待一段时间或使用缓存

## 更多资源

- [EIA Open Data官方文档](https://www.eia.gov/opendata/)
- [EIA数据浏览器](https://www.eia.gov/opendata/browser/)
- [eia-python GitHub仓库](https://github.com/mra1385/EIA-python)

## 注意事项

1. EIA数据通常有1-2天的延迟
2. 库存数据每周三发布，反映前一周的情况
3. 部分数据可能会有修正，建议定期更新历史数据
4. 使用数据时请遵守EIA的使用条款
