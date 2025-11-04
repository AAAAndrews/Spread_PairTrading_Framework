"""
简单的EIA数据获取测试
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data_fetcher import DataFetcher

# 创建数据获取器
fetcher = DataFetcher()

# 测试获取美国原油库存数据
print("测试获取美国原油库存数据...")
crude_stock = fetcher.fetch_eia_data(series_id='PET.WCRSTUS1.W')

if not crude_stock.empty:
    print(f"\n✓ 成功获取数据!")
    print(f"数据条数: {len(crude_stock)}")
    print(f"时间范围: {crude_stock.index.min()} 至 {crude_stock.index.max()}")
    print(f"\n最新5条数据:")
    print(crude_stock.tail())
else:
    print("\n✗ 数据获取失败")
