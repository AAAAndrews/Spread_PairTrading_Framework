"""
EIA数据获取示例
展示如何使用EIA Python Library获取能源数据
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data_fetcher import DataFetcher
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    
    # 创建数据获取器
    fetcher = DataFetcher()
    
    # EIA API密钥（请替换为您自己的密钥）
    api_key = "O8N0mhrHyhUV3pg6qus3cSZ62EMYteG4Ar2SvgAc"
    
    print("=" * 60)
    print("EIA数据获取示例")
    print("=" * 60)
    
    # 示例1：获取美国原油库存数据
    print("\n1. 获取美国原油库存数据（周度）")
    crude_stock = fetcher.fetch_eia_data(
        series_id='PET.WCRSTUS1.W',
        api_key=api_key
    )
    if not crude_stock.empty:
        print(f"   数据条数: {len(crude_stock)}")
        print(f"   时间范围: {crude_stock.index.min()} 至 {crude_stock.index.max()}")
        print(f"   最新数据:\n{crude_stock.tail()}")
    
    # 示例2：获取美国汽油库存数据
    print("\n2. 获取美国汽油库存数据（周度）")
    gasoline_stock = fetcher.fetch_eia_data(
        series_id='PET.WGTSTUS1.W',
        api_key=api_key
    )
    if not gasoline_stock.empty:
        print(f"   数据条数: {len(gasoline_stock)}")
        print(f"   最新数据:\n{gasoline_stock.tail()}")
    
    # 示例3：获取美国馏分油库存数据
    print("\n3. 获取美国馏分油库存数据（周度）")
    distillate_stock = fetcher.fetch_eia_data(
        series_id='PET.WDISTUS1.W',
        api_key=api_key
    )
    if not distillate_stock.empty:
        print(f"   数据条数: {len(distillate_stock)}")
        print(f"   最新数据:\n{distillate_stock.tail()}")
    
    # 示例4：获取WTI原油现货价格
    print("\n4. 获取WTI原油现货价格（日度）")
    wti_price = fetcher.fetch_eia_data(
        series_id='PET.RWTC.D',
        api_key=api_key
    )
    if not wti_price.empty:
        print(f"   数据条数: {len(wti_price)}")
        print(f"   最新数据:\n{wti_price.tail()}")
    
    print("\n" + "=" * 60)
    print("常用EIA序列ID参考：")
    print("=" * 60)
    print("原油库存相关：")
    print("  PET.WCRSTUS1.W  - 美国原油库存（周度）")
    print("  PET.WCESTUS1.W  - 美国原油库欣地区库存（周度）")
    print("\n成品油库存相关：")
    print("  PET.WGTSTUS1.W  - 美国汽油库存（周度）")
    print("  PET.WDISTUS1.W  - 美国馏分油库存（周度）")
    print("\n价格相关：")
    print("  PET.RWTC.D      - WTI原油现货价格（日度）")
    print("  PET.RBRTE.D     - 布伦特原油现货价格（日度）")
    print("\n产量相关：")
    print("  PET.WCRFPUS2.W  - 美国原油产量（周度）")
    print("  PET.WGFRPUS2.W  - 美国汽油产量（周度）")
    print("\n更多序列ID请访问: https://www.eia.gov/opendata/")
    print("=" * 60)


if __name__ == "__main__":
    main()
