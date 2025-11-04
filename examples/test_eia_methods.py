"""
测试不同的EIA API方法
"""
from eia import API
import json

api_key = "O8N0mhrHyhUV3pg6qus3cSZ62EMYteG4Ar2SvgAc"
series_id = 'PET.WCRSTUS1.W'

api = API(api_key)

print("=" * 60)
print("测试EIA API不同方法")
print("=" * 60)

# 方法1：使用 query 方法
print("\n方法1: api.query()")
try:
    result = api.query(series_id)
    print(f"结果类型: {type(result)}")
    if result:
        print(f"数据长度: {len(result)}")
        print(f"前3条数据: {result[:3]}")
except Exception as e:
    print(f"错误: {e}")

# 方法2：使用 data 方法
print("\n方法2: api.data()")
try:
    result = api.data(series_id)
    print(f"结果类型: {type(result)}")
    if result:
        print(f"数据长度: {len(result)}")
        print(f"前3条数据: {result[:3]}")
except Exception as e:
    print(f"错误: {e}")

# 方法3：检查API对象的所有方法
print("\n方法3: 查看API对象的所有方法")
methods = [m for m in dir(api) if not m.startswith('_')]
print("可用方法:", methods)
