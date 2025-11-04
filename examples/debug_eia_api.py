"""
调试EIA API响应格式
"""
from eia import API

api_key = "O8N0mhrHyhUV3pg6qus3cSZ62EMYteG4Ar2SvgAc"
series_id = 'PET.WCRSTUS1.W'

api = API(api_key)

print("测试EIA API...")
print(f"序列ID: {series_id}\n")

try:
    result = api.data_by_series(series=series_id)
    print("API返回结果类型:", type(result))
    print("\n完整结果:")
    print(result)
    
    if isinstance(result, list) and len(result) > 0:
        print("\n第一个元素的键:")
        print(result[0].keys())
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
