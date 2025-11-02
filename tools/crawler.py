import requests
import pandas as pd
import time
import json
from datetime import datetime

def fetch_price_data(seqno, page_number=1, page_size=15):
    """
    获取商务部商品价格数据
    
    Parameters:
    seqno (str): 商品序列号，如"270"
    page_number (int): 页码，默认1
    page_size (int): 每页条数，默认15
    """
    
    # 请求URL
    url = "https://price.mofcom.gov.cn/datamofcom/front/price/pricequotation/priceQueryList"
    
    # 请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
        'Origin': 'https://price.mofcom.gov.cn',
        'Referer': f'https://price.mofcom.gov.cn/price_2021/pricequotation/pricequotationdetail.shtml?seqno={seqno}',
        'X-Requested-With': 'XMLHttpRequest',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }
    
    # 请求参数 - 只包含必需的参数，开始时间和结束时间为空
    data = {
        'seqno': str(seqno),
        'startTime': '',  # 空字符串
        'endTime': '',    # 空字符串
        'pageNumber': str(page_number),
        'pageSize': str(page_size)
    }
    
    try:
        # 发送POST请求
        response = requests.post(url, data=data, headers=headers, timeout=10)
        
        # 检查请求是否成功
        if response.status_code == 200:
            result = response.json()
            
            if 'rows' in result and result['rows']:
                df = pd.DataFrame(result['rows'])
                
                # 处理日期字段：将yyyy, mm, dd合并为标准日期格式
                df = process_date_columns(df)
                
                total_count = result.get('total', 0)
                max_page = result.get('maxPageNum', 0)
                
                print(f"第{page_number}页获取成功，本页{len(df)}条数据，总计{total_count}条，最大页数{max_page}")
                return df, total_count, max_page
            else:
                print(f"第{page_number}页没有数据")
                print(f"返回结果: {result}")
                return pd.DataFrame(), 0, 0
        else:
            print(f"请求失败，状态码：{response.status_code}")
            print(f"响应内容：{response.text}")
            return pd.DataFrame(), 0, 0
            
    except Exception as e:
        print(f"请求发生异常：{str(e)}")
        return pd.DataFrame(), 0, 0

def process_date_columns(df):
    """
    处理日期字段，将yyyy, mm, dd合并为标准日期
    """
    if all(col in df.columns for col in ['yyyy', 'mm', 'dd']):
        # 确保月份和日期是两位数
        df['mm'] = df['mm'].astype(str).str.zfill(2)
        df['dd'] = df['dd'].astype(str).str.zfill(2)
        
        # 合并为标准日期格式
        df['date'] = df['yyyy'] + '-' + df['mm'] + '-' + df['dd']
        
        # 转换为日期类型
        try:
            df['date'] = pd.to_datetime(df['date'])
        except:
            print("日期转换失败，保留字符串格式")
        
        # 按日期排序（最新的在前）
        df = df.sort_values('date', ascending=False).reset_index(drop=True)
    
    return df

def get_all_price_data(seqno, page_size=15, max_pages=None):
    """
    获取所有分页的数据
    
    Parameters:
    seqno (str): 商品序列号
    page_size (int): 每页条数
    max_pages (int): 最大获取页数（用于测试或限制数据量）
    """
    all_data = []
    page_number = 1
    total_count = 0
    max_page_num = 0
    
    while True:
        print(f"正在获取第{page_number}页...")
        
        df, current_total, current_max_page = fetch_price_data(
            seqno=seqno,
            page_number=page_number,
            page_size=page_size
        )
        
        # 第一页时更新总信息
        if page_number == 1:
            total_count = current_total
            max_page_num = current_max_page
            if total_count > 0:
                print(f"数据统计：总计{total_count}条，共{max_page_num}页")
        
        if not df.empty:
            all_data.append(df)
            
            # 判断是否还有更多数据
            if (max_page_num > 0 and page_number >= max_page_num) or (len(df) < page_size):
                print(f"已到达最后一页，停止获取")
                break
            elif max_pages and page_number >= max_pages:
                print(f"已达到最大页数限制({max_pages})，停止获取")
                break
            else:
                page_number += 1
                time.sleep(0.5)  # 礼貌性延迟，避免请求过于频繁
        else:
            print("没有获取到数据，停止爬取")
            break
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # 最终数据处理
        if 'date' in final_df.columns:
            final_df = final_df.sort_values('date', ascending=False).reset_index(drop=True)
        
        print(f"\n数据获取完成！共获取 {len(final_df)} 条记录")
        if 'date' in final_df.columns and len(final_df) > 0:
            min_date = final_df['date'].min()
            max_date = final_df['date'].max()
            min_date_str = min_date.strftime('%Y-%m-%d') if hasattr(min_date, 'strftime') else min_date
            max_date_str = max_date.strftime('%Y-%m-%d') if hasattr(max_date, 'strftime') else max_date
            print(f"时间范围: {min_date_str} 至 {max_date_str}")
        
        return final_df
    else:
        print("没有获取到任何数据")
        return pd.DataFrame()

def analyze_data(df):
    """
    分析获取的数据
    """
    if df.empty:
        print("没有数据可分析")
        return
    
    print("\n=== 数据统计分析 ===")
    print(f"总记录数: {len(df)}")
    
    if 'date' in df.columns:
        min_date = df['date'].min()
        max_date = df['date'].max()
        min_date_str = min_date.strftime('%Y-%m-%d') if hasattr(min_date, 'strftime') else min_date
        max_date_str = max_date.strftime('%Y-%m-%d') if hasattr(max_date, 'strftime') else max_date
        print(f"时间范围: {min_date_str} 至 {max_date_str}")
    
    if 'prod_name' in df.columns:
        print(f"商品名称: {df['prod_name'].iloc[0]}")
    if 'prod_spec' in df.columns:
        print(f"规格: {df['prod_spec'].iloc[0]}")
    if 'region' in df.columns:
        print(f"地区: {df['region'].iloc[0]}")
    if 'unit' in df.columns:
        print(f"单位: {df['unit'].iloc[0]}")
    
    if 'price' in df.columns:
        # 转换价格列为数值类型
        df['price_numeric'] = pd.to_numeric(df['price'], errors='coerce')
        valid_prices = df['price_numeric'].dropna()
        if len(valid_prices) > 0:
            unit = df['unit'].iloc[0] if 'unit' in df.columns else ''
            print(f"价格范围: {valid_prices.min():.2f} - {valid_prices.max():.2f} {unit}")
            print(f"平均价格: {valid_prices.mean():.2f} {unit}")

# 使用示例
if __name__ == "__main__":
    # 设置查询参数
    seqno = "130"  # 商品序列号
    
    print(f"开始获取商品数据 (seqno: {seqno})")
    print("=" * 50)
    
    # 先测试获取第一页数据
    print("第一步：测试获取第一页数据...")
    test_data, total, max_page = fetch_price_data(seqno, page_number=1, page_size=5)
    
    if not test_data.empty:
        print("\n测试数据预览：")
        print(test_data.head())
        
        # 获取所有数据
        print("\n第二步：获取所有数据...")
        all_data = get_all_price_data(seqno, page_size=15)
        
        if not all_data.empty:
            # 数据分析
            analyze_data(all_data)
            
            # 显示数据预览
            print("\n最终数据预览：")
            print(all_data.head(10))
            
            # 保存到CSV文件
            commodity_name = all_data['prod_name'].iloc[0] if 'prod_name' in all_data.columns else f"商品_{seqno}"
            filename = f"{commodity_name}_价格数据.csv"
            all_data.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\n数据已保存到：{filename}")
            
        else:
            print("未能获取到完整数据")
    else:
        print("测试失败，请检查：")
        print("1. seqno参数是否正确")
        print("2. 网络连接是否正常")
        print("3. 该商品是否有价格数据")