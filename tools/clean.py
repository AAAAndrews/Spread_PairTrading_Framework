import pandas as pd

def clean_price_data(input_csv, output_csv=None):
    """
    清洗价格数据：只保留date和price列，并将时间顺序从逆序改为正序
    
    Parameters:
    input_csv (str): 输入的原始CSV文件路径
    output_csv (str): 输出的清洗后CSV文件路径，如果为None则自动生成
    """
    
    # 读取原始数据
    try:
        raw_df = pd.read_csv(input_csv, encoding='utf-8-sig')
        print(f"成功读取数据: {input_csv}")
        print(f"原始数据形状: {raw_df.shape}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 检查必要的列是否存在
    if 'date' not in raw_df.columns:
        print("错误：原始数据中缺少date列")
        print(f"可用列: {list(raw_df.columns)}")
        return
    
    if 'price' not in raw_df.columns:
        print("错误：原始数据中缺少price列")
        print(f"可用列: {list(raw_df.columns)}")
        return
    
    # 创建清洗后的DataFrame，只保留date和price列
    cleaned_df = pd.DataFrame()
    cleaned_df['date'] = raw_df['date']
    cleaned_df['price'] = raw_df['price']
    
    # 确保date列是日期类型
    try:
        cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
    except Exception as e:
        print(f"日期转换警告: {e}")
        # 如果转换失败，保持原样
    
    # 按日期正序排列（从早到晚）
    cleaned_df = cleaned_df.sort_values('date', ascending=True).reset_index(drop=True)
    
    # 生成输出文件名
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_清洗后.csv')
    
    # 保存清洗后的数据
    try:
        cleaned_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"清洗完成！数据已保存到: {output_csv}")
        print(f"清洗后数据形状: {cleaned_df.shape}")
        print(f"时间范围: {cleaned_df['date'].min()} 至 {cleaned_df['date'].max()}")
        print(f"数据条数: {len(cleaned_df)}")
        
        # 显示前后对比
        print("\n原始数据前5条（逆序）:")
        print(raw_df[['date', 'price']].head())
        print("\n清洗后数据前5条（正序）:")
        print(cleaned_df.head())
        
    except Exception as e:
        print(f"保存文件失败: {e}")

# 使用示例
if __name__ == "__main__":
    # 直接指定输入输出文件路径
    input_file = "棕榈油_价格数据.csv"  # 替换为您的原始CSV文件路径
    output_file = "棕榈油_价格数据_清洗后.csv"  # 替换为您想要的输出文件路径

    clean_price_data(input_file, output_file)