"""
导入棕榈油港口库存数据到数据库
"""
import pandas as pd
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.database import DatabaseManager

def process_palm_oil_inventory(csv_path: str, db_path: str):
    """
    处理棕榈油港口库存数据并存入数据库
    
    Args:
        csv_path: CSV文件路径
        db_path: 数据库路径
    """
    print("="*60)
    print("开始处理棕榈油港口库存数据")
    print("="*60)
    
    # 1. 读取CSV文件
    print("\n[1/5] 读取CSV文件...")
    # 指定编码，处理中文列名
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except:
        df = pd.read_csv(csv_path, encoding='gbk')
    
    print(f"  ✓ 读取 {len(df)} 行数据")
    print(f"  原始列名: {df.columns.tolist()}")
    
    # 2. 重命名列为英文和Python合法名称
    print("\n[2/5] 重命名列...")
    # 检测列名并重命名
    original_columns = df.columns.tolist()
    column_mapping = {}
    
    # 尝试不同的列名格式
    for col in original_columns:
        col_clean = col.strip()
        if '时间' in col_clean or '日期' in col_clean or 'date' in col_clean.lower():
            column_mapping[col] = 'date'
        elif '棕榈油' in col_clean and '库存' in col_clean:
            column_mapping[col] = 'palm_oil_port_inventory'
    
    # 如果没有找到映射，使用默认
    if 'date' not in column_mapping.values():
        column_mapping[original_columns[0]] = 'date'
    if 'palm_oil_port_inventory' not in column_mapping.values():
        column_mapping[original_columns[1]] = 'palm_oil_port_inventory'
    
    df = df.rename(columns=column_mapping)
    print(f"  ✓ 新列名: {df.columns.tolist()}")
    print(f"  ✓ 列映射: {column_mapping}")
    
    # 3. 处理日期和时区
    print("\n[3/5] 处理日期和时区...")
    # 转换日期列为datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # 删除无效日期的行
    df = df.dropna(subset=['date'])
    # 添加东八区时区 (Asia/Shanghai = UTC+8)
    df['date'] = df['date'].dt.tz_localize('Asia/Shanghai')
    print(f"  ✓ 日期时区: {df['date'].dt.tz}")
    print(f"  ✓ 日期范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"  ✓ 有效日期数: {len(df)}")
    
    # 4. 清理数值列（去除空格，转换为float）
    print("\n[4/5] 清理和转换数值列...")
    numeric_columns = ['palm_oil_port_inventory']
    
    for col in numeric_columns:
        # 去除空格
        df[col] = df[col].astype(str).str.strip()
        # 替换空字符串为NaN
        df[col] = df[col].replace('', pd.NA)
        df[col] = df[col].replace('nan', pd.NA)
        # 转换为float
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        non_null_count = df[col].notna().sum()
        print(f"  ✓ {col}: {non_null_count} 个有效值")
    
    # 5. 设置日期为索引
    df.set_index('date', inplace=True)
    
    # 6. 存入数据库
    print("\n[5/5] 存入数据库...")
    db = DatabaseManager(db_path)
    
    # 遍历每一行，存入fundamental_data表
    success_count = 0
    for date, row in df.iterrows():
        try:
            # 准备数据字典（只包含非空值）
            data_dict = {}
            for col in numeric_columns:
                if pd.notna(row[col]):
                    data_dict[col] = float(row[col])
            
            # 只有当有数据时才插入
            if data_dict:
                db.insert_fundamental_data(
                    data_source='PALM_OIL_INVENTORY',
                    report_date=date.strftime('%Y-%m-%d'),
                    publish_date=date.strftime('%Y-%m-%d'),
                    data_dict=data_dict
                )
                success_count += 1
                
        except Exception as e:
            print(f"  ✗ 插入失败 {date}: {e}")
    
    print(f"  ✓ 成功插入 {success_count} 条记录")
    
    # 7. 验证数据
    print("\n[6/6] 验证数据...")
    verify_df = db.get_fundamental_data(
        data_source='PALM_OIL_INVENTORY',
        start_date='2002-01-01'
    )
    
    if not verify_df.empty:
        print(f"  ✓ 数据库中共有 {len(verify_df)} 条记录")
        print(f"  ✓ 日期范围: {verify_df['report_date'].min()} 至 {verify_df['report_date'].max()}")
        # 显示数据样本
        print("\n  数据样本:")
        print(verify_df.head(5))
    else:
        print("  ✗ 数据库中无数据")
    
    print("\n" + "="*60)
    print("✅ 棕榈油数据处理和导入完成！")
    print("="*60)
    
    return df


if __name__ == '__main__':
    # 文件路径
    csv_path = "outside_data_import/棕榈油港口库存_合计.csv"
    db_path = "data/trading_data.db"
    
    # 处理数据
    process_palm_oil_inventory(csv_path, db_path)
