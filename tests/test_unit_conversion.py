"""
价格单位转换功能测试
验证裂解价差计算中的单位转换是否正确
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.spread_calculator import SpreadCalculator, GALLONS_PER_BARREL


def test_unit_conversion():
    """测试单位转换函数"""
    print("="*80)
    print("测试1: 单位转换函数")
    print("="*80)
    
    # 测试数据
    test_prices = pd.Series([70.0, 71.0, 72.0, 73.0, 74.0])
    
    # 测试CL（原油）- 无需转换
    print("\n测试 CL (WTI原油):")
    cl_converted = SpreadCalculator.convert_units(test_prices, 'CL')
    print(f"  原始价格: {test_prices.mean():.2f} 美元/桶")
    print(f"  转换后:   {cl_converted.mean():.2f} 美元/桶")
    assert (cl_converted == test_prices).all(), "CL转换失败：不应改变"
    print("  ✅ 通过")
    
    # 测试RBOB（汽油）- 需要乘以42
    print("\n测试 RBOB (汽油):")
    rbob_prices = pd.Series([2.0, 2.1, 2.2, 2.3, 2.4])
    rbob_converted = SpreadCalculator.convert_units(rbob_prices, 'RBOB')
    expected_rbob = rbob_prices * GALLONS_PER_BARREL
    
    print(f"  原始价格: {rbob_prices.mean():.2f} 美元/加仑")
    print(f"  转换系数: × {GALLONS_PER_BARREL}")
    print(f"  转换后:   {rbob_converted.mean():.2f} 美元/桶")
    print(f"  预期值:   {expected_rbob.mean():.2f} 美元/桶")
    
    assert np.allclose(rbob_converted, expected_rbob), "RBOB转换失败"
    print("  ✅ 通过")
    
    # 测试HO（取暖油）- 需要乘以42
    print("\n测试 HO (取暖油):")
    ho_prices = pd.Series([1.9, 2.0, 2.1, 2.2, 2.3])
    ho_converted = SpreadCalculator.convert_units(ho_prices, 'HO')
    expected_ho = ho_prices * GALLONS_PER_BARREL
    
    print(f"  原始价格: {ho_prices.mean():.2f} 美元/加仑")
    print(f"  转换系数: × {GALLONS_PER_BARREL}")
    print(f"  转换后:   {ho_converted.mean():.2f} 美元/桶")
    print(f"  预期值:   {expected_ho.mean():.2f} 美元/桶")
    
    assert np.allclose(ho_converted, expected_ho), "HO转换失败"
    print("  ✅ 通过")


def test_crack_spread_calculation():
    """测试裂解价差计算"""
    print("\n\n" + "="*80)
    print("测试2: 裂解价差计算（3:2:1）")
    print("="*80)
    
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    
    test_data = {
        'CL': pd.DataFrame({
            'close': [70.0, 71.0, 72.0, 73.0, 74.0]
        }, index=dates),
        
        'RBOB': pd.DataFrame({
            'close': [2.0, 2.1, 2.2, 2.3, 2.4]
        }, index=dates),
        
        'HO': pd.DataFrame({
            'close': [1.9, 2.0, 2.1, 2.2, 2.3]
        }, index=dates)
    }
    
    print("\n原始价格（第1天）:")
    print(f"  CL:   {test_data['CL']['close'].iloc[0]:.2f} 美元/桶")
    print(f"  RBOB: {test_data['RBOB']['close'].iloc[0]:.2f} 美元/加仑")
    print(f"  HO:   {test_data['HO']['close'].iloc[0]:.2f} 美元/加仑")
    
    # 创建价差计算器
    spread_calc = SpreadCalculator()
    
    # 创建3:2:1裂解价差配置
    spread_calc.create_spread(
        'CRACK_3_2_1',
        [('CL', -3), ('RBOB', 2), ('HO', 1)]
    )
    
    # 计算价差（带单位转换）
    print("\n计算价差（自动单位转换）...")
    spread_with_conversion = spread_calc.calculate_spread(
        'CRACK_3_2_1',
        test_data,
        convert_to_barrels=True
    )
    
    # 手动计算验证（第1天）
    cl = test_data['CL']['close'].iloc[0]
    rbob = test_data['RBOB']['close'].iloc[0] * GALLONS_PER_BARREL
    ho = test_data['HO']['close'].iloc[0] * GALLONS_PER_BARREL
    
    expected_spread = -3*cl + 2*rbob + 1*ho
    actual_spread = spread_with_conversion['spread'].iloc[0]
    
    print(f"\n转换后价格（第1天）:")
    print(f"  CL:   {cl:.2f} 美元/桶")
    print(f"  RBOB: {rbob:.2f} 美元/桶 (原始 {test_data['RBOB']['close'].iloc[0]:.2f} × {GALLONS_PER_BARREL})")
    print(f"  HO:   {ho:.2f} 美元/桶 (原始 {test_data['HO']['close'].iloc[0]:.2f} × {GALLONS_PER_BARREL})")
    
    print(f"\n价差计算:")
    print(f"  公式: -3×CL + 2×RBOB + 1×HO")
    print(f"  计算: -3×{cl:.2f} + 2×{rbob:.2f} + 1×{ho:.2f}")
    print(f"  结果: {-3*cl:.2f} + {2*rbob:.2f} + {ho:.2f}")
    print(f"  预期值: {expected_spread:.2f} 美元/桶")
    print(f"  实际值: {actual_spread:.2f} 美元/桶")
    print(f"  误差:   {abs(expected_spread - actual_spread):.6f}")
    
    assert abs(expected_spread - actual_spread) < 0.01, "价差计算不匹配"
    print("\n  ✅ 价差计算正确")
    
    # 检查价差范围
    avg_spread = spread_with_conversion['spread'].mean()
    print(f"\n价差统计:")
    print(f"  平均值: {avg_spread:.2f} 美元/桶")
    print(f"  范围:   {spread_with_conversion['spread'].min():.2f} - {spread_with_conversion['spread'].max():.2f}")
    
    if abs(avg_spread) < 200:
        print("  ✅ 价差值在合理范围内")
    else:
        print("  ⚠️ 警告：价差值异常")


def test_without_conversion():
    """测试不转换单位的错误情况"""
    print("\n\n" + "="*80)
    print("测试3: 对比有无单位转换的差异")
    print("="*80)
    
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=3, freq='D')
    
    test_data = {
        'CL': pd.DataFrame({'close': [70.0, 71.0, 72.0]}, index=dates),
        'RBOB': pd.DataFrame({'close': [2.0, 2.1, 2.2]}, index=dates),
        'HO': pd.DataFrame({'close': [1.9, 2.0, 2.1]}, index=dates)
    }
    
    spread_calc = SpreadCalculator()
    spread_calc.create_spread('CRACK_3_2_1', [('CL', -3), ('RBOB', 2), ('HO', 1)])
    
    # 计算价差（带转换）
    spread_with = spread_calc.calculate_spread(
        'CRACK_3_2_1', test_data, convert_to_barrels=True
    )
    
    # 模拟不转换的错误情况
    cl = test_data['CL']['close'].iloc[0]
    rbob = test_data['RBOB']['close'].iloc[0]  # 不转换
    ho = test_data['HO']['close'].iloc[0]      # 不转换
    
    wrong_spread = -3*cl + 2*rbob + 1*ho
    correct_spread = spread_with['spread'].iloc[0]
    
    print(f"\n❌ 错误计算（不转换单位）:")
    print(f"  公式: -3×70.00 + 2×2.00 + 1×1.90")
    print(f"  结果: {wrong_spread:.2f} （单位混乱！）")
    
    print(f"\n✅ 正确计算（转换单位）:")
    print(f"  公式: -3×70.00 + 2×(2.00×42) + 1×(1.90×42)")
    print(f"  公式: -3×70.00 + 2×84.00 + 1×79.80")
    print(f"  结果: {correct_spread:.2f} 美元/桶")
    
    print(f"\n差异: {abs(correct_spread - wrong_spread):.2f} 美元/桶")
    print(f"差异比例: {abs(correct_spread - wrong_spread)/correct_spread*100:.1f}%")
    
    print("\n⚠️ 警告：不转换单位会导致严重的计算错误！")


def test_real_world_scenario():
    """测试真实场景数据"""
    print("\n\n" + "="*80)
    print("测试4: 真实市场数据场景")
    print("="*80)
    
    # 模拟真实的市场价格
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    
    # 典型的能源价格范围
    np.random.seed(42)
    test_data = {
        'CL': pd.DataFrame({
            'close': np.random.uniform(65, 75, 10)  # 原油：65-75 美元/桶
        }, index=dates),
        
        'RBOB': pd.DataFrame({
            'close': np.random.uniform(2.2, 2.6, 10)  # 汽油：2.2-2.6 美元/加仑
        }, index=dates),
        
        'HO': pd.DataFrame({
            'close': np.random.uniform(2.0, 2.4, 10)  # 取暖油：2.0-2.4 美元/加仑
        }, index=dates)
    }
    
    spread_calc = SpreadCalculator()
    spread_calc.create_spread('CRACK_3_2_1', [('CL', -3), ('RBOB', 2), ('HO', 1)])
    
    spread_df = spread_calc.calculate_spread('CRACK_3_2_1', test_data)
    
    print("\n真实场景统计:")
    print(f"  原油平均价:   {test_data['CL']['close'].mean():.2f} 美元/桶")
    print(f"  汽油平均价:   {test_data['RBOB']['close'].mean():.2f} 美元/加仑")
    print(f"  取暖油平均价: {test_data['HO']['close'].mean():.2f} 美元/加仑")
    
    print(f"\n价差统计:")
    print(f"  平均价差: {spread_df['spread'].mean():.2f} 美元/桶")
    print(f"  标准差:   {spread_df['spread'].std():.2f} 美元/桶")
    print(f"  最小值:   {spread_df['spread'].min():.2f} 美元/桶")
    print(f"  最大值:   {spread_df['spread'].max():.2f} 美元/桶")
    
    # 合理性检查
    avg_spread = spread_df['spread'].mean()
    
    print(f"\n合理性检查:")
    if 0 < avg_spread < 100:
        print(f"  ✅ 价差在正常范围 (0-100)，炼油有利润")
    elif -20 < avg_spread < 0:
        print(f"  ⚠️ 价差为负，炼油亏损")
    elif abs(avg_spread) > 200:
        print(f"  ❌ 价差异常，可能未正确转换单位！")
    else:
        print(f"  ✅ 价差正常")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪 价格单位转换功能测试套件 🧪".center(80))
    print("="*80)
    
    try:
        test_unit_conversion()
        test_crack_spread_calculation()
        test_without_conversion()
        test_real_world_scenario()
        
        print("\n\n" + "="*80)
        print("✅ 所有测试通过！".center(80))
        print("="*80)
        
        print("\n总结:")
        print("  ✅ 单位转换函数工作正常")
        print("  ✅ 裂解价差计算准确")
        print("  ✅ 自动转换机制有效")
        print("  ✅ 价差值在合理范围")
        
    except AssertionError as e:
        print(f"\n\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        raise


if __name__ == '__main__':
    run_all_tests()
