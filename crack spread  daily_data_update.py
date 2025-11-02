"""
每日数据更新脚本
用途：每天运行以更新数据库中的所有数据
用法：python scripts/daily_data_update.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.core.database import DatabaseManager
from src.core.data_fetcher import DataFetcher

# 配置日志
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'daily_update.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def update_futures_data(db: DatabaseManager, fetcher: DataFetcher, lookback_days: int = 30):
    """
    更新期货数据
    
    Args:
        db: 数据库管理器
        fetcher: 数据抓取器
        lookback_days: 回溯天数（防止遗漏数据）
    """
    logger.info("="*60)
    logger.info("更新期货数据")
    logger.info("="*60)
    
    # 期货品种配置
    symbols_config = {
        'CL': 'CL=F',    # WTI原油
        'RBOB': 'RB=F',  # RBOB汽油
        'HO': 'HO=F'     # 取暖油/柴油
    }
    
    # 计算起始日期
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    for name, symbol in symbols_config.items():
        try:
            logger.info(f"更新 {name} ({symbol}) 数据...")
            
            df = fetcher.fetch_yfinance_data(
                symbol, 
                start_date=start_date,
                interval='1d'
            )
            
            if not df.empty:
                # 保存到数据库（数据库会自动跳过重复数据）
                db.insert_price_data(df, name, 'adjusted')
                logger.info(f"✓ {name}: 获取 {len(df)} 条记录，已更新到数据库")
            else:
                logger.warning(f"✗ {name}: 未获取到新数据")
                
        except Exception as e:
            logger.error(f"✗ {name} 更新失败: {e}")
    
    logger.info("期货数据更新完成\n")


def update_macro_data(db: DatabaseManager, fetcher: DataFetcher, lookback_days: int = 30):
    """
    更新宏观数据
    
    Args:
        db: 数据库管理器
        fetcher: 数据抓取器
        lookback_days: 回溯天数
    """
    logger.info("="*60)
    logger.info("更新宏观数据")
    logger.info("="*60)
    
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    # VIX波动率指数
    try:
        logger.info("更新 VIX 数据...")
        vix_data = fetcher.fetch_index_data('VIX', start_date=start_date)
        if not vix_data.empty:
            db.insert_price_data(vix_data, 'VIX', 'index')
            logger.info(f"✓ VIX: 获取 {len(vix_data)} 条记录")
        else:
            logger.warning("✗ VIX: 未获取到新数据")
    except Exception as e:
        logger.error(f"✗ VIX 更新失败: {e}")
    
    # 美元指数
    try:
        logger.info("更新 DXY 数据...")
        dxy_data = fetcher.fetch_index_data('DXY', start_date=start_date)
        if not dxy_data.empty:
            db.insert_price_data(dxy_data, 'DXY', 'index')
            logger.info(f"✓ DXY: 获取 {len(dxy_data)} 条记录")
        else:
            logger.warning("✗ DXY: 未获取到新数据")
    except Exception as e:
        logger.error(f"✗ DXY 更新失败: {e}")
    
    logger.info("宏观数据更新完成\n")


def update_fundamental_data(db: DatabaseManager, fetcher: DataFetcher):
    """
    更新基本面数据
    
    Args:
        db: 数据库管理器
        fetcher: 数据抓取器
    """
    logger.info("="*60)
    logger.info("更新基本面数据")
    logger.info("="*60)
    
    try:
        logger.info("更新 EIA 数据...")
        eia_data = fetcher.fetch_eia_data('PET.WCRSTUS1.W')
        
        if not eia_data.empty:
            for date, row in eia_data.iterrows():
                db.insert_fundamental_data(
                    'EIA',
                    date.strftime('%Y-%m-%d'),
                    date.strftime('%Y-%m-%d'),
                    row.to_dict()
                )
            logger.info(f"✓ EIA: 获取 {len(eia_data)} 条记录")
        else:
            logger.warning("✗ EIA: 未获取到新数据")
            
    except Exception as e:
        logger.error(f"✗ EIA 更新失败: {e}")
    
    logger.info("基本面数据更新完成\n")


def main():
    """主函数"""
    logger.info("\n" + "="*80)
    logger.info(f"开始每日数据更新 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80 + "\n")
    
    # 初始化
    db_path = "data/trading_data.db"
    db = DatabaseManager(db_path)
    fetcher = DataFetcher()
    
    try:
        # 1. 更新期货数据（回溯30天）
        update_futures_data(db, fetcher, lookback_days=30)
        
        # 2. 更新宏观数据（回溯30天）
        update_macro_data(db, fetcher, lookback_days=30)
        
        # 3. 更新基本面数据
        update_fundamental_data(db, fetcher)
        
        logger.info("="*80)
        logger.info("✅ 所有数据更新完成！")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ 数据更新过程中发生错误: {e}")
        raise
    
    finally:
        db.conn.close()


if __name__ == "__main__":
    main()