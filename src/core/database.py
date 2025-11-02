"""
数据库模块：负责SQLite3数据库的创建、管理和操作
"""
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
import pytz
from typing import Optional, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库管理类"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        """
        初始化数据库管理器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """初始化数据库，创建必要的表"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()
        logger.info(f"数据库已初始化: {self.db_path}")
    
    def _create_tables(self):
        """创建数据表"""
        cursor = self.conn.cursor()
        
        # 行情数据表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            open_interest REAL,
            adjusted_close REAL,
            data_type TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(symbol, date)
        )
        ''')
        
        # 基本面数据表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fundamental_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_source TEXT NOT NULL,
            report_date TEXT NOT NULL,
            publish_date TEXT NOT NULL,
            data_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(data_source, report_date)
        )
        ''')
        
        # 指标数据表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS indicator_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            indicator_name TEXT NOT NULL,
            symbol TEXT,
            date TEXT NOT NULL,
            value REAL,
            metadata TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(indicator_name, symbol, date)
        )
        ''')
        
        # 价差配比表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS spread_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            spread_name TEXT NOT NULL UNIQUE,
            config_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_symbol_date ON price_data(symbol, date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fundamental_source_date ON fundamental_data(data_source, report_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_indicator_name_date ON indicator_data(indicator_name, date)')
        
        self.conn.commit()
        logger.info("数据表创建完成")
    
    def insert_price_data(self, df: pd.DataFrame, symbol: str, data_type: str = "spot"):
        """
        插入或更新行情数据
        
        Args:
            df: 包含OHLCV数据的DataFrame
            symbol: 品种代码
            data_type: 数据类型（spot/futures/adjusted等）
        """
        # 确保列名统一
        df = df.copy()
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'Volume': 'volume', 'Open Interest': 'open_interest'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # 只保留我们需要的列
        valid_cols = ['open', 'high', 'low', 'close', 'volume', 'open_interest', 'adjusted_close']
        df_filtered = pd.DataFrame()
        
        for col in valid_cols:
            if col in df.columns:
                df_filtered[col] = df[col]
        
        df = df_filtered
        
        # 确保必需字段存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"缺少字段 {col}，将设置为None")
                df[col] = None
        
        # 添加元数据
        df['symbol'] = symbol
        df['data_type'] = data_type
        df['created_at'] = datetime.now(pytz.UTC).isoformat()
        
        # 重置索引，将日期作为列
        if isinstance(df.index, pd.DatetimeIndex):
            df['date'] = df.index.strftime('%Y-%m-%d %H:%M:%S%z')
        elif 'date' not in df.columns:
            df['date'] = df.index.astype(str)
        
        # 插入数据（忽略重复）
        try:
            df.to_sql('price_data', self.conn, if_exists='append', index=False)
            logger.info(f"成功插入 {len(df)} 条 {symbol} 的行情数据")
        except sqlite3.IntegrityError:
            # 如果有重复，逐行插入
            inserted = 0
            for _, row in df.iterrows():
                try:
                    row.to_frame().T.to_sql('price_data', self.conn, if_exists='append', index=False)
                    inserted += 1
                except sqlite3.IntegrityError:
                    continue
            logger.info(f"插入了 {inserted} 条新数据，跳过了 {len(df) - inserted} 条重复数据")
    
    def insert_fundamental_data(self, data_source: str, report_date: str, 
                               publish_date: str, data_dict: Dict[str, Any]):
        """
        插入基本面数据
        
        Args:
            data_source: 数据源名称（如EIA、COT）
            report_date: 报告日期
            publish_date: 发布日期
            data_dict: 数据内容字典
        """
        import json
        
        cursor = self.conn.cursor()
        created_at = datetime.now(pytz.UTC).isoformat()
        data_json = json.dumps(data_dict, ensure_ascii=False)
        
        try:
            cursor.execute('''
            INSERT INTO fundamental_data (data_source, report_date, publish_date, data_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ''', (data_source, report_date, publish_date, data_json, created_at))
            self.conn.commit()
            logger.info(f"成功插入基本面数据: {data_source} - {report_date}")
        except sqlite3.IntegrityError:
            logger.warning(f"数据已存在: {data_source} - {report_date}")
    
    def insert_indicator_data(self, indicator_name: str, df: pd.DataFrame, 
                            symbol: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        插入指标数据
        
        Args:
            indicator_name: 指标名称
            df: 包含日期和值的DataFrame
            symbol: 关联品种（可选）
            metadata: 元数据（可选）
        """
        import json
        
        df = df.copy()
        
        # 只保留value列（假设第一列是值，或者列名为value）
        if 'value' in df.columns:
            value_col = 'value'
        elif len(df.columns) > 0:
            value_col = df.columns[0]
        else:
            logger.error("DataFrame没有可用的数据列")
            return
        
        # 创建新的DataFrame只包含必需列
        result_df = pd.DataFrame()
        result_df['indicator_name'] = indicator_name
        result_df['symbol'] = symbol
        result_df['value'] = df[value_col]
        result_df['created_at'] = datetime.now(pytz.UTC).isoformat()
        result_df['metadata'] = json.dumps(metadata) if metadata else None
        
        # 确保有date列
        if isinstance(df.index, pd.DatetimeIndex):
            result_df['date'] = df.index.strftime('%Y-%m-%d %H:%M:%S%z')
        elif 'date' not in df.columns:
            result_df['date'] = df.index.astype(str)
        else:
            result_df['date'] = df['date']
        
        # 插入数据
        try:
            result_df.to_sql('indicator_data', self.conn, if_exists='append', index=False)
            logger.info(f"成功插入指标数据: {indicator_name}")
        except sqlite3.IntegrityError:
            inserted = 0
            for _, row in result_df.iterrows():
                try:
                    row.to_frame().T.to_sql('indicator_data', self.conn, if_exists='append', index=False)
                    inserted += 1
                except sqlite3.IntegrityError:
                    continue
            logger.info(f"插入了 {inserted} 条新指标数据")
    
    def get_price_data(self, symbol: str, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None, 
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        提取行情数据
        
        Args:
            symbol: 品种代码
            start_date: 开始日期
            end_date: 结束日期
            columns: 需要的列名列表
        
        Returns:
            DataFrame
        """
        query = "SELECT * FROM price_data WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        if columns:
            available_cols = [col for col in columns if col in df.columns]
            df = df[available_cols]
        
        return df
    
    def get_fundamental_data(self, data_source: str, start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        提取基本面数据
        
        Args:
            data_source: 数据源名称
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame
        """
        query = "SELECT * FROM fundamental_data WHERE data_source = ?"
        params = [data_source]
        
        if start_date:
            query += " AND report_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND report_date <= ?"
            params.append(end_date)
        
        query += " ORDER BY report_date"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        
        if not df.empty:
            # 解析JSON数据
            import json
            df['data'] = df['data_json'].apply(json.loads)
            df['report_date'] = pd.to_datetime(df['report_date'])
        
        return df
    
    def get_indicator_data(self, indicator_name: str, symbol: Optional[str] = None,
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        提取指标数据
        
        Args:
            indicator_name: 指标名称
            symbol: 品种代码（可选）
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame
        """
        query = "SELECT * FROM indicator_data WHERE indicator_name = ?"
        params = [indicator_name]
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        return df
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")
    
    def __del__(self):
        """析构函数"""
        self.close()
