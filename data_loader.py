import akshare as ak
import pandas as pd
from datetime import datetime

def fetch_csi300_data(start_date="20200101", end_date="20231231"):
    """
    从新浪财经获取沪深300指数历史行情数据（通过akshare库解析）
    
    参数:
        start_date (str): 起始日期，格式YYYYMMDD
        end_date (str): 结束日期，格式YYYYMMDD
    
    返回:
        pd.DataFrame: 包含日期、开盘价、收盘价等字段的DataFrame
    """
    # 获取原始数据并检查列名
    df = ak.stock_zh_index_daily(symbol="sh000300")
    
    # 强制统一日期列为 pandas.Timestamp
    # 处理可能的混合类型（字符串或 datetime.date）
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date  # 先转为 datetime.date
    df["date"] = pd.to_datetime(df["date"])  # 再转为 pandas.Timestamp
    
    # 输入参数转换为 pandas.Timestamp
    start_date = pd.to_datetime(start_date, format="%Y%m%d")
    end_date = pd.to_datetime(end_date, format="%Y%m%d")
    
    # 过滤日期范围
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    
    # 调试：验证类型一致性
    if not df.empty:
        print("df['date'].dtype:", df["date"].dtype)          # 应输出 datetime64[ns]
        print("start_date type:", type(start_date))           # 应输出 pandas.Timestamp
    return df

if __name__ == "__main__":
    try:
        data = fetch_csi300_data()
        data.to_csv("data/csi300.csv", index=False)
        print("数据已保存至 data/csi300.csv")
    except Exception as e:
        print(f"数据获取失败，错误信息: {e}")