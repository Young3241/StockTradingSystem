import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess_data(file_path="data/csi300.csv"):
    # 加载数据
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # 处理缺失值（线性插值）
    df.interpolate(method="linear", inplace=True)

    # 计算技术指标（以MA和RSI为例）
    df["MA5"] = df["close"].rolling(window=5).mean()
    df["MA20"] = df["close"].rolling(window=20).mean()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # 标准化处理
    scaler_price = MinMaxScaler(feature_range=(0, 1))
    df[["close_norm"]] = scaler_price.fit_transform(df[["close"]])

    scaler_rsi = StandardScaler()
    df[["RSI_norm"]] = scaler_rsi.fit_transform(df[["RSI"]])

    # 保存处理后的数据
    df.to_csv("data/csi300_processed.csv")
    return df

if __name__ == "__main__":
    preprocess_data()