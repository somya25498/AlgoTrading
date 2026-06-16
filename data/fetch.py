import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "GS", "BAC", "MS", "V", "MA", "UNH", "JNJ", "XOM", "CVX",
    "PFE", "ABBV", "WMT", "HD", "PG", "KO", "PEP", "MCD", "NFLX",
    "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "AVGO", "NOW",
    "SNOW", "PLTR", "SQ", "SHOP", "UBER", "ABNB", "COIN", "SOFI",
    "RIVN", "NIO", "BABA", "TSM", "PYPL", "SNAP", "SPOT", "RBLX"
]
RAW_DIR=Path("data/raw")
PROCESSED_DIR=Path("data/processed")
RAW_DIR.mkdir(parents=True,exist_ok=True)
PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
def fetch_raw(ticker:str):

    df=yf.download(ticker,period="5y",auto_adjust=True,progress=True)
    df.columns = [c[0].lower() if isinstance(c,tuple) else c.lower() for  c in df.columns]
    
    df.dropna(inplace=True)
    return df
def add_features(df):
    close=df["close"]
    volume=df["volume"]
    df["rsi"]=RSIIndicator(close=close,window=14).rsi()
    mac=MACD(close=close)
    df["macd"]=mac.macd()
    df["macd_signal"]=mac.macd_signal()
    df["macd_diff"]=mac.macd_diff()
    bb=BollingerBands(close=close,window=20)
    df["bb_upper"]=bb.bollinger_hband()
    df["bb_lower"]=bb.bollinger_lband()
    df["bb_pct"]=bb.bollinger_pband()
    df["momentum_5"]=close.pct_change(5)
    df["momentum_10"]=close.pct_change(10)
    df["momentum_20"]=close.pct_change(20)
    df["volatility_20"]=close.pct_change().rolling(20).std()
    df["volume_zscore"]=(volume-volume.rolling(20).mean())/volume.rolling(20).std()
    df["return_1d"] = close.pct_change(1).shift(-1)
    df["target"] = (df["return_1d"] > 0).astype(int)
    df.dropna(inplace=True)
    return df
def run():
    all_frames=[]
    for ticker in TICKERS:
        raw=fetch_raw(ticker)
        raw.to_parquet(RAW_DIR/f"{ticker}.parquet")
        features=add_features(raw)
        features["ticker"]=ticker
        all_frames.append(features)
        print("Done")
   
    combined=pd.concat(all_frames)
    combined.to_parquet(PROCESSED_DIR/"all_stocks.parquet")

if __name__=="__main__":
    run()      



