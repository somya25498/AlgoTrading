# AlgoTrading System

An end-to-end quantitative trading research system built with Python, Rust, and LLMs.

## What it does

- Fetches and cleans real market data for 50+ stocks
- ML model (XGBoost) generates buy/sell signals from technical features
- Rust-powered risk engine computes options Greeks and Value at Risk
- LLM sentiment pipeline reads SEC earnings call transcripts via RAG
- Custom backtester validates strategy performance (Sharpe, Drawdown, Calmar)
- Streamlit dashboard with AI chatbot for portfolio Q&A
- Portfolio allocator: input your budget → get optimized stock allocation

## Tech Stack

| Layer | Technology |
|---|---|
| Data | Python, yfinance, pandas |
| Signal Generation | scikit-learn, XGBoost |
| Risk Engine | Rust, QuantLib, pyo3 |
| Sentiment | LangChain, RAG, SEC EDGAR API |
| Backtesting | Custom-built (no library) |
| Dashboard | Streamlit, Plotly |

## Project Structure