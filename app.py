import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go

# Page setup / 页面设置
st.set_page_config(
    page_title="Market Risk Dashboard | 市场风险监控",
    layout="wide"
)

# Main title / 主标题
st.title("S&P 500 + VIX Market Risk Dashboard | 标普500 + VIX 市场风险监控系统")

st.markdown("""
This dashboard monitors market risk using S&P 500 and VIX data.  
本系统通过标普500指数和VIX恐慌指数，动态监控市场风险。

Key risk metrics include price trend, volatility, historical VaR, and drawdown.  
核心风险指标包括：价格走势、年化波动率、历史VaR以及最大回撤。
""")

# Tickers / 数据代码
ticker = "^GSPC"      # S&P 500 Index / 标普500指数
vix_ticker = "^VIX"   # VIX Index / 恐慌指数

# Sidebar settings / 侧边栏参数
window = st.sidebar.slider(
    "Rolling Window | 滚动窗口",
    10, 120, 20
)

confidence = st.sidebar.selectbox(
    "VaR Confidence Level | VaR置信水平",
    [0.95, 0.99]
)

# Download data / 下载市场数据
data = yf.download(ticker, period="2y", interval="1d", auto_adjust=False)
vix = yf.download(vix_ticker, period="2y", interval="1d", auto_adjust=False)

# Fix multi-index columns from yfinance / 修复 yfinance 多层列名问题
data.columns = data.columns.get_level_values(0)
vix.columns = vix.columns.get_level_values(0)

# Calculate log return / 计算对数收益率
data["Return"] = np.log(data["Close"] / data["Close"].shift(1))

# Annualized volatility / 年化波动率
data["Rolling Vol"] = data["Return"].rolling(window).std() * np.sqrt(252)

# Historical VaR / 历史模拟法 VaR
alpha = 1 - confidence
data["Historical VaR"] = -data["Return"].rolling(window).quantile(alpha)

# Drawdown / 回撤计算
data["Peak"] = data["Close"].cummax()
data["Drawdown"] = data["Close"] / data["Peak"] - 1

# Latest metrics / 最新指标
latest_price = float(data["Close"].iloc[-1])
latest_vol = float(data["Rolling Vol"].iloc[-1])
latest_var = float(data["Historical VaR"].iloc[-1])
latest_dd = float(data["Drawdown"].iloc[-1])
latest_vix = float(vix["Close"].iloc[-1])

# Risk signal logic / 风险信号逻辑
if latest_vix > 30 or latest_vol > 0.30 or latest_var > 0.03:
    risk_signal = "High Risk | 高风险"
    risk_explanation = "Market stress is elevated. Volatility or VaR is above the warning threshold. 市场压力较高，波动率或VaR超过预警水平。"
elif latest_vix > 20 or latest_vol > 0.20 or latest_var > 0.02:
    risk_signal = "Medium Risk | 中等风险"
    risk_explanation = "Market risk is moderate. Investors should monitor volatility and downside risk. 市场风险处于中等水平，需要关注波动率和下行风险。"
else:
    risk_signal = "Low Risk | 低风险"
    risk_explanation = "Market conditions are relatively stable based on current indicators. 根据当前指标，市场状态相对稳定。"

# Metric cards / 指标卡片
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("S&P 500 | 标普500", f"{latest_price:.2f}")
col2.metric("VIX | 恐慌指数", f"{latest_vix:.2f}")
col3.metric("Annualized Vol | 年化波动率", f"{latest_vol:.2%}")
col4.metric(f"Historical VaR {int(confidence*100)}% | 历史VaR", f"{latest_var:.2%}")
col5.metric("Current Drawdown | 当前回撤", f"{latest_dd:.2%}")

# Risk signal display / 风险信号展示
st.subheader(f"Risk Signal | 风险信号: {risk_signal}")
st.info(risk_explanation)

# Explanation section / 指标解释
with st.expander("Metric Explanation | 指标解释"):
    st.markdown("""
**S&P 500 | 标普500**  
Represents the overall performance of large-cap U.S. equities.  
代表美国大型股票市场的整体表现。

**VIX | 恐慌指数**  
Measures expected market volatility. A higher VIX usually indicates higher market uncertainty.  
衡量市场预期波动率。VIX越高，通常代表市场不确定性越强。

**Annualized Volatility | 年化波动率**  
Measures how volatile daily returns are after annualization.  
衡量每日收益率波动经过年化后的风险水平。

**Historical VaR | 历史VaR**  
Estimates the potential one-day loss under a selected confidence level based on historical returns.  
基于历史收益率，在指定置信水平下估计未来一天可能出现的损失。

**Drawdown | 回撤**  
Measures how much the index has fallen from its historical peak.  
衡量指数从历史高点下跌的幅度。
""")

# Price chart / 价格走势图
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(
    x=data.index,
    y=data["Close"],
    name="S&P 500 | 标普500"
))
fig_price.update_layout(
    title="S&P 500 Price Trend | 标普500价格走势",
    xaxis_title="Date | 日期",
    yaxis_title="Index Level | 指数点位"
)
st.plotly_chart(fig_price, use_container_width=True)

# VIX chart / VIX走势图
fig_vix = go.Figure()
fig_vix.add_trace(go.Scatter(
    x=vix.index,
    y=vix["Close"],
    name="VIX | 恐慌指数"
))
fig_vix.update_layout(
    title="VIX Index Trend | VIX恐慌指数走势",
    xaxis_title="Date | 日期",
    yaxis_title="VIX Level | VIX点位"
)
st.plotly_chart(fig_vix, use_container_width=True)

# Risk metrics chart / 风险指标图
fig_risk = go.Figure()
fig_risk.add_trace(go.Scatter(
    x=data.index,
    y=data["Rolling Vol"],
    name="Rolling Volatility | 滚动波动率"
))
fig_risk.add_trace(go.Scatter(
    x=data.index,
    y=data["Historical VaR"],
    name="Historical VaR | 历史VaR"
))
fig_risk.update_layout(
    title="Volatility and VaR | 波动率与VaR",
    xaxis_title="Date | 日期",
    yaxis_title="Risk Level | 风险水平"
)
st.plotly_chart(fig_risk, use_container_width=True)

# Drawdown chart / 回撤图
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=data.index,
    y=data["Drawdown"],
    name="Drawdown | 回撤"
))
fig_dd.update_layout(
    title="Drawdown Analysis | 回撤分析",
    xaxis_title="Date | 日期",
    yaxis_title="Drawdown | 回撤"
)
st.plotly_chart(fig_dd, use_container_width=True)

# Footer / 说明
st.markdown("""
---
**Note | 说明**  
This dashboard is for analytical and educational purposes only. It is not investment advice.  
本系统仅用于数据分析和学习展示，不构成任何投资建议。
""")