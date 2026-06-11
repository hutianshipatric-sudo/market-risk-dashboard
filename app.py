import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ============================================================
# Page Setup | 页面设置
# ============================================================

st.set_page_config(
    page_title="Mining & Fund Risk Dashboard | 矿业与基金风控系统",
    layout="wide"
)

# ============================================================
# Auto Refresh | 自动刷新
# ============================================================

refresh_seconds = st.sidebar.selectbox(
    "Auto Refresh Interval | 自动刷新频率",
    [5, 10, 30, 60],
    index=0
)

st_autorefresh(
    interval=refresh_seconds * 1000,
    key="auto_refresh"
)

# ============================================================
# Title | 标题
# ============================================================

st.title("Global Mining & Fund Risk Intelligence Dashboard")
st.subheader("全球矿业商品与基金风险智能监控系统")

st.caption(
    f"Last Updated | 最后更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

st.markdown("""
This dashboard monitors global market, commodity, and portfolio risk indicators.  
本系统用于动态监控全球市场、矿业商品以及基金组合风险指标。

Current version uses Yahoo Finance data for prototype demonstration.  
当前版本使用 Yahoo Finance 数据作为原型展示，后续可接入 Bloomberg、Refinitiv、IBKR 或机构级行情 API。
""")

# ============================================================
# Asset Universe | 资产池
# ============================================================

assets = {
    "S&P 500 | 标普500": "^GSPC",
    "Nasdaq 100 | 纳斯达克100": "^NDX",
    "VIX | 恐慌指数": "^VIX",
    "Gold Futures | 黄金期货": "GC=F",
    "Copper Futures | 铜期货": "HG=F",
    "Crude Oil Futures | 原油期货": "CL=F",
    "Silver Futures | 白银期货": "SI=F",
    "US 10Y Yield | 美国10年期国债": "^TNX",
    "USD/CNY | 美元人民币": "CNY=X",
    "AUD/USD | 澳元美元": "AUDUSD=X"
}

selected_assets = st.sidebar.multiselect(
    "Select Assets | 选择监控资产",
    list(assets.keys()),
    default=[
        "S&P 500 | 标普500",
        "VIX | 恐慌指数",
        "Gold Futures | 黄金期货",
        "Copper Futures | 铜期货",
        "Crude Oil Futures | 原油期货",
        "USD/CNY | 美元人民币"
    ]
)

risk_window = st.sidebar.slider(
    "Risk Calculation Window | 风险计算窗口",
    10, 120, 20
)

confidence = st.sidebar.selectbox(
    "VaR Confidence Level | VaR置信水平",
    [0.95, 0.99],
    index=0
)

# ============================================================
# Data Functions | 数据函数
# ============================================================

@st.cache_data(ttl=refresh_seconds)
def download_data(ticker):
    data = yf.download(
        ticker,
        period="1y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def calculate_metrics(data, window, confidence):
    data = data.copy()

    data["Return"] = np.log(data["Close"] / data["Close"].shift(1))
    data["Rolling Vol"] = data["Return"].rolling(window).std() * np.sqrt(252)

    alpha = 1 - confidence
    data["Historical VaR"] = -data["Return"].rolling(window).quantile(alpha)

    data["Peak"] = data["Close"].cummax()
    data["Drawdown"] = data["Close"] / data["Peak"] - 1

    latest_price = float(data["Close"].iloc[-1])
    previous_price = float(data["Close"].iloc[-2])
    daily_change = (latest_price / previous_price - 1)

    latest_vol = float(data["Rolling Vol"].iloc[-1])
    latest_var = float(data["Historical VaR"].iloc[-1])
    latest_dd = float(data["Drawdown"].iloc[-1])

    return {
        "price": latest_price,
        "daily_change": daily_change,
        "vol": latest_vol,
        "var": latest_var,
        "drawdown": latest_dd,
        "data": data
    }


# ============================================================
# Load Data | 加载数据
# ============================================================

results = {}

for asset_name in selected_assets:
    ticker = assets[asset_name]
    raw_data = download_data(ticker)

    if raw_data is not None and len(raw_data) > risk_window + 2:
        results[asset_name] = calculate_metrics(raw_data, risk_window, confidence)

# ============================================================
# Top Market Ticker | 顶部行情栏
# ============================================================

st.markdown("## Live Market Monitor | 实时市场监控")

cols = st.columns(len(results))

for col, (asset_name, metric) in zip(cols, results.items()):
    col.metric(
        label=asset_name,
        value=f"{metric['price']:.2f}",
        delta=f"{metric['daily_change']:.2%}"
    )

# ============================================================
# Overall Risk Signal | 综合风险信号
# ============================================================

avg_vol = np.mean([m["vol"] for m in results.values()])
avg_var = np.mean([m["var"] for m in results.values()])
avg_dd = np.mean([abs(m["drawdown"]) for m in results.values()])

vix_value = None
for name, metric in results.items():
    if "VIX" in name:
        vix_value = metric["price"]

risk_score = 0

if avg_vol > 0.30:
    risk_score += 2
elif avg_vol > 0.20:
    risk_score += 1

if avg_var > 0.03:
    risk_score += 2
elif avg_var > 0.02:
    risk_score += 1

if avg_dd > 0.15:
    risk_score += 2
elif avg_dd > 0.08:
    risk_score += 1

if vix_value is not None:
    if vix_value > 30:
        risk_score += 2
    elif vix_value > 20:
        risk_score += 1

if risk_score >= 5:
    risk_level = "🔴 High Risk | 高风险"
    risk_comment = "Market stress is elevated. Portfolio and commodity exposure should be closely monitored. 当前市场压力较高，需要重点关注商品和基金组合风险敞口。"
elif risk_score >= 3:
    risk_level = "🟡 Medium Risk | 中等风险"
    risk_comment = "Risk level is moderate. Volatility, VaR and drawdown should be monitored. 当前风险处于中等水平，需要持续关注波动率、VaR和回撤。"
else:
    risk_level = "🟢 Low Risk | 低风险"
    risk_comment = "Market conditions are relatively stable based on selected indicators. 根据当前指标，市场整体状态相对稳定。"

st.markdown("## AI Risk Signal | 智能风险信号")
st.subheader(risk_level)
st.info(risk_comment)

# ============================================================
# Risk Summary Table | 风险汇总表
# ============================================================

summary_rows = []

for asset_name, metric in results.items():
    summary_rows.append({
        "Asset | 资产": asset_name,
        "Price | 价格": round(metric["price"], 4),
        "Daily Change | 日涨跌": f"{metric['daily_change']:.2%}",
        "Annualized Vol | 年化波动率": f"{metric['vol']:.2%}",
        f"Historical VaR {int(confidence * 100)}% | 历史VaR": f"{metric['var']:.2%}",
        "Drawdown | 回撤": f"{metric['drawdown']:.2%}"
    })

summary_df = pd.DataFrame(summary_rows)

st.markdown("## Risk Summary | 风险指标汇总")
st.dataframe(summary_df, use_container_width=True)

# ============================================================
# Commodity Performance Chart | 商品表现图
# ============================================================

st.markdown("## Commodity / Market Performance | 商品与市场表现")

performance_df = pd.DataFrame([
    {
        "Asset": asset_name,
        "Daily Change": metric["daily_change"] * 100
    }
    for asset_name, metric in results.items()
])

fig_perf = go.Figure()
fig_perf.add_trace(go.Bar(
    x=performance_df["Asset"],
    y=performance_df["Daily Change"],
    name="Daily Change % | 日涨跌幅"
))
fig_perf.update_layout(
    title="Daily Performance | 日度表现",
    xaxis_title="Asset | 资产",
    yaxis_title="Daily Change % | 日涨跌幅"
)
st.plotly_chart(fig_perf, width="stretch")

# ============================================================
# Price Charts | 价格走势图
# ============================================================

st.markdown("## Price Trend | 价格走势")

for asset_name, metric in results.items():
    data = metric["data"]

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=data.index,
        y=data["Close"],
        name=asset_name
    ))
    fig_price.update_layout(
        title=f"{asset_name} Price Trend | 价格走势",
        xaxis_title="Date | 日期",
        yaxis_title="Price | 价格"
    )

    st.plotly_chart(fig_price, width="stretch")

# ============================================================
# Risk Charts | 风险图表
# ============================================================

st.markdown("## Volatility and VaR | 波动率与VaR")

for asset_name, metric in results.items():
    data = metric["data"]

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
        title=f"{asset_name} Risk Metrics | 风险指标",
        xaxis_title="Date | 日期",
        yaxis_title="Risk Level | 风险水平"
    )

    st.plotly_chart(fig_risk, width="stretch")

# ============================================================
# Correlation Heatmap | 相关性热力图
# ============================================================

st.markdown("## Correlation Heatmap | 相关性热力图")

returns_dict = {}

for asset_name, metric in results.items():
    data = metric["data"]
    returns_dict[asset_name] = data["Return"]

returns_df = pd.DataFrame(returns_dict).dropna()

if len(returns_df) > 10:
    corr = returns_df.corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu",
        zmin=-1,
        zmax=1
    ))

    fig_corr.update_layout(
        title="Asset Return Correlation | 资产收益率相关性"
    )

    st.plotly_chart(fig_corr, width="stretch")
else:
    st.warning("Not enough data to calculate correlation. 数据不足，无法计算相关性。")

# ============================================================
# Metric Explanation | 指标解释
# ============================================================

with st.expander("Metric Explanation | 指标解释"):
    st.markdown("""
**Daily Change | 日涨跌幅**  
Measures the latest one-day price change.  
衡量最新一个交易日的价格变化。

**Annualized Volatility | 年化波动率**  
Measures the annualized standard deviation of daily returns.  
衡量每日收益率波动经过年化后的风险水平。

**Historical VaR | 历史VaR**  
Estimates potential one-day loss under a selected confidence level based on historical returns.  
基于历史收益率，在指定置信水平下估计未来一天可能出现的损失。

**Drawdown | 回撤**  
Measures how much the asset has fallen from its historical peak.  
衡量资产价格从历史高点下跌的幅度。

**Correlation Heatmap | 相关性热力图**  
Shows how different assets move together.  
展示不同资产之间的联动关系。
""")

# ============================================================
# Footer | 页脚说明
# ============================================================

st.markdown("""
---
**Disclaimer | 免责声明**  
This dashboard is for analytical and educational purposes only. It is not investment advice.  
本系统仅用于数据分析、风险展示和学习用途，不构成任何投资建议。
""")