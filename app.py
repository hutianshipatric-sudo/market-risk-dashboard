import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Global Mining & Fund Risk Terminal",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0b0f19;
    color: #e5e7eb;
}
h1, h2, h3 {
    color: #facc15;
}
[data-testid="stMetricValue"] {
    color: #22c55e;
}
[data-testid="stMetricDelta"] {
    color: #38bdf8;
}
.block-container {
    padding-top: 1.5rem;
}
.risk-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #374151;
}
</style>
""", unsafe_allow_html=True)

refresh_seconds = st.sidebar.selectbox(
    "Auto Refresh | 自动刷新",
    [5, 10, 30, 60],
    index=0
)

st_autorefresh(
    interval=refresh_seconds * 1000,
    key="auto_refresh"
)

st.title("GLOBAL MINING & FUND RISK TERMINAL")
st.caption("全球矿业商品与基金风险智能监控终端")
st.caption(f"Last Updated | 最后更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

assets = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "VIX": "^VIX",
    "Gold": "GC=F",
    "Copper": "HG=F",
    "Crude Oil": "CL=F",
    "Silver": "SI=F",
    "US 10Y Yield": "^TNX",
    "USD/CNY": "CNY=X",
    "AUD/USD": "AUDUSD=X"
}

selected_assets = st.sidebar.multiselect(
    "Assets | 监控资产",
    list(assets.keys()),
    default=["S&P 500", "VIX", "Gold", "Copper", "Crude Oil", "USD/CNY"]
)

window = st.sidebar.slider(
    "Risk Window | 风险窗口",
    10, 120, 20
)

confidence = st.sidebar.selectbox(
    "VaR Confidence | VaR置信水平",
    [0.95, 0.99],
    index=0
)

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
    daily_change = latest_price / previous_price - 1

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

results = {}

for name in selected_assets:
    ticker = assets[name]
    data = download_data(ticker)

    if data is not None and len(data) > window + 2:
        results[name] = calculate_metrics(data, window, confidence)

st.markdown("## LIVE MARKET BOARD | 实时市场行情")

if not results:
    st.error("No data loaded. 数据加载失败。")
    st.stop()

cols = st.columns(len(results))

for col, (name, metric) in zip(cols, results.items()):
    col.metric(
        label=name,
        value=f"{metric['price']:.2f}",
        delta=f"{metric['daily_change']:.2%}"
    )

avg_vol = np.mean([m["vol"] for m in results.values()])
avg_var = np.mean([m["var"] for m in results.values()])
avg_dd = np.mean([abs(m["drawdown"]) for m in results.values()])

vix_value = results["VIX"]["price"] if "VIX" in results else None

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
    risk_level = "🔴 HIGH RISK | 高风险"
    risk_comment = """
    Market stress is elevated. Commodity and fund exposure should be closely monitored.
    当前市场压力较高，建议重点关注商品价格波动、基金组合敞口和下行风险。
    """
elif risk_score >= 3:
    risk_level = "🟡 MEDIUM RISK | 中等风险"
    risk_comment = """
    Risk level is moderate. Volatility, VaR, drawdown and commodity correlation should be monitored.
    当前风险处于中等水平，需要持续跟踪波动率、VaR、回撤以及商品联动关系。
    """
else:
    risk_level = "🟢 LOW RISK | 低风险"
    risk_comment = """
    Market conditions are relatively stable based on selected indicators.
    根据当前指标，市场整体状态相对稳定。
    """

left, right = st.columns([2, 1])

with left:
    st.markdown("## MARKET PERFORMANCE | 市场表现")

    perf_df = pd.DataFrame([
        {
            "Asset": name,
            "Daily Change": metric["daily_change"] * 100
        }
        for name, metric in results.items()
    ])

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Bar(
        x=perf_df["Asset"],
        y=perf_df["Daily Change"],
        name="Daily Change %"
    ))
    fig_perf.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        title="Daily Asset Performance | 资产日度表现",
        xaxis_title="Asset",
        yaxis_title="Daily Change %"
    )
    st.plotly_chart(fig_perf, width="stretch")

with right:
    st.markdown("## AI RISK SIGNAL | 智能风险信号")
    st.markdown(f"""
    <div class="risk-card">
    <h2>{risk_level}</h2>
    <p>{risk_comment}</p>
    <hr>
    <p><b>Average Volatility | 平均波动率:</b> {avg_vol:.2%}</p>
    <p><b>Average VaR | 平均VaR:</b> {avg_var:.2%}</p>
    <p><b>Average Drawdown | 平均回撤:</b> {avg_dd:.2%}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("## RISK SUMMARY | 风险汇总")

summary = []

for name, metric in results.items():
    summary.append({
        "Asset | 资产": name,
        "Price | 价格": round(metric["price"], 4),
        "Daily Change | 日涨跌": f"{metric['daily_change']:.2%}",
        "Annualized Vol | 年化波动率": f"{metric['vol']:.2%}",
        f"Historical VaR {int(confidence * 100)}% | 历史VaR": f"{metric['var']:.2%}",
        "Drawdown | 回撤": f"{metric['drawdown']:.2%}"
    })

summary_df = pd.DataFrame(summary)
st.dataframe(summary_df, width="stretch")

st.markdown("## PRICE TREND | 价格走势")

main_asset = st.selectbox(
    "Select asset for detailed chart | 选择详细图表资产",
    list(results.keys())
)

main_data = results[main_asset]["data"]

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(
    x=main_data.index,
    y=main_data["Close"],
    name=main_asset
))
fig_price.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0b0f19",
    plot_bgcolor="#0b0f19",
    title=f"{main_asset} Price Trend | 价格走势",
    xaxis_title="Date | 日期",
    yaxis_title="Price | 价格"
)
st.plotly_chart(fig_price, width="stretch")

st.markdown("## VOLATILITY & VaR | 波动率与VaR")

fig_risk = go.Figure()
fig_risk.add_trace(go.Scatter(
    x=main_data.index,
    y=main_data["Rolling Vol"],
    name="Rolling Volatility"
))
fig_risk.add_trace(go.Scatter(
    x=main_data.index,
    y=main_data["Historical VaR"],
    name="Historical VaR"
))
fig_risk.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0b0f19",
    plot_bgcolor="#0b0f19",
    title=f"{main_asset} Risk Metrics | 风险指标",
    xaxis_title="Date | 日期",
    yaxis_title="Risk Level | 风险水平"
)
st.plotly_chart(fig_risk, width="stretch")

st.markdown("## CORRELATION HEATMAP | 相关性热力图")

returns_dict = {}

for name, metric in results.items():
    returns_dict[name] = metric["data"]["Return"]

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
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        title="Asset Return Correlation | 资产收益率相关性"
    )

    st.plotly_chart(fig_corr, width="stretch")
else:
    st.warning("Not enough data to calculate correlation. 数据不足，无法计算相关性。")

with st.expander("Methodology | 方法说明"):
    st.markdown("""
**Annualized Volatility | 年化波动率**  
Calculated from daily log returns and annualized by multiplying by √252.  
基于日对数收益率计算，并乘以 √252 年化。

**Historical VaR | 历史VaR**  
Uses historical return quantiles to estimate potential one-day downside loss.  
使用历史收益率分位数估算未来一天潜在下行损失。

**Drawdown | 回撤**  
Measures the percentage decline from the historical peak.  
衡量资产价格相对于历史高点的下跌幅度。

**Correlation Heatmap | 相关性热力图**  
Shows co-movement among equities, commodities, rates and FX.  
展示股票、商品、利率和汇率之间的联动关系。
""")

st.markdown("""
---
**Disclaimer | 免责声明**  
This dashboard is a prototype for investment risk monitoring and commodity market analysis.  
It is for analytical and educational purposes only and does not constitute investment advice.  

本系统为投资风险监控与商品市场分析原型，仅用于数据展示、研究和学习，不构成任何投资建议。
""")