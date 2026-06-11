import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Mining & Investment Risk Dashboard",
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

top_left, top_right = st.columns([4, 1])

with top_right:
    lang = st.selectbox("Language / 语言", ["中文", "English"], index=0)

is_cn = lang == "中文"

txt = {
    "title": "全球资产风险监控平台" if is_cn else "Global Asset Risk Dashboard",
    "subtitle": "标普500、VIX、黄金、铜、原油与美元人民币风险监控" if is_cn else "Risk monitoring for SPY, VIX, gold, copper, crude oil and USD/CNY",
    "last_update": "最后更新时间" if is_cn else "Last Updated",
    "refresh": "自动刷新频率" if is_cn else "Auto Refresh Interval",
    "risk_window": "风险计算窗口" if is_cn else "Risk Window",
    "confidence": "风险置信水平" if is_cn else "Confidence Level",
    "mc_days": "模拟未来交易日" if is_cn else "Future Simulation Days",
    "mc_paths": "模拟路径数量" if is_cn else "Number of Simulation Paths",
    "market_board": "实时市场行情" if is_cn else "Live Market Board",
    "risk_summary": "风险指标汇总" if is_cn else "Risk Summary",
    "price": "当前价格" if is_cn else "Price",
    "change": "今日涨跌幅" if is_cn else "Daily Change",
    "vol": "市场波动风险" if is_cn else "Market Volatility",
    "var": "未来一天最大可能亏损" if is_cn else "One-Day Potential Loss",
    "cvar": "极端情况下平均亏损" if is_cn else "Expected Shortfall",
    "drawdown": "距离历史高点跌幅" if is_cn else "Drawdown from Peak",
    "performance": "今日市场涨跌情况" if is_cn else "Daily Market Performance",
    "risk_signal": "智能风险提示" if is_cn else "AI Risk Signal",
    "price_trend": "价格走势分析" if is_cn else "Price Trend Analysis",
    "risk_chart": "风险指标走势" if is_cn else "Risk Metrics Trend",
    "correlation": "资产联动关系图" if is_cn else "Asset Correlation Heatmap",
    "mc_title": "蒙特卡洛上涨/下跌概率模拟" if is_cn else "Monte Carlo Gain/Loss Probability Simulation",
    "select_mc": "选择模拟资产" if is_cn else "Select Asset for Simulation",
    "prob_up": "上涨概率" if is_cn else "Probability of Gain",
    "prob_down": "下跌概率" if is_cn else "Probability of Loss",
    "mc_var": "模拟最大可能亏损" if is_cn else "Monte Carlo VaR",
    "mc_cvar": "极端情景平均亏损" if is_cn else "Monte Carlo CVaR",
    "method": "指标说明" if is_cn else "Methodology",
}

refresh_seconds = st.sidebar.selectbox(txt["refresh"], [5, 10, 30, 60], index=0)

st_autorefresh(
    interval=refresh_seconds * 1000,
    key="auto_refresh"
)

window = st.sidebar.slider(txt["risk_window"], 10, 120, 20)

confidence = st.sidebar.selectbox(txt["confidence"], [0.95, 0.99], index=0)

mc_days = st.sidebar.slider(txt["mc_days"], 5, 120, 30)

mc_paths = st.sidebar.slider(txt["mc_paths"], 500, 10000, 3000, step=500)

with top_left:
    st.title(txt["title"])
    st.caption(txt["subtitle"])
    st.caption(f"{txt['last_update']}：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

assets = {
    "标普500ETF" if is_cn else "S&P 500 ETF": "SPY",
    "VIX恐慌指数" if is_cn else "VIX Fear Index": "^VIX",
    "黄金期货" if is_cn else "Gold Futures": "GC=F",
    "铜期货" if is_cn else "Copper Futures": "HG=F",
    "原油期货" if is_cn else "Crude Oil Futures": "CL=F",
    "美元兑人民币" if is_cn else "USD/CNY": "CNY=X"
}

@st.cache_data(ttl=refresh_seconds)
def download_data(ticker):
    data = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if data is None or data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns:
        return None

    data = data.dropna(subset=["Close"])

    if data.empty:
        return None

    return data


def calculate_metrics(data, window, confidence):
    data = data.copy()

    data["Return"] = np.log(data["Close"] / data["Close"].shift(1))
    data["Rolling Vol"] = data["Return"].rolling(window).std() * np.sqrt(252)

    alpha = 1 - confidence

    data["Historical VaR"] = -data["Return"].rolling(window).quantile(alpha)

    data["CVaR"] = data["Return"].rolling(window).apply(
        lambda x: -x[x <= np.quantile(x, alpha)].mean()
        if len(x[x <= np.quantile(x, alpha)]) > 0 else np.nan,
        raw=False
    )

    data["Peak"] = data["Close"].cummax()
    data["Drawdown"] = data["Close"] / data["Peak"] - 1

    clean = data.dropna(subset=[
        "Close",
        "Return",
        "Rolling Vol",
        "Historical VaR",
        "CVaR",
        "Drawdown"
    ])

    if len(clean) < 2:
        return None

    latest_price = float(clean["Close"].iloc[-1])
    previous_price = float(clean["Close"].iloc[-2])

    return {
        "price": latest_price,
        "daily_change": latest_price / previous_price - 1,
        "vol": float(clean["Rolling Vol"].iloc[-1]),
        "var": float(clean["Historical VaR"].iloc[-1]),
        "cvar": float(clean["CVaR"].iloc[-1]),
        "drawdown": float(clean["Drawdown"].iloc[-1]),
        "data": data,
        "clean": clean
    }


results = {}

for name, ticker in assets.items():
    raw_data = download_data(ticker)

    if raw_data is not None and len(raw_data) > window + 10:
        metric = calculate_metrics(raw_data, window, confidence)
        if metric is not None:
            results[name] = metric

if not results:
    st.error("数据加载失败，请检查网络或稍后重试。" if is_cn else "Data loading failed. Please check the network or try again.")
    st.stop()

st.markdown(f"## {txt['market_board']}")

cols = st.columns(len(results))

for col, (name, metric) in zip(cols, results.items()):
    col.metric(
        label=name,
        value=f"{metric['price']:.2f}",
        delta=f"{metric['daily_change']:.2%}"
    )

avg_vol = np.mean([m["vol"] for m in results.values()])
avg_var = np.mean([m["var"] for m in results.values()])
avg_cvar = np.mean([m["cvar"] for m in results.values()])
avg_dd = np.mean([abs(m["drawdown"]) for m in results.values()])

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

vix_key = "VIX恐慌指数" if is_cn else "VIX Fear Index"

if vix_key in results:
    vix_value = results[vix_key]["price"]
    if vix_value > 30:
        risk_score += 2
    elif vix_value > 20:
        risk_score += 1

if is_cn:
    if risk_score >= 5:
        risk_level = "🔴 风险较高"
        risk_comment = "当前市场压力较高，建议重点关注商品价格波动、VIX恐慌指数、基金组合敞口和下行风险。"
    elif risk_score >= 3:
        risk_level = "🟡 风险中等"
        risk_comment = "当前市场存在一定不确定性，建议持续跟踪黄金、铜、原油、汇率和美股风险变化。"
    else:
        risk_level = "🟢 风险较低"
        risk_comment = "根据当前指标，市场整体状态相对稳定，但仍需持续观察商品价格和汇率变化。"
else:
    if risk_score >= 5:
        risk_level = "🔴 High Risk"
        risk_comment = "Market stress is elevated. Commodity prices, VIX, portfolio exposure and downside risk should be closely monitored."
    elif risk_score >= 3:
        risk_level = "🟡 Medium Risk"
        risk_comment = "Market uncertainty remains. Gold, copper, crude oil, FX and US equity risk should be monitored."
    else:
        risk_level = "🟢 Low Risk"
        risk_comment = "Market conditions are relatively stable based on current indicators."

left, right = st.columns([2, 1])

with left:
    st.markdown(f"## {txt['performance']}")

    perf_df = pd.DataFrame([
        {
            "资产" if is_cn else "Asset": name,
            "涨跌幅" if is_cn else "Daily Change": metric["daily_change"] * 100
        }
        for name, metric in results.items()
    ])

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Bar(
        x=perf_df["资产" if is_cn else "Asset"],
        y=perf_df["涨跌幅" if is_cn else "Daily Change"],
        name=txt["change"]
    ))
    fig_perf.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        xaxis_title="资产" if is_cn else "Asset",
        yaxis_title="涨跌幅（%）" if is_cn else "Daily Change (%)"
    )
    st.plotly_chart(fig_perf, use_container_width=True)

with right:
    st.markdown(f"## {txt['risk_signal']}")
    st.markdown(f"""
    <div class="risk-card">
    <h2>{risk_level}</h2>
    <p>{risk_comment}</p>
    <hr>
    <p><b>{txt["vol"]}：</b>{avg_vol:.2%}</p>
    <p><b>{txt["var"]}：</b>{avg_var:.2%}</p>
    <p><b>{txt["cvar"]}：</b>{avg_cvar:.2%}</p>
    <p><b>{txt["drawdown"]}：</b>{avg_dd:.2%}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"## {txt['risk_summary']}")

summary = []

for name, metric in results.items():
    summary.append({
        "资产" if is_cn else "Asset": name,
        txt["price"]: round(metric["price"], 4),
        txt["change"]: f"{metric['daily_change']:.2%}",
        txt["vol"]: f"{metric['vol']:.2%}",
        f"{txt['var']}（{int(confidence * 100)}%）": f"{metric['var']:.2%}",
        txt["cvar"]: f"{metric['cvar']:.2%}",
        txt["drawdown"]: f"{metric['drawdown']:.2%}"
    })

st.dataframe(pd.DataFrame(summary), use_container_width=True)

main_asset = st.selectbox(
    "选择查看资产" if is_cn else "Select Asset",
    list(results.keys())
)

main_data = results[main_asset]["data"]

st.markdown(f"## {txt['price_trend']}")

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
    title=f"{main_asset} - {txt['price_trend']}",
    xaxis_title="日期" if is_cn else "Date",
    yaxis_title=txt["price"]
)
st.plotly_chart(fig_price, use_container_width=True)

st.markdown(f"## {txt['risk_chart']}")

fig_risk = go.Figure()
fig_risk.add_trace(go.Scatter(
    x=main_data.index,
    y=main_data["Rolling Vol"],
    name=txt["vol"]
))
fig_risk.add_trace(go.Scatter(
    x=main_data.index,
    y=main_data["Historical VaR"],
    name=txt["var"]
))
fig_risk.add_trace(go.Scatter(
    x=main_data.index,
    y=main_data["CVaR"],
    name=txt["cvar"]
))
fig_risk.add_trace(go.Scatter(
    x=main_data.index,
    y=main_data["Drawdown"],
    name=txt["drawdown"]
))
fig_risk.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0b0f19",
    plot_bgcolor="#0b0f19",
    title=f"{main_asset} - {txt['risk_chart']}",
    xaxis_title="日期" if is_cn else "Date",
    yaxis_title="风险水平" if is_cn else "Risk Level"
)
st.plotly_chart(fig_risk, use_container_width=True)

st.markdown(f"## {txt['correlation']}")

returns_df = pd.DataFrame({
    name: metric["data"]["Return"]
    for name, metric in results.items()
}).dropna()

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
        title="资产走势关联分析" if is_cn else "Asset Return Correlation Analysis"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown(f"## {txt['mc_title']}")

mc_asset = st.selectbox(
    txt["select_mc"],
    list(results.keys()),
    key="mc_asset"
)

mc_data = results[mc_asset]["clean"]

recent_returns = mc_data["Return"].dropna()

mu = recent_returns.mean()
sigma = recent_returns.std()
latest_price = results[mc_asset]["price"]

np.random.seed(42)

random_returns = np.random.normal(
    loc=mu,
    scale=sigma,
    size=(mc_days, mc_paths)
)

price_paths = np.zeros((mc_days + 1, mc_paths))
price_paths[0] = latest_price

for t in range(1, mc_days + 1):
    price_paths[t] = price_paths[t - 1] * np.exp(random_returns[t - 1])

terminal_prices = price_paths[-1]
terminal_returns = terminal_prices / latest_price - 1

prob_up = np.mean(terminal_returns > 0)
prob_down = np.mean(terminal_returns < 0)
mc_var = -np.quantile(terminal_returns, 1 - confidence)
mc_cvar = -terminal_returns[
    terminal_returns <= np.quantile(terminal_returns, 1 - confidence)
].mean()

m1, m2, m3, m4 = st.columns(4)

m1.metric(txt["prob_up"], f"{prob_up:.2%}")
m2.metric(txt["prob_down"], f"{prob_down:.2%}")
m3.metric(txt["mc_var"], f"{mc_var:.2%}")
m4.metric(txt["mc_cvar"], f"{mc_cvar:.2%}")

path_index = pd.date_range(
    start=mc_data.index[-1],
    periods=mc_days + 1,
    freq="B"
)

fig_mc = go.Figure()

sample_paths = min(100, mc_paths)

for i in range(sample_paths):
    fig_mc.add_trace(go.Scatter(
        x=path_index,
        y=price_paths[:, i],
        mode="lines",
        line=dict(width=1),
        opacity=0.20,
        showlegend=False
    ))

fig_mc.add_trace(go.Scatter(
    x=path_index,
    y=np.mean(price_paths, axis=1),
    mode="lines",
    line=dict(width=4),
    name="平均模拟路径" if is_cn else "Average Simulated Path"
))

fig_mc.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0b0f19",
    plot_bgcolor="#0b0f19",
    title=f"{mc_asset} - {txt['mc_title']}",
    xaxis_title="日期" if is_cn else "Date",
    yaxis_title="模拟价格" if is_cn else "Simulated Price"
)

st.plotly_chart(fig_mc, use_container_width=True)

if is_cn:
    st.markdown(f"""
### AI 模拟解读

当前选择资产为 **{mc_asset}**，当前价格为 **{latest_price:.2f}**。  
基于过去两年的历史收益率，我们使用蒙特卡洛方法模拟未来 **{mc_days}** 个交易日、共 **{mc_paths}** 条可能路径。

模拟结果显示：

- 上涨概率约为 **{prob_up:.2%}**
- 下跌概率约为 **{prob_down:.2%}**
- 在 {int(confidence * 100)}% 置信水平下，模拟最大可能亏损约为 **{mc_var:.2%}**
- 极端情景下的平均亏损约为 **{mc_cvar:.2%}**

该结果可用于辅助判断商品价格、汇率、美股和市场恐慌指数的未来风险方向。
""")
else:
    st.markdown(f"""
### AI Simulation Interpretation

The selected asset is **{mc_asset}**, with a current price of **{latest_price:.2f}**.  
Based on the past two years of historical returns, the Monte Carlo engine simulates **{mc_paths}** possible paths over the next **{mc_days}** trading days.

Simulation results:

- Probability of gain: **{prob_up:.2%}**
- Probability of loss: **{prob_down:.2%}**
- Monte Carlo VaR at {int(confidence * 100)}% confidence: **{mc_var:.2%}**
- Monte Carlo CVaR under extreme scenarios: **{mc_cvar:.2%}**

This can support forward-looking risk assessment for commodities, FX, US equities and market fear indicators.
""")

with st.expander(txt["method"]):
    if is_cn:
        st.markdown("""
**数据支撑**  
本系统使用历史价格、历史收益率、波动率、VaR、CVaR、最大回撤、相关性和蒙特卡洛模拟作为风险分析基础。

**市场波动风险**  
衡量资产价格波动的剧烈程度。数值越高，市场越不稳定。

**未来一天最大可能亏损**  
基于历史收益率估算在指定置信水平下未来一天可能出现的最大亏损。

**极端情况下平均亏损**  
如果市场进入最差的一小部分情景，平均可能亏损是多少。

**资产联动关系图**  
用于观察标普500、VIX、黄金、铜、原油和美元人民币之间的联动关系。

**蒙特卡洛模拟**  
利用历史收益率的均值和波动率，随机生成大量未来价格路径，计算上涨概率、下跌概率和极端亏损概率。
""")
    else:
        st.markdown("""
**Data Support**  
This dashboard uses historical prices, returns, volatility, VaR, CVaR, drawdown, correlation and Monte Carlo simulation.

**Market Volatility**  
Measures the intensity of price fluctuations.

**One-Day Potential Loss**  
Estimates potential downside loss under the selected confidence level.

**Expected Shortfall**  
Measures average loss under extreme downside scenarios.

**Asset Correlation**  
Shows the co-movement among SPY, VIX, gold, copper, crude oil and USD/CNY.

**Monte Carlo Simulation**  
Uses historical return mean and volatility to simulate future price paths and estimate gain/loss probabilities.
""")

if is_cn:
    st.markdown("""
---
**免责声明**  
本系统仅用于市场分析、风险监控和投资研究展示，不构成任何投资建议。  
投资有风险，决策需谨慎。
""")
else:
    st.markdown("""
---
**Disclaimer**  
This dashboard is for market analysis, risk monitoring and investment research demonstration only.  
It does not constitute investment advice. Investing involves risk.
""")
