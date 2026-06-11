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

# =========================
# Language Switch
# =========================

top_left, top_right = st.columns([4, 1])

with top_right:
    lang = st.selectbox(
        "Language / 语言",
        ["中文", "English"],
        index=0
    )

is_cn = lang == "中文"

txt = {
    "title": "矿业与投资风险监控平台" if is_cn else "Mining & Investment Risk Dashboard",
    "subtitle": "实时监控全球商品价格、市场风险与投资机会" if is_cn else "Real-time monitoring of commodity prices, market risk, and investment opportunities",
    "last_update": "最后更新时间" if is_cn else "Last Updated",
    "refresh": "自动刷新频率" if is_cn else "Auto Refresh Interval",
    "select_assets": "选择监控资产" if is_cn else "Select Assets",
    "risk_window": "风险计算窗口" if is_cn else "Risk Calculation Window",
    "confidence": "风险置信水平" if is_cn else "Risk Confidence Level",
    "live_market": "实时市场行情" if is_cn else "Live Market Board",
    "market_perf": "市场表现概览" if is_cn else "Market Performance Overview",
    "daily_perf": "今日市场涨跌情况" if is_cn else "Daily Market Performance",
    "asset_name": "资产名称" if is_cn else "Asset",
    "daily_change": "今日涨跌幅" if is_cn else "Daily Change",
    "risk_signal": "智能风险提示" if is_cn else "AI Risk Signal",
    "avg_vol": "平均市场波动风险" if is_cn else "Average Market Volatility",
    "avg_var": "平均未来一天最大可能亏损" if is_cn else "Average One-Day Potential Loss",
    "avg_dd": "平均距离历史高点跌幅" if is_cn else "Average Drawdown from Peak",
    "risk_summary": "风险指标汇总" if is_cn else "Risk Summary",
    "price": "当前价格" if is_cn else "Price",
    "vol": "市场波动风险" if is_cn else "Market Volatility",
    "var": "未来一天最大可能亏损" if is_cn else "One-Day Potential Loss",
    "drawdown": "距离历史高点跌幅" if is_cn else "Drawdown from Historical Peak",
    "price_trend": "价格走势分析" if is_cn else "Price Trend Analysis",
    "select_chart": "选择查看资产" if is_cn else "Select Asset",
    "risk_analysis": "风险分析" if is_cn else "Risk Analysis",
    "risk_level": "风险水平" if is_cn else "Risk Level",
    "correlation": "资产联动关系图" if is_cn else "Asset Correlation Heatmap",
    "corr_title": "资产走势关联分析" if is_cn else "Asset Return Correlation Analysis",
    "method": "指标说明" if is_cn else "Methodology",
    "disclaimer": "免责声明" if is_cn else "Disclaimer",
    "data_fail": "数据加载失败，请检查网络或稍后重试。" if is_cn else "Data loading failed. Please check the network or try again later.",
    "data_not_enough": "数据不足，暂时无法计算资产联动关系。" if is_cn else "Not enough data to calculate asset correlations."
}

asset_map = {
    "SPY": {"cn": "标普500ETF", "en": "S&P 500 ETF"},
    "QQQ": {"cn": "纳斯达克100ETF", "en": "Nasdaq 100 ETF"},
    "^VIX": {"cn": "VIX恐慌指数", "en": "VIX Fear Index"},
    "GC=F": {"cn": "黄金期货", "en": "Gold Futures"},
    "HG=F": {"cn": "铜期货", "en": "Copper Futures"},
    "CL=F": {"cn": "原油期货", "en": "Crude Oil Futures"},
    "SI=F": {"cn": "白银期货", "en": "Silver Futures"},
    "^TNX": {"cn": "美国10年期国债收益率", "en": "US 10Y Treasury Yield"},
    "CNY=X": {"cn": "美元兑人民币", "en": "USD/CNY"},
    "AUDUSD=X": {"cn": "澳元兑美元", "en": "AUD/USD"}
}

assets = {
    v["cn"] if is_cn else v["en"]: k
    for k, v in asset_map.items()
}

default_assets = [
    asset_map["SPY"]["cn"] if is_cn else asset_map["SPY"]["en"],
    asset_map["^VIX"]["cn"] if is_cn else asset_map["^VIX"]["en"],
    asset_map["GC=F"]["cn"] if is_cn else asset_map["GC=F"]["en"],
    asset_map["HG=F"]["cn"] if is_cn else asset_map["HG=F"]["en"],
    asset_map["CL=F"]["cn"] if is_cn else asset_map["CL=F"]["en"],
    asset_map["CNY=X"]["cn"] if is_cn else asset_map["CNY=X"]["en"]
]

# =========================
# Sidebar
# =========================

refresh_seconds = st.sidebar.selectbox(
    txt["refresh"],
    [5, 10, 30, 60],
    index=0
)

st_autorefresh(
    interval=refresh_seconds * 1000,
    key="auto_refresh"
)

selected_assets = st.sidebar.multiselect(
    txt["select_assets"],
    list(assets.keys()),
    default=default_assets
)

window = st.sidebar.slider(
    txt["risk_window"],
    10, 120, 20
)

confidence = st.sidebar.selectbox(
    txt["confidence"],
    [0.95, 0.99],
    index=0
)

# =========================
# Header
# =========================

with top_left:
    st.title(txt["title"])
    st.caption(txt["subtitle"])
    st.caption(f"{txt['last_update']}：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =========================
# Data Functions
# =========================

@st.cache_data(ttl=refresh_seconds)
def download_data(ticker):
    data = yf.download(
        ticker,
        period="1y",
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
    data = data.dropna(subset=["Close"])

    data["Return"] = np.log(data["Close"] / data["Close"].shift(1))
    data["Rolling Vol"] = data["Return"].rolling(window).std() * np.sqrt(252)

    alpha = 1 - confidence
    data["Historical VaR"] = -data["Return"].rolling(window).quantile(alpha)

    data["Peak"] = data["Close"].cummax()
    data["Drawdown"] = data["Close"] / data["Peak"] - 1

    clean = data.dropna(subset=[
        "Close",
        "Return",
        "Rolling Vol",
        "Historical VaR",
        "Drawdown"
    ])

    if len(clean) < 2:
        return None

    latest_price = float(clean["Close"].iloc[-1])
    previous_price = float(clean["Close"].iloc[-2])
    daily_change = latest_price / previous_price - 1

    latest_vol = float(clean["Rolling Vol"].iloc[-1])
    latest_var = float(clean["Historical VaR"].iloc[-1])
    latest_dd = float(clean["Drawdown"].iloc[-1])

    return {
        "price": latest_price,
        "daily_change": daily_change,
        "vol": latest_vol,
        "var": latest_var,
        "drawdown": latest_dd,
        "data": data
    }

# =========================
# Load Data
# =========================

results = {}

for name in selected_assets:
    ticker = assets[name]
    data = download_data(ticker)

    if data is not None and len(data) > window + 5:
        metric = calculate_metrics(data, window, confidence)
        if metric is not None:
            results[name] = metric

st.markdown(f"## {txt['live_market']}")

if not results:
    st.error(txt["data_fail"])
    st.stop()

cols = st.columns(len(results))

for col, (name, metric) in zip(cols, results.items()):
    col.metric(
        label=name,
        value=f"{metric['price']:.2f}",
        delta=f"{metric['daily_change']:.2%}"
    )

# =========================
# Risk Score
# =========================

avg_vol = np.mean([m["vol"] for m in results.values()])
avg_var = np.mean([m["var"] for m in results.values()])
avg_dd = np.mean([abs(m["drawdown"]) for m in results.values()])

vix_name = asset_map["^VIX"]["cn"] if is_cn else asset_map["^VIX"]["en"]
vix_value = results[vix_name]["price"] if vix_name in results else None

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

if is_cn:
    if risk_score >= 5:
        risk_level = "🔴 风险较高"
        risk_comment = """
        当前市场压力较高，建议重点关注商品价格波动、基金组合敞口和下行风险。
        对矿业相关资产而言，需要特别留意黄金、铜、原油以及汇率变化对利润和估值的影响。
        """
    elif risk_score >= 3:
        risk_level = "🟡 风险中等"
        risk_comment = """
        当前风险处于中等水平，市场仍有一定不确定性。
        建议持续跟踪价格波动、未来一天最大可能亏损、距离历史高点跌幅以及资产之间的联动关系。
        """
    else:
        risk_level = "🟢 风险较低"
        risk_comment = """
        根据当前选定指标，市场整体状态相对稳定。
        但仍建议持续观察大宗商品价格、美元汇率和主要股指变化。
        """
else:
    if risk_score >= 5:
        risk_level = "🔴 High Risk"
        risk_comment = """
        Market stress is elevated. Commodity price movements, portfolio exposure, and downside risk should be closely monitored.
        For mining-related assets, gold, copper, crude oil, and FX movements may materially affect profitability and valuation.
        """
    elif risk_score >= 3:
        risk_level = "🟡 Medium Risk"
        risk_comment = """
        Current risk is moderate, and market uncertainty remains.
        Volatility, one-day potential loss, drawdown from peak, and asset correlations should be monitored continuously.
        """
    else:
        risk_level = "🟢 Low Risk"
        risk_comment = """
        Based on the selected indicators, overall market conditions are relatively stable.
        However, commodity prices, FX movements, and major equity indices should still be monitored.
        """

# =========================
# Main Layout
# =========================

left, right = st.columns([2, 1])

with left:
    st.markdown(f"## {txt['market_perf']}")

    perf_df = pd.DataFrame([
        {
            txt["asset_name"]: name,
            txt["daily_change"]: metric["daily_change"] * 100
        }
        for name, metric in results.items()
    ])

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Bar(
        x=perf_df[txt["asset_name"]],
        y=perf_df[txt["daily_change"]],
        name=txt["daily_change"]
    ))
    fig_perf.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        title=txt["daily_perf"],
        xaxis_title=txt["asset_name"],
        yaxis_title=f"{txt['daily_change']} (%)"
    )
    st.plotly_chart(fig_perf, width="stretch")

with right:
    st.markdown(f"## {txt['risk_signal']}")
    st.markdown(f"""
    <div class="risk-card">
    <h2>{risk_level}</h2>
    <p>{risk_comment}</p>
    <hr>
    <p><b>{txt["avg_vol"]}：</b> {avg_vol:.2%}</p>
    <p><b>{txt["avg_var"]}：</b> {avg_var:.2%}</p>
    <p><b>{txt["avg_dd"]}：</b> {avg_dd:.2%}</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Risk Summary
# =========================

st.markdown(f"## {txt['risk_summary']}")

summary = []

for name, metric in results.items():
    summary.append({
        txt["asset_name"]: name,
        txt["price"]: round(metric["price"], 4),
        txt["daily_change"]: f"{metric['daily_change']:.2%}",
        txt["vol"]: f"{metric['vol']:.2%}",
        f"{txt['var']}（{int(confidence * 100)}%）": f"{metric['var']:.2%}",
        txt["drawdown"]: f"{metric['drawdown']:.2%}"
    })

summary_df = pd.DataFrame(summary)
st.dataframe(summary_df, width="stretch")

# =========================
# Price Trend
# =========================

st.markdown(f"## {txt['price_trend']}")

main_asset = st.selectbox(
    txt["select_chart"],
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
    title=f"{main_asset} - {txt['price_trend']}",
    xaxis_title="Date" if not is_cn else "日期",
    yaxis_title=txt["price"]
)
st.plotly_chart(fig_price, width="stretch")

# =========================
# Risk Chart
# =========================

st.markdown(f"## {txt['risk_analysis']}")

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
fig_risk.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0b0f19",
    plot_bgcolor="#0b0f19",
    title=f"{main_asset} - {txt['risk_analysis']}",
    xaxis_title="Date" if not is_cn else "日期",
    yaxis_title=txt["risk_level"]
)
st.plotly_chart(fig_risk, width="stretch")

# =========================
# Correlation
# =========================

st.markdown(f"## {txt['correlation']}")

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
        title=txt["corr_title"]
    )

    st.plotly_chart(fig_corr, width="stretch")
else:
    st.warning(txt["data_not_enough"])

# =========================
# Methodology
# =========================

with st.expander(txt["method"]):
    if is_cn:
        st.markdown("""
**市场波动风险**  
衡量价格波动的剧烈程度。数值越高，说明市场越不稳定。

**未来一天最大可能亏损**  
根据历史价格变化，估算在当前置信水平下未来一天可能出现的最大损失。  
例如 95% 概率下的最大可能亏损为 2%，表示在正常市场情况下，未来一天亏损超过 2% 的概率相对较低。

**距离历史高点跌幅**  
衡量当前价格相比过去一段时间最高点下跌了多少。  
跌幅越大，说明资产距离高点越远。

**资产联动关系图**  
用于观察不同资产是否存在同步上涨、同步下跌或反向波动的关系。  
对矿业和基金投资来说，可以帮助判断黄金、铜、原油、汇率和股市之间的联动风险。
""")
    else:
        st.markdown("""
**Market Volatility**  
Measures how strongly prices fluctuate. A higher value indicates a more unstable market.

**One-Day Potential Loss**  
Estimates the potential one-day downside loss based on historical price movements and the selected confidence level.

**Drawdown from Historical Peak**  
Measures how far the current price has fallen from its historical peak.

**Asset Correlation Heatmap**  
Shows whether different assets tend to move together or in opposite directions.  
For mining and fund investment, this helps evaluate the relationship among gold, copper, crude oil, FX, and equity markets.
""")

# =========================
# Disclaimer
# =========================

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
This dashboard is for market analysis, risk monitoring, and investment research demonstration only.  
It does not constitute investment advice. Investing involves risk.
""")
