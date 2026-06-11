iimport streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="矿业与投资风险监控平台",
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
    "自动刷新频率",
    [5, 10, 30, 60],
    index=0
)

st_autorefresh(
    interval=refresh_seconds * 1000,
    key="auto_refresh"
)

st.title("矿业与投资风险监控平台")
st.caption("实时监控全球商品价格、市场风险与投资机会")
st.caption(f"最后更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

assets = {
    "标普500ETF": "SPY",
    "纳斯达克100ETF": "QQQ",
    "VIX恐慌指数": "^VIX",
    "黄金期货": "GC=F",
    "铜期货": "HG=F",
    "原油期货": "CL=F",
    "白银期货": "SI=F",
    "美国10年期国债收益率": "^TNX",
    "美元兑人民币": "CNY=X",
    "澳元兑美元": "AUDUSD=X"
}

selected_assets = st.sidebar.multiselect(
    "选择监控资产",
    list(assets.keys()),
    default=[
        "标普500ETF",
        "VIX恐慌指数",
        "黄金期货",
        "铜期货",
        "原油期货",
        "美元兑人民币"
    ]
)

window = st.sidebar.slider(
    "风险计算窗口",
    10, 120, 20
)

confidence = st.sidebar.selectbox(
    "风险置信水平",
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


results = {}

for name in selected_assets:
    ticker = assets[name]
    data = download_data(ticker)

    if data is not None and len(data) > window + 5:
        metric = calculate_metrics(data, window, confidence)
        if metric is not None:
            results[name] = metric

st.markdown("## 实时市场行情")

if not results:
    st.error("数据加载失败，请检查网络或稍后重试。")
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

vix_value = results["VIX恐慌指数"]["price"] if "VIX恐慌指数" in results else None

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

left, right = st.columns([2, 1])

with left:
    st.markdown("## 市场表现概览")

    perf_df = pd.DataFrame([
        {
            "资产名称": name,
            "今日涨跌幅": metric["daily_change"] * 100
        }
        for name, metric in results.items()
    ])

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Bar(
        x=perf_df["资产名称"],
        y=perf_df["今日涨跌幅"],
        name="今日涨跌幅"
    ))
    fig_perf.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        title="今日市场涨跌情况",
        xaxis_title="资产名称",
        yaxis_title="今日涨跌幅（%）"
    )
    st.plotly_chart(fig_perf, width="stretch")

with right:
    st.markdown("## 智能风险提示")
    st.markdown(f"""
    <div class="risk-card">
    <h2>{risk_level}</h2>
    <p>{risk_comment}</p>
    <hr>
    <p><b>平均市场波动风险：</b> {avg_vol:.2%}</p>
    <p><b>平均未来一天最大可能亏损：</b> {avg_var:.2%}</p>
    <p><b>平均距离历史高点跌幅：</b> {avg_dd:.2%}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("## 风险指标汇总")

summary = []

for name, metric in results.items():
    summary.append({
        "资产名称": name,
        "当前价格": round(metric["price"], 4),
        "今日涨跌幅": f"{metric['daily_change']:.2%}",
        "市场波动风险": f"{metric['vol']:.2%}",
        f"未来一天最大可能亏损（{int(confidence * 100)}%概率）": f"{metric['var']:.2%}",
        "距离历史高点跌幅": f"{metric['drawdown']:.2%}"
    })

summary_df = pd.DataFrame(summary)
st.dataframe(summary_df, width="stretch")

st.markdown("## 价格走势分析")

main_asset = st.selectbox(
    "选择查看资产",
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
    title=f"{main_asset} 价格走势",
    xaxis_title="日期",
    yaxis_title="价格"
)
st.plotly_chart(fig_price, width="stretch")

st.markdown("## 风险分析")

fig_risk = go.Figure()
fig_risk.add_trace(go.Scatter(
    x=main_data.index,
    y=main_data["Rolling Vol"],
    name="市场波动风险"
))
fig_risk.add_trace(go.Scatter(
    x=main_data.index,
    y=main_data["Historical VaR"],
    name="未来一天最大可能亏损"
))
fig_risk.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0b0f19",
    plot_bgcolor="#0b0f19",
    title=f"{main_asset} 风险指标走势",
    xaxis_title="日期",
    yaxis_title="风险水平"
)
st.plotly_chart(fig_risk, width="stretch")

st.markdown("## 资产联动关系图")

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
        title="资产走势关联分析"
    )

    st.plotly_chart(fig_corr, width="stretch")
else:
    st.warning("数据不足，暂时无法计算资产联动关系。")

with st.expander("指标说明"):
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

st.markdown("""
---
**免责声明**  
本系统仅用于市场分析、风险监控和投资研究展示，不构成任何投资建议。  
投资有风险，决策需谨慎。
""")