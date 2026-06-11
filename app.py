import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

try:
    import akshare as ak
except Exception:
    ak = None


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
    "subtitle": "实时监控全球商品价格、市场风险、上证指数与投资机会" if is_cn else "Real-time monitoring of commodities, market risk, SSE Composite, and investment opportunities",
    "last_update": "最后更新时间" if is_cn else "Last Updated",
    "refresh": "自动刷新频率" if is_cn else "Auto Refresh Interval",
    "tab_global": "全球资产监控" if is_cn else "Global Asset Monitor",
    "tab_sse": "上证指数风险模拟" if is_cn else "SSE Composite Risk Simulation",
    "select_assets": "选择监控资产" if is_cn else "Select Assets",
    "risk_window": "风险计算窗口" if is_cn else "Risk Window",
    "confidence": "风险置信水平" if is_cn else "Risk Confidence Level",
    "live_market": "实时市场行情" if is_cn else "Live Market Board",
    "market_perf": "市场表现概览" if is_cn else "Market Performance Overview",
    "risk_signal": "智能风险提示" if is_cn else "AI Risk Signal",
    "risk_summary": "风险指标汇总" if is_cn else "Risk Summary",
    "price_trend": "价格走势分析" if is_cn else "Price Trend Analysis",
    "risk_analysis": "风险分析" if is_cn else "Risk Analysis",
    "correlation": "资产联动关系图" if is_cn else "Asset Correlation Heatmap",
    "method": "指标说明" if is_cn else "Methodology",
    "disclaimer": "免责声明" if is_cn else "Disclaimer",
    "asset": "资产名称" if is_cn else "Asset",
    "price": "当前价格" if is_cn else "Price",
    "change": "今日涨跌幅" if is_cn else "Daily Change",
    "vol": "市场波动风险" if is_cn else "Market Volatility",
    "var": "未来一天最大可能亏损" if is_cn else "One-Day Potential Loss",
    "cvar": "极端情况下平均亏损" if is_cn else "Expected Shortfall",
    "dd": "距离历史高点跌幅" if is_cn else "Drawdown from Peak",
    "data_fail": "数据加载失败，请检查网络或稍后重试。" if is_cn else "Data loading failed. Please check the network or try again later.",
}

refresh_seconds = st.sidebar.selectbox(
    txt["refresh"],
    [5, 10, 30, 60],
    index=0
)

st_autorefresh(
    interval=refresh_seconds * 1000,
    key="auto_refresh"
)

with top_left:
    st.title(txt["title"])
    st.caption(txt["subtitle"])
    st.caption(f"{txt['last_update']}：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def normalize_yf_data(data):
    if data is None or data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns:
        return None

    data = data.dropna(subset=["Close"])
    return data if not data.empty else None


@st.cache_data(ttl=refresh_seconds)
def download_yfinance(ticker):
    data = yf.download(
        ticker,
        period="1y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False
    )
    return normalize_yf_data(data)


@st.cache_data(ttl=refresh_seconds)
def download_sse_index():
    if ak is not None:
        try:
            df = ak.index_zh_a_hist(
                symbol="000001",
                period="daily",
                start_date="20200101",
                end_date=datetime.now().strftime("%Y%m%d")
            )

            df = df.rename(columns={
                "日期": "Date",
                "开盘": "Open",
                "收盘": "Close",
                "最高": "High",
                "最低": "Low",
                "成交量": "Volume"
            })

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
            df = df.set_index("Date")
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna(subset=["Close"])

            if not df.empty:
                return df

        except Exception:
            pass

    try:
        data = yf.download(
            "000001.SS",
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )
        return normalize_yf_data(data)
    except Exception:
        return None


def calculate_metrics(data, window, confidence):
    data = data.copy()
    data = data.dropna(subset=["Close"])

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

    clean = data.dropna(subset=["Close", "Return", "Rolling Vol", "Historical VaR", "Drawdown"])

    if len(clean) < 2:
        return None

    latest_price = float(clean["Close"].iloc[-1])
    previous_price = float(clean["Close"].iloc[-2])

    return {
        "price": latest_price,
        "daily_change": latest_price / previous_price - 1,
        "vol": float(clean["Rolling Vol"].iloc[-1]),
        "var": float(clean["Historical VaR"].iloc[-1]),
        "cvar": float(clean["CVaR"].iloc[-1]) if not pd.isna(clean["CVaR"].iloc[-1]) else np.nan,
        "drawdown": float(clean["Drawdown"].iloc[-1]),
        "data": data
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

tab_global, tab_sse = st.tabs([txt["tab_global"], txt["tab_sse"]])


with tab_global:
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

    results = {}

    for name in selected_assets:
        ticker = assets[name]
        data = download_yfinance(ticker)

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
            risk_comment = "当前市场压力较高，建议重点关注商品价格波动、基金组合敞口和下行风险。"
        elif risk_score >= 3:
            risk_level = "🟡 风险中等"
            risk_comment = "当前风险处于中等水平，建议持续关注波动率、潜在亏损和资产联动关系。"
        else:
            risk_level = "🟢 风险较低"
            risk_comment = "根据当前指标，市场整体状态相对稳定。"
    else:
        if risk_score >= 5:
            risk_level = "🔴 High Risk"
            risk_comment = "Market stress is elevated. Commodity exposure and downside risk should be closely monitored."
        elif risk_score >= 3:
            risk_level = "🟡 Medium Risk"
            risk_comment = "Risk is moderate. Volatility, potential loss, and correlations should be monitored."
        else:
            risk_level = "🟢 Low Risk"
            risk_comment = "Overall market conditions are relatively stable."

    left, right = st.columns([2, 1])

    with left:
        st.markdown(f"## {txt['market_perf']}")

        perf_df = pd.DataFrame([
            {txt["asset"]: name, txt["change"]: metric["daily_change"] * 100}
            for name, metric in results.items()
        ])

        fig_perf = go.Figure()
        fig_perf.add_trace(go.Bar(
            x=perf_df[txt["asset"]],
            y=perf_df[txt["change"]],
            name=txt["change"]
        ))
        fig_perf.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            title="今日市场涨跌情况" if is_cn else "Daily Market Performance",
            xaxis_title=txt["asset"],
            yaxis_title=f"{txt['change']} (%)"
        )
        st.plotly_chart(fig_perf, width="stretch")

    with right:
        st.markdown(f"## {txt['risk_signal']}")
        st.markdown(f"""
        <div class="risk-card">
        <h2>{risk_level}</h2>
        <p>{risk_comment}</p>
        <hr>
        <p><b>{txt["vol"]}：</b> {avg_vol:.2%}</p>
        <p><b>{txt["var"]}：</b> {avg_var:.2%}</p>
        <p><b>{txt["dd"]}：</b> {avg_dd:.2%}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"## {txt['risk_summary']}")

    summary = []

    for name, metric in results.items():
        summary.append({
            txt["asset"]: name,
            txt["price"]: round(metric["price"], 4),
            txt["change"]: f"{metric['daily_change']:.2%}",
            txt["vol"]: f"{metric['vol']:.2%}",
            f"{txt['var']}（{int(confidence * 100)}%）": f"{metric['var']:.2%}",
            txt["dd"]: f"{metric['drawdown']:.2%}"
        })

    st.dataframe(pd.DataFrame(summary), width="stretch")

    st.markdown(f"## {txt['price_trend']}")

    main_asset = st.selectbox(
        "选择查看资产" if is_cn else "Select Asset",
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
        xaxis_title="日期" if is_cn else "Date",
        yaxis_title=txt["price"]
    )
    st.plotly_chart(fig_price, width="stretch")

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
        xaxis_title="日期" if is_cn else "Date",
        yaxis_title="风险水平" if is_cn else "Risk Level"
    )
    st.plotly_chart(fig_risk, width="stretch")

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
        st.plotly_chart(fig_corr, width="stretch")


with tab_sse:
    st.markdown("## 上证指数风险可视化与蒙特卡洛模拟" if is_cn else "SSE Composite Risk Visualization & Monte Carlo Simulation")

    col_a, col_b, col_c = st.columns(3)

    sse_window = col_a.slider(
        "上证风险窗口" if is_cn else "SSE Risk Window",
        10, 120, 20
    )

    mc_days = col_b.slider(
        "模拟未来交易日" if is_cn else "Future Trading Days",
        5, 120, 30
    )

    mc_paths = col_c.slider(
        "模拟路径数量" if is_cn else "Number of Simulation Paths",
        500, 10000, 3000,
        step=500
    )

    sse_data = download_sse_index()

    if sse_data is None or len(sse_data) < sse_window + 30:
        st.error("上证指数数据加载失败，请检查 akshare / yfinance 数据源。" if is_cn else "Failed to load SSE Composite data.")
        st.stop()

    sse_metric = calculate_metrics(sse_data, sse_window, confidence)

    if sse_metric is None:
        st.error("上证指数风险指标计算失败。" if is_cn else "Failed to calculate SSE risk metrics.")
        st.stop()

    sse_df = sse_metric["data"].dropna(subset=["Return"])

    latest_price = sse_metric["price"]
    daily_change = sse_metric["daily_change"]
    annual_vol = sse_metric["vol"]
    historical_var = sse_metric["var"]
    historical_cvar = sse_metric["cvar"]
    drawdown = sse_metric["drawdown"]

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("上证指数" if is_cn else "SSE Composite", f"{latest_price:.2f}", f"{daily_change:.2%}")
    c2.metric("市场波动风险" if is_cn else "Market Volatility", f"{annual_vol:.2%}")
    c3.metric("未来一天最大可能亏损" if is_cn else "One-Day Potential Loss", f"{historical_var:.2%}")
    c4.metric("极端情况下平均亏损" if is_cn else "Expected Shortfall", f"{historical_cvar:.2%}")
    c5.metric("距离历史高点跌幅" if is_cn else "Drawdown from Peak", f"{drawdown:.2%}")

    st.markdown("### 上证指数价格走势" if is_cn else "### SSE Composite Price Trend")

    fig_sse_price = go.Figure()
    fig_sse_price.add_trace(go.Scatter(
        x=sse_data.index,
        y=sse_data["Close"],
        name="上证指数" if is_cn else "SSE Composite"
    ))
    fig_sse_price.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        xaxis_title="日期" if is_cn else "Date",
        yaxis_title="指数点位" if is_cn else "Index Level"
    )
    st.plotly_chart(fig_sse_price, width="stretch")

    st.markdown("### 上证指数风险走势" if is_cn else "### SSE Composite Risk Trend")

    fig_sse_risk = go.Figure()
    fig_sse_risk.add_trace(go.Scatter(
        x=sse_df.index,
        y=sse_df["Rolling Vol"],
        name="市场波动风险" if is_cn else "Market Volatility"
    ))
    fig_sse_risk.add_trace(go.Scatter(
        x=sse_df.index,
        y=sse_df["Historical VaR"],
        name="未来一天最大可能亏损" if is_cn else "One-Day Potential Loss"
    ))
    fig_sse_risk.add_trace(go.Scatter(
        x=sse_df.index,
        y=sse_df["Drawdown"],
        name="距离历史高点跌幅" if is_cn else "Drawdown"
    ))
    fig_sse_risk.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        xaxis_title="日期" if is_cn else "Date",
        yaxis_title="风险水平" if is_cn else "Risk Level"
    )
    st.plotly_chart(fig_sse_risk, width="stretch")

    st.markdown("### 蒙特卡洛未来路径模拟" if is_cn else "### Monte Carlo Future Path Simulation")

    recent_returns = sse_df["Return"].dropna()

    mu = recent_returns.mean()
    sigma = recent_returns.std()

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

    mc_var_95 = -np.quantile(terminal_returns, 0.05)
    mc_cvar_95 = -terminal_returns[terminal_returns <= np.quantile(terminal_returns, 0.05)].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("上涨概率" if is_cn else "Probability of Gain", f"{prob_up:.2%}")
    m2.metric("下跌概率" if is_cn else "Probability of Loss", f"{prob_down:.2%}")
    m3.metric("模拟最大可能亏损" if is_cn else "Monte Carlo VaR", f"{mc_var_95:.2%}")
    m4.metric("极端情景平均亏损" if is_cn else "Monte Carlo CVaR", f"{mc_cvar_95:.2%}")

    path_index = pd.date_range(
        start=sse_data.index[-1],
        periods=mc_days + 1,
        freq="B"
    )

    fig_mc = go.Figure()

    sample_paths = min(80, mc_paths)

    for i in range(sample_paths):
        fig_mc.add_trace(go.Scatter(
            x=path_index,
            y=price_paths[:, i],
            mode="lines",
            line=dict(width=1),
            opacity=0.25,
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
        title="上证指数未来路径模拟" if is_cn else "SSE Composite Future Path Simulation",
        xaxis_title="日期" if is_cn else "Date",
        yaxis_title="指数点位" if is_cn else "Index Level"
    )
    st.plotly_chart(fig_mc, width="stretch")

    if is_cn:
        st.markdown(f"""
### AI 解读

当前上证指数为 **{latest_price:.2f}**，最近一个交易日涨跌幅为 **{daily_change:.2%}**。  
基于历史收益率估算，当前年化波动风险为 **{annual_vol:.2%}**，95%置信水平下未来一天最大可能亏损约为 **{historical_var:.2%}**。  

在未来 **{mc_days}** 个交易日的蒙特卡洛模拟中：
- 上涨概率约为 **{prob_up:.2%}**
- 下跌概率约为 **{prob_down:.2%}**
- 模拟情景下的最大可能亏损约为 **{mc_var_95:.2%}**
- 极端情景平均亏损约为 **{mc_cvar_95:.2%}**

本模块适合用于判断中国权益市场风险偏好、基金仓位压力以及矿业相关资产的市场环境。
""")
    else:
        st.markdown(f"""
### AI Interpretation

The SSE Composite is currently at **{latest_price:.2f}**, with the latest daily change of **{daily_change:.2%}**.  
Based on historical returns, the current annualized volatility is **{annual_vol:.2%}**, and the estimated one-day potential loss at the 95% confidence level is **{historical_var:.2%}**.

Over the next **{mc_days}** trading days under Monte Carlo simulation:
- Probability of gain: **{prob_up:.2%}**
- Probability of loss: **{prob_down:.2%}**
- Monte Carlo VaR: **{mc_var_95:.2%}**
- Monte Carlo CVaR: **{mc_cvar_95:.2%}**

This module can support China equity market risk monitoring, fund allocation decisions, and macro risk assessment for mining-related investment exposure.
""")


with st.expander(txt["method"]):
    if is_cn:
        st.markdown("""
**市场波动风险**  
衡量价格波动的剧烈程度。数值越高，说明市场越不稳定。

**未来一天最大可能亏损**  
根据历史价格变化，估算在当前置信水平下未来一天可能出现的最大损失。

**极端情况下平均亏损**  
如果市场进入最差的 5% 情景，平均可能亏损是多少。

**蒙特卡洛模拟**  
利用历史收益率的均值和波动率，随机生成大量未来价格路径，用于估计未来上涨、下跌和极端亏损概率。

**说明**  
当前版本使用 Yahoo Finance 和 AkShare 数据源作为展示原型。若用于机构级实时监控，后续可以接入 Wind、Bloomberg、Refinitiv、交易所行情或券商 API。
""")
    else:
        st.markdown("""
**Market Volatility**  
Measures the intensity of price fluctuations.

**One-Day Potential Loss**  
Estimates potential one-day downside loss under the selected confidence level.

**Expected Shortfall**  
Measures the average loss under the worst 5% scenarios.

**Monte Carlo Simulation**  
Uses historical return mean and volatility to simulate many possible future price paths and estimate gain/loss probabilities.

**Note**  
This prototype uses Yahoo Finance and AkShare data sources. For institutional-grade real-time monitoring, it can be upgraded with Wind, Bloomberg, Refinitiv, exchange feeds, or broker APIs.
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
This dashboard is for market analysis, risk monitoring, and investment research demonstration only.  
It does not constitute investment advice. Investing involves risk.
""")
