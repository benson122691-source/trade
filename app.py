"""Cross-asset investment simulator for a student project prototype."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from model import forecast_return


st.set_page_config(page_title="跨資產投資模擬器", page_icon="📈", layout="wide")

ASSETS = {
    "台股 ETF｜元大台灣50 (0050)": {"ticker": "0050.TW", "currency": "TWD", "color": "#00a884"},
    "美股 ETF｜S&P 500 (SPY)": {"ticker": "SPY", "currency": "USD", "color": "#4c78a8"},
    "比特幣 (BTC)": {"ticker": "BTC-USD", "currency": "USD", "color": "#f2a900"},
    "黃金 ETF｜SPDR Gold Shares (GLD)": {"ticker": "GLD", "currency": "USD", "color": "#d9a400"},
}
@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data(start: date, end: date) -> tuple[pd.DataFrame, pd.Series]:
    """Download adjusted closes and USD/TWD from Yahoo Finance via yfinance."""
    symbols = [item["ticker"] for item in ASSETS.values()] + ["TWD=X"]
    raw = yf.download(
        symbols,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise ValueError("資料來源沒有回傳資料。請稍後重試或縮短日期範圍。")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})
    close = close.sort_index().ffill()
    if "TWD=X" not in close.columns:
        raise ValueError("匯率資料缺漏，暫時無法將海外資產換算為新台幣。")
    fx = close["TWD=X"].rename("USD/TWD").dropna()
    prices = pd.DataFrame(index=close.index)
    for item in ASSETS.values():
        ticker = item["ticker"]
        if ticker not in close.columns:
            continue
        series = close[ticker]
        prices[ticker] = series if item["currency"] == "TWD" else series * close["TWD=X"]
    prices = prices.dropna(how="all").ffill().dropna()
    return prices, fx


@st.cache_data(ttl=1800, show_spinner=False)
def get_forecast(price: pd.Series, horizon: int) -> dict[str, float | int | str]:
    """Cache model comparisons for the same price series and forecast horizon."""
    return forecast_return(price, horizon).to_dict()


def backtest(price: pd.Series, principal: float, monthly: bool) -> pd.DataFrame:
    """Historical buy-and-hold or monthly contribution simulation, in TWD."""
    normalized = price / price.iloc[0]
    if not monthly:
        return pd.DataFrame({"資產價值": principal * normalized}, index=price.index)
    # Buy an equal TWD contribution on the first observed trading day of each month.
    month_groups = pd.Series(np.arange(len(price)), index=price.index).groupby(price.index.to_period("M"))
    contribution_dates = [price.index[int(group.iloc[0])] for _, group in month_groups]
    contribution = principal / max(len(contribution_dates), 1)
    units = 0.0
    values = []
    next_date_i = 0
    for i, (dt, value) in enumerate(price.items()):
        if next_date_i < len(contribution_dates) and dt == contribution_dates[next_date_i]:
            units += contribution / value
            next_date_i += 1
        values.append(units * value)
    return pd.DataFrame({"資產價值": values}, index=price.index)


st.title("📈 跨資產投資模擬器")
st.caption("比較台股 ETF、美股 ETF、比特幣與黃金 ETF。歷史回測和模型估算分開呈現，預測不代表保證收益。")

with st.sidebar:
    st.header("模擬設定")
    principal = st.number_input("總投入金額（新台幣）", min_value=1000, max_value=100_000_000, value=100_000, step=10_000)
    years = st.slider("歷史資料期間（年）", min_value=2, max_value=10, value=5)
    horizon = st.select_slider("預估期間（交易日）", options=[5, 10, 20, 60, 120, 252], value=20, format_func=lambda x: f"{x} 日（約 {max(1, round(x/21))} 個月）")
    method = st.radio("投入方式", ["一次投入", "定期定額"], horizontal=False)
    selected = st.multiselect("比較標的", list(ASSETS), default=list(ASSETS))
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years + 15)
    st.caption("資料來源：Yahoo Finance（透過 yfinance）。海外資產依每日 USD/TWD 換算；以調整後價格近似含息報酬。")

if not selected:
    st.info("請在左側選取至少一個標的。")
    st.stop()

try:
    with st.spinner("正在下載市場資料並計算模擬結果…"):
        prices, fx = get_market_data(start_date, end_date)
except Exception as exc:
    st.error(f"無法取得市場資料：{exc}")
    st.info("請確認網路連線，或稍後再試。此原型需要 Yahoo Finance 提供歷史價格與匯率資料。")
    st.stop()

labels_by_ticker = {item["ticker"]: label for label, item in ASSETS.items()}
chosen_tickers = [ASSETS[label]["ticker"] for label in selected if ASSETS[label]["ticker"] in prices]
if not chosen_tickers:
    st.error("所選標的目前沒有可用資料。")
    st.stop()
prices = prices[chosen_tickers].dropna(how="all").ffill().dropna()
actual_days = max((prices.index[-1] - prices.index[0]).days, 1)
actual_years = actual_days / 365.25

st.subheader("歷史模擬")
st.caption(f"期間：{prices.index[0].date()} 至 {prices.index[-1].date()}｜資產價格以新台幣表示")
hist_frames = []
metric_cols = st.columns(len(chosen_tickers))
for col, ticker in zip(metric_cols, chosen_tickers):
    series = prices[ticker].dropna()
    sim = backtest(series, float(principal), method == "定期定額")
    hist_frames.append(pd.DataFrame({"日期": sim.index, "資產價值（元）": sim["資產價值"].values, "標的": labels_by_ticker[ticker]}))
    returns = series.pct_change().dropna()
    final_value = float(sim["資產價值"].iloc[-1])
    invested = float(principal)
    net_return = final_value / invested - 1 if invested else 0
    annual_vol = float(returns.std() * np.sqrt(252))
    wealth = sim["資產價值"]
    max_drawdown = float((wealth / wealth.cummax() - 1).min())
    with col:
        st.metric(labels_by_ticker[ticker], f"NT$ {final_value:,.0f}", f"累計投入 NT$ {invested:,.0f}｜報酬 {net_return:+.1%}")
        st.caption(f"年化波動 {annual_vol:.1%}　·　最大回撤 {max_drawdown:.1%}")

history = pd.concat(hist_frames, ignore_index=True)
fig = px.line(history, x="日期", y="資產價值（元）", color="標的", title="歷史投資價值（新台幣）")
fig.update_layout(legend_title_text="標的", hovermode="x unified", yaxis_tickprefix="NT$ ")
st.plotly_chart(fig, use_container_width=True)
if method == "定期定額":
    st.caption("定期定額採每月第一個有交易資料的日期投入；總投入金額隨回測月份數增加。")

st.subheader("模型情境估算")
st.caption(f"Random Forest 與 Ridge 在訓練資料內的時間順序驗證段比較，驗證 MAE 較低者產生估算；保留測試段只用來報告評估指標。MAE、RMSE 以對數報酬誤差計算。情境區間不是統計保證，期末估算假設全部總投入金額目前一次投入。")
forecast_rows = []
forecast_plot_rows = []
for ticker in chosen_tickers:
    series = prices[ticker].dropna()
    try:
        estimate = get_forecast(series, horizon)
        current = float(series.iloc[-1])
        # Estimate a portfolio-sized value under a single asset allocation.
        central = principal * (1 + estimate["prediction"])
        lower = principal * (1 + estimate["lower"])
        upper = principal * (1 + estimate["upper"])
        forecast_rows.append({"標的": labels_by_ticker[ticker], "採用模型": estimate["selected_model"], "模型估計報酬": estimate["prediction"], "情境下界報酬": estimate["lower"], "情境上界報酬": estimate["upper"], "Random Forest MAE": estimate["random_forest_mae"], "Ridge MAE": estimate["ridge_mae"], "零報酬基準 MAE": estimate["zero_baseline_mae"], "平均報酬基準 MAE": estimate["mean_baseline_mae"], "模型 RMSE": estimate["rmse"], "期末估算（中央）": central, "期末估算區間": f"NT$ {lower:,.0f} – {upper:,.0f}"})
        forecast_plot_rows.append({"標的": labels_by_ticker[ticker], "lower": lower, "central": central, "upper": upper})
    except Exception as exc:
        st.warning(f"{labels_by_ticker[ticker]}：{exc}")
if forecast_rows:
    forecast = pd.DataFrame(forecast_rows)
    st.dataframe(
        forecast,
        hide_index=True,
        use_container_width=True,
        column_config={
            "採用模型": st.column_config.TextColumn(),
            "模型估計報酬": st.column_config.NumberColumn(format="百分比"),
            "情境下界報酬": st.column_config.NumberColumn(format="百分比"),
            "情境上界報酬": st.column_config.NumberColumn(format="百分比"),
            "Random Forest MAE": st.column_config.NumberColumn(format="百分比"),
            "Ridge MAE": st.column_config.NumberColumn(format="百分比"),
            "模型 RMSE": st.column_config.NumberColumn(format="百分比"),
            "零報酬基準 MAE": st.column_config.NumberColumn(format="百分比"),
            "平均報酬基準 MAE": st.column_config.NumberColumn(format="百分比"),
            "期末估算（中央）": st.column_config.NumberColumn(format="NT$ %.0f"),
        },
    )
    st.markdown("**預估期末金額區間**")
    fig_forecast = go.Figure()
    for row in forecast_plot_rows:
        fig_forecast.add_trace(go.Scatter(
            x=[row["lower"],row["upper"]],y=[row["標的"],row["標的"]],mode="lines+markers",name=f"{row['標的']} 情境區間",
            line={"color":"#8bbab0","width":5},marker={"size":7,"color":"#8bbab0"},
            customdata=[[row["lower"],row["upper"]],[row["lower"],row["upper"]]],
            hovertemplate=f"{row['標的']}<br>經驗情境：NT$ {row['lower']:,.0f}–{row['upper']:,.0f}<extra></extra>",
        ))
        fig_forecast.add_trace(go.Scatter(
            x=[row["central"]],y=[row["標的"]],mode="markers+text",name=f"{row['標的']} 中央估算",
            text=[f"NT$ {row['central']:,.0f}"],textposition="middle right",marker={"size":12,"color":"#0f766e"},
            hovertemplate=f"{row['標的']}<br>模型中央估算：NT$ {row['central']:,.0f}<extra></extra>",
        ))
    fig_forecast.update_layout(
        height=max(260,70*len(forecast_plot_rows)),showlegend=False,hovermode="closest",
        xaxis_title="模擬期末金額（新台幣）",yaxis_title="",margin={"l":10,"r":30,"t":20,"b":40},
    )
    st.plotly_chart(fig_forecast,use_container_width=True)

with st.expander("計算方式與限制"):
    st.markdown(
        """
- 這是教育用途的專題原型；歷史回測只說明過去資料下的結果，不代表未來績效。
- 一次投入回測將總投入金額在期間起點投入；定期定額則把同一總金額平均分配到回測期間的每月第一個交易日。
- 未來模型情境以總投入金額在目前一次投入計算，不會模擬未來逐月扣款；定期定額的比較以歷史回測為準。
- 海外資產使用 USD/TWD 匯率換算為新台幣，價格使用調整後收盤價；手續費、稅負、滑價與匯兌成本尚未納入。
- Random Forest 和 Ridge 在訓練期內比較後選模，再以保留測試期報告 MAE、RMSE 和簡單基準。金融市場變化可能使模型失效；區間是測試資料的經驗分位數，不是保證區間。
- 資料由 Yahoo Finance 提供，可能延遲、缺漏或修訂；本系統不提供個人化投資建議。
        """
    )