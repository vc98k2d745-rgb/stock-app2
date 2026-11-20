import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

CONFIG_FILE = "holdings.json"

st.set_page_config(page_title="个人股票助手", layout="wide")

def load_holdings():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_holdings(h):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(h, f, ensure_ascii=False, indent=2)

@st.cache_data(ttl=60)
def get_daily(symbol):
    return ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")

@st.cache_data(ttl=60)
def get_weekly(symbol):
    return ak.stock_zh_a_hist(symbol=symbol, period="weekly", adjust="qfq")

@st.cache_data(ttl=60)
def get_60min(symbol):
    return ak.stock_zh_a_hist_min_em(symbol=symbol, period="60", adjust="qfq")

@st.cache_data(ttl=30)
def get_spot():
    return ak.stock_zh_a_spot_em()

def get_price(symbol):
    spot = get_spot()
    row = spot[spot['代码'] == symbol]
    if row.empty:
        return None
    return float(row['最新价'].iloc[0])

def calc_indicators(df):
    df = df.copy()
    rename_map = {"收盘": "close", "开盘": "open", "最高": "high", "最低": "low", "成交量": "volume"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "close" not in df.columns:
        for c in df.columns:
            if "收盘" in c:
                df["close"] = df[c]
                break
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    exp12 = df["close"].ewm(span=12, adjust=False).mean()
    exp26 = df["close"].ewm(span=26, adjust=False).mean()
    df["dif"] = exp12 - exp26
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd"] = 2 * (df["dif"] - df["dea"])
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def safe_last(df, col):
    if df is None or df.empty or col not in df.columns:
        return np.nan
    v = df[col].iloc[-1]
    return float(v) if pd.notna(v) else np.nan

def analyze_holding(symbol, cost, shares, price):
    try:
        d = calc_indicators(get_daily(symbol))
        w = calc_indicators(get_weekly(symbol))
        m = calc_indicators(get_60min(symbol))
    except Exception as e:
        return {"error": f"数据获取失败: {e}"}

    ld, lw, lm = d.iloc[-1], w.iloc[-1], m.iloc[-1]
    score = 0

    if ld["ma5"] > ld["ma10"] > ld["ma20"]:
        score += 2
    elif ld["ma5"] < ld["ma10"] < ld["ma20"]:
        score -= 2
    if ld["macd"] > 0 and ld["dif"] > ld["dea"]:
        score += 1
    if ld["macd"] < 0:
        score -= 1
    if ld["rsi"] > 70:
        score -= 1
    elif ld["rsi"] < 30:
        score += 1

    if pd.notna(lw["ma5"]) and pd.notna(lw["ma20"]) and lw["ma5"] < lw["ma20"]:
        score -= 1

    if pd.notna(lm["ma10"]) and price < lm["ma10"] and lm["macd"] < 0:
        reduce_price = round(price, 2)
        reduce_note = "60分钟转弱，现价可减"
    else:
        reduce_price = round(price * 1.02, 2)
        reduce_note = "等反弹约2%减仓"

    stop_price = round(max(cost * 0.92, price * 0.95), 2) if cost and cost > 0 else round(price * 0.95, 2)

    pnl_pct = ((price - cost) / cost * 100) if cost and cost > 0 else None
    pnl_amt = (price - cost) * shares if cost and shares else None

    if score <= -2:
        action = "⚠️ 趋势偏空，建议减仓"
    elif score < 0:
        action = "🟡 走弱迹象，准备减仓"
    else:
        action = "🟢 持有观察"

    return {
        "现价": price,
        "盈亏%": f"{pnl_pct:.2f}%" if pnl_pct is not None else "未填成本",
        "盈亏金额": f"{pnl_amt:.2f}" if pnl_amt is not None else "-",
        "综合评分": score,
        "建议": action,
        "减仓价": reduce_price,
        "减仓说明": reduce_note,
        "止损价": stop_price,
        "日线": f"MA5={safe_last(d,'ma5'):.2f} MA10={safe_last(d,'ma10'):.2f} MA20={safe_last(d,'ma20'):.2f} MACD={safe_last(d,'macd'):.2f} RSI={safe_last(d,'rsi'):.1f}",
        "周线": f"MA5={safe_last(w,'ma5'):.2f} MA20={safe_last(w,'ma20'):.2f} RSI={safe_last(w,'rsi'):.1f}",
        "60分": f"MA10={safe_last(m,'ma10'):.2f} MACD={safe_last(m,'macd'):.2f} RSI={safe_last(m,'rsi'):.1f}",
    }

def analyze_buy(symbol, price):
    try:
        d = calc_indicators(get_daily(symbol))
        w = calc_indicators(get_weekly(symbol))
        m = calc_indicators(get_60min(symbol))
    except Exception as e:
        return {"error": f"数据获取失败: {e}"}

    ld, lw, lm = d.iloc[-1], w.iloc[-1], m.iloc[-1]
    score = 0

    if ld["ma5"] > ld["ma10"] > ld["ma20"]:
        score += 3
    if ld["macd"] > 0:
        score += 2
    if ld["rsi"] < 50:
        score += 1
    if ld["rsi"] > 70:
        score -= 2
    if pd.notna(lw["ma5"]) and pd.notna(lw["ma20"]) and lw["ma5"] > lw["ma20"]:
        score += 2

    candidates = [x for x in [safe_last(m, "ma10"), safe_last(m, "ma20")] if pd.notna(x)]
    candidates.append(price * 0.98)
    buy_price = round(min(candidates), 2)

    prob = int(max(0, min(100, 50 + score * 8)))

    return {
        "现价": price,
        "买入概率": f"{prob}%",
        "综合评分": score,
        "建议买入价": buy_price,
        "日线": f"MA5={safe_last(d,'ma5'):.2f} MA10={safe_last(d,'ma10'):.2f} MA20={safe_last(d,'ma20'):.2f} MACD={safe_last(d,'macd'):.2f} RSI={safe_last(d,'rsi'):.1f}",
        "周线": f"MA5={safe_last(w,'ma5'):.2f} MA20={safe_last(w,'ma20'):.2f} RSI={safe_last(w,'rsi'):.1f}",
        "60分": f"MA10={safe_last(m,'ma10'):.2f} MACD={safe_last(m,'macd'):.2f} RSI={safe_last(m,'rsi'):.1f}",
    }

st.title("📱 个人股票助手")
st.caption(f"数据刷新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ｜ 每次刷新页面自动更新")

holdings = load_holdings()
mode = st.radio("模式", ["持仓分析", "买入分析"], horizontal=True)

if mode == "持仓分析":
    st.subheader("我的持仓")
    with st.form("holding_form"):
        cols = st.columns(3)
        sym = cols[0].text_input("代码", value="")
        cost = cols[1].number_input("成本", value=0.0, step=0.01)
        shares = cols[2].number_input("股数", value=0, step=100)
        saved = st.form_submit_button("保存到持仓")
        if saved and sym:
            holdings[sym.strip()] = {"cost": cost, "shares": shares}
            save_holdings(holdings)
            st.success(f"{sym} 已保存")

    if holdings:
        st.write("已保存持仓：")
        for s, info in list(holdings.items()):
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            c1.write(f"**{s}**")
            c2.write(f"成本 {info['cost']} / {info['shares']}股")
            if c3.button("分析", key=f"an_{s}"):
                st.session_state["active"] = s
            if c4.button("删除", key=f"del_{s}"):
                del holdings[s]
                save_holdings(holdings)
                st.rerun()

        active = st.session_state.get("active")
        if active and active in holdings:
            price = get_price(active)
            if price is None:
                st.error("未获取到现价，请检查代码")
            else:
                res = analyze_holding(active, holdings[active]["cost"], holdings[active]["shares"], price)
                st.markdown(f"### {active} 分析结果")
                if "error" in res:
                    st.error(res["error"])
                else:
                    a, b, c, d = st.columns(4)
                    a.metric("现价", res["现价"])
                    b.metric("盈亏%", res["盈亏%"])
                    c.metric("盈亏金额", res["盈亏金额"])
                    d.metric("评分", res["综合评分"])
                    st.info(res["建议"])
                    e, f = st.columns(2)
                    e.metric("减仓价", res["减仓价"], res["减仓说明"])
                    f.metric("止损价", res["止损价"])
                    st.write("**日线**", res["日线"])
                    st.write("**周线**", res["周线"])
                    st.write("**60分钟**", res["60分"])
    else:
        st.info("还没有保存持仓，先在上方添加。")

else:
    st.subheader("买入分析")
    symbols = st.text_input("代码（多个用逗号分隔）", "600519,000001")
    if st.button("分析"):
        for sym in [s.strip() for s in symbols.split(",") if s.strip()]:
            price = get_price(sym)
            if price is None:
                st.warning(f"{sym} 未获取到现价")
                continue
            res = analyze_buy(sym, price)
            st.markdown(f"### {sym}")
            if "error" in res:
                st.error(res["error"])
                continue
            a, b, c, d = st.columns(4)
            a.metric("现价", res["现价"])
            b.metric("买入概率", res["买入概率"])
            c.metric("建议买入价", res["建议买入价"])
            d.metric("评分", res["综合评分"])
            st.write("**日线**", res["日线"])
            st.write("**周线**", res["周线"])
            st.write("**60分钟**", res["60分"])
            st.divider()
