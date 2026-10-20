import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Use Matplotlib's bundled cross-platform font for chart labels.
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# 1. 頁面配置
st.set_page_config(page_title="Candle Lab - K-Line Trainer", layout="wide", initial_sidebar_state="collapsed")

# 現代暗黑風格 CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #c9d1d9; font-size: 20px; }
    .stApp p,
    .stApp label,
    .stApp li,
    .stApp input,
    .stApp button,
    .stApp [data-testid="stMarkdownContainer"] {
        font-size: 20px !important;
        line-height: 1.55;
    }
    .stApp h1 { font-size: 2.4rem; }
    .stApp h2 { font-size: 2rem; }
    .stApp h3 { font-size: 1.65rem; }
    .stApp [data-testid="stMetricValue"] { font-size: 30px; }
    .stApp [data-testid="stMetricLabel"] { font-size: 18px; }
    .stApp [data-testid="stButton"] button { min-height: 52px; }
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 16px 20px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# 2. 型態知識庫 (集中管理，未來擴充只要在這裡新增字典即可)
PATTERN_DB = {
    "鎚子線 (Hammer / Pin Bar)": {
        "desc": "下影線至少為實體的 2 倍，上影線不超過實體長度。",
        "psychology": "盤中空頭強力打壓，但隨後遭遇強大買盤全面推升收復失土，代表底部強勢拒絕破位。"
    },
    "流星線 (Shooting Star)": {
        "desc": "上影線至少為實體的 2 倍，下影線不超過實體長度。",
        "psychology": "盤中多頭曾試圖拉高，但隨後遭遇沉重賣壓打回原形，代表上方阻力極大。"
    },
    "看漲吞噬 (Bullish Engulfing)": {
        "desc": "雙 K 組合，今日的陽棒實體完全包覆前一日的陰棒實體。",
        "psychology": "多頭動能全面爆發，徹底扭轉前一日的空頭優勢，屬於經典的反轉架構。"
    },
    "十字星 (Doji)": {
        "desc": "開盤價與收盤價的差距不超過全日高低價區間的 10%。",
        "psychology": "多空雙方力量極度拉鋸、勢均力敵，市場處於動能暫停或猶豫期。"
    }
}

# 3. 狀態初始化
if "total_count" not in st.session_state:
    st.session_state.total_count = 0
    st.session_state.correct_count = 0
    st.session_state.streak = 0
    st.session_state.current_quiz = None
    st.session_state.submitted = False

# 數據快取取得 (SPY & QQQ 日 K)
@st.cache_resource(show_spinner=False)
def load_data():
    tickers = ["SPY", "QQQ"]
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # 計算 EMA 20 與 EMA 50
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df = df.dropna()
        data[ticker] = df
    return data

# 根據真實 OHLC 價格分類最後一根 K 棒；Doji 優先處理與影線型態重疊的情況。
def detect_pattern(df, idx):
    current = df.iloc[idx]
    open_p = float(current["Open"])
    high_p = float(current["High"])
    low_p = float(current["Low"])
    close_p = float(current["Close"])

    values = (open_p, high_p, low_p, close_p)
    if not all(np.isfinite(value) for value in values) or high_p <= low_p:
        return None
    if high_p < max(open_p, close_p) or low_p > min(open_p, close_p):
        return None

    body = abs(close_p - open_p)
    candle_range = high_p - low_p
    upper_wick = high_p - max(open_p, close_p)
    lower_wick = min(open_p, close_p) - low_p

    if body <= candle_range * 0.1:
        return "十字星 (Doji)"

    if idx > 0:
        previous = df.iloc[idx - 1]
        prev_open = float(previous["Open"])
        prev_close = float(previous["Close"])
        if (
            np.isfinite(prev_open)
            and np.isfinite(prev_close)
            and prev_close < prev_open
            and close_p > open_p
            and open_p <= prev_close
            and close_p >= prev_open
        ):
            return "看漲吞噬 (Bullish Engulfing)"

    if lower_wick >= body * 2 and upper_wick <= body:
        return "鎚子線 (Hammer / Pin Bar)"
    if upper_wick >= body * 2 and lower_wick <= body:
        return "流星線 (Shooting Star)"
    return None


# 隨機從歷史中已偵測到的型態產生題目
def generate_quiz():
    data_dict = load_data()
    matches = {pattern: [] for pattern in PATTERN_DB}
    for ticker, df in data_dict.items():
        for idx in range(24, len(df)):
            pattern = detect_pattern(df, idx)
            if pattern is not None:
                matches[pattern].append((ticker, idx))

    available_patterns = [pattern for pattern, hits in matches.items() if hits]
    if not available_patterns:
        raise RuntimeError("目前下載的歷史資料中找不到符合條件的 K 線型態。")

    correct_pattern = random.choice(available_patterns)
    ticker, target_idx = random.choice(matches[correct_pattern])
    df = data_dict[ticker]
    sub_df = df.iloc[target_idx - 24 : target_idx + 1].copy().reset_index()

    return {
        "ticker": ticker,
        "df": sub_df,
        "correct": correct_pattern,
        "date": str(sub_df.iloc[-1]["Date"])[:10],
    }

if st.session_state.current_quiz is None:
    st.session_state.current_quiz = generate_quiz()

quiz = st.session_state.current_quiz

# 4. 頂部儀表板
st.markdown("## ⚡ CANDLE LAB | K 線型態訓練器")
acc = 0.0 if st.session_state.total_count == 0 else (st.session_state.correct_count / st.session_state.total_count) * 100
col1, col2, col3 = st.columns(3)
col1.metric("累積題數", f"{st.session_state.total_count} 題")
col2.metric("辨識準確率", f"{acc:.1f} %")
col3.metric("連續答對 (Streak)", f"🔥 {st.session_state.streak} 題")

st.markdown("---")

# 5. 左右雙欄配置
col_chart, col_quiz = st.columns([65, 35])

# --- 左側：繪製專業暗黑 K 線圖 ---
with col_chart:
    st.subheader(f"📊 標的走勢：{quiz['ticker']} (日 K 局部)")
    
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0e1117')
    ax.set_facecolor('#161b22')
    
    df_plot = quiz['df']
    x = np.arange(len(df_plot))
    
    # 繪製 EMA 均線
    ax.plot(x, df_plot['EMA20'], color='#2962FF', linewidth=1.8, label="EMA 20 (Dynamic Support)")
    ax.plot(x, df_plot['EMA50'], color='#FF6D00', linewidth=1.8, label="EMA 50 (Trend Line)")
    
    # 繪製 K 棒
    for i in range(len(df_plot)):
        open_p = df_plot.loc[i, 'Open']
        close_p = df_plot.loc[i, 'Close']
        high_p = df_plot.loc[i, 'High']
        low_p = df_plot.loc[i, 'Low']
        
        color = '#26a69a' if close_p >= open_p else '#ef5350'
        
        # 影線
        ax.plot([i, i], [low_p, high_p], color=color, linewidth=1.2)
        # 實體
        lower = min(open_p, close_p)
        height = max(abs(close_p - open_p), 0.05)
        ax.bar(i, height, bottom=lower, color=color, width=0.6)
        
    # 考題棒箭頭標記 (最後一根)
    target_idx = len(df_plot) - 1
    target_low = df_plot.loc[target_idx, 'Low']
    ax.annotate('Signal Bar ⬆', 
                xy=(target_idx, target_low), 
                xytext=(target_idx, target_low - (df_plot['High'].max() - df_plot['Low'].min()) * 0.1),
                arrowprops=dict(facecolor='#00E676', shrink=0.08, width=1.5, headwidth=6),
                color='#00E676', ha='center', fontsize=14, fontweight='bold')
    
    ax.grid(True, color='#21262d', linestyle='--', alpha=0.5)
    ax.tick_params(colors='#8b949e', labelsize=12)
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    plt.tight_layout()
    try:
        st.pyplot(fig)
    finally:
        plt.close(fig)

# --- 右側：作答與即時反饋 ---
with col_quiz:
    st.subheader("❓ 型態判斷題")
    st.markdown(f"**題目**：請觀察箭頭所標註的最後一根 K 棒（日期：{quiz['date']}），屬於何種型態？")
    
    options = list(PATTERN_DB.keys())
    choice = st.radio("請選擇答案：", options, index=None, disabled=st.session_state.submitted)
    
    if not st.session_state.submitted:
        if st.button("確認提交答案 (Submit)", type="primary", use_container_width=True):
            if choice is None:
                st.warning("請先選取一個選項！")
            else:
                st.session_state.submitted = True
                st.session_state.total_count += 1
                if choice == quiz['correct']:
                    st.session_state.correct_count += 1
                    st.session_state.streak += 1
                    st.success("✅ **回答正確！**")
                else:
                    st.session_state.streak = 0
                    st.error(f"❌ **回答錯誤！** 正確答案是：**{quiz['correct']}**")
                
                # 印出詳細結構特徵反饋
                pattern_info = PATTERN_DB[quiz['correct']]
                st.markdown(f"**💡 結構解析：**\n* {pattern_info['desc']}")
                st.markdown(f"**🔥 多空力道心理：**\n* {pattern_info['psychology']}")
                st.rerun()
    else:
        # 已送出後顯示解析
        if choice == quiz['correct']:
            st.success("✅ **回答正確！**")
        else:
            st.error(f"❌ **回答錯誤！** 正確答案是：**{quiz['correct']}**")
            
        pattern_info = PATTERN_DB[quiz['correct']]
        st.markdown(f"**💡 結構解析：**\n* {pattern_info['desc']}")
        st.markdown(f"**🔥 多空力道心理：**\n* {pattern_info['psychology']}")
        
        if st.button("下一題 ➔ (Next)", use_container_width=True):
            st.session_state.current_quiz = generate_quiz()
            st.session_state.submitted = False
            st.rerun()
