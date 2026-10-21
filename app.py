import base64
from io import BytesIO
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ft2font
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import yfinance as yf


APP_DIR = Path(__file__).resolve().parent
FONT_FAMILY = ["Microsoft JhengHei", "SimHei", "sans-serif"]
plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["axes.unicode_minus"] = False

WINDOW_SIZE = 25
MAX_QUIZ_ATTEMPTS = 500
OHLC_COLUMNS = ["Open", "High", "Low", "Close"]


@st.cache_resource(show_spinner=False)
def get_chart_font():
    """Use an installed CJK font, or ASCII labels with bundled DejaVu Sans.

    Generic sans-serif alone does not supply Chinese glyphs on Linux.
    Resolve fonts before rendering rather than suppressing missing-font warnings.
    """
    preferred = FONT_FAMILY[:-1] + [
        "Noto Sans CJK TC", "Noto Sans TC", "Noto Sans CJK SC",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Source Han Sans TC",
    ]
    priority = {name: index for index, name in enumerate(preferred)}
    fonts = sorted(
        font_manager.fontManager.ttflist,
        key=lambda font: priority.get(font.name, len(priority)),
    )
    required = {ord(character) for character in "日考題棒"}
    for font in fonts:
        try:
            glyphs = ft2font.FT2Font(font.fname).get_charmap()
        except (OSError, RuntimeError):
            continue
        if required.issubset(glyphs):
            return font.name, True
    return "DejaVu Sans", False


PATTERN_DB = {
    "鎚子線 (Hammer / Pin Bar)": {
        "desc": "下影線至少為實體 2 倍，上影線不超過實體；實體須大於全日波幅 10%，以排除十字星。",
        "psychology": "空頭盤中打壓後買盤收復失土，下方出現承接；若位於跌勢末端，可能預示反轉，仍需後續確認。",
    },
    "流星線 (Shooting Star)": {
        "desc": "上影線至少為實體 2 倍，下影線不超過實體；實體須大於全日波幅 10%，以排除十字星。",
        "psychology": "多頭拉高後遭賣壓打回，上方供給強勁；若出現在漲勢末端，可能代表上攻動能轉弱。",
    },
    "看漲吞噬 (Bullish Engulfing)": {
        "desc": "前日陰 K、今日陽 K；今日開盤不高於前日收盤，收盤不低於前日開盤，實體完全吞噬前日。",
        "psychology": "買方收復前日賣方控制的整段實體，顯示多方取得主動權，跌勢可能轉折。",
    },
    "十字星 (Doji)": {
        "desc": "開收盤差距不超過當日高低波幅的 10%；零波幅或無效價格不列為型態。",
        "psychology": "多空拉鋸後價格回到開盤附近，顯示猶豫與動能暫停；方向需由趨勢與後續 K 棒確認。",
    },
    "看跌吞噬 (Bearish Engulfing)": {
        "desc": "前日陽 K、今日陰 K；今日開盤不低於前日收盤，收盤不高於前日開盤，實體完全吞噬前日。",
        "psychology": "賣方從高處全面壓回，吞沒前日買盤成果，顯示空方取得主導權。",
    },
    "倒鎚子線 (Inverted Hammer)": {
        "desc": "前三日收盤整體下降，當日實體中點位於前三日高低區間下半部；上影線至少為實體 2 倍，下影線不超過實體 0.5 倍。",
        "psychology": "下跌後買方開始試探上方價位，雖遭壓回仍顯示反攻意圖，需後續陽線確認。",
    },
    "上吊線 (Hanging Man)": {
        "desc": "前三日收盤整體上升，當日實體中點位於前三日高低區間上半部；下影線至少為實體 2 倍，上影線不超過實體 0.5 倍。",
        "psychology": "高檔曾出現明顯拋售，雖收復多數跌幅，仍暴露賣壓；後續跌破可確認轉弱。",
    },
    "大陽線 (Long Bullish)": {
        "desc": "收盤高於開盤，陽線實體佔當日高低波幅至少 80%。",
        "psychology": "買盤幾乎控制整個交易區間，顯示強勁上攻動能；高檔追價仍需考量位置。",
    },
    "大陰線 (Long Bearish)": {
        "desc": "收盤低於開盤，陰線實體佔當日高低波幅至少 80%。",
        "psychology": "賣盤控制大部分交易區間，反彈承接不足，顯示下行動能強勁。",
    },
    "看漲孕線 (Bullish Harami)": {
        "desc": "前日為實體大於波幅 50% 的陰 K，今日陽 K 實體完全包含於前日實體內。",
        "psychology": "強烈下跌後賣壓收斂，買方在前日實體內反攻，可能是跌勢休止或反轉前兆。",
    },
    "看跌孕線 (Bearish Harami)": {
        "desc": "前日為實體大於波幅 50% 的陽 K，今日陰 K 實體完全包含於前日實體內。",
        "psychology": "強烈上漲後買方後繼乏力，賣方開始反擊，可能是漲勢停頓或轉弱訊號。",
    },
    "刺透型態 (Piercing Line)": {
        "desc": "前日陰 K，今日陽 K 開盤低於前日低點或收盤；今日收盤高於前日實體中點。完全覆蓋者優先歸為看漲吞噬。",
        "psychology": "跳低開盤後買方反攻，收復前日實體過半，顯示低檔承接與空方動能衰退。",
    },
    "烏雲罩頂 (Dark Cloud Cover)": {
        "desc": "前日陽 K，今日陰 K 開盤高於前日高點或收盤；今日收盤低於前日實體中點。完全覆蓋者優先歸為看跌吞噬。",
        "psychology": "跳高開盤後賣壓湧現，侵蝕前日漲幅過半，顯示高檔供給開始壓倒買盤。",
    },
    "早晨之星 (Morning Star)": {
        "desc": "前兩日長陰 K（實體 > 波幅 50%）；前日小星線（實體 < 波幅 25%，開盤低於長陰 K 收盤）；今日長陽 K（實體 > 波幅 50%），收盤高於長陰 K 實體中點。",
        "psychology": "強烈賣壓後出現猶豫，接著買盤收復大半失地，呈現空方退潮、多方接手的三日轉折。",
    },
    "黃昏之星 (Evening Star)": {
        "desc": "前兩日長陽 K（實體 > 波幅 50%）；前日小星線（實體 < 波幅 25%，開盤高於長陽 K 收盤）；今日長陰 K（實體 > 波幅 50%），收盤低於長陽 K 實體中點。",
        "psychology": "強烈買盤後上攻停滯，接著賣盤壓回大半漲幅，呈現多方衰退、空方接手的三日轉折。",
    },
}


def _candle(row):
    """Normalize and validate one real OHLC bar without modifying the input."""
    if row is None:
        return None
    try:
        open_p, high_p, low_p, close_p = (
            float(row[column]) for column in OHLC_COLUMNS
        )
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return None
    if not all(np.isfinite(v) for v in (open_p, high_p, low_p, close_p)):
        return None
    if high_p <= low_p or high_p < max(open_p, close_p) or low_p > min(open_p, close_p):
        return None
    return {
        "open": open_p, "high": high_p, "low": low_p, "close": close_p,
        "body": abs(close_p - open_p), "range": high_p - low_p,
        "upper": high_p - max(open_p, close_p),
        "lower": min(open_p, close_p) - low_p,
        "mid": (open_p + close_p) / 2,
    }


def identify_pattern(curr, prev=None, prev2=None, prev3=None):
    """Deterministic OHLC classifier; prev3 supplies three *preceding* days.

    Priority: Doji, three-bar reversals, engulfing, piercing/cloud,
    harami, contextual shadows, original shadows, long bodies.
    Overall three-day trend compares prev.Close with prev3.Close.
    High/low location compares the current body midpoint with the
    midpoint of the preceding three days' combined high-low range.
    Randomness is used only to select historical quiz windows.
    """
    c, p, p2, p3 = (_candle(row) for row in (curr, prev, prev2, prev3))
    if c is None:
        return None

    if c["body"] <= c["range"] * 0.1:
        return "十字星 (Doji)"

    bullish = c["close"] > c["open"]
    bearish = c["close"] < c["open"]
    if p is not None and p2 is not None:
        star = p["body"] < p["range"] * 0.25
        if star and p2["body"] > p2["range"] * 0.5 and c["body"] > c["range"] * 0.5:
            if (
                p2["close"] < p2["open"] and p["open"] < p2["close"]
                and bullish and c["close"] > p2["mid"]
            ):
                return "早晨之星 (Morning Star)"
            if (
                p2["close"] > p2["open"] and p["open"] > p2["close"]
                and bearish and c["close"] < p2["mid"]
            ):
                return "黃昏之星 (Evening Star)"

    if p is not None:
        prev_bullish = p["close"] > p["open"]
        prev_bearish = p["close"] < p["open"]
        if prev_bearish and bullish and c["open"] <= p["close"] and c["close"] >= p["open"]:
            return "看漲吞噬 (Bullish Engulfing)"
        if prev_bullish and bearish and c["open"] >= p["close"] and c["close"] <= p["open"]:
            return "看跌吞噬 (Bearish Engulfing)"
        if (
            prev_bearish and bullish
            and (c["open"] < p["low"] or c["open"] < p["close"])
            and c["close"] > p["mid"]
        ):
            return "刺透型態 (Piercing Line)"
        if (
            prev_bullish and bearish
            and (c["open"] > p["high"] or c["open"] > p["close"])
            and c["close"] < p["mid"]
        ):
            return "烏雲罩頂 (Dark Cloud Cover)"
        if p["body"] > p["range"] * 0.5:
            if prev_bearish and bullish and c["open"] >= p["close"] and c["close"] <= p["open"]:
                return "看漲孕線 (Bullish Harami)"
            if prev_bullish and bearish and c["open"] <= p["close"] and c["close"] >= p["open"]:
                return "看跌孕線 (Bearish Harami)"

    if p is not None and p2 is not None and p3 is not None:
        context_mid = (
            max(bar["high"] for bar in (p3, p2, p))
            + min(bar["low"] for bar in (p3, p2, p))
        ) / 2
        if (
            p["close"] > p3["close"] and c["mid"] >= context_mid
            and c["lower"] >= c["body"] * 2 and c["upper"] <= c["body"] * 0.5
        ):
            return "上吊線 (Hanging Man)"
        if (
            p["close"] < p3["close"] and c["mid"] <= context_mid
            and c["upper"] >= c["body"] * 2 and c["lower"] <= c["body"] * 0.5
        ):
            return "倒鎚子線 (Inverted Hammer)"

    if c["lower"] >= c["body"] * 2 and c["upper"] <= c["body"]:
        return "鎚子線 (Hammer / Pin Bar)"
    if c["upper"] >= c["body"] * 2 and c["lower"] <= c["body"]:
        return "流星線 (Shooting Star)"
    if c["body"] >= c["range"] * 0.8:
        if bullish:
            return "大陽線 (Long Bullish)"
        if bearish:
            return "大陰線 (Long Bearish)"
    return None


def detect_pattern(df, idx):
    """Compatibility wrapper that provides all needed preceding candles."""
    if not isinstance(idx, (int, np.integer)) or idx < 0 or idx >= len(df):
        return None
    return identify_pattern(
        df.iloc[idx],
        df.iloc[idx - 1] if idx >= 1 else None,
        df.iloc[idx - 2] if idx >= 2 else None,
        df.iloc[idx - 3] if idx >= 3 else None,
    )


@st.cache_resource(show_spinner=False)
def load_data():
    """Cache SPY/QQQ in server memory across reruns and sessions.

    This is not disk persistence: restart/clear requires another download.
    Network timeouts and unavailable responses are handled below.
    """
    data = {}
    for ticker in ("SPY", "QQQ"):
        try:
            df = yf.download(
                ticker, period="5y", interval="1d", progress=False,
                auto_adjust=True, threads=False, timeout=15,
            )
            if df is None or df.empty:
                continue
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not set(OHLC_COLUMNS).issubset(df.columns):
                continue
            df = df.sort_index()
            df = df.loc[~df.index.duplicated(keep="last")]
            df[OHLC_COLUMNS] = df[OHLC_COLUMNS].apply(pd.to_numeric, errors="coerce")
            df = df.dropna(subset=OHLC_COLUMNS)
            finite = np.isfinite(df[OHLC_COLUMNS]).all(axis=1)
            valid = (
                finite & (df["High"] > df["Low"])
                & (df["High"] >= df[["Open", "Close"]].max(axis=1))
                & (df["Low"] <= df[["Open", "Close"]].min(axis=1))
            )
            df = df.loc[valid].copy()
            df.index.name = "Date"
            df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
            if len(df) >= WINDOW_SIZE:
                data[ticker] = df
        except Exception:
            # One failed provider response must not disable the other symbol.
            continue
    return data


def _quiz_slice(ticker, df, idx, pattern, fallback=False):
    sub_df = df.iloc[idx - WINDOW_SIZE + 1:idx + 1].copy()
    date = str(sub_df.index[-1])[:10]
    sub_df.index.name = "Date"
    options = []
    if pattern in PATTERN_DB:
        distractors = random.sample(
            [name for name in PATTERN_DB if name != pattern], 3
        )
        options = [pattern, *distractors]
        random.shuffle(options)
    return {
        "ticker": ticker,
        "df": sub_df.reset_index(),
        "correct": pattern,
        "options": options,
        "date": date,
        "fallback": fallback,
    }


def generate_quiz():
    """Inspect at most 500 randomly sampled historical bars; never invent labels."""
    eligible = {
        ticker: df for ticker, df in load_data().items()
        if len(df) >= WINDOW_SIZE and set(OHLC_COLUMNS).issubset(df.columns)
    }
    if not eligible:
        return None

    tickers = list(eligible)
    # Choose a desired category to improve exposure to rarer multi-bar patterns.
    desired = random.choice(list(PATTERN_DB))
    matches = {}
    for _ in range(MAX_QUIZ_ATTEMPTS):
        ticker = random.choice(tickers)
        df = eligible[ticker]
        idx = random.randrange(WINDOW_SIZE - 1, len(df))
        pattern = detect_pattern(df, idx)
        if pattern in PATTERN_DB:
            candidate = (ticker, idx)
            matches.setdefault(pattern, []).append(candidate)
            if pattern == desired:
                return _quiz_slice(ticker, df, idx, pattern)

    # After 500 attempts, choose a real observed category if possible.
    # No further scanning or retry loop is allowed.
    if matches:
        pattern = random.choice(list(matches))
        ticker, idx = random.choice(matches[pattern])
        return _quiz_slice(ticker, eligible[ticker], idx, pattern, fallback=True)

    # A random safe window with no label becomes ungraded observation practice.
    ticker = random.choice(tickers)
    df = eligible[ticker]
    idx = random.randrange(WINDOW_SIZE - 1, len(df))
    return _quiz_slice(ticker, df, idx, None, fallback=True)


@st.cache_resource(show_spinner=False)
def apple_touch_icon():
    icon = APP_DIR / "kline_trainer.ico"
    if not icon.is_file():
        return ""
    try:
        with Image.open(icon) as source:
            output = BytesIO()
            source.convert("RGBA").resize((180, 180)).save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")
    except OSError:
        return ""


def draw_chart(quiz):
    """Close every allocated figure, including when plotting/rendering fails."""
    font_name, supports_chinese = get_chart_font()
    with plt.rc_context({"font.family": [font_name], "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#0e1117")
        try:
            ax.set_facecolor("#161b22")
            df = quiz["df"]
            x = np.arange(len(df))
            ax.plot(x, df["EMA20"], color="#2962FF", linewidth=1.8, label="EMA 20")
            ax.plot(x, df["EMA50"], color="#FF6D00", linewidth=1.8, label="EMA 50")
            for i, row in df.iterrows():
                color = "#26a69a" if row["Close"] >= row["Open"] else "#ef5350"
                ax.plot([i, i], [row["Low"], row["High"]], color=color, linewidth=1.2)
                body = abs(row["Close"] - row["Open"])
                if body == 0:
                    ax.hlines(row["Open"], i - 0.3, i + 0.3, color=color, linewidth=1.5)
                else:
                    ax.bar(i, body, bottom=min(row["Open"], row["Close"]), color=color, width=0.6)
            target = len(df) - 1
            low = df.loc[target, "Low"]
            chart_range = max(float(df["High"].max() - df["Low"].min()), 0.01)
            ax.annotate(
                "考題棒" if supports_chinese else "Signal Bar", xy=(target, low),
                xytext=(target, low - chart_range * 0.12),
                arrowprops=dict(facecolor="#00E676", shrink=0.08, width=1.5, headwidth=6),
                color="#00E676", ha="center", fontsize=12, fontweight="bold",
            )
            ax.set_title(
                f"{quiz['ticker']} {'日 K' if supports_chinese else 'Daily K'} | {quiz['date']}",
                color="#c9d1d9", fontsize=15,
            )
            ax.grid(True, color="#21262d", linestyle="--", alpha=0.5)
            ax.tick_params(colors="#8b949e", labelsize=11)
            ticks = list(range(0, len(df), 5))
            if target not in ticks:
                ticks.append(target)
            ax.set_xticks(ticks)
            ax.set_xticklabels(pd.to_datetime(df.iloc[ticks]["Date"]).dt.strftime("%m/%d"))
            ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
            fig.tight_layout()
            st.pyplot(fig)
        finally:
            plt.close(fig)


def main():
    icon_path = APP_DIR / "kline_trainer.ico"
    st.set_page_config(
        page_title="Candle Lab - K-Line Trainer",
        page_icon=str(icon_path) if icon_path.is_file() else "📈",
        layout="wide", initial_sidebar_state="collapsed",
    )
    icon = apple_touch_icon()
    st.markdown(
        '<head><link rel="apple-touch-icon" sizes="180x180" '
        f'href="data:image/png;base64,{icon}">'
        '<meta name="apple-mobile-web-app-capable" content="yes"></head>',
        unsafe_allow_html=True,
    )
    st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #c9d1d9; font-size: 20px; }
.stApp p, .stApp label, .stApp li, .stApp input, .stApp button,
.stApp [data-testid="stMarkdownContainer"] { font-size: 20px !important; line-height: 1.55; }
.stApp h1 { font-size: 2.4rem; }
.stApp h2 { font-size: 2rem; }
.stApp h3 { font-size: 1.65rem; }
.stApp [data-testid="stMetricValue"] { font-size: 30px; }
.stApp [data-testid="stMetricLabel"] { font-size: 18px; }
.stApp [data-testid="stButton"] button { min-height: 52px; }
div[data-testid="metric-container"] {
    background-color: #161b22; border: 1px solid #30363d;
    padding: 16px 20px; border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)
    defaults = {
        "total_count": 0, "correct_count": 0, "streak": 0,
        "current_quiz": None, "submitted": False, "quiz_number": 0,
        "submitted_choice": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.markdown("## ⚡ CANDLE LAB | K 線型態訓練器")
    if st.session_state.current_quiz is None:
        with st.spinner("正在載入 SPY / QQQ 五年日 K 並抽取題目…"):
            st.session_state.current_quiz = generate_quiz()
    quiz = st.session_state.current_quiz
    if quiz is None:
        st.warning("暫時無法取得足夠的 SPY / QQQ 歷史資料，請確認網路後重試。")
        if st.button("重新載入資料", type="primary"):
            load_data.clear()
            st.rerun()
        return

    acc = 100 * st.session_state.correct_count / st.session_state.total_count if st.session_state.total_count else 0.0
    col1, col2, col3 = st.columns(3)
    col1.metric("累積題數", f"{st.session_state.total_count} 題")
    col2.metric("辨識準確率", f"{acc:.1f} %")
    col3.metric("連續答對 (Streak)", f"🔥 {st.session_state.streak} 題")
    st.markdown("---")

    col_chart, col_quiz = st.columns([65, 35])
    with col_chart:
        st.subheader(f"📊 標的走勢：{quiz['ticker']} (日 K 局部)")
        draw_chart(quiz)
        st.caption("請一併觀察最後三根 K 棒與先前趨勢。陽線為綠色，陰線為紅色。")
    with col_quiz:
        st.subheader("❓ 型態判斷題")
        if quiz["correct"] is None:
            st.info("500 次抽樣未找到符合規則的型態，已切換安全歷史切片。此題供觀察練習，不計分。")
        else:
            st.markdown(f"**題目**：箭頭所標註的最後一根 K 棒（日期：{quiz['date']}），屬於何種型態？")
            st.caption("重疊判定：十字星 → 三日反轉 → 吞噬 → 刺透／烏雲 → 孕線 → 趨勢影線 → 一般影線 → 長實體。")
            choice = st.radio(
                "請選擇答案：", quiz["options"], index=None,
                disabled=st.session_state.submitted,
                key=f"answer_{st.session_state.quiz_number}",
            )
            if not st.session_state.submitted:
                if st.button("確認提交答案 (Submit)", type="primary", use_container_width=True):
                    if choice is None:
                        st.warning("請先選取一個選項！")
                    else:
                        st.session_state.submitted = True
                        st.session_state.submitted_choice = choice
                        st.session_state.total_count += 1
                        if choice == quiz["correct"]:
                            st.session_state.correct_count += 1
                            st.session_state.streak += 1
                        else:
                            st.session_state.streak = 0
                        st.rerun()
            else:
                if st.session_state.submitted_choice == quiz["correct"]:
                    st.success("✅ **回答正確！**")
                else:
                    st.error(f"❌ **回答錯誤！** 正確答案是：**{quiz['correct']}**")
                info = PATTERN_DB[quiz["correct"]]
                st.markdown(f"**💡 結構解析：**\n\n{info['desc']}")
                st.markdown(f"**🔥 多空力道心理：**\n\n{info['psychology']}")

        if quiz["correct"] is None or st.session_state.submitted:
            if st.button("下一題 ➔ (Next)", use_container_width=True):
                old_key = f"answer_{st.session_state.quiz_number}"
                st.session_state.pop(old_key, None)
                st.session_state.current_quiz = None
                st.session_state.submitted = False
                st.session_state.submitted_choice = None
                st.session_state.quiz_number += 1
                st.rerun()

    with st.expander("📖 15 種經典 K 線型態學習庫"):
        for name, info in PATTERN_DB.items():
            st.markdown(f"**{name}**\n\n{info['desc']}\n\n多空心理：{info['psychology']}")


if __name__ == "__main__":
    main()
