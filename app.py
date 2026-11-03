import base64
from io import BytesIO
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import yfinance as yf


APP_DIR = Path(__file__).resolve().parent
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

WINDOW_SIZE = 25
MAX_QUIZ_ATTEMPTS = 500
OHLC_COLUMNS = ["Open", "High", "Low", "Close"]


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
    """Cache SPY, QQQ and GLD in server memory across reruns and sessions.

    This is not disk persistence: restart/clear requires another download.
    Network timeouts and unavailable responses are handled below.
    """
    data = {}
    for ticker in ("SPY", "QQQ", "GLD"):
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


def draw_chart(quiz, marker="Signal Bar ⬆"):
    """Close every allocated figure, including when plotting/rendering fails."""
    with plt.rc_context({
        "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "axes.unicode_minus": False,
    }):
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
                marker, xy=(target, low),
                xytext=(target, low - chart_range * 0.12),
                arrowprops=dict(facecolor="#00E676", shrink=0.08, width=1.5, headwidth=6),
                color="#00E676", ha="center", fontsize=12, fontweight="bold",
            )
            ax.set_title(
                f"{quiz['ticker']} - Daily Chart",
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


def _decision_checks(df):
    """Evaluate the objective SOP conditions at the current decision bar."""
    if len(df) < 6:
        return {"trend": False, "pullback": False, "signal": False, "risk": False}
    current = df.iloc[-1]
    trend = bool(
        current["Close"] > current["EMA20"] > current["EMA50"]
        and current["EMA20"] > df.iloc[-6]["EMA20"]
    )
    ema20 = float(current["EMA20"])
    pullback = bool(
        ema20 > 0
        and abs(float(current["Low"]) - ema20) / ema20 <= 0.02
        and float(current["Close"]) / ema20 <= 1.03
    )
    signal = detect_pattern(df, len(df) - 1) in {
        "鎚子線 (Hammer / Pin Bar)", "看漲吞噬 (Bullish Engulfing)"
    }
    risk = bool(float(current["High"]) > float(current["Low"]))
    return {"trend": trend, "pullback": pullback, "signal": signal, "risk": risk}


def _decision_signal_explanation(df):
    """Return the actual candle classification and a specific long-side reason."""
    pattern = detect_pattern(df, len(df) - 1) if len(df) else None
    current = _candle(df.iloc[-1]) if len(df) else None
    previous = _candle(df.iloc[-2]) if len(df) > 1 else None
    # Give the decision report explicit bearish names even where the quiz
    # classifier's existing priority assigns an overlapping candle elsewhere.
    if current is not None and current["close"] < current["open"]:
        if (
            current["body"] >= current["range"] * 0.8
            and pattern not in {"看跌吞噬 (Bearish Engulfing)"}
        ):
            pattern = "大陰線 (Long Bearish)"
        if (
            previous is not None
            and previous["close"] > previous["open"]
            and current["open"] >= previous["close"]
            and current["close"] <= previous["open"]
        ):
            pattern = "看跌吞噬 (Bearish Engulfing)"
    name = pattern or "普通 K 棒 (無反轉特徵)"
    bearish_patterns = {
        "流星線 (Shooting Star)",
        "上吊線 (Hanging Man)",
        "大陰線 (Long Bearish)",
        "看跌吞噬 (Bearish Engulfing)",
        "看跌孕線 (Bearish Harami)",
        "烏雲罩頂 (Dark Cloud Cover)",
        "黃昏之星 (Evening Star)",
    }
    indecision_patterns = {
        "十字星 (Doji)",
        "看漲孕線 (Bullish Harami)",
        "看跌孕線 (Bearish Harami)",
    }
    if pattern in indecision_patterns:
        reason = "多空處於僵持猶豫，尚未出現多頭表態訊號。"
    elif pattern in bearish_patterns:
        reason = "當前出現明顯賣壓/空頭型態，此處做多極易遭摜壓。"
    elif pattern is None:
        reason = "下影線長度不足（未達實體 2 倍），亦非看漲吞噬，缺少多頭強勢拒絕破位的扳機訊號。"
    else:
        reason = "雖辨識到型態，但它不在本 SOP 允許的做多訊號清單內。"
    return name, reason


def generate_decision_quiz():
    eligible = {
        ticker: df for ticker, df in load_data().items()
        if len(df) >= WINDOW_SIZE + 15
    }
    if not eligible:
        return None
    ticker = random.choice(list(eligible))
    df = eligible[ticker]
    idx = random.randrange(WINDOW_SIZE - 1, len(df) - 15)
    window = df.iloc[idx - WINDOW_SIZE + 1:idx + 1].copy()
    return {
        "ticker": ticker,
        "df": window.reset_index(),
        "future": df.iloc[idx + 1:idx + 16].copy(),
        "checks": _decision_checks(window),
        "date": str(df.index[idx])[:10],
    }


def settle_long(decision):
    """Simulate a long order and scan future bars in chronological order."""
    current = decision["df"].iloc[-1]
    entry = float(current["High"])
    stop = float(current["Low"]) * 0.997
    risk = entry - stop
    target = entry + 2.0 * risk
    future = decision["future"]
    details = {"entry": entry, "stop": stop, "target": target}
    if future.empty or risk <= 0:
        return {**details, "r": 0.0, "label": "0.0 R (無後續資料)"}
    first = future.iloc[0]
    signal_low = float(current["Low"])
    if float(first["High"]) < entry and float(first["Low"]) < signal_low:
        return {**details, "r": 0.0, "label": "0.0 R (未觸發取消)"}
    for _, bar in future.iterrows():
        # Daily OHLC cannot reveal intraday order when both levels are touched.
        # Use the conservative stop-first convention for same-bar ambiguity.
        hit_stop = float(bar["Low"]) <= stop
        hit_target = float(bar["High"]) >= target
        if hit_stop:
            return {**details, "r": -1.0, "label": "-1.0 R (觸發停損)"}
        if hit_target:
            return {**details, "r": 2.0, "label": "+2.0 R (停利達標)"}
    final_r = (float(future.iloc[-1]["Close"]) - entry) / risk
    return {**details, "r": final_r, "label": f"{final_r:+.2f} R (15 棒浮動結算)"}


def _decision_feedback(decision, action):
    checks = decision["checks"]
    sop_ok = all(checks.values())
    buy = action == "buy"
    score = 100 if buy == sop_ok else 0
    if buy and sop_ok:
        message = "符合 SOP；即使停損，虧損仍是正常成本，請維持正期望值。"
    elif buy:
        message = "破壞紀律，僥倖獲利是長期毒藥。"
    elif sop_ok:
        message = "錯失標準機會，應依 SOP 勇敢執行。"
    else:
        message = "自律避險，觀望符合交易紀律。"
    result = settle_long(decision) if buy else {
        "r": 0.0, "label": "未進場 (觀望)",
        "entry": None, "stop": None, "target": None,
    }
    return {"score": score, "message": message, "result": result}


def _render_pattern_tab():
    defaults = {
        "quiz": None, "submitted": False, "choice": None,
        "total": 0, "correct": 0, "streak": 0, "number": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(f"pattern_{key}", value)
    if st.session_state.pattern_quiz is None:
        st.session_state.pattern_quiz = generate_quiz()
    quiz = st.session_state.pattern_quiz
    if quiz is None:
        st.warning("暫時無法取得 SPY、QQQ、GLD 歷史資料。")
        return
    accuracy = 100 * st.session_state.pattern_correct / st.session_state.pattern_total if st.session_state.pattern_total else 0
    a, b, c = st.columns(3)
    a.metric("作答總數", f"{st.session_state.pattern_total} 題")
    b.metric("答對率", f"{accuracy:.1f}%")
    c.metric("連續答對 Streak", f"{st.session_state.pattern_streak} 題")
    left, right = st.columns([65, 35])
    with left:
        st.subheader(f"📊 {quiz['ticker']} 日 K 局部")
        draw_chart(quiz)
    with right:
        st.subheader("❓ 型態判斷題")
        if quiz["correct"] is None:
            st.info("本題為安全觀察切片，不計分。")
        else:
            st.markdown(f"**日期：** {quiz['date']}，請判斷最後一根 K 棒。")
            choice = st.radio(
                "請選擇答案：", quiz["options"], index=None,
                disabled=st.session_state.pattern_submitted,
                key=f"pattern_answer_{st.session_state.pattern_number}",
            )
            if not st.session_state.pattern_submitted and st.button(
                "確認提交答案", type="primary", use_container_width=True,
                key="pattern_submit",
            ):
                if choice is None:
                    st.warning("請先選取一個選項！")
                else:
                    st.session_state.pattern_submitted = True
                    st.session_state.pattern_choice = choice
                    st.session_state.pattern_total += 1
                    if choice == quiz["correct"]:
                        st.session_state.pattern_correct += 1
                        st.session_state.pattern_streak += 1
                    else:
                        st.session_state.pattern_streak = 0
                    st.rerun()
            if st.session_state.pattern_submitted:
                if st.session_state.pattern_choice == quiz["correct"]:
                    st.success("✅ 回答正確！")
                else:
                    st.error(f"❌ 正確答案：{quiz['correct']}")
                info = PATTERN_DB[quiz["correct"]]
                st.info(f"結構解析：{info['desc']}\n\n多空力道心理：{info['psychology']}")
        if quiz["correct"] is None or st.session_state.pattern_submitted:
            if st.button("下一題 ➔", key="pattern_next", use_container_width=True):
                st.session_state.pattern_quiz = None
                st.session_state.pattern_submitted = False
                st.session_state.pattern_choice = None
                st.session_state.pattern_number += 1
                st.rerun()
    with st.expander("📖 15 種經典 K 線型態學習庫"):
        for name, info in PATTERN_DB.items():
            st.markdown(f"**{name}**\n\n{info['desc']}\n\n多空心理：{info['psychology']}")


def _render_decision_tab():
    defaults = {
        "quiz": None, "feedback": None, "total": 0,
        "r": 0.0, "perfect": 0, "number": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(f"decision_{key}", value)
    if st.session_state.decision_quiz is None:
        st.session_state.decision_quiz = generate_decision_quiz()
    decision = st.session_state.decision_quiz
    if decision is None:
        st.warning("暫時無法取得足夠的歷史資料。")
        return
    rate = 100 * st.session_state.decision_perfect / st.session_state.decision_total if st.session_state.decision_total else 0
    a, b, c = st.columns(3)
    a.metric("模擬總筆數", st.session_state.decision_total)
    b.metric("累積 R 值", f"{st.session_state.decision_r:+.2f} R")
    c.metric("SOP 滿分執行率", f"{rate:.1f}%")
    left, right = st.columns([65, 35])
    with left:
        st.subheader(f"📊 {decision['ticker']} Decision Setup")
        draw_chart(
            {"df": decision["df"], "ticker": decision["ticker"], "date": decision["date"]},
            marker="Decision Bar ⬆",
        )
    with right:
        st.subheader("⚡ 做多 vs 觀望")
        if st.session_state.decision_feedback is None:
            buy_col, pass_col = st.columns(2)
            if buy_col.button("🟢 進場做多 (Buy)", key="decision_buy", use_container_width=True):
                feedback = _decision_feedback(decision, "buy")
                st.session_state.decision_feedback = feedback
                st.session_state.decision_total += 1
                st.session_state.decision_r += feedback["result"]["r"]
                st.session_state.decision_perfect += int(feedback["score"] == 100)
                st.rerun()
            if pass_col.button("⚪ 觀望略過 (Pass)", key="decision_pass", use_container_width=True):
                feedback = _decision_feedback(decision, "pass")
                st.session_state.decision_feedback = feedback
                st.session_state.decision_total += 1
                st.session_state.decision_perfect += int(feedback["score"] == 100)
                st.rerun()
        else:
            feedback = st.session_state.decision_feedback
            message = f"決策評分：{feedback['score']} 分\n\n{feedback['message']}"
            if feedback["score"] == 100:
                st.success(message)
            else:
                st.error(message)
            result = feedback["result"]
            st.metric("市場結果", result["label"])
            st.markdown("**SOP 檢核明細**")
            labels = {
                "trend": "趨勢方向", "pullback": "回踩位置",
                "signal": "訊號棒辨識", "risk": "風控設定",
            }
            current = decision["df"].iloc[-1]
            trend_text = (
                f"Close {current['Close']:.2f} > EMA20 {current['EMA20']:.2f} "
                f"> EMA50 {current['EMA50']:.2f}; EMA20 五日前 "
                f"{decision['df'].iloc[-6]['EMA20']:.2f}"
            )
            distance = abs(float(current["Low"]) - float(current["EMA20"])) / float(current["EMA20"]) * 100 if current["EMA20"] else float("inf")
            signal_name, signal_reason = _decision_signal_explanation(decision["df"])
            details = {
                "trend": trend_text,
                "pullback": f"Low 到 EMA20 距離 {distance:.2f}%；收盤偏離 {float(current['Close']) / float(current['EMA20']) - 1:.2%}" if current["EMA20"] else "EMA20 無效",
                "signal": f"最後一棒判定：{signal_name}",
                "risk": f"預設進場 {float(current['High']):.2f}；停損 {float(current['Low']) * 0.997:.2f}；停利 {float(current['High']) + 2 * (float(current['High']) - float(current['Low']) * 0.997):.2f}（1:2）",
            }
            for key, label in labels.items():
                passed = decision["checks"][key]
                st.write(f"{'✅' if passed else '❌'} {label}：{'通過' if passed else '未通過'}；{details[key]}")
                if key == "signal" and not passed:
                    st.warning(signal_reason)
            st.info("做多進場僅認可【鎚子線 (Pin Bar)】或【看漲吞噬 (Bullish Engulfing)】。")
            if result["entry"] is not None:
                st.caption(f"進場 {result['entry']:.2f} | 停損 {result['stop']:.2f} | 停利 {result['target']:.2f}")
            if st.button("下一局決策 ➔", key="decision_next", use_container_width=True):
                st.session_state.decision_quiz = None
                st.session_state.decision_feedback = None
                st.session_state.decision_number += 1
                st.rerun()


def main():
    st.set_page_config(page_title="Candle Lab", page_icon="📈", layout="wide")
    st.markdown(
        '<head><link rel="apple-touch-icon" '
        'href="https://img.icons8.com/fluency/180/candlestick-chart.png">'
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
    st.markdown("## ⚡ CANDLE LAB | 金融訓練中心")
    pattern_tab, decision_tab = st.tabs([
        "🎯 K 線型態辨識 (認型態)",
        "⚡ 實戰決策模擬 (練 SOP)",
    ])
    with pattern_tab:
        _render_pattern_tab()
    with decision_tab:
        _render_decision_tab()


if __name__ == "__main__":
    main()
