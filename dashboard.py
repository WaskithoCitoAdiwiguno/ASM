"""
Dashboard Simulasi Paper Trading & Signal Monitor
==================================================
Jalankan dengan:
    streamlit run dashboard.py

Tab "Sinyal & Rekomendasi" memakai signal engine mode VOTING yang
mereplikasi metodologi Technical Ratings ala TradingView & Technical
Summary ala Investing.com:
    - 15 moving averages + 11 oscillator memberi suara -1/0/+1
    - rating rata-rata di-map ke 5 kategori (Strong Sell .. Strong Buy)
    - sinyal BUY/SELL dikirim saat selisih suara melewati ambang voting
    - tiap sinyal disertai rincian suara per indikator & kalimat rekomendasi

--------------------------------------------------------------------------
ASUMSI SKEMA DATABASE (SQLite) — sama seperti sebelumnya, semua nama
tabel & kolom didefinisikan di CONFIG di bawah.

Tabel 1 - paper_trades (wajib)
Tabel 2 - ohlcv_data  (opsional, untuk chart & perhitungan sinyal)
--------------------------------------------------------------------------

Dependensi:
    pip install streamlit plotly pandas numpy ta
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from technical_indicators import build_feature_set
from signal_engine import SignalEngine

# ============================================================
# CONFIG — sesuaikan dengan skema database kamu di sini
# ============================================================
CONFIG = {
    "trades_table": "paper_trades",
    "ohlcv_table": "ohlcv_data",
    "trades_cols": {
        "id": "trade_id",
        "ticker": "ticker",
        "market": "market",
        "signal": "signal_type",
        "entry_date": "entry_timestamp",
        "entry_price": "entry_price",
        "exit_date": "exit_timestamp",
        "exit_price": "exit_price",
        "stop_loss": "stop_loss_price",
        "take_profit": "take_profit_price",
        "quantity": "position_size",
        "confidence": "confidence_score",
        "status": "status",
        "pnl": "pnl",
    },
    "ohlcv_cols": {
        "ticker": "ticker",
        "market": "market",
        "date": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    },
}

DEFAULT_STARTING_CAPITAL = 100_000_000

# ============================================================
# DESIGN TOKENS (satu sumber warna untuk CSS + chart)
# ============================================================
COLORS = {
    "bg": "#f5f7fa",
    "surface": "#ffffff",
    "border": "#e4e9f0",
    "ink": "#0f172a",
    "muted": "#64748b",
    "accent": "#2563eb",
    "strong_buy": "#059669",
    "buy": "#34d399",
    "neutral": "#94a3b8",
    "sell": "#f87171",
    "strong_sell": "#dc2626",
}

CATEGORY_COLORS = {
    "STRONG BUY": COLORS["strong_buy"],
    "BUY": COLORS["buy"],
    "NEUTRAL": COLORS["neutral"],
    "SELL": COLORS["sell"],
    "STRONG SELL": COLORS["strong_sell"],
}

CATEGORY_ICONS = {
    "STRONG BUY": "▲▲",
    "BUY": "▲",
    "NEUTRAL": "■",
    "SELL": "▼",
    "STRONG SELL": "▼▼",
}

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #0f172a; --muted: #64748b; --border: #e4e9f0;
    --surface: #ffffff; --accent: #2563eb;
}
/* ---------- App background & typography ---------- */
.stApp {
    background: #f5f7fa;
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
    color: var(--ink);
}
.block-container { padding-top: 1.6rem; max-width: 1400px; }

/* ---------- Hero panel ---------- */
.hero {
    background: linear-gradient(115deg, #0b1220 0%, #14264d 55%, #1d3a8a 100%);
    border-radius: 18px;
    padding: 26px 32px 22px 32px;
    color: #f8fafc;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(13, 30, 66, 0.25);
}
.hero-title {
    font-size: 1.65rem; font-weight: 800; letter-spacing: -0.02em; margin: 0;
}
.hero-sub { color: #b6c2e2; font-size: 0.92rem; margin-top: 6px; }
.hero-stats { display: flex; gap: 36px; margin-top: 18px; flex-wrap: wrap; }
.hero-stat .v { font-size: 1.35rem; font-weight: 700; }
.hero-stat .k {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.09em;
    color: #8fa3cc; margin-top: 2px;
}
.hero-badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.18);
    font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase;
    margin-bottom: 10px; color: #c7d5f5;
}

/* ---------- Section titles ---------- */
.section-title {
    display: flex; align-items: center; gap: 10px;
    font-weight: 700; font-size: 1.02rem; color: var(--ink);
    margin: 6px 0 10px 0;
}
.section-title::before {
    content: ''; width: 4px; height: 18px; border-radius: 2px;
    background: var(--accent); display: inline-block;
}

/* ---------- Metric cards ---------- */
div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 18px 10px 18px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
}
div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 700; letter-spacing: -0.01em;
}

/* ---------- Tabs as pills ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: #eef2f7; padding: 5px; border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px; padding: 7px 16px; font-weight: 600;
    font-size: 0.88rem; color: var(--muted); background: transparent;
}
.stTabs [aria-selected="true"] {
    background: var(--surface); color: var(--ink);
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- Signal cards ---------- */
.signal-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 18px 20px; height: 100%;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}
.signal-card .head {
    display: flex; justify-content: space-between; align-items: baseline;
    border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 12px;
}
.signal-card .ticker { font-size: 1.25rem; font-weight: 800; letter-spacing: -0.01em; }
.signal-card .market {
    font-size: 0.68rem; font-weight: 700; color: var(--muted);
    border: 1px solid var(--border); border-radius: 6px; padding: 1px 7px;
    letter-spacing: 0.08em;
}
.signal-card .price { color: var(--muted); font-size: 0.86rem; margin-top: 2px; }

.rating-chip {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    color: #fff; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.05em;
}

.vote-bar { display: flex; height: 10px; border-radius: 6px; overflow: hidden; margin: 10px 0 4px 0; }
.vote-bar .b { background: #10b981; }
.vote-bar .n { background: #cbd5e1; }
.vote-bar .s { background: #ef4444; }
.vote-legend { display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--muted); }

.suggestion-box {
    border-radius: 10px; padding: 12px 14px; font-size: 0.86rem; line-height: 1.5;
    margin-top: 12px; border: 1px solid;
}
.suggestion-box .s-title { font-weight: 700; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }

/* ---------- Vote detail table ---------- */
table.votes { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
table.votes th {
    text-align: left; padding: 7px 10px; color: var(--muted);
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em;
    border-bottom: 1px solid var(--border);
}
table.votes td { padding: 6px 10px; border-bottom: 1px solid #f1f5f9; }
table.votes td.grp { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
.v-buy { color: #059669; font-weight: 700; }
.v-sell { color: #dc2626; font-weight: 700; }
.v-neutral { color: #94a3b8; font-weight: 600; }

/* ---------- Misc ---------- */
[data-testid="stToolbar"] { visibility: hidden; height: 0; }
[data-testid="stDecoration"] { background-image: none; }
div[data-testid="stExpander"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
}
.stCaption, small { color: var(--muted); }
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 6px; }
::-webkit-scrollbar-track { background: transparent; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=False)
def load_table(db_path: str, table_name: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        if cur.fetchone() is None:
            return pd.DataFrame()
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    return df


def prepare_trades(df: pd.DataFrame) -> pd.DataFrame:
    c = CONFIG["trades_cols"]
    if df.empty:
        return df

    rename_map = {v: k for k, v in c.items() if v in df.columns}
    df = df.rename(columns=rename_map)

    for col in ["entry_date", "exit_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["entry_price", "exit_price", "stop_loss", "take_profit",
                "quantity", "confidence", "pnl"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "status" not in df.columns:
        df["status"] = np.where(df.get("exit_date").notna(), "CLOSED", "OPEN")
    df["status"] = df["status"].astype(str).str.upper()
    df["status"] = df["status"].replace({"OPEN": "OPEN", "CLOSED": "CLOSED"})

    if "pnl" not in df.columns or df["pnl"].isna().all():
        qty = df.get("quantity", pd.Series(1, index=df.index)).fillna(1)
        direction = np.where(df.get("signal", "BUY").astype(str).str.upper() == "SELL", -1, 1)
        df["pnl"] = np.where(
            df["status"] == "CLOSED",
            (df["exit_price"] - df["entry_price"]) * direction * qty,
            np.nan,
        )

    if "stop_loss" in df.columns:
        qty = df.get("quantity", pd.Series(1, index=df.index)).fillna(1)
        risk = (df["entry_price"] - df["stop_loss"]).abs() * qty
        df["risk_amount"] = risk.replace(0, np.nan)
        df["r_multiple"] = df["pnl"] / df["risk_amount"]
    else:
        df["r_multiple"] = np.nan

    return df


def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisasi tabel OHLCV ke kolom kanonik (timestamp, open, ...)."""
    c = CONFIG["ohlcv_cols"]
    if df.empty:
        return df
    rename_map = {v: "timestamp" for k, v in c.items() if v in df.columns and k == "date"}
    rename_map.update({v: k for k, v in c.items() if v in df.columns and k != "date"})
    df = df.rename(columns=rename_map)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ============================================================
# SIGNAL COMPUTATION (voting engine)
# ============================================================
@st.cache_data(show_spinner=False, ttl=300)
def compute_ticker_signals(raw_ohlcv: pd.DataFrame) -> dict:
    """
    Hitung rating & sinyal voting untuk ticker terakhir-data di raw_ohlcv.
    Return dict per ticker: {ticker: {…engine result…, 'market', 'price',
    'price_change_pct', 'as_of'}}
    """
    if raw_ohlcv.empty:
        return {}
    engine = SignalEngine()
    out: dict = {}
    for ticker, g in raw_ohlcv.dropna(subset=["timestamp"]).groupby("ticker"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) < 60:
            continue
        try:
            features = build_feature_set(g)
        except Exception:
            continue
        if len(features) < 2:
            continue
        latest = features.iloc[-1]
        prev = features.iloc[-2]

        row = latest.copy()
        for col in ["rsi_14", "cci_20", "williams_r", "momentum_10", "adx",
                    "macd_hist", "close"]:
            row[f"{col}_prev"] = prev[col]

        res = engine.evaluate_row(row)
        price_change = (
            (latest["close"] - prev["close"]) / prev["close"] * 100
            if prev["close"] else 0.0
        )
        out[ticker] = {
            **res,
            "market": latest.get("market", ""),
            "price": float(latest["close"]),
            "price_change_pct": float(price_change),
            "as_of": latest["timestamp"],
        }
    return out


def vote_bar_html(buy: int, sell: int, neutral: int) -> str:
    total = max(buy + sell + neutral, 1)
    pb, ps, pn = buy / total * 100, sell / total * 100, neutral / total * 100
    return (
        f'<div class="vote-bar">'
        f'<div class="b" style="width:{pb:.1f}%"></div>'
        f'<div class="n" style="width:{pn:.1f}%"></div>'
        f'<div class="s" style="width:{ps:.1f}%"></div>'
        f'</div>'
        f'<div class="vote-legend"><span>🟢 {buy} Buy</span>'
        f'<span>⚪ {neutral} Neutral</span><span>🔴 {sell} Sell</span></div>'
    )


def suggestion_box_html(category: str, suggestion: str) -> str:
    color = CATEGORY_COLORS.get(category, COLORS["neutral"])
    icon = CATEGORY_ICONS.get(category, "•")
    bg = {"STRONG BUY": "#ecfdf5", "BUY": "#f0fdf4", "NEUTRAL": "#f8fafc",
          "SELL": "#fef2f2", "STRONG SELL": "#fef2f2"}.get(category, "#f8fafc")
    return (
        f'<div class="suggestion-box" style="border-color:{color}55;background:{bg}">'
        f'<div class="s-title" style="color:{color}">{icon} Rekomendasi Sistem</div>'
        f'{suggestion}</div>'
    )


def votes_table_html(vote_table: list) -> str:
    rows = ""
    for v in vote_table:
        label = v["label"]
        css = {"Buy": "v-buy", "Sell": "v-sell"}.get(label, "v-neutral")
        icon = {"Buy": "▲", "Sell": "▼"}.get(label, "—")
        grp = "Moving Average" if v["group"] == "ma" else "Oscillator"
        rows += (
            f"<tr><td>{v['name']}</td><td class='grp'>{grp}</td>"
            f"<td class='{css}'>{icon} {label}</td></tr>"
        )
    return (
        "<table class='votes'><thead><tr><th>Indikator</th><th>Grup</th>"
        f"<th>Vote</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def rating_gauge_fig(rating: float, category: str, height: int = 190) -> go.Figure:
    color = CATEGORY_COLORS.get(category, COLORS["neutral"])
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rating,
            number={"valueformat": ".2f", "font": {"size": 30, "family": "Inter"}},
            title={"text": "rating", "font": {"size": 10, "color": COLORS["muted"]}},
            gauge={
                "axis": {"range": [-1, 1], "tickwidth": 1,
                         "tickvals": [-1, -0.5, -0.1, 0.1, 0.5, 1],
                         "tickfont": {"size": 9, "color": COLORS["muted"]}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#eef2f7",
                "borderwidth": 0,
                "steps": [
                    {"range": [-1, -0.5], "color": "#fee2e2"},
                    {"range": [-0.5, -0.1], "color": "#fef2f2"},
                    {"range": [-0.1, 0.1], "color": "#f1f5f9"},
                    {"range": [0.1, 0.5], "color": "#f0fdf4"},
                    {"range": [0.5, 1], "color": "#d1fae5"},
                ],
                "threshold": {"line": {"color": color, "width": 2},
                              "thickness": 0.8, "value": rating},
            },
        )
    )
    fig.update_layout(height=height, margin=dict(l=18, r=18, t=6, b=6),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ============================================================
# METRIC HELPERS (dipertahankan dari versi sebelumnya)
# ============================================================
def compute_equity_curve(closed: pd.DataFrame, starting_capital: float) -> pd.DataFrame:
    if closed.empty:
        return pd.DataFrame(columns=["date", "equity"])
    s = closed.dropna(subset=["exit_date", "pnl"]).sort_values("exit_date")
    equity = starting_capital + s["pnl"].cumsum()
    out = pd.DataFrame({"date": s["exit_date"].values, "equity": equity.values})
    if not out.empty:
        start_row = pd.DataFrame({"date": [out["date"].min()], "equity": [starting_capital]})
        out = pd.concat([start_row, out], ignore_index=True)
    return out


def compute_max_drawdown(equity_df: pd.DataFrame) -> float:
    if equity_df.empty:
        return 0.0
    eq = equity_df["equity"]
    running_max = eq.cummax()
    drawdown = (eq - running_max) / running_max
    return float(drawdown.min())


def confidence_bucket_stats(closed: pd.DataFrame) -> pd.DataFrame:
    if closed.empty or "confidence" not in closed.columns:
        return pd.DataFrame(columns=["bucket", "n_signals", "win_rate"])
    d = closed.dropna(subset=["confidence", "pnl"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["bucket", "n_signals", "win_rate"])
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    d["bucket"] = pd.cut(d["confidence"], bins=bins, labels=labels, right=False)
    grouped = d.groupby("bucket", observed=False).agg(
        n_signals=("pnl", "count"),
        win_rate=("pnl", lambda x: (x > 0).mean() if len(x) else np.nan),
    ).reset_index()
    return grouped


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="ASM · Market Monitor",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(APP_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    db_path = st.text_input("Path database SQLite", value="trading_system.db")
    starting_capital = st.number_input(
        "Modal awal (untuk equity curve)",
        min_value=0.0,
        value=float(DEFAULT_STARTING_CAPITAL),
        step=1_000_000.0,
        format="%.0f",
    )
    st.caption(
        "Equity curve dihitung dari akumulasi PnL trade CLOSED, "
        "diurutkan berdasarkan tanggal exit."
    )
    with st.expander("📖 Metodologi Sinyal", expanded=False):
        st.markdown(
            """
            Engine sinyal memakai sistem **voting multi-indikator** yang
            mereplikasi publik platform analisis teknikal:

            - **15 Moving Averages** (SMA 10–200, EMA 10–200, Hull MA9,
              VWMA20, Ichimoku) → vote berdasarkan posisi harga vs MA
            - **11 Oscillators** (RSI, Stochastic, Stoch RSI, CCI, ADX,
              MACD, Awesome Osc, Momentum, Williams %R, Bull Bear Power,
              Ultimate Osc) → vote dengan aturan oversold/overbought
            - Rating = rata-rata suara dalam rentang **−1 … +1**
            - Kategori: Strong Sell / Sell / Neutral / Buy / Strong Buy
            - Sinyal dikirim saat selisih suara ≥ ambang voting

            *Bukan nasihat keuangan — murni simulasi & edukasi.*
            """
        )

if not Path(db_path).exists():
    st.warning(f"File database `{db_path}` tidak ditemukan. Cek path di sidebar.")
    st.stop()

raw_trades = load_table(db_path, CONFIG["trades_table"])
raw_ohlcv = load_table(db_path, CONFIG["ohlcv_table"])

if raw_trades.empty:
    st.error(
        f"Tabel `{CONFIG['trades_table']}` tidak ditemukan atau kosong di database. "
        "Cek nama tabel di bagian CONFIG pada dashboard.py."
    )
    st.stop()

trades = prepare_trades(raw_trades)
ohlcv = prepare_ohlcv(raw_ohlcv)
signals = compute_ticker_signals(raw_ohlcv)

open_trades = trades[trades["status"] == "OPEN"]
closed_trades = trades[trades["status"] == "CLOSED"]

# ------------------------------------------------------------
# HERO PANEL
# ------------------------------------------------------------
n_bull = sum(1 for s in signals.values() if s["rating_category"] in ("BUY", "STRONG BUY"))
n_bear = sum(1 for s in signals.values() if s["rating_category"] in ("SELL", "STRONG SELL"))
if signals:
    if n_bull > n_bear:
        sentiment, sent_color = "Bullish", CATEGORY_COLORS["BUY"]
    elif n_bear > n_bull:
        sentiment, sent_color = "Bearish", CATEGORY_COLORS["SELL"]
    else:
        sentiment, sent_color = "Netral", COLORS["neutral"]
else:
    sentiment, sent_color = "—", COLORS["neutral"]

hero_stats = "".join(
    f'<div class="hero-stat"><div class="v">{v}</div><div class="k">{k}</div></div>'
    for k, v in [
        ("Ticker Dipantau", len(signals)),
        ("Open Trades", len(open_trades)),
        ("Closed Trades", len(closed_trades)),
        ("Sentimen Agregat", sentiment),
    ]
)
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-badge">Signal Engine · Multi-Indicator Voting</div>
        <p class="hero-title">Market Monitor — Paper Trading &amp; Signal Simulator</p>
        <p class="hero-sub">Rating teknikal 26 indikator ala TradingView/Investing.com ·
        simulasi order tanpa eksekusi riil · bukan nasihat keuangan</p>
        <div class="hero-stats">{hero_stats}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab_signal, tab_valid, tab_chart, tab_trades, tab_equity, tab_metrics = st.tabs(
    ["🎯 Sinyal", "🧪 Validasi", "🕯️ Chart", "📋 Trades", "📈 Equity", "📐 Metrik"]
)

# ============================================================
# TAB 1 — SINYAL & REKOMENDASI
# ============================================================
with tab_signal:
    st.markdown('<div class="section-title">Ringkasan Rating Semua Ticker</div>',
                unsafe_allow_html=True)

    if not signals:
        st.info(
            "Sinyal belum bisa dihitung — butuh tabel OHLCV yang terisi "
            f"(`{CONFIG['ohlcv_table']}`) dengan data cukup (≥ 60 bar) per ticker."
        )
    else:
        # ---- tabel ringkasan lintas ticker ----
        summary_rows = []
        for tk, s in sorted(signals.items()):
            summary_rows.append({
                "Ticker": tk,
                "Market": s.get("market", ""),
                "Harga": s["price"],
                "Δ 1 bar (%)": round(s["price_change_pct"], 2),
                "Rating": round(s["rating"], 2),
                "Kategori": s["rating_category"],
                "Buy": s["buy_votes"],
                "Neutral": s["neutral_votes"],
                "Sell": s["sell_votes"],
                "Sinyal": s["signal"],
            })
        summary_df = pd.DataFrame(summary_rows)

        def _color_signal(val):
            if val == "BUY":
                return "color: #059669; font-weight:700"
            if val == "SELL":
                return "color: #dc2626; font-weight:700"
            return "color: #64748b"

        st.dataframe(
            summary_df.style.map(_color_signal, subset=["Sinyal"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Harga": st.column_config.NumberColumn(format="%.2f"),
                "Rating": st.column_config.ProgressColumn(
                    format="%+.2f", min_value=-1.0, max_value=1.0
                ),
            },
        )
        st.caption(
            "Rating ∈ [−1, +1]: rata-rata suara 26 indikator (15 MA + 11 oscillator). "
            "≥ +0.5 Strong Buy · ≥ +0.1 Buy · −0.1…+0.1 Neutral · ≤ −0.1 Sell · ≤ −0.5 Strong Sell."
        )

        st.markdown('<div class="section-title">Kartu Sinyal per Ticker</div>',
                    unsafe_allow_html=True)

        # ---- kartu sinyal, 2 kolom ----
        tickers_sorted = sorted(signals.keys())
        for i in range(0, len(tickers_sorted), 2):
            cols = st.columns(2)
            for j, tk in enumerate(tickers_sorted[i:i + 2]):
                with cols[j]:
                    s = signals[tk]
                    cat = s["rating_category"]
                    color = CATEGORY_COLORS.get(cat, COLORS["neutral"])
                    arrow = "▲" if s["price_change_pct"] >= 0 else "▼"
                    arrow_color = "#059669" if s["price_change_pct"] >= 0 else "#dc2626"

                    st.markdown(
                        f"""
                        <div class="signal-card">
                            <div class="head">
                                <div>
                                    <span class="ticker">{tk}</span>
                                    <span class="market">{s.get('market', '')}</span>
                                    <div class="price">
                                        {s['price']:,.2f}
                                        <span style="color:{arrow_color};font-weight:700">
                                            {arrow} {abs(s['price_change_pct']):.2f}%
                                        </span>
                                        <span style="color:#94a3b8"> · {pd.Timestamp(s['as_of']).date()}</span>
                                    </div>
                                </div>
                                <span class="rating-chip" style="background:{color}">
                                    {CATEGORY_ICONS.get(cat, '')} {cat}
                                </span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    g1, g2 = st.columns([1.1, 1])
                    with g1:
                        st.plotly_chart(
                            rating_gauge_fig(s["rating"], cat), use_container_width=True
                        )
                    with g2:
                        st.markdown(
                            vote_bar_html(s["buy_votes"], s["sell_votes"], s["neutral_votes"]),
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"MA rating **{s['ma_rating']:+.2f}** · "
                            f"Oscillator rating **{s['osc_rating']:+.2f}** · "
                            f"Sinyal: **{s['signal']}**"
                        )

                    st.markdown(
                        suggestion_box_html(cat, s["suggestion"]), unsafe_allow_html=True
                    )

                    with st.expander("Rincian voting 26 indikator"):
                        st.markdown(votes_table_html(s["vote_table"]), unsafe_allow_html=True)
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ============================================================
# TAB 2 — VALIDASI SINYAL MINGGUAN
# ============================================================
@st.cache_data(show_spinner=False, ttl=60)
def load_validation_data(db_path: str):
    """Baca run & snapshot validasi langsung dari SQLite (aman jika tabel belum ada)."""
    try:
        with sqlite3.connect(db_path) as conn:
            has_runs = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='signal_runs'"
            ).fetchone()
            if not has_runs:
                return pd.DataFrame(), pd.DataFrame()
            runs = pd.read_sql_query(
                "SELECT * FROM signal_runs ORDER BY run_id DESC", conn
            )
            snaps = pd.read_sql_query(
                "SELECT * FROM signal_snapshots ORDER BY run_id DESC, ticker", conn
            )
        return runs, snaps
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def _hit_badge(signal_type: str, outcome_pct) -> str:
    """Badge ✓ / ✗ / ⏳ untuk satu hasil evaluasi."""
    if outcome_pct is None or pd.isna(outcome_pct):
        return '<span style="color:#94a3b8">⏳ pending</span>'
    if signal_type == "HOLD":
        return '<span style="color:#94a3b8">— hold</span>'
    hit = outcome_pct > 0
    color = "#059669" if hit else "#dc2626"
    icon = "✓" if hit else "✗"
    return f'<span style="color:{color};font-weight:700">{icon} {outcome_pct:+.2f}%</span>'


with tab_valid:
    runs_df, snaps_df = load_validation_data(db_path)

    st.markdown('<div class="section-title">Pemantauan Validasi Sinyal (Siklus Mingguan)</div>',
                unsafe_allow_html=True)
    st.caption(
        "Setiap hari sistem menyimpan snapshot sinyal, lalu otomatis mengecek apakah arah sinyal "
        "terbukti benar 1, 3, dan 5 bar trading kemudian. Setelah 7 hari, run disimpulkan."
    )

    if runs_df.empty:
        st.info(
            "Belum ada run validasi. Jalankan `python orchestrator.py` (atau jadwalkan harian) — "
            "siklus validasi dimulai otomatis pada run pertama."
        )
    else:
        active = runs_df[runs_df["status"] == "active"]
        concluded = runs_df[runs_df["status"] == "concluded"]

        # ---------- RUN AKTIF ----------
        if not active.empty:
            run = active.iloc[0]
            run_id = int(run["run_id"])
            run_snaps = snaps_df[snaps_df["run_id"] == run_id].copy()

            started = pd.Timestamp(run["started_at"])
            age_days = (pd.Timestamp.now() - started).total_seconds() / 86400
            days_left = max(7.0 - age_days, 0.0)
            progress = min(age_days / 7.0, 1.0)

            directional = run_snaps[run_snaps["signal_type"].isin(["BUY", "SELL"])]
            holds = run_snaps[run_snaps["signal_type"] == "HOLD"]

            st.markdown(
                f'<div class="section-title">Run Aktif #{run_id} — hari ke-{age_days:.0f} dari 7</div>',
                unsafe_allow_html=True,
            )
            st.progress(progress, text=f"Sisa {days_left:.1f} hari sampai penyimpulan otomatis")

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Snapshot Tersimpan", len(run_snaps))
            k2.metric("Sinyal Arah (BUY/SELL)", len(directional))
            k3.metric("HOLD", len(holds))
            k4.metric("Mulai", started.strftime("%d %b %Y"))
            k5.metric("Tanggal Pasar", str(run["as_of_date"]))

            if not run_snaps.empty:
                st.markdown('<div class="section-title">Snapshot Sinyal & Hasil Validasi</div>',
                            unsafe_allow_html=True)

                rows_html = ""
                for _, s in run_snaps.iterrows():
                    sig_color = {"BUY": "#059669", "SELL": "#dc2626"}.get(s["signal_type"], "#64748b")
                    rows_html += (
                        f"<tr>"
                        f"<td><b>{s['ticker']}</b></td>"
                        f"<td style='color:{sig_color};font-weight:700'>{s['signal_type']}</td>"
                        f"<td>{s['rating_category'] or '-'}</td>"
                        f"<td>{s['rating']:+.2f}</td>"
                        f"<td>{s['price_at_signal']:,.2f}</td>"
                        f"<td>{s['signal_date']}</td>"
                        f"<td>{_hit_badge(s['signal_type'], s['day1_outcome_pct'])}</td>"
                        f"<td>{_hit_badge(s['signal_type'], s['day3_outcome_pct'])}</td>"
                        f"<td>{_hit_badge(s['signal_type'], s['day5_outcome_pct'])}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    """
                    <table class='votes'>
                    <thead><tr>
                        <th>Ticker</th><th>Sinyal</th><th>Kategori</th><th>Rating</th>
                        <th>Harga</th><th>Tanggal</th><th>Day 1</th><th>Day 3</th><th>Day 5</th>
                    </tr></thead>
                    <tbody>""" + rows_html + "</tbody></table>",
                    unsafe_allow_html=True,
                )

                # ---------- akurasi berjalan per horizon ----------
                if not directional.empty:
                    st.markdown('<div class="section-title">Akurasi Berjalan (sinyal arah)</div>',
                                unsafe_allow_html=True)
                    acc_cols = st.columns(3)
                    for col, n in zip(acc_cols, [1, 3, 5]):
                        ev = directional[directional[f"day{n}_outcome_pct"].notna()]
                        if len(ev):
                            acc = (ev[f"day{n}_outcome_pct"] > 0).mean() * 100
                            avg = ev[f"day{n}_outcome_pct"].mean()
                            col.metric(
                                f"Akurasi Day {n}", f"{acc:.0f}%",
                                delta=f"rata-rata {avg:+.2f}%",
                                delta_color="off" if abs(avg) < 0.005 else ("normal" if avg > 0 else "inverse"),
                            )
                        else:
                            col.metric(f"Akurasi Day {n}", "⏳",
                                       delta="menunggu data pasar", delta_color="off")
                    st.caption(
                        "✓ berarti arah sinyal terbukti benar pada horizon itu; "
                        "✗ berarti salah; ⏳ berarti data pasar belum tersedia."
                    )
            else:
                st.info("Belum ada snapshot pada run ini (menunggu run orchestrator berikutnya).")

        # ---------- RUN SELESAI (KESIMPULAN) ----------
        st.markdown('<div class="section-title">Run Selesai & Kesimpulan</div>', unsafe_allow_html=True)
        if concluded.empty:
            st.caption("Belum ada run yang disimpulkan — run pertama selesai setelah 7 hari siklus.")
        else:
            for _, run in concluded.iterrows():
                run_id = int(run["run_id"])
                try:
                    summary = json.loads(run["summary"]) if run["summary"] else None
                except Exception:
                    summary = None

                verdict_color = COLORS["neutral"]
                if summary:
                    ct = summary.get("conclusion_text", "")
                    if "TERBUKTI VALID" in ct:
                        verdict_color = CATEGORY_COLORS["STRONG BUY"]
                    elif "BELUM VALID" in ct:
                        verdict_color = CATEGORY_COLORS["STRONG SELL"]
                    elif "KURANG VALID" in ct:
                        verdict_color = CATEGORY_COLORS["SELL"]

                header = (
                    f"Run #{run_id} · {run['as_of_date']} — "
                    + (summary["conclusion_text"][:90] + "…" if summary and len(summary.get("conclusion_text", "")) > 90
                       else (summary.get("conclusion_text", "") if summary else "disimpulkan"))
                )
                with st.expander(header):
                    if summary:
                        st.markdown(
                            suggestion_box_html(
                                "STRONG BUY" if verdict_color == CATEGORY_COLORS["STRONG BUY"]
                                else "STRONG SELL" if verdict_color == CATEGORY_COLORS["STRONG SELL"]
                                else "SELL" if verdict_color == CATEGORY_COLORS["SELL"]
                                else "NEUTRAL",
                                summary["conclusion_text"],
                            ),
                            unsafe_allow_html=True,
                        )

                        bh = summary.get("by_horizon", {})
                        if bh:
                            c1, c2, c3 = st.columns(3)
                            for col, n in zip([c1, c2, c3], ["1d", "3d", "5d"]):
                                h = bh.get(n, {})
                                if h.get("accuracy_pct") is not None:
                                    col.metric(
                                        f"Akurasi {n}", f"{h['accuracy_pct']}%",
                                        delta=f"{h['hits']}/{h['evaluated']} sinyal · avg {h.get('avg_outcome_pct', 0):+.2f}%",
                                        delta_color="off",
                                    )
                                else:
                                    col.metric(f"Akurasi {n}", "—", delta="tidak ada data", delta_color="off")

                        bc, bt = st.columns(2)
                        with bc:
                            st.markdown("**Akurasi per Kategori Rating**")
                            cat_rows = summary.get("by_category", {})
                            if cat_rows:
                                st.dataframe(
                                    pd.DataFrame([
                                        {"Kategori": k, "Sinyal": v["signals"],
                                         "Dievaluasi": v["evaluated"], "Benar": v["hits"],
                                         "Akurasi %": v["accuracy_pct"]}
                                        for k, v in cat_rows.items()
                                    ]),
                                    hide_index=True, use_container_width=True,
                                )
                        with bt:
                            st.markdown("**Akurasi per Ticker**")
                            tk_rows = summary.get("by_ticker", {})
                            if tk_rows:
                                st.dataframe(
                                    pd.DataFrame([
                                        {"Ticker": k, "Sinyal": v["signals"], "Akurasi %": v["accuracy_pct"]}
                                        for k, v in tk_rows.items()
                                    ]),
                                    hide_index=True, use_container_width=True,
                                )
                    else:
                        st.caption("Ringkasan tidak tersedia untuk run ini.")

# ============================================================
# TAB 3 — CHART & TEKNIKAL
# ============================================================
with tab_chart:
    st.markdown('<div class="section-title">Candlestick dengan Marker Sinyal & Overlay Indikator</div>',
                unsafe_allow_html=True)

    if ohlcv.empty:
        st.info(
            f"Tabel `{CONFIG['ohlcv_table']}` tidak ditemukan/kosong — chart candlestick "
            "butuh data OHLCV historis. Sesuaikan CONFIG di dashboard.py jika nama "
            "tabelnya berbeda."
        )
    else:
        ticker_options = sorted(ohlcv["ticker"].dropna().unique().tolist())
        if not ticker_options:
            st.info("Tidak ada ticker pada tabel OHLCV.")
        else:
            sel = st.selectbox("Pilih ticker", ticker_options)
            ohlcv_t = ohlcv[ohlcv["ticker"] == sel].sort_values("timestamp")
            trades_t = trades[trades["ticker"] == sel] if "ticker" in trades else pd.DataFrame()

            # Overlay indikator dihitung dari data OHLCV ticker terpilih.
            features_t = pd.DataFrame()
            try:
                features_t = build_feature_set(ohlcv_t)
            except Exception:
                pass

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                show_ma = st.checkbox("Moving Averages (20/50/200)", value=True)
            with c2:
                show_bb = st.checkbox("Bollinger Bands", value=False)
            with c3:
                show_volume = st.checkbox("Volume", value=True)
            with c4:
                show_levels = st.checkbox("Level SL/TP (open trades)", value=True)

            fig_c = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.78, 0.22] if show_volume else [1.0],
                vertical_spacing=0.02,
            )

            fig_c.add_trace(
                go.Candlestick(
                    x=ohlcv_t["timestamp"],
                    open=ohlcv_t["open"], high=ohlcv_t["high"],
                    low=ohlcv_t["low"], close=ohlcv_t["close"],
                    name=sel,
                    increasing_line_color="#10b981", decreasing_line_color="#ef4444",
                    increasing_fillcolor="#10b981", decreasing_fillcolor="#ef4444",
                ),
                row=1, col=1,
            )

            if show_ma and not features_t.empty:
                for col, cname, ccolor in [
                    ("sma_20", "SMA20", "#94a3b8"),
                    ("sma_50", "SMA50", "#f59e0b"),
                    ("sma_200", "SMA200", "#2563eb"),
                ]:
                    if col in features_t.columns:
                        fig_c.add_trace(
                            go.Scatter(
                                x=features_t["timestamp"], y=features_t[col],
                                name=cname, line=dict(width=1.4, color=ccolor),
                                hovertemplate=f"{cname}" + ": %{y:.2f}<extra></extra>",
                            ),
                            row=1, col=1,
                        )

            if show_bb and not features_t.empty and "bb_upper" in features_t.columns:
                band_fill = "rgba(37, 99, 235, 0.06)"
                for col, cname in [("bb_upper", "BB Upper"), ("bb_lower", "BB Lower")]:
                    fig_c.add_trace(
                        go.Scatter(
                            x=features_t["timestamp"], y=features_t[col],
                            name=cname, line=dict(width=1, color="rgba(37,99,235,0.5)", dash="dot"),
                        ),
                        row=1, col=1,
                    )
                fig_c.add_trace(
                    go.Scatter(
                        x=features_t["timestamp"], y=features_t["bb_middle"],
                        name="BB Mid", line=dict(width=0.8, color="rgba(37,99,235,0.35)"),
                        fill="tonexty" if False else None,
                    ),
                    row=1, col=1,
                )

            if show_volume and not ohlcv_t.empty:
                vol_colors = np.where(
                    ohlcv_t["close"] >= ohlcv_t["open"], "#10b98155", "#ef444455"
                )
                fig_c.add_trace(
                    go.Bar(
                        x=ohlcv_t["timestamp"], y=ohlcv_t["volume"],
                        name="Volume", marker_color=vol_colors, showlegend=False,
                    ),
                    row=2, col=1,
                )

            if not trades_t.empty:
                buy_entries = trades_t[trades_t["signal"].astype(str).str.upper() == "BUY"]
                sell_entries = trades_t[trades_t["signal"].astype(str).str.upper() == "SELL"]
                exits = trades_t[trades_t["status"] == "CLOSED"]

                if not buy_entries.empty:
                    fig_c.add_trace(
                        go.Scatter(
                            x=buy_entries["entry_date"], y=buy_entries["entry_price"],
                            mode="markers", name="BUY entry",
                            marker=dict(symbol="triangle-up", size=13, color="#059669",
                                        line=dict(width=1, color="white")),
                            text=buy_entries["confidence"].apply(
                                lambda c: f"confidence={c:.2f}" if pd.notna(c) else ""
                            ),
                            hovertemplate="BUY entry<br>%{x}<br>Harga: %{y}<br>%{text}<extra></extra>",
                        ),
                        row=1, col=1,
                    )
                if not sell_entries.empty:
                    fig_c.add_trace(
                        go.Scatter(
                            x=sell_entries["entry_date"], y=sell_entries["entry_price"],
                            mode="markers", name="SELL entry",
                            marker=dict(symbol="triangle-down", size=13, color="#dc2626",
                                        line=dict(width=1, color="white")),
                            text=sell_entries["confidence"].apply(
                                lambda c: f"confidence={c:.2f}" if pd.notna(c) else ""
                            ),
                            hovertemplate="SELL entry<br>%{x}<br>Harga: %{y}<br>%{text}<extra></extra>",
                        ),
                        row=1, col=1,
                    )
                if not exits.empty:
                    fig_c.add_trace(
                        go.Scatter(
                            x=exits["exit_date"], y=exits["exit_price"],
                            mode="markers", name="Exit",
                            marker=dict(symbol="x", size=10, color="#0f172a"),
                            text=exits["pnl"].apply(
                                lambda p: f"pnl={p:,.0f}" if pd.notna(p) else ""
                            ),
                            hovertemplate="Exit<br>%{x}<br>Harga: %{y}<br>%{text}<extra></extra>",
                        ),
                        row=1, col=1,
                    )

                # Level SL/TP untuk posisi yang masih open.
                if show_levels:
                    opens = trades_t[trades_t["status"] == "OPEN"]
                    for _, ot in opens.iterrows():
                        for lvl_col, lvl_name, lvl_color in [
                            ("stop_loss", "SL", "#dc2626"),
                            ("take_profit", "TP", "#059669"),
                        ]:
                            if lvl_col in ot and pd.notna(ot[lvl_col]):
                                fig_c.add_hline(
                                    y=float(ot[lvl_col]), line_dash="dash",
                                    line_color=lvl_color, line_width=1,
                                    annotation_text=f"{lvl_name} {ot['ticker']}",
                                    annotation_font_size=9,
                                    row=1, col=1,
                                )

            fig_c.update_layout(
                height=620, xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.01,
                            xanchor="left", x=0, font=dict(size=11)),
                hovermode="x unified",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig_c.update_yaxes(title_text="Harga", row=1, col=1, gridcolor="#eef2f7")
            fig_c.update_xaxes(gridcolor="#eef2f7")
            st.plotly_chart(fig_c, use_container_width=True)

# ============================================================
# TAB 4 — TRADES
# ============================================================
with tab_trades:
    st.markdown('<div class="section-title">Semua Trades (Open & Closed)</div>',
                unsafe_allow_html=True)

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        tickers = sorted(trades["ticker"].dropna().unique().tolist()) if "ticker" in trades else []
        sel_ticker = st.multiselect("Ticker", tickers, default=[])
    with col_f2:
        markets = sorted(trades["market"].dropna().unique().tolist()) if "market" in trades else []
        sel_market = st.multiselect("Market", markets, default=[])
    with col_f3:
        sel_status = st.multiselect("Status", ["OPEN", "CLOSED"], default=[])
    with col_f4:
        if "entry_date" in trades and trades["entry_date"].notna().any():
            min_d = trades["entry_date"].min().date()
            max_d = trades["entry_date"].max().date()
            date_range = st.date_input("Rentang tanggal entry", value=(min_d, max_d))
        else:
            date_range = None

    filtered = trades.copy()
    if sel_ticker:
        filtered = filtered[filtered["ticker"].isin(sel_ticker)]
    if sel_market:
        filtered = filtered[filtered["market"].isin(sel_market)]
    if sel_status:
        filtered = filtered[filtered["status"].isin(sel_status)]
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
        filtered = filtered[
            (filtered["entry_date"].dt.date >= start_d)
            & (filtered["entry_date"].dt.date <= end_d)
        ]

    st.dataframe(
        filtered.sort_values("entry_date", ascending=False),
        use_container_width=True,
        column_config={
            "entry_price": st.column_config.NumberColumn("Entry", format="%.2f"),
            "exit_price": st.column_config.NumberColumn("Exit", format="%.2f"),
            "stop_loss": st.column_config.NumberColumn("SL", format="%.2f"),
            "take_profit": st.column_config.NumberColumn("TP", format="%.2f"),
            "confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "pnl": st.column_config.NumberColumn("PnL", format="%.0f"),
        },
        hide_index=True,
    )
    st.caption(f"Menampilkan {len(filtered)} dari {len(trades)} total trade.")

# ============================================================
# TAB 5 — EQUITY CURVE
# ============================================================
with tab_equity:
    st.markdown('<div class="section-title">Equity Curve Simulasi (mengikuti semua sinyal)</div>',
                unsafe_allow_html=True)
    equity_df = compute_equity_curve(closed_trades, starting_capital)

    if equity_df.empty:
        st.info("Belum ada trade CLOSED untuk membentuk equity curve.")
    else:
        final_equity = equity_df["equity"].iloc[-1]
        total_return_pct = (final_equity - starting_capital) / starting_capital * 100
        max_dd = compute_max_drawdown(equity_df)

        k1, k2, k3 = st.columns(3)
        k1.metric("Equity Akhir", f"{final_equity:,.0f}",
                  delta=f"{total_return_pct:+.2f}%")
        k2.metric("Return Total", f"{total_return_pct:,.2f}%")
        k3.metric("Max Drawdown", f"{max_dd*100:,.1f}%")

        eq_fig = go.Figure()
        eq_fig.add_trace(
            go.Scatter(
                x=equity_df["date"], y=equity_df["equity"],
                mode="lines", name="Equity",
                line=dict(color=COLORS["accent"], width=2.4),
                fill="tozeroy" if False else None,
            )
        )
        eq_fig.add_hline(y=starting_capital, line_dash="dash", line_color="#94a3b8",
                         annotation_text="Modal awal", annotation_font_size=10)
        eq_fig.update_layout(
            height=430, margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Portfolio Value", xaxis_title="Tanggal",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        eq_fig.update_yaxes(gridcolor="#eef2f7")
        eq_fig.update_xaxes(gridcolor="#eef2f7")
        st.plotly_chart(eq_fig, use_container_width=True)

# ============================================================
# TAB 6 — METRIK
# ============================================================
with tab_metrics:
    st.markdown('<div class="section-title">Metrik Ringkasan</div>', unsafe_allow_html=True)
    closed = closed_trades

    if closed.empty:
        st.info("Belum ada trade CLOSED untuk dihitung metriknya.")
    else:
        win_rate = (closed["pnl"] > 0).mean()
        avg_r = closed["r_multiple"].mean(skipna=True)
        equity_df = compute_equity_curve(closed, starting_capital)
        max_dd = compute_max_drawdown(equity_df)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Win Rate", f"{win_rate*100:,.1f}%")
        m2.metric(
            "Avg Risk-Reward Realized",
            f"{avg_r:,.2f} R" if pd.notna(avg_r) else "N/A",
        )
        m3.metric("Max Drawdown", f"{max_dd*100:,.1f}%")
        m4.metric("Total Trade Closed", f"{len(closed)}")

        st.markdown('<div class="section-title">Win Rate per Bucket Confidence Score</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Cek apakah confidence score yang lebih tinggi memang berkorelasi "
            "dengan win rate yang lebih tinggi."
        )
        bucket_df = confidence_bucket_stats(closed)

        if bucket_df.empty:
            st.info("Kolom confidence_score tidak tersedia atau kosong.")
        else:
            fig_bucket = go.Figure()
            fig_bucket.add_trace(
                go.Bar(
                    x=bucket_df["bucket"].astype(str),
                    y=bucket_df["win_rate"] * 100,
                    name="Win Rate (%)",
                    marker_color=COLORS["accent"],
                    text=bucket_df["n_signals"].apply(lambda n: f"n={n}"),
                    textposition="outside",
                )
            )
            fig_bucket.update_layout(
                xaxis_title="Confidence Score Bucket",
                yaxis_title="Win Rate (%)",
                height=400,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_bucket.update_yaxes(gridcolor="#eef2f7")
            st.plotly_chart(fig_bucket, use_container_width=True)
            st.dataframe(bucket_df, use_container_width=True, hide_index=True)

st.markdown(
    "<div style='text-align:center;color:#94a3b8;font-size:0.78rem;padding:18px 0 6px 0'>"
    "ASM Market Monitor · data via yfinance · engine: multi-indicator voting "
    "(TradingView/Investing.com methodology) · bukan nasihat keuangan</div>",
    unsafe_allow_html=True,
)
