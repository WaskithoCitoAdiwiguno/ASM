"""
signal_engine.py

Multi-indicator voting/confluence signal engine — mereplikasi metodologi
"Technical Ratings" ala TradingView dan "Technical Summary" ala
Investing.com, yang dipakai platform publik untuk mengirim sinyal
BUY/SELL ke user.

Metodologi (hasil riset terhadap sistem yang sudah ada):
---------------------------------------------------------------------
1. TradingView Technical Ratings (https://www.tradingview.com/support/
   solutions/43000614331-technical-ratings/):
   - Setiap indikator konstituen memberi SUARA: -1 (sell), 0 (neutral),
     atau +1 (buy) pada setiap bar.
   - Rating = rata-rata suara, bergerak di antara -1 dan +1.
   - Kategori: Strong Sell (< -0.5), Sell (< -0.1), Neutral (-0.1..0.1),
     Buy (> 0.1), Strong Buy (> 0.5).
   - Grup Moving Averages (15): SMA 10/20/30/50/100/200, EMA 10/20/30/50/
     100/200, Hull MA(9), VWMA(20), Ichimoku Cloud.
     Aturan semua MA  : buy jika price > MA, sell jika price < MA.
   - Grup Oscillators (11): RSI(14), Stochastic(14,3,3), CCI(20),
     ADX(14,14), Awesome Oscillator, Momentum(10), MACD(12,26,9),
     Stochastic RSI(3,3,14,14), Williams %R(14), Bull Bear Power(13),
     Ultimate Oscillator(7,14,28) — masing-masing dengan aturan
     buy/sell/neutral yang spesifik (lihat _VOTE_RULES di bawah).
2. Investing.com Technical Summary: menghitung BERAPA indikator yang
   setuju tiap arah (mis. "10 Buy / 2 Sell = Strong Buy") lalu menampilkan
   gauge per indikator + summary 5 kategori yang sama.
3. Sinyal final dikirim ke user dalam bentuk: rating angka, kategori
   5 tingkat, rincian suara per indikator, dan alasan text (rekomendasi)
   — persis seperti tabel "Technicals" pada platform tersebut.

Output engine ini kompatibel dengan modul lain di sistem:
    - risk_manager.py          -> pakai key "direction"
    - paper_trading_simulator  -> pakai key "signal_type",
                                  "confidence_score", "triggered_conditions"
API lama berbasis skor aditif tetap tersedia lewat `generate_signal_legacy`.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# Rating thresholds (mengikuti TradingView Technical Ratings)
# ============================================================
RATING_THRESHOLDS = {
    "strong_sell": -0.5,
    "sell": -0.1,
    "neutral": 0.1,
    "buy": 0.5,
}  # <= -0.5 Strong Sell; <= -0.1 Sell; < 0.1 Neutral; < 0.5 Buy; else Strong Buy


DEFAULT_CONFIG: Dict[str, Any] = {
    # Mapping nama kolom logis -> nama kolom aktual di DataFrame.
    "columns": {
        "timestamp": "timestamp",
        "ticker": "ticker",
        "close": "close",
        "close_prev": "close_prev",
        "rsi": "rsi_14",
        "macd": "macd",
        "macd_signal": "macd_signal",
        "macd_hist": "macd_hist",
        "macd_hist_prev": "macd_hist_prev",
        "sma_200": "sma_200",
        "sma_50": "sma_50",
        "volume_spike_ratio": "volume_spike_ratio",
        "bb_lower": "bb_lower",
        "bb_upper": "bb_upper",
    },
    # Mode rating/voting (baru, default).
    "voting": {
        "min_votes_to_signal": 4,   # minimal selisih (buy - sell) untuk memicu sinyal
    },
    # Level osilator utk mode voting (mengikuti standar TradingView).
    "groups": {
        "rsi_overbought": 70,
        "rsi_oversold": 30,
    },
    # Konfigurasi engine lama (dipakai generate_signal_legacy).
    "buy": {
        "threshold": 0.5,
        "rsi_oversold_level": 30,
        "volume_spike_ratio_threshold": 1.5,
        "bb_tolerance_pct": 0.002,
        "weights": {
            "rsi_oversold": 0.25,
            "macd_bullish_crossover": 0.30,
            "uptrend_above_sma200": 0.20,
            "volume_spike": 0.15,
            "near_bb_lower": 0.10,
        },
    },
    "sell": {
        "threshold": 0.5,
        "rsi_overbought_level": 70,
        "volume_spike_ratio_threshold": 1.5,
        "bb_tolerance_pct": 0.002,
        "weights": {
            "rsi_overbought": 0.25,
            "macd_bearish_crossover": 0.30,
            "weak_below_sma50": 0.20,
            "near_bb_upper": 0.15,
            "volume_spike_distribution": 0.10,
        },
    },
}


# Shorthand untuk aturan per-indikator ala TradingView.
# Setiap rule: (nama_tampilan, kolom_data, fungsi_vote) -> +1 buy / -1 sell / 0 neutral
def _make_vote_rules(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    cols = config["columns"]

    def g(row: pd.Series, logical: str) -> Any:
        col = cols.get(logical, logical)
        if col not in row.index or pd.isna(row[col]):
            return np.nan
        return row[col]

    def _grp(key: str, default: float) -> float:
        return float(config.get("groups", {}).get(key, default))

    # --- Moving averages: buy jika price > MA, sell jika price < MA (aturan TV) ---
    rules: List[Dict[str, Any]] = []
    ma_specs = [
        ("SMA10", "sma_10"), ("SMA20", "sma_20"), ("SMA30", "sma_30"),
        ("SMA50", "sma_50"), ("SMA100", "sma_100"), ("SMA200", "sma_200"),
        ("EMA10", "ema_10"), ("EMA20", "ema_20"), ("EMA30", "ema_30"),
        ("EMA50", "ema_50"), ("EMA100", "ema_100"), ("EMA200", "ema_200"),
        ("HullMA9", "hull_9"), ("VWMA20", "vwma_20"),
    ]
    for name, col in ma_specs:
        col_actual = cols.get(col, col)

        def vote(row: pd.Series, _col: str = col_actual) -> int:
            close, ma = g(row, "close"), g(row, _col)
            if not _valid(close, ma):
                return 0
            return 1 if close > ma else (-1 if close < ma else 0)

        rules.append({"group": "ma", "name": name, "vote": vote})

    def vote_ichimoku(row: pd.Series) -> int:
        close = g(row, "close")
        conv, base = g(row, "ichimoku_conversion"), g(row, "ichimoku_base")
        span_a, span_b = g(row, "ichimoku_span_a"), g(row, "ichimoku_span_b")
        if not _valid(close, conv, base, span_a, span_b):
            return 0
        if span_a > span_b and base > span_a and conv > base and close > conv:
            return 1
        if span_a < span_b and base < span_a and conv < base and close < conv:
            return -1
        return 0

    rules.append({"group": "ma", "name": "Ichimoku Cloud", "vote": vote_ichimoku})

    # --- Oscillators (aturan buy/sell mengikuti TradingView) ---
    def vote_rsi(row: pd.Series) -> int:
        lvl = _grp("rsi_overbought", 70), _grp("rsi_oversold", 30)
        rsi, rsi_prev = g(row, "rsi"), g(row, "rsi_prev")
        if not _valid(rsi):
            return 0
        if rsi < lvl[1] and _valid(rsi_prev) and rsi > rsi_prev:
            return 1
        if rsi > lvl[0] and _valid(rsi_prev) and rsi < rsi_prev:
            return -1
        return 0

    def vote_stoch(row: pd.Series) -> int:
        k, d = g(row, "stoch_k"), g(row, "stoch_d")
        if not _valid(k, d):
            return 0
        if k < 20 and d < 20 and k > d:
            return 1
        if k > 80 and d > 80 and k < d:
            return -1
        return 0

    def vote_cci(row: pd.Series) -> int:
        cci, cci_prev = g(row, "cci_20"), g(row, "cci_20_prev")
        if not _valid(cci):
            return 0
        if cci < -100 and _valid(cci_prev) and cci > cci_prev:
            return 1
        if cci > 100 and _valid(cci_prev) and cci < cci_prev:
            return -1
        return 0

    def vote_adx(row: pd.Series) -> int:
        adx, adx_prev = g(row, "adx"), g(row, "adx_prev")
        plus_di, minus_di = g(row, "adx_pos"), g(row, "adx_neg")
        if not _valid(adx, plus_di, minus_di):
            return 0
        if plus_di > minus_di and adx > 20 and _valid(adx_prev) and adx > adx_prev:
            return 1
        if plus_di < minus_di and adx > 20 and _valid(adx_prev) and adx < adx_prev:
            return -1
        return 0

    def vote_ao(row: pd.Series) -> int:
        ao = g(row, "awesome_oscillator")
        if not _valid(ao):
            return 0
        return 1 if ao > 0 else (-1 if ao < 0 else 0)

    def vote_momentum(row: pd.Series) -> int:
        mom, mom_prev = g(row, "momentum_10"), g(row, "momentum_10_prev")
        if not _valid(mom):
            return 0
        if _valid(mom_prev):
            return 1 if mom > mom_prev else (-1 if mom < mom_prev else 0)
        return 1 if mom > 0 else (-1 if mom < 0 else 0)

    def vote_macd(row: pd.Series) -> int:
        macd, sig = g(row, "macd"), g(row, "macd_signal")
        if not _valid(macd, sig):
            return 0
        return 1 if macd > sig else (-1 if macd < sig else 0)

    def vote_stochrsi(row: pd.Series) -> int:
        k, d = g(row, "stochrsi_k"), g(row, "stochrsi_d")
        if not _valid(k, d):
            return 0
        if k < 20 and d < 20 and k > d:
            return 1
        if k > 80 and d > 80 and k < d:
            return -1
        return 0

    def vote_willr(row: pd.Series) -> int:
        wr, wr_prev = g(row, "williams_r"), g(row, "williams_r_prev")
        if not _valid(wr):
            return 0
        if wr < -80 and _valid(wr_prev) and wr > wr_prev:
            return 1
        if wr > -20 and _valid(wr_prev) and wr < wr_prev:
            return -1
        return 0

    def vote_bbpower(row: pd.Series) -> int:
        bull, bull_prev = g(row, "bull_power"), g(row, "bull_power_prev")
        bear, bear_prev = g(row, "bear_power"), g(row, "bear_power_prev")
        if not _valid(bull, bear):
            return 0
        # Trend filter sederhana: close vs SMA50 (pengganti "uptrend/downtrend" TV)
        close, sma50 = g(row, "close"), g(row, "sma_50")
        if not _valid(close, sma50):
            return 0
        uptrend, downtrend = close > sma50, close < sma50
        if uptrend and bear < 0 and _valid(bear_prev) and bear > bear_prev:
            return 1
        if downtrend and bull > 0 and _valid(bull_prev) and bull < bull_prev:
            return -1
        return 0

    def vote_uo(row: pd.Series) -> int:
        uo = g(row, "ultimate_oscillator")
        if not _valid(uo):
            return 0
        if uo > 70:
            return 1
        if uo < 30:
            return -1
        return 0

    osc_specs = [
        ("RSI(14)", vote_rsi),
        ("Stoch %K(14,3,3)", vote_stoch),
        ("CCI(20)", vote_cci),
        ("ADX(14)", vote_adx),
        ("Awesome Osc", vote_ao),
        ("Momentum(10)", vote_momentum),
        ("MACD(12,26,9)", vote_macd),
        ("Stoch RSI Fast(3,3,14,14)", vote_stochrsi),
        ("Williams %R(14)", vote_willr),
        ("Bull Bear Power", vote_bbpower),
        ("Ultimate Osc(7,14,28)", vote_uo),
    ]
    for name, fn in osc_specs:
        rules.append({"group": "osc", "name": name, "vote": fn})

    return rules


def _valid(*values: Any) -> bool:
    """True jika semua value bukan NaN/None (kondisi bisa dievaluasi)."""
    for v in values:
        if v is None:
            return False
        if isinstance(v, float) and np.isnan(v):
            return False
        if pd.isna(v):
            return False
    return True


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge `override` ke dalam salinan `base`. Tidak memutasi input."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _rating_category(rating: float) -> str:
    """Map rating [-1, +1] ke kategori 5 tingkat (logika TradingView)."""
    if rating <= RATING_THRESHOLDS["strong_sell"]:
        return "STRONG SELL"
    if rating <= RATING_THRESHOLDS["sell"]:
        return "SELL"
    if rating < RATING_THRESHOLDS["neutral"]:
        return "NEUTRAL"
    if rating < RATING_THRESHOLDS["buy"]:
        return "BUY"
    return "STRONG BUY"


class SignalEngine:
    """
    Voting/confluence signal engine (baseline, bukan ML).

    Mode default: multi-indicator voting ala TradingView Technical Ratings
    & Investing.com Technical Summary:
        - 15 moving averages + 11 oscillators memberi suara -1/0/+1.
        - rating = mean(votes) dalam rentang [-1, +1].
        - kategori: STRONG SELL / SELL / NEUTRAL / BUY / STRONG BUY.
        - sinyal BUY/SELL dihasilkan jika selisih suara mencapai
          `voting.min_votes_to_signal`.

    Parameters
    ----------
    config : dict, optional
        Override sebagian/seluruh DEFAULT_CONFIG (mapping kolom, threshold
        voting, level osilator, dll). Akan di-deep-merge di atas default.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = (
            _deep_merge(DEFAULT_CONFIG, config) if config else copy.deepcopy(DEFAULT_CONFIG)
        )
        self._vote_rules = _make_vote_rules(self.config)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _get_val(self, row: pd.Series, logical_name: str) -> Any:
        """Ambil nilai kolom lewat mapping config['columns']; NaN jika tidak ada."""
        col_name = self.config["columns"].get(logical_name, logical_name)
        if col_name not in row.index or pd.isna(row[col_name]):
            return np.nan
        return row[col_name]

    # ------------------------------------------------------------------ #
    # Voting engine (mode utama — ala TradingView / Investing.com)
    # ------------------------------------------------------------------ #
    def evaluate_row(self, row: pd.Series) -> Dict[str, Any]:
        """
        Evaluasi satu baris (dengan indikator lengkap) menjadi rating & sinyal.

        Returns
        -------
        dict dengan key:
            signal                : "BUY" | "SELL" | "HOLD"
            rating                : float [-1, +1]
            rating_category       : STRONG SELL / SELL / NEUTRAL / BUY / STRONG BUY
            ma_rating, osc_rating : rating per grup
            buy_votes, sell_votes, neutral_votes
            vote_table            : list[{name, group, vote, label}]
            suggestion            : kalimat rekomendasi siap tampilkan
            reasons               : list alasan (indikator yang setuju)
            confidence_score      : normalisasi rating ke 0..1 (kompatibel lama)
            triggered_conditions  : alias reasons (kompatibel lama)
            direction, signal_type: kompatibel modul lain
            timestamp, ticker
        """
        close = self._get_val(row, "close")
        votes: List[Dict[str, Any]] = []

        for rule in self._vote_rules:
            try:
                v = int(rule["vote"](row))
            except Exception:
                v = 0
            votes.append({
                "name": rule["name"],
                "group": rule["group"],
                "vote": v,
                "label": {1: "Buy", 0: "Neutral", -1: "Sell"}[v],
            })

        n_votes = len(votes)
        buy_votes = sum(1 for v in votes if v["vote"] > 0)
        sell_votes = sum(1 for v in votes if v["vote"] < 0)
        neutral_votes = n_votes - buy_votes - sell_votes

        ma_votes = [v["vote"] for v in votes if v["group"] == "ma"]
        osc_votes = [v["vote"] for v in votes if v["group"] == "osc"]
        ma_rating = float(np.mean(ma_votes)) if ma_votes else 0.0
        osc_rating = float(np.mean(osc_votes)) if osc_votes else 0.0
        rating = float(np.mean([v["vote"] for v in votes])) if n_votes else 0.0

        category = _rating_category(rating)

        # Sinyal eksplisit butuh selisih suara minimal (min_votes_to_signal).
        min_diff = int(self.config["voting"].get("min_votes_to_signal", 4))
        vote_diff = buy_votes - sell_votes
        if vote_diff >= min_diff:
            signal = "BUY"
        elif vote_diff <= -min_diff:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Confidence score 0..1: seberapa banyak indikator searah dengan sinyal.
        if n_votes:
            directional = max(buy_votes, sell_votes)
            confidence = round(directional / n_votes, 6)
        else:
            confidence = 0.0

        # Reasons: indikator yang setuju dengan arah dominan.
        dominant = 1 if buy_votes >= sell_votes else -1
        group_label = {"ma": "MA", "osc": "OSC"}
        reasons = [
            f"{v['name']} ({group_label.get(v['group'], v['group'])}): {v['label']}"
            for v in votes
            if v["vote"] != 0 and (v["vote"] > 0) == (dominant > 0)
        ]

        suggestion = self._build_suggestion(
            signal=signal, category=category, rating=rating,
            buy_votes=buy_votes, sell_votes=sell_votes,
            neutral_votes=neutral_votes, reasons=reasons, close=close,
        )

        return {
            "signal": signal,
            "rating": round(rating, 6),
            "rating_category": category,
            "ma_rating": round(ma_rating, 6),
            "osc_rating": round(osc_rating, 6),
            "buy_votes": buy_votes,
            "sell_votes": sell_votes,
            "neutral_votes": neutral_votes,
            "vote_table": votes,
            "suggestion": suggestion,
            "reasons": reasons,
            # ---- kompatibilitas dengan modul lain ----
            "confidence_score": float(confidence),
            "triggered_conditions": reasons,
            "direction": signal if signal != "HOLD" else None,   # risk_manager.py
            "signal_type": signal,                               # paper_trading_simulator.py
            "timestamp": self._get_val(row, "timestamp"),
            "ticker": self._get_val(row, "ticker"),
        }

    def _build_suggestion(
        self,
        signal: str,
        category: str,
        rating: float,
        buy_votes: int,
        sell_votes: int,
        neutral_votes: int,
        reasons: List[str],
        close: Any,
    ) -> str:
        """Susun kalimat rekomendasi gaya platform publik (bahasa Indonesia)."""
        vote_summary = f"{buy_votes} Buy / {sell_votes} Sell / {neutral_votes} Neutral"
        close_txt = f"{close:,.2f}" if _valid(close) else "harga saat ini"

        if category in ("STRONG BUY", "BUY"):
            headline = (
                f"{category} — {vote_summary}. Sebagian besar indikator teknikal "
                f"menunjukkan tekanan beli pada {close_txt}."
            )
            action = (
                "Pertimbangkan posisi beli dengan stop loss di bawah support terdekat; "
                "sinyal dikirim karena konfluensi indikator sudah melewati ambang voting."
                if signal == "BUY"
                else "Rating condong bullish namun belum melewati ambang konfirmasi voting — tunggu konfirmasi tambahan sebelum entry."
            )
        elif category in ("STRONG SELL", "SELL"):
            headline = (
                f"{category} — {vote_summary}. Sebagian besar indikator teknikal "
                f"menunjukkan tekanan jual pada {close_txt}."
            )
            action = (
                "Pertimbangkan mengurangi posisi / hindari entry baru; sinyal jual "
                "terkirim karena konfluensi indikator sudah melewati ambang voting."
                if signal == "SELL"
                else "Rating condong bearish namun belum melewati ambang konfirmasi voting — tunggu konfirmasi tambahan."
            )
        else:
            headline = (
                f"NEUTRAL — {vote_summary}. Indikator teknikal tidak menunjukkan "
                f"keunggulan arah yang jelas pada {close_txt}."
            )
            action = "Tidak ada sinyal — disarankan menunggu (HOLD) sampai konfluensi indikator terbentuk."

        top_reasons = "; ".join(reasons[:3]) + ("; dst." if len(reasons) > 3 else "")
        if top_reasons:
            return f"{headline} {action} Alasan utama: {top_reasons}."
        return f"{headline} {action}"

    # ------------------------------------------------------------------ #
    # Public API utama
    # ------------------------------------------------------------------ #
    def generate_signal(self, row: pd.Series) -> Dict[str, Any]:
        """
        Hasilkan sinyal untuk satu baris data yang sudah berisi indikator
        teknikal (gunakan build_feature_set / build_extended_feature_set).

        Kode lama yang memanggil generate_signal() akan otomatis mendapat
        format baru; field lama tetap tersedia (signal_type, direction,
        confidence_score, triggered_conditions).
        """
        return self.evaluate_row(row)

    # ------------------------------------------------------------------ #
    # Engine lama (skor aditif) — disimpan untuk perbandingan/backtest
    # ------------------------------------------------------------------ #
    def generate_signal_legacy(self, row: pd.Series) -> Dict[str, Any]:
        """Sinyal versi lama berbasis skor aditif (5 kondisi BUY, 5 SELL)."""
        buy_score, buy_triggers = self._score_buy(row)
        sell_score, sell_triggers = self._score_sell(row)

        buy_valid = buy_score >= self.config["buy"]["threshold"]
        sell_valid = sell_score >= self.config["sell"]["threshold"]

        if buy_valid and sell_valid:
            if buy_score >= sell_score:
                signal, confidence, triggered = "BUY", buy_score, buy_triggers
            else:
                signal, confidence, triggered = "SELL", sell_score, sell_triggers
        elif buy_valid:
            signal, confidence, triggered = "BUY", buy_score, buy_triggers
        elif sell_valid:
            signal, confidence, triggered = "SELL", sell_score, sell_triggers
        else:
            signal = "HOLD"
            confidence = round(max(buy_score, sell_score), 6)
            triggered = buy_triggers + sell_triggers

        return {
            "signal": signal,
            "confidence_score": float(confidence),
            "triggered_conditions": triggered,
            "timestamp": self._get_val(row, "timestamp"),
            "ticker": self._get_val(row, "ticker"),
            "direction": signal if signal != "HOLD" else None,
            "signal_type": signal,
        }

    def _score_buy(self, row: pd.Series) -> tuple[float, List[str]]:
        cfg = self.config["buy"]
        w = cfg["weights"]
        score = 0.0
        triggered: List[str] = []

        rsi = self._get_val(row, "rsi")
        if _valid(rsi) and rsi < cfg["rsi_oversold_level"]:
            score += w["rsi_oversold"]
            triggered.append(f"RSI oversold ({rsi:.2f} < {cfg['rsi_oversold_level']})")

        macd_hist = self._get_val(row, "macd_hist")
        macd_hist_prev = self._get_val(row, "macd_hist_prev")
        if _valid(macd_hist, macd_hist_prev) and macd_hist_prev < 0 <= macd_hist:
            score += w["macd_bullish_crossover"]
            triggered.append("MACD histogram bullish crossover (neg -> pos)")

        close = self._get_val(row, "close")
        sma_200 = self._get_val(row, "sma_200")
        if _valid(close, sma_200) and close > sma_200:
            score += w["uptrend_above_sma200"]
            triggered.append("Close > SMA200 (uptrend jangka panjang)")

        vol_ratio = self._get_val(row, "volume_spike_ratio")
        if _valid(vol_ratio) and vol_ratio > cfg["volume_spike_ratio_threshold"]:
            score += w["volume_spike"]
            triggered.append(f"Volume spike ratio {vol_ratio:.2f} > {cfg['volume_spike_ratio_threshold']}")

        bb_lower = self._get_val(row, "bb_lower")
        if _valid(close, bb_lower) and close <= bb_lower * (1 + cfg["bb_tolerance_pct"]):
            score += w["near_bb_lower"]
            triggered.append("Close mendekati/menyentuh Bollinger lower band")

        return round(score, 6), triggered

    def _score_sell(self, row: pd.Series) -> tuple[float, List[str]]:
        cfg = self.config["sell"]
        w = cfg["weights"]
        score = 0.0
        triggered: List[str] = []

        rsi = self._get_val(row, "rsi")
        if _valid(rsi) and rsi > cfg["rsi_overbought_level"]:
            score += w["rsi_overbought"]
            triggered.append(f"RSI overbought ({rsi:.2f} > {cfg['rsi_overbought_level']})")

        macd_hist = self._get_val(row, "macd_hist")
        macd_hist_prev = self._get_val(row, "macd_hist_prev")
        if _valid(macd_hist, macd_hist_prev) and macd_hist_prev > 0 >= macd_hist:
            score += w["macd_bearish_crossover"]
            triggered.append("MACD histogram bearish crossover (pos -> neg)")

        close = self._get_val(row, "close")
        sma_50 = self._get_val(row, "sma_50")
        if _valid(close, sma_50) and close < sma_50:
            score += w["weak_below_sma50"]
            triggered.append("Close < SMA50 (momentum melemah)")

        bb_upper = self._get_val(row, "bb_upper")
        if _valid(close, bb_upper) and close >= bb_upper * (1 - cfg["bb_tolerance_pct"]):
            score += w["near_bb_upper"]
            triggered.append("Close mendekati/menyentuh Bollinger upper band")

        vol_ratio = self._get_val(row, "volume_spike_ratio")
        close_prev = self._get_val(row, "close_prev")
        price_falling = _valid(close, close_prev) and close < close_prev
        if _valid(vol_ratio) and vol_ratio > cfg["volume_spike_ratio_threshold"] and price_falling:
            score += w["volume_spike_distribution"]
            triggered.append(
                f"Volume spike ratio {vol_ratio:.2f} saat harga turun (distribusi/jual besar)"
            )

        return round(score, 6), triggered

    # ------------------------------------------------------------------ #
    # Batch
    # ------------------------------------------------------------------ #
    def batch_generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Terapkan `generate_signal` (mode voting) ke seluruh baris DataFrame.

        Kolom *_prev (rsi_prev, cci_20_prev, williams_r_prev, momentum_10_prev,
        adx_prev, macd_hist_prev, close_prev) dihitung otomatis lewat shift(1)
        jika belum tersedia, dengan asumsi df terurut ascending per ticker.

        Returns
        -------
        pd.DataFrame dengan kolom: signal, rating, rating_category, ma_rating,
        osc_rating, buy_votes, sell_votes, neutral_votes, suggestion, reasons,
        confidence_score, triggered_conditions, timestamp, ticker.
        """
        cols = self.config["columns"]
        df = df.copy()

        ticker_col = cols.get("ticker")
        group_key = ticker_col if ticker_col in df.columns else None

        auto_prev_cols = [
            "rsi_14", "cci_20", "williams_r", "momentum_10", "adx",
            "macd_hist", "close",
        ]
        for col in auto_prev_cols:
            prev_col = f"{col}_prev"
            if prev_col not in df.columns and col in df.columns:
                if group_key:
                    df[prev_col] = df.groupby(group_key)[col].shift(1)
                else:
                    df[prev_col] = df[col].shift(1)

        results = [self.generate_signal(row) for _, row in df.iterrows()]
        out = pd.DataFrame(
            [
                {k: v for k, v in r.items() if k != "vote_table"}
                for r in results
            ],
            index=df.index,
        )
        return out


if __name__ == "__main__":
    # Contoh penggunaan singkat.
    sample = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="D"),
            "ticker": ["BBCA", "BBCA", "BBCA"],
            "close": [9000, 8800, 9200],
            "rsi_14": [28, 45, 72],
            "macd": [10, 8, 6],
            "macd_signal": [9, 9, 8],
            "macd_hist": [0.5, -0.2, -0.6],
            "sma_200": [8700, 8700, 8700],
            "sma_50": [8900, 8900, 8900],
            "volume_spike_ratio": [1.8, 1.0, 1.9],
            "bb_lower": [9010, 8850, 8800],
            "bb_upper": [9300, 9250, 9210],
            # Kolom MA/osilator lain boleh absen -> vote neutral otomatis.
        }
    )

    engine = SignalEngine()
    hasil = engine.batch_generate(sample)
    print(hasil[["timestamp", "signal", "rating", "rating_category",
                 "buy_votes", "sell_votes", "neutral_votes"]].to_string(index=False))
    print("\nContoh suggestion:")
    print(hasil.iloc[-1]["suggestion"])
