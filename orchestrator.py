"""
orchestrator.py
================

Menyambungkan seluruh modul jadi satu pipeline simulasi harian:

    trading_data_layer  -> technical_indicators -> signal_engine
        -> risk_manager -> paper_trading_simulator

Cara pakai:
    1. Sesuaikan TICKERS di bawah (tambah/kurangi ticker & market).
    2. Sesuaikan STARTING_CAPITAL.
    3. Jalankan manual: python orchestrator.py
    4. Untuk simulasi 1 minggu berjalan otomatis, jadwalkan file ini
       lewat cron (Linux/Mac) atau Task Scheduler (Windows) untuk jalan
       1x per hari setelah market tutup.
    5. Cek hasilnya di dashboard: streamlit run dashboard.py
       (pastikan db_path di sidebar dashboard sama dengan DB_PATH di bawah)

PENTING: script ini TIDAK melakukan eksekusi order riil apa pun — murni
fetch data, generate sinyal, evaluasi risiko, dan catat ke database
simulasi (paper_trades).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from trading_data_layer import (
    DataFetchError,
    fetch_ohlcv,
    load_from_db,
    save_to_db,
)
from technical_indicators import build_feature_set
from signal_engine import SignalEngine
from risk_manager import RiskManager
from paper_trading_simulator import run_daily_simulation_job
from signal_validator import SignalValidator, compute_signals_from_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("orchestrator")


# ============================================================
# KONFIGURASI — sesuaikan di sini
# ============================================================

# Satu database untuk OHLCV historis DAN paper_trades, karena dashboard.py
# membaca kedua tabel dari satu db_path yang sama.
DB_PATH = "trading_system.db"

# Universal: campur market apa saja, format {ticker: market}
TICKERS = {
    "BBCA": "IDX",
    "TLKM": "IDX",
    "AAPL": "US",
    "MSFT": "US",
}

STARTING_CAPITAL = 100_000_000  # sesuaikan dengan modal simulasi kamu
LOOKBACK_DAYS = 400             # histori yang di-fetch, cukup untuk SMA200 (butuh min. ~200 hari data)
TIMEFRAME = "1d"

HORIZONS = (1, 3, 5)            # evaluasi sinyal 1/3/5 bar trading setelah sinyal
RUN_DURATION_DAYS = 7           # lama siklus validasi mingguan

# Instansiasi modul (parameter risk management bisa kamu tuning di sini)
signal_engine = SignalEngine()
risk_manager = RiskManager(
    max_position_size_pct=0.05,
    atr_multiplier_stoploss=2.0,
    min_risk_reward_ratio=2.0,
    max_daily_drawdown_pct=0.03,
    max_weekly_drawdown_pct=0.08,
    risk_per_trade_pct=0.01,
)
risk_manager.capital = STARTING_CAPITAL  # WAJIB di-set

# Pemantauan validasi sinyal mingguan (snapshot harian -> evaluasi -> kesimpulan)
signal_validator = SignalValidator(DB_PATH, horizons=HORIZONS, run_duration_days=RUN_DURATION_DAYS)


# ============================================================
# ADAPTER: DataFetcher
# ============================================================
def data_fetcher(ticker: str) -> dict:
    """
    Sync data terbaru dari yfinance ke DB, lalu load histori dari DB.
    Return: {"price": float, "ohlcv": list[dict], "timestamp": ...}
    """
    market = TICKERS[ticker]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    # Sync data terbaru (best-effort — kalau fetch gagal, tetap coba pakai data lama di DB)
    try:
        fresh = fetch_ohlcv(ticker, market, start_date, end_date, TIMEFRAME)
        save_to_db(fresh, DB_PATH)
    except DataFetchError as exc:
        logger.warning("Gagal sync data terbaru untuk %s: %s (pakai data DB yang ada)", ticker, exc)

    df = load_from_db(ticker, market, start_date, end_date, DB_PATH, TIMEFRAME)
    if df.empty:
        raise ValueError(f"Tidak ada data historis untuk {ticker} ({market}) di database.")
    if len(df) < 200:
        logger.warning(
            "%s hanya punya %d baris data (< 200) — indikator SMA200 kemungkinan NaN.",
            ticker, len(df),
        )

    df = df.sort_values("timestamp").reset_index(drop=True)
    latest = df.iloc[-1]

    return {
        "price": float(latest["close"]),
        "ohlcv": df.to_dict(orient="records"),
        "timestamp": latest["timestamp"],
    }


# ============================================================
# ADAPTER: IndicatorCalculator
# ============================================================
def indicator_calculator(ohlcv: list) -> dict:
    """
    Hitung seluruh indikator teknikal, lalu kembalikan dict indikator dari
    baris terakhir (+ nilai _prev untuk deteksi crossover di signal_engine).
    """
    df = pd.DataFrame(ohlcv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    features = build_feature_set(df)
    if len(features) < 2:
        raise ValueError("Data tidak cukup untuk menghitung indikator (butuh >= warm-up period + 1).")

    latest = features.iloc[-1]
    prev = features.iloc[-2]

    indicators = latest.to_dict()
    # Nilai periode sebelumnya untuk deteksi 'berbalik arah' (turning) pada
    # aturan voting ala TradingView (RSI, CCI, Williams %R, Momentum, ADX).
    for col in ["rsi_14", "cci_20", "williams_r", "momentum_10", "adx"]:
        indicators[f"{col}_prev"] = prev[col]
    indicators["macd_hist_prev"] = prev["macd_hist"]
    indicators["close_prev"] = prev["close"]
    return indicators


# ============================================================
# ADAPTER: SignalGenerator
# ============================================================
def signal_generator(ticker: str, market_data: dict, indicators: dict) -> Optional[dict]:
    """
    Bungkus indicators jadi pd.Series lalu panggil SignalEngine.
    Return None kalau HOLD (tidak ada trade baru).
    """
    row = pd.Series(indicators)
    row["ticker"] = ticker
    row["timestamp"] = market_data.get("timestamp")

    result = signal_engine.generate_signal(row)

    if result["signal"] == "HOLD":
        return None

    result["market"] = TICKERS[ticker]
    result["entry_price"] = market_data["price"]
    return result


# ============================================================
# VALIDASI SINYAL MINGGUAN (otomatis, tanpa network)
# ============================================================
def run_signal_validation_cycle() -> dict:
    """
    Pemantauan validasi otomatis:
        1. Hitung sinyal terkini dari DB (data sudah disinkkan data_fetcher).
        2. Simpan snapshot harian ke run mingguan aktif.
        3. Evaluasi snapshot lama dgn data pasar terbaru (1/3/5 bar).
        4. Auto-conclude jika run sudah berusia >= RUN_DURATION_DAYS.
    """
    signals = compute_signals_from_db(TICKERS, DB_PATH, TIMEFRAME)
    cycle = signal_validator.run_daily_cycle(signals)

    logger.info(
        "Validasi sinyal: action=%s run=%s usia=%s hari (sisa %s hari), "
        "snapshot +%d/-%d, evaluasi diperbarui %d",
        cycle["action"], cycle["run_id"], cycle["run_age_days"],
        cycle["days_remaining"], cycle["merge"]["inserted"],
        cycle["merge"]["updated"], cycle["snapshots_updated"],
    )
    return cycle


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    logger.info("=== Menjalankan simulasi harian untuk %d ticker ===", len(TICKERS))

    result = run_daily_simulation_job(
        ticker_list=list(TICKERS.keys()),
        db_path=DB_PATH,
        data_fetcher=data_fetcher,
        indicator_calculator=indicator_calculator,
        signal_generator=signal_generator,
        risk_manager=risk_manager,  # RiskManager.evaluate() dipanggil otomatis oleh run_daily_simulation_job
    )

    logger.info("Hasil: %d entry baru, %d ditutup, %d dilewati, %d error",
                len(result["new_trades"]), len(result["closed_trades"]),
                len(result["skipped"]), len(result["errors"]))

    for e in result["errors"]:
        logger.error("Error pada %s: %s", e["ticker"], e["error"])

    # --- Pemantauan validasi sinyal (jalan tiap hari setelah pipeline utama) ---
    validation = run_signal_validation_cycle()

    print(result)
    print("validation:", validation)


if __name__ == "__main__":
    main()
