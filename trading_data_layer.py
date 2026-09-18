"""
trading_data_layer.py
======================

Modul data layer untuk sistem trading:
  - fetch_ohlcv     : ambil data OHLCV dari yfinance (US stocks & IDX .JK)
  - init_db         : setup schema SQLite
  - save_to_db      : simpan DataFrame ke SQLite (duplicate-safe, ON CONFLICT IGNORE)
  - load_from_db    : ambil data dari SQLite sebagai DataFrame
  - daily_sync_job  : job harian untuk update banyak ticker sekaligus (cocok dijadwalkan cron)

Dependencies: pandas, yfinance
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Package 'yfinance' belum terinstall. Jalankan: pip install yfinance"
    ) from exc


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("trading_data_layer")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "ticker",
    "market",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "timeframe",
]

# Mapping timeframe -> parameter `interval` yfinance
_TIMEFRAME_MAP = {
    "1d": "1d",
    "1h": "1h",
    "1wk": "1wk",
    "1mo": "1mo",
    "15m": "15m",
    "5m": "5m",
    "1m": "1m",
}

# Market yang butuh suffix khusus di ticker yfinance
_MARKET_SUFFIX = {
    "IDX": ".JK",
    "US": "",
}


class DataFetchError(Exception):
    """Dilempar ketika fetch data OHLCV gagal (ticker tidak ditemukan / data kosong)."""


# ---------------------------------------------------------------------------
# 1. fetch_ohlcv
# ---------------------------------------------------------------------------

def _resolve_yf_symbol(ticker: str, market: str) -> str:
    """Ubah (ticker, market) jadi simbol yang dikenali yfinance.

    Contoh: ("BBCA", "IDX") -> "BBCA.JK"
             ("AAPL", "US")  -> "AAPL"
    """
    market_upper = market.upper()
    suffix = _MARKET_SUFFIX.get(market_upper, "")

    # Kalau user sudah menyertakan suffix sendiri, jangan dobel-tambahkan.
    if suffix and ticker.upper().endswith(suffix):
        return ticker.upper()

    return f"{ticker.upper()}{suffix}"


def fetch_ohlcv(
    ticker: str,
    market: str,
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
) -> pd.DataFrame:
    """
    Ambil data OHLCV dari yfinance.

    Parameters
    ----------
    ticker : str
        Kode saham, misal "BBCA" atau "AAPL" (tanpa suffix .JK).
    market : str
        "IDX" untuk Bursa Efek Indonesia, "US" untuk saham AS.
        Market lain akan diperlakukan tanpa suffix tambahan.
    start_date, end_date : str
        Format "YYYY-MM-DD".
    timeframe : str
        Salah satu dari: 1m, 5m, 15m, 1h, 1d, 1wk, 1mo (default "1d").

    Returns
    -------
    pd.DataFrame
        Kolom: ticker, market, timestamp, open, high, low, close, volume, timeframe

    Raises
    ------
    DataFetchError
        Jika ticker tidak ditemukan, timeframe tidak dikenali, atau data kosong.
    """
    if timeframe not in _TIMEFRAME_MAP:
        msg = (
            f"Timeframe '{timeframe}' tidak didukung. "
            f"Pilihan valid: {list(_TIMEFRAME_MAP.keys())}"
        )
        logger.error(msg)
        raise DataFetchError(msg)

    yf_symbol = _resolve_yf_symbol(ticker, market)
    interval = _TIMEFRAME_MAP[timeframe]

    logger.info(
        "Fetching OHLCV: ticker=%s market=%s symbol=%s start=%s end=%s timeframe=%s",
        ticker, market, yf_symbol, start_date, end_date, timeframe,
    )

    try:
        raw = yf.download(
            yf_symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as exc:  # noqa: BLE001 - yfinance bisa lempar berbagai exception
        logger.error(
            "Fetch gagal untuk ticker=%s market=%s: %s", ticker, market, exc
        )
        raise DataFetchError(
            f"Gagal fetch data untuk {ticker} ({market}): {exc}"
        ) from exc

    if raw is None or raw.empty:
        msg = f"Data kosong untuk ticker '{ticker}' ({yf_symbol}), market={market}."
        logger.error(msg)
        raise DataFetchError(msg)

    # yfinance kadang mengembalikan MultiIndex kolom (khusus multi-ticker) -> flatten.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()

    # Nama kolom index waktu bisa "Date" (daily) atau "Datetime" (intraday).
    time_col = "Date" if "Date" in raw.columns else "Datetime"
    if time_col not in raw.columns:
        msg = f"Kolom waktu tidak ditemukan pada hasil fetch {ticker} ({yf_symbol})."
        logger.error(msg)
        raise DataFetchError(msg)

    missing_cols = {"Open", "High", "Low", "Close", "Volume"} - set(raw.columns)
    if missing_cols:
        msg = f"Kolom hasil fetch tidak lengkap untuk {ticker}: hilang {missing_cols}"
        logger.error(msg)
        raise DataFetchError(msg)

    df = pd.DataFrame({
        "ticker": ticker.upper(),
        "market": market.upper(),
        "timestamp": pd.to_datetime(raw[time_col]),
        "open": raw["Open"].astype(float),
        "high": raw["High"].astype(float),
        "low": raw["Low"].astype(float),
        "close": raw["Close"].astype(float),
        "volume": raw["Volume"].fillna(0).astype("int64"),
        "timeframe": timeframe,
    })

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    if df.empty:
        msg = f"Setelah cleaning, data kosong untuk ticker '{ticker}' ({yf_symbol})."
        logger.error(msg)
        raise DataFetchError(msg)

    logger.info(
        "Fetch sukses: ticker=%s baris=%d rentang=%s s/d %s",
        ticker, len(df), df["timestamp"].min(), df["timestamp"].max(),
    )

    return df[REQUIRED_COLUMNS]


# ---------------------------------------------------------------------------
# 2. Setup database SQLite
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ohlcv_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    market TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    timeframe TEXT NOT NULL,
    UNIQUE(ticker, timestamp, timeframe)
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup
ON ohlcv_data (ticker, market, timeframe, timestamp);
"""


@contextmanager
def _get_connection(db_path: str) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    """Buat tabel `ohlcv_data` beserta index-nya jika belum ada."""
    logger.info("Init database di %s", db_path)
    with _get_connection(db_path) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(_CREATE_INDEX_SQL)
        conn.commit()


# ---------------------------------------------------------------------------
# 3. save_to_db
# ---------------------------------------------------------------------------

_INSERT_SQL = """
INSERT INTO ohlcv_data
    (ticker, market, timestamp, open, high, low, close, volume, timeframe)
VALUES
    (:ticker, :market, :timestamp, :open, :high, :low, :close, :volume, :timeframe)
ON CONFLICT(ticker, timestamp, timeframe) DO NOTHING;
"""


def save_to_db(df: pd.DataFrame, db_path: str) -> int:
    """
    Simpan DataFrame OHLCV ke SQLite. Baris duplikat (ticker+timestamp+timeframe
    sama) diabaikan lewat ON CONFLICT DO NOTHING.

    Parameters
    ----------
    df : pd.DataFrame
        Harus mengandung kolom REQUIRED_COLUMNS.
    db_path : str
        Path file database SQLite.

    Returns
    -------
    int
        Jumlah baris baru yang benar-benar ter-insert (duplikat tidak dihitung).
    """
    if df is None or df.empty:
        logger.warning("save_to_db dipanggil dengan DataFrame kosong, dilewati.")
        return 0

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        msg = f"DataFrame kekurangan kolom wajib: {missing}"
        logger.error(msg)
        raise ValueError(msg)

    init_db(db_path)

    df_to_save = df.copy()
    df_to_save["timestamp"] = pd.to_datetime(df_to_save["timestamp"]).astype(str)

    records = df_to_save[REQUIRED_COLUMNS].to_dict(orient="records")

    with _get_connection(db_path) as conn:
        cur = conn.cursor()
        before = cur.execute("SELECT changes()").fetchone()[0]
        try:
            cur.executemany(_INSERT_SQL, records)
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            logger.error("Gagal menyimpan ke database %s: %s", db_path, exc)
            raise

        inserted = conn.total_changes - before

    logger.info(
        "save_to_db: %d baris diproses, %d baris baru tersimpan (sisanya duplikat).",
        len(records), inserted,
    )
    return inserted


# ---------------------------------------------------------------------------
# 4. load_from_db
# ---------------------------------------------------------------------------

def load_from_db(
    ticker: str,
    market: str,
    start_date: str,
    end_date: str,
    db_path: str,
    timeframe: str | None = None,
) -> pd.DataFrame:
    """
    Muat data OHLCV dari SQLite untuk satu ticker/market dalam rentang tanggal.

    Parameters
    ----------
    timeframe : str, optional
        Jika diisi, filter tambahan per timeframe. Jika None, semua timeframe
        untuk ticker tsb. dikembalikan.

    Returns
    -------
    pd.DataFrame
        Kolom sama seperti REQUIRED_COLUMNS, terurut berdasarkan timestamp.
    """
    query = """
        SELECT ticker, market, timestamp, open, high, low, close, volume, timeframe
        FROM ohlcv_data
        WHERE ticker = :ticker
          AND market = :market
          AND timestamp >= :start_date
          AND timestamp <= :end_date
    """
    params = {
        "ticker": ticker.upper(),
        "market": market.upper(),
        "start_date": start_date,
        "end_date": end_date,
    }

    if timeframe is not None:
        query += " AND timeframe = :timeframe"
        params["timeframe"] = timeframe

    query += " ORDER BY timestamp ASC;"

    logger.info(
        "load_from_db: ticker=%s market=%s start=%s end=%s timeframe=%s",
        ticker, market, start_date, end_date, timeframe,
    )

    try:
        with _get_connection(db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["timestamp"])
    except sqlite3.Error as exc:
        logger.error("Gagal load dari database %s: %s", db_path, exc)
        raise

    logger.info("load_from_db: %d baris ditemukan untuk %s", len(df), ticker)
    return df


# ---------------------------------------------------------------------------
# 5. daily_sync_job
# ---------------------------------------------------------------------------

@dataclass
class SyncResult:
    ticker: str
    market: str
    status: str          # "success" | "failed" | "no_new_data"
    rows_inserted: int = 0
    error: str | None = None


def daily_sync_job(
    ticker_list: list[dict],
    db_path: str,
    timeframe: str = "1d",
    lookback_days: int = 7,
) -> list[SyncResult]:
    """
    Job harian untuk update data OHLCV banyak ticker sekaligus.
    Cocok dipanggil dari scheduler (cron, APScheduler, Airflow, dll).

    Untuk tiap ticker: fetch `lookback_days` hari terakhir (mengcover weekend/
    hari libur bursa dan potensi data yang terlewat), lalu simpan ke DB dengan
    duplicate-safe insert.

    Parameters
    ----------
    ticker_list : list[dict]
        Contoh: [{"ticker": "BBCA", "market": "IDX"}, {"ticker": "AAPL", "market": "US"}]
    db_path : str
        Path database SQLite.
    timeframe : str
        Timeframe yang di-sync, default "1d".
    lookback_days : int
        Berapa hari ke belakang yang di-fetch ulang tiap run, untuk jaga-jaga
        data yang belum sempat masuk di run sebelumnya (default 7).

    Returns
    -------
    list[SyncResult]
        Ringkasan hasil sync per ticker (untuk logging/monitoring/alerting).
    """
    init_db(db_path)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    logger.info(
        "=== daily_sync_job dimulai: %d ticker, rentang %s s/d %s, timeframe=%s ===",
        len(ticker_list), start_date, end_date, timeframe,
    )

    results: list[SyncResult] = []

    for item in ticker_list:
        ticker = item.get("ticker")
        market = item.get("market")

        if not ticker or not market:
            msg = f"Item ticker_list tidak valid, harus ada 'ticker' & 'market': {item}"
            logger.error(msg)
            results.append(SyncResult(
                ticker=str(ticker), market=str(market),
                status="failed", error=msg,
            ))
            continue

        try:
            df = fetch_ohlcv(
                ticker=ticker,
                market=market,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
            )
            inserted = save_to_db(df, db_path)

            status = "success" if inserted > 0 else "no_new_data"
            results.append(SyncResult(
                ticker=ticker, market=market,
                status=status, rows_inserted=inserted,
            ))

        except DataFetchError as exc:
            logger.error("Sync gagal untuk %s (%s): %s", ticker, market, exc)
            results.append(SyncResult(
                ticker=ticker, market=market,
                status="failed", error=str(exc),
            ))
        except Exception as exc:  # noqa: BLE001 - jangan sampai satu ticker gagal hentikan job
            logger.error(
                "Sync gagal tak terduga untuk %s (%s): %s", ticker, market, exc
            )
            results.append(SyncResult(
                ticker=ticker, market=market,
                status="failed", error=str(exc),
            ))

    success_count = sum(1 for r in results if r.status == "success")
    failed_count = sum(1 for r in results if r.status == "failed")
    logger.info(
        "=== daily_sync_job selesai: %d sukses, %d gagal, %d tanpa data baru ===",
        success_count, failed_count, len(results) - success_count - failed_count,
    )

    return results


# ---------------------------------------------------------------------------
# Contoh penggunaan (jalankan langsung: python trading_data_layer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DB_PATH = "trading_data.db"

    tickers = [
        {"ticker": "BBCA", "market": "IDX"},
        {"ticker": "TLKM", "market": "IDX"},
        {"ticker": "AAPL", "market": "US"},
        {"ticker": "TICKERTIDAKADA", "market": "US"},  # contoh kasus gagal
    ]

    hasil = daily_sync_job(tickers, DB_PATH, timeframe="1d", lookback_days=10)
    for r in hasil:
        print(r)

    contoh = load_from_db(
        ticker="BBCA",
        market="IDX",
        start_date="2024-01-01",
        end_date="2024-12-31",
        db_path=DB_PATH,
    )
    print(contoh.tail())
