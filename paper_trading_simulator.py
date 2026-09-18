"""
paper_trading_simulator.py

Modul SIMULASI paper trading — hanya mencatat (logging) hipotetis posisi ke
SQLite dan mengevaluasinya terhadap harga pasar. Modul ini TIDAK melakukan:
  - koneksi ke broker / exchange
  - pengiriman order riil (buy/sell) dalam bentuk apa pun
  - autentikasi ke akun trading riil

Semua "posisi" yang tercatat murni baris di tabel `paper_trades` untuk
keperluan backtesting/forward-testing strategi secara aman.

Komponen:
  - CREATE_TABLE_SQL          : skema tabel paper_trades
  - PaperTradingSimulator     : class utama (open_trade, check_open_trades,
                                 get_performance_summary, has_open_trade)
  - run_daily_simulation_job  : orchestrator harian (dependency-injected,
                                 tidak mengimplementasikan fetch data /
                                 indikator / sinyal / risk management secara
                                 konkret — komponen tsb harus disuntikkan dari
                                 modul lain di sistem Anda)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any, Callable, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skema
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS paper_trades (
    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    market TEXT,
    signal_type TEXT NOT NULL,           -- 'BUY' | 'SELL'
    confidence_score REAL,
    triggered_conditions TEXT,           -- JSON string
    entry_timestamp DATETIME NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss_price REAL,
    take_profit_price REAL,
    position_size REAL,
    exit_timestamp DATETIME,
    exit_price REAL,
    exit_reason TEXT,                    -- 'stop_loss_hit' | 'take_profit_hit' | 'manual' | 'still_open'
    outcome_pct REAL,
    status TEXT NOT NULL DEFAULT 'open'  -- 'open' | 'closed'
);
"""

_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker_status ON paper_trades(ticker, status);",
    "CREATE INDEX IF NOT EXISTS idx_paper_trades_entry_ts ON paper_trades(entry_timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_paper_trades_exit_ts ON paper_trades(exit_timestamp);",
]


# ---------------------------------------------------------------------------
# Helper kecil
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    # SQLite DATETIME disimpan sbg string 'YYYY-MM-DD HH:MM:SS'
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    raise ValueError(f"Format timestamp tidak dikenali: {value!r}")


def _calc_outcome_pct(signal_type: str, entry_price: float, exit_price: float) -> float:
    if signal_type == "BUY":
        return (exit_price - entry_price) / entry_price * 100
    else:  # SELL (posisi short)
        return (entry_price - exit_price) / entry_price * 100


def _round(value: Optional[float], ndigits: int = 2) -> Optional[float]:
    return round(value, ndigits) if value is not None else None


def _max_drawdown_pct(closed_trades: list[dict]) -> Optional[float]:
    """Max drawdown dari kurva ekuitas hipotetis (compounding), berdasarkan
    urutan exit_timestamp trade-trade yang closed."""
    if not closed_trades:
        return None
    ordered = sorted(closed_trades, key=lambda t: t.get("exit_timestamp") or "")
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for t in ordered:
        pct = t.get("outcome_pct") or 0.0
        equity *= (1 + pct / 100)
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)
    return max_dd


# ---------------------------------------------------------------------------
# PaperTradingSimulator
# ---------------------------------------------------------------------------

class PaperTradingSimulator:
    """Mencatat & mengevaluasi posisi simulasi (paper trade) di SQLite.

    Sekali lagi: class ini TIDAK berkomunikasi dengan broker/exchange apa pun.
    Semua operasi hanya baca/tulis ke database lokal.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    # -- infrastruktur ------------------------------------------------

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)
            for stmt in _INDEX_SQL:
                conn.execute(stmt)

    # -- API utama ------------------------------------------------------

    def has_open_trade(self, ticker: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM paper_trades WHERE ticker = ? AND status = 'open' LIMIT 1",
                (ticker,),
            ).fetchone()
        return row is not None

    def open_trade(self, signal: dict, trade_plan: dict) -> int:
        """Catat entry baru sebagai simulasi (TIDAK mengirim order apa pun).

        signal (dict), minimal berisi:
            ticker (str), signal_type ('BUY'|'SELL'), market (opsional),
            confidence_score (float, opsional),
            triggered_conditions (list/dict, opsional -> disimpan sbg JSON)

        trade_plan (dict), minimal berisi:
            entry_price (float),
            stop_loss_price (float, opsional),
            take_profit_price (float, opsional),
            position_size (float, opsional),
            entry_timestamp (str, opsional — default: waktu sekarang UTC)

        Return: trade_id (int)
        """
        ticker = signal["ticker"]
        signal_type = str(signal["signal_type"]).upper()
        if signal_type not in ("BUY", "SELL"):
            raise ValueError(f"signal_type harus 'BUY' atau 'SELL', dapat: {signal_type!r}")

        entry_price = trade_plan["entry_price"]
        entry_ts = trade_plan.get("entry_timestamp") or _now_iso()

        triggered_conditions = signal.get("triggered_conditions")
        triggered_conditions_json = (
            json.dumps(triggered_conditions, ensure_ascii=False)
            if triggered_conditions is not None
            else None
        )

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO paper_trades (
                    ticker, market, signal_type, confidence_score, triggered_conditions,
                    entry_timestamp, entry_price, stop_loss_price, take_profit_price,
                    position_size, exit_reason, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    signal.get("market"),
                    signal_type,
                    signal.get("confidence_score"),
                    triggered_conditions_json,
                    entry_ts,
                    entry_price,
                    trade_plan.get("stop_loss_price"),
                    trade_plan.get("take_profit_price"),
                    trade_plan.get("position_size"),
                    "still_open",
                    "open",
                ),
            )
            trade_id = int(cur.lastrowid)

        logger.info(
            "Paper trade dibuka: id=%s ticker=%s type=%s entry=%.4f",
            trade_id, ticker, signal_type, entry_price,
        )
        return trade_id

    def check_open_trades(self, current_prices: dict[str, float]) -> list[dict]:
        """Cek semua trade status='open'. Jika harga terkini sudah menembus
        stop_loss atau take_profit, trade ditutup (update baris, bukan order
        riil). Return list trade yang BARU ditutup pada panggilan ini.
        """
        closed_now: list[dict] = []
        now_iso = _now_iso()

        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM paper_trades WHERE status = 'open'").fetchall()

            for row in rows:
                ticker = row["ticker"]
                if ticker not in current_prices:
                    continue

                price = current_prices[ticker]
                signal_type = row["signal_type"]
                sl = row["stop_loss_price"]
                tp = row["take_profit_price"]
                entry_price = row["entry_price"]

                exit_reason = None
                if signal_type == "BUY":
                    if sl is not None and price <= sl:
                        exit_reason = "stop_loss_hit"
                    elif tp is not None and price >= tp:
                        exit_reason = "take_profit_hit"
                else:  # SELL / short
                    if sl is not None and price >= sl:
                        exit_reason = "stop_loss_hit"
                    elif tp is not None and price <= tp:
                        exit_reason = "take_profit_hit"

                if exit_reason is None:
                    continue

                outcome_pct = _calc_outcome_pct(signal_type, entry_price, price)

                conn.execute(
                    """
                    UPDATE paper_trades
                    SET exit_timestamp = ?, exit_price = ?, exit_reason = ?,
                        outcome_pct = ?, status = 'closed'
                    WHERE trade_id = ?
                    """,
                    (now_iso, price, exit_reason, outcome_pct, row["trade_id"]),
                )

                closed_row = dict(row)
                closed_row.update(
                    exit_timestamp=now_iso,
                    exit_price=price,
                    exit_reason=exit_reason,
                    outcome_pct=outcome_pct,
                    status="closed",
                )
                closed_now.append(closed_row)

                logger.info(
                    "Paper trade ditutup: id=%s ticker=%s reason=%s outcome=%.2f%%",
                    row["trade_id"], ticker, exit_reason, outcome_pct,
                )

        return closed_now

    def close_trade_manual(self, trade_id: int, exit_price: float,
                            exit_timestamp: Optional[str] = None) -> None:
        """Tutup satu trade secara manual (mis. dibatalkan/di-review manusia),
        bukan karena SL/TP tersentuh. Tetap murni update baris DB."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT signal_type, entry_price FROM paper_trades WHERE trade_id = ?",
                (trade_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"trade_id {trade_id} tidak ditemukan")

            outcome_pct = _calc_outcome_pct(row["signal_type"], row["entry_price"], exit_price)
            conn.execute(
                """
                UPDATE paper_trades
                SET exit_timestamp = ?, exit_price = ?, exit_reason = 'manual',
                    outcome_pct = ?, status = 'closed'
                WHERE trade_id = ?
                """,
                (exit_timestamp or _now_iso(), exit_price, outcome_pct, trade_id),
            )

    def get_performance_summary(
        self,
        start_date: str,
        end_date: str,
        price_fetcher: Optional[Callable[[str, datetime], Optional[float]]] = None,
    ) -> dict:
        """Hitung metrik performa untuk trade yang entry_timestamp-nya jatuh
        pada rentang [start_date, end_date] (inklusif, format 'YYYY-MM-DD').

        price_fetcher (opsional): callable(ticker, at_datetime) -> harga
            historis pada tanggal tsb, dipakai untuk menghitung
            directional_accuracy_by_horizon (arah harga N hari setelah entry
            dibanding arah yang diprediksi sinyal). Tabel paper_trades sendiri
            tidak menyimpan harga di setiap horizon, jadi tanpa price_fetcher
            metrik ini akan bernilai None untuk tiap horizon.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM paper_trades
                WHERE date(entry_timestamp) >= date(?) AND date(entry_timestamp) <= date(?)
                ORDER BY entry_timestamp ASC
                """,
                (start_date, end_date),
            ).fetchall()

        trades = [dict(r) for r in rows]
        closed = [t for t in trades if t["status"] == "closed" and t["outcome_pct"] is not None]

        total_trades = len(trades)
        wins = [t["outcome_pct"] for t in closed if t["outcome_pct"] > 0]
        losses = [t["outcome_pct"] for t in closed if t["outcome_pct"] <= 0]

        win_rate = (len(wins) / len(closed) * 100) if closed else None
        avg_win_pct = mean(wins) if wins else None
        avg_loss_pct = mean(losses) if losses else None

        # Risk:reward yang TERREALISASI = (outcome_pct aktual) / (risiko% yang direncanakan)
        rr_list = []
        for t in closed:
            entry = t["entry_price"]
            sl = t["stop_loss_price"]
            if entry is None or sl is None:
                continue
            risk_pct = abs(entry - sl) / entry * 100
            if risk_pct == 0:
                continue
            rr_list.append(t["outcome_pct"] / risk_pct)
        avg_risk_reward_realized = mean(rr_list) if rr_list else None

        max_drawdown = _max_drawdown_pct(closed)

        directional_accuracy_by_horizon: dict[str, Optional[float]] = {"1d": None, "3d": None, "5d": None}
        if price_fetcher is not None:
            for horizon_label, days in (("1d", 1), ("3d", 3), ("5d", 5)):
                correct = 0
                evaluated = 0
                for t in trades:
                    entry_dt = _parse_dt(t["entry_timestamp"])
                    target_dt = entry_dt + timedelta(days=days)
                    future_price = price_fetcher(t["ticker"], target_dt)
                    if future_price is None:
                        continue
                    predicted_up = t["signal_type"] == "BUY"
                    actual_up = future_price > t["entry_price"]
                    evaluated += 1
                    if actual_up == predicted_up:
                        correct += 1
                directional_accuracy_by_horizon[horizon_label] = (
                    (correct / evaluated * 100) if evaluated else None
                )

        return {
            "total_trades": total_trades,
            "win_rate": _round(win_rate),
            "avg_win_pct": _round(avg_win_pct),
            "avg_loss_pct": _round(avg_loss_pct),
            "avg_risk_reward_realized": _round(avg_risk_reward_realized),
            "max_drawdown": _round(max_drawdown),
            "directional_accuracy_by_horizon": {
                k: _round(v) for k, v in directional_accuracy_by_horizon.items()
            },
        }


# ---------------------------------------------------------------------------
# Interface komponen yang harus disuntikkan (TIDAK diimplementasikan di sini)
# ---------------------------------------------------------------------------
#
# Modul ini sengaja tidak mengimplementasikan fetch data pasar, kalkulasi
# indikator, generator sinyal, maupun risk manager — supaya tetap murni
# sebagai lapisan simulasi/logging dan tidak bergantung pada sumber data /
# strategi tertentu. Sambungkan implementasi Anda sendiri lewat dependency
# injection ke run_daily_simulation_job() di bawah.

class DataFetcher(Protocol):
    def __call__(self, ticker: str) -> dict:
        """Return minimal: {"price": float, "ohlcv": list[...]}"""
        ...


class IndicatorCalculator(Protocol):
    def __call__(self, ohlcv: list) -> dict:
        """Return dict indikator teknikal (mis. {'rsi': .., 'macd': ..})."""
        ...


class SignalGenerator(Protocol):
    def __call__(self, ticker: str, market_data: dict, indicators: dict) -> Optional[dict]:
        """Return None jika tidak ada sinyal, atau dict sinyal (lihat
        PaperTradingSimulator.open_trade untuk skema `signal`)."""
        ...


class RiskManager(Protocol):
    def evaluate(self, signal: dict, indicators: dict) -> Optional[dict]:
        """Return None jika sinyal ditolak, atau dict trade_plan (lihat
        PaperTradingSimulator.open_trade untuk skema `trade_plan`)."""
        ...


# ---------------------------------------------------------------------------
# Orchestrator harian
# ---------------------------------------------------------------------------

def run_daily_simulation_job(
    ticker_list: list[str],
    db_path: str,
    data_fetcher: DataFetcher,
    indicator_calculator: IndicatorCalculator,
    signal_generator: SignalGenerator,
    risk_manager: RiskManager,
) -> dict:
    """Orchestrate pipeline simulasi harian:

        untuk setiap ticker:
            1. fetch data terbaru       (data_fetcher)
            2. hitung indikator         (indicator_calculator)
            3. generate sinyal          (signal_generator)
            4. jika ada sinyal BUY/SELL baru DAN belum ada open trade utk
               ticker itu -> evaluasi lewat RiskManager
            5. jika disetujui -> open_trade()
        lalu:
            6. update semua open_trades yang sudah ada (check_open_trades)

    Semua komponen (data_fetcher, indicator_calculator, signal_generator,
    risk_manager) adalah dependency injection dari modul lain di sistem
    Anda — fungsi ini TIDAK berisi kode koneksi broker atau eksekusi order
    riil apa pun, murni orchestration + logging simulasi.

    Return dict ringkasan hasil run: new_trades, closed_trades, skipped, errors.
    """
    sim = PaperTradingSimulator(db_path)
    result: dict[str, list] = {
        "new_trades": [],
        "closed_trades": [],
        "skipped": [],
        "errors": [],
    }

    current_prices: dict[str, float] = {}

    for ticker in ticker_list:
        try:
            market_data = data_fetcher(ticker)
            price = market_data.get("price")
            if price is not None:
                current_prices[ticker] = price

            ohlcv = market_data.get("ohlcv", [])
            indicators = indicator_calculator(ohlcv)

            signal = signal_generator(ticker, market_data, indicators)
            if signal is None:
                result["skipped"].append({"ticker": ticker, "reason": "no_signal"})
                continue

            if sim.has_open_trade(ticker):
                result["skipped"].append({"ticker": ticker, "reason": "open_trade_exists"})
                continue

            trade_plan = risk_manager.evaluate(signal, indicators)
            if not trade_plan:
                result["skipped"].append({"ticker": ticker, "reason": "risk_manager_rejected"})
                continue

            trade_id = sim.open_trade(signal, trade_plan)
            result["new_trades"].append({"ticker": ticker, "trade_id": trade_id})

        except Exception as exc:  # noqa: BLE001 - defensif, 1 ticker gagal tak menghentikan job
            logger.exception("Gagal memproses ticker %s", ticker)
            result["errors"].append({"ticker": ticker, "error": str(exc)})

    if current_prices:
        closed = sim.check_open_trades(current_prices)
        result["closed_trades"] = [
            {"trade_id": t["trade_id"], "ticker": t["ticker"], "exit_reason": t["exit_reason"]}
            for t in closed
        ]

    logger.info(
        "run_daily_simulation_job selesai: %d entry baru, %d ditutup, %d dilewati, %d error",
        len(result["new_trades"]), len(result["closed_trades"]),
        len(result["skipped"]), len(result["errors"]),
    )
    return result


# ---------------------------------------------------------------------------
# Contoh pemakaian (dummy — HANYA untuk demonstrasi struktur, bukan strategi
# riil). Jalankan file ini langsung untuk melihat alurnya bekerja end-to-end
# dengan data palsu.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    def dummy_data_fetcher(ticker: str) -> dict:
        price = round(random.uniform(1000, 5000), 2)
        return {"price": price, "ohlcv": [{"close": price}] * 20}

    def dummy_indicator_calculator(ohlcv: list) -> dict:
        closes = [c["close"] for c in ohlcv]
        return {"sma_20": mean(closes) if closes else None}

    def dummy_signal_generator(ticker: str, market_data: dict, indicators: dict) -> Optional[dict]:
        if random.random() < 0.5:
            return None
        signal_type = random.choice(["BUY", "SELL"])
        return {
            "ticker": ticker,
            "market": "IDX",
            "signal_type": signal_type,
            "confidence_score": round(random.uniform(0.5, 0.95), 2),
            "triggered_conditions": ["dummy_condition_1", "dummy_condition_2"],
        }

    class DummyRiskManager:
        def evaluate(self, signal: dict, indicators: dict) -> Optional[dict]:
            entry_price = round(random.uniform(1000, 5000), 2)
            if signal["signal_type"] == "BUY":
                sl = round(entry_price * 0.97, 2)
                tp = round(entry_price * 1.06, 2)
            else:
                sl = round(entry_price * 1.03, 2)
                tp = round(entry_price * 0.94, 2)
            return {
                "entry_price": entry_price,
                "stop_loss_price": sl,
                "take_profit_price": tp,
                "position_size": 10,
            }

    demo_db = "demo_paper_trades.db"
    outcome = run_daily_simulation_job(
        ticker_list=["BBCA", "TLKM", "ASII"],
        db_path=demo_db,
        data_fetcher=dummy_data_fetcher,
        indicator_calculator=dummy_indicator_calculator,
        signal_generator=dummy_signal_generator,
        risk_manager=DummyRiskManager(),
    )
    print(json.dumps(outcome, indent=2, ensure_ascii=False))

    sim = PaperTradingSimulator(demo_db)
    summary = sim.get_performance_summary("2000-01-01", "2100-01-01")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
