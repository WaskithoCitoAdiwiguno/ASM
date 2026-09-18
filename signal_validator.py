"""
signal_validator.py
===================

Sistem pemantauan otomatis untuk memvalidasi sinyal selama seminggu.

Alur kerja (weekly validation cycle):
    1. Setiap hari (dipanggil dari orchestrator setelah market tutup), sistem
       menyimpan SNAPSHOT sinyal semua ticker: arah sinyal, rating, jumlah
       suara vote, harga saat sinyal, alasan, dan saran.
    2. Snapshot disimpan ke SATU validation run yang aktif selama 7 hari.
    3. Setiap hari juga, sistem mengevaluasi ulang snapshot lama terhadap
       data pasar TERBARU di database: apakah arah sinyal terbukti benar
       1, 3, dan 5 bar trading setelah sinyal dibuat?
       - Sinyal BUY  benar jika close naik vs harga sinyal pada horizon tsb.
       - Sinyal SELL benar jika close turun vs harga sinyal pada horizon tsb.
       - Sinyal HOLD tidak dinilai benar/salah (tidak ada arah), hanya dicatat.
    4. Setelah 7 hari, run DISENSUS (concluded): dihitung akurasi per horizon,
       per kategori rating, per ticker, lalu dibuat kalimat kesimpulan.
    5. Siklus baru dimulai otomatis pada run orchestrator berikutnya.

Horizon memakai hitungan BAR TRADING (bukan hari kalender) sehingga aman
terhadap weekend/hari libur bursa: "1 bar" = bar perdagangan berikutnya.

Tabel database:
    signal_runs      : 1 baris per siklus validasi mingguan
    signal_snapshots : 1 baris per (run, ticker), berisi sinyal + hasil
                       evaluasi day1/day3/day5

Modul ini murni baca/tulis SQLite + perhitungan pandas — tidak ada koneksi
broker dan tidak ada eksekusi order.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from trading_data_layer import load_from_db

logger = logging.getLogger("signal_validator")

DEFAULT_HORIZONS = (1, 3, 5)          # bar trading setelah sinyal
RUN_DURATION_DAYS = 7                  # lama siklus validasi


# ---------------------------------------------------------------------------
# Skema database
# ---------------------------------------------------------------------------
CREATE_RUNS_SQL = """
CREATE TABLE IF NOT EXISTS signal_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,            -- UTC ISO: mulai siklus
    as_of_date TEXT NOT NULL,            -- tanggal pasar dari snapshot pertama
    status TEXT NOT NULL DEFAULT 'active',   -- 'active' | 'concluded'
    concluded_at TEXT,
    summary TEXT                         -- JSON hasil kesimpulan
);
"""

CREATE_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS signal_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    market TEXT,
    signal_date TEXT NOT NULL,           -- tanggal bar saat sinyal dibuat
    signal_type TEXT NOT NULL,           -- 'BUY' | 'SELL' | 'HOLD'
    rating REAL,
    rating_category TEXT,
    buy_votes INTEGER,
    sell_votes INTEGER,
    neutral_votes INTEGER,
    price_at_signal REAL NOT NULL,
    suggestion TEXT,
    reasons TEXT,                        -- JSON list of str
    day1_close REAL, day1_outcome_pct REAL,
    day3_close REAL, day3_outcome_pct REAL,
    day5_close REAL, day5_outcome_pct REAL,
    evaluated_at TEXT,
    error_note TEXT,
    UNIQUE(run_id, ticker),
    FOREIGN KEY(run_id) REFERENCES signal_runs(run_id)
);
"""

_CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_sig_snap_run ON signal_snapshots(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_sig_runs_status ON signal_runs(status);",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _date_str(value: Any) -> str:
    """Normalisasi datetime/Timestamp/str ke 'YYYY-MM-DD'."""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _hit(outcome_pct: Optional[float]) -> Optional[bool]:
    """True jika outcome searah dengan sinyal (positif setelah penyesuaian arah)."""
    if outcome_pct is None:
        return None
    return outcome_pct > 0


# ---------------------------------------------------------------------------
# Pengambilan sinyal terkini dari DB (tanpa network)
# ---------------------------------------------------------------------------
def compute_signals_from_db(
    tickers: Dict[str, str],
    db_path: str,
    timeframe: str = "1d",
) -> List[Dict[str, Any]]:
    """
    Hitung sinyal voting TERKINI untuk semua ticker langsung dari tabel
    OHLCV di database (tanpa fetch network — data diasumsikan sudah
    disinkkan oleh orchestrator).

    Returns
    -------
    list[dict], satu per ticker:
        {ticker, market, signal, rating, rating_category, buy/sell/neutral_votes,
         price, signal_date, suggestion, reasons}
    """
    from technical_indicators import build_feature_set  # import lokal utk hindari siklus
    from signal_engine import SignalEngine

    engine = SignalEngine()
    out: List[Dict[str, Any]] = []

    for ticker, market in tickers.items():
        try:
            df = load_from_db(ticker, market, "2020-01-01", "2100-01-01", db_path, timeframe)
            if df.empty or len(df) < 60:
                logger.warning("Validator: data %s kurang (%d bar), dilewati.", ticker, len(df))
                continue

            features = build_feature_set(df)
            if len(features) < 2:
                continue

            latest, prev = features.iloc[-1], features.iloc[-2]
            row = latest.copy()
            for col in ["rsi_14", "cci_20", "williams_r", "momentum_10", "adx",
                        "macd_hist", "close"]:
                row[f"{col}_prev"] = prev[col]

            res = engine.evaluate_row(row)
            out.append({
                "ticker": ticker,
                "market": market,
                "signal": res["signal"],
                "rating": res["rating"],
                "rating_category": res["rating_category"],
                "buy_votes": res["buy_votes"],
                "sell_votes": res["sell_votes"],
                "neutral_votes": res["neutral_votes"],
                "price": float(latest["close"]),
                "signal_date": _date_str(latest["timestamp"]),
                "suggestion": res["suggestion"],
                "reasons": res["reasons"],
            })
        except Exception as exc:  # noqa: BLE001 — satu ticker gagal tak menghentikan cycle
            logger.exception("Validator: gagal hitung sinyal %s", ticker)

    return out


# ---------------------------------------------------------------------------
# SignalValidator
# ---------------------------------------------------------------------------
class SignalValidator:
    """Mengelola siklus validasi sinyal mingguan di SQLite."""

    def __init__(self, db_path: str, horizons: tuple = DEFAULT_HORIZONS,
                 run_duration_days: int = RUN_DURATION_DAYS):
        import sqlite3
        self.db_path = db_path
        self.horizons = tuple(horizons)
        self.run_duration_days = run_duration_days
        self._init_db()

    # -- infrastruktur ------------------------------------------------
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(CREATE_RUNS_SQL)
            conn.execute(CREATE_SNAPSHOTS_SQL)
            for stmt in _CREATE_INDEX_SQL:
                conn.execute(stmt)

    def _connect(self):
        # didefinisikan sebagai method biasa agar bisa dipakai `with`
        return _ValidatorConnection(self.db_path)

    # -- manajemen run ------------------------------------------------
    def get_active_run(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM signal_runs WHERE status = 'active' "
                "ORDER BY run_id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def start_new_run(self, as_of_date: str, started_at: Optional[str] = None) -> int:
        """Buka run validasi baru (aktif 7 hari)."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO signal_runs (started_at, as_of_date, status) VALUES (?, ?, 'active')",
                (started_at or _now_iso(), _date_str(as_of_date)),
            )
            run_id = int(cur.lastrowid)
        logger.info("Validator: run validasi baru dibuka id=%s (as_of=%s)", run_id, as_of_date)
        return run_id

    def record_signals(self, run_id: int, signals: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Simpan/merge snapshot sinyal harian ke dalam run.

        Aturan merge per (run, ticker) — first-directional-wins:
            - belum ada baris            -> INSERT (semua sinyal, termasuk HOLD)
            - sudah HOLD, datang arah    -> UPDATE ke sinyal baru (sinyal serius menimpa tunggu)
            - sudah punya arah, datang HOLD -> biarkan (sinyal yang sedang divalidasi tetap)
            - sudah punya arah, datang arah sama/beda -> biarkan sinyal awal
        """
        inserted = updated = unchanged = 0
        with self._connect() as conn:
            for s in signals:
                existing = conn.execute(
                    "SELECT snapshot_id, signal_type FROM signal_snapshots "
                    "WHERE run_id = ? AND ticker = ?",
                    (run_id, s["ticker"]),
                ).fetchone()

                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO signal_snapshots (
                            run_id, ticker, market, signal_date, signal_type,
                            rating, rating_category, buy_votes, sell_votes, neutral_votes,
                            price_at_signal, suggestion, reasons
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id, s["ticker"], s.get("market"),
                            _date_str(s.get("signal_date") or datetime.now()),
                            str(s.get("signal", "HOLD")).upper(),
                            s.get("rating"), s.get("rating_category"),
                            s.get("buy_votes"), s.get("sell_votes"), s.get("neutral_votes"),
                            float(s["price"]), s.get("suggestion"),
                            json.dumps(s.get("reasons", []), ensure_ascii=False),
                        ),
                    )
                    inserted += 1
                elif existing["signal_type"] == "HOLD" and str(s.get("signal", "HOLD")).upper() != "HOLD":
                    conn.execute(
                        """
                        UPDATE signal_snapshots SET
                            signal_type = ?, rating = ?, rating_category = ?,
                            buy_votes = ?, sell_votes = ?, neutral_votes = ?,
                            price_at_signal = ?, suggestion = ?, reasons = ?,
                            signal_date = ?
                        WHERE snapshot_id = ?
                        """,
                        (
                            str(s.get("signal")).upper(), s.get("rating"), s.get("rating_category"),
                            s.get("buy_votes"), s.get("sell_votes"), s.get("neutral_votes"),
                            float(s["price"]), s.get("suggestion"),
                            json.dumps(s.get("reasons", []), ensure_ascii=False),
                            _date_str(s.get("signal_date") or datetime.now()),
                            existing["snapshot_id"],
                        ),
                    )
                    updated += 1
                else:
                    unchanged += 1

        logger.info(
            "Validator: snapshot run=%s -> %d insert, %d update, %d unchanged",
            run_id, inserted, updated, unchanged,
        )
        return {"inserted": inserted, "updated": updated, "unchanged": unchanged}

    # -- evaluasi terhadap pasar --------------------------------------
    def update_validation(self) -> int:
        """
        Evaluasi semua snapshot yang BELUM terevaluasi pada run aktif,
        memakai harga close dari tabel ohlcv_data.

        Horizon = bar trading ke-N SETELAH bar sinyal (aman dari weekend).
        Return jumlah snapshot yang barisnya baru saja diperbarui.
        """
        run = self.get_active_run()
        if run is None:
            return 0

        updated_rows = 0
        with self._connect() as conn:
            snaps = conn.execute(
                "SELECT * FROM signal_snapshots WHERE run_id = ?", (run["run_id"],)
            ).fetchall()

        for snap in snaps:
            updates: Dict[str, Any] = {}
            future_bars = self._load_future_bars(snap["ticker"], snap["market"], snap["signal_date"])
            if future_bars is None:
                continue  # tidak ada data historis utk ticker ini

            direction = 1 if snap["signal_type"] == "BUY" else (-1 if snap["signal_type"] == "SELL" else 0)

            for n in self.horizons:
                close_col, out_col = f"day{n}_close", f"day{n}_outcome_pct"
                if snap[out_col] is not None:
                    continue  # sudah dievaluasi sebelumnya (immutable)
                if len(future_bars) < n:
                    continue  # data belum tersedia — coba lagi besok

                close_n = float(future_bars.iloc[n - 1]["close"])
                if direction == 0:
                    outcome = None  # HOLD: tidak dinilai
                else:
                    outcome = (close_n - snap["price_at_signal"]) / snap["price_at_signal"] * 100 * direction
                updates[close_col] = close_n
                updates[out_col] = outcome

            if updates:
                updates["evaluated_at"] = _now_iso()
                set_clause = ", ".join(f"{k} = ?" for k in updates)
                with self._connect() as conn:
                    conn.execute(
                        f"UPDATE signal_snapshots SET {set_clause} WHERE snapshot_id = ?",
                        (*updates.values(), snap["snapshot_id"]),
                    )
                updated_rows += 1

        if updated_rows:
            logger.info("Validator: %d snapshot diperbarui hasil evaluasinya.", updated_rows)
        return updated_rows

    def _load_future_bars(self, ticker: str, market: Optional[str], signal_date: str) -> Optional[pd.DataFrame]:
        """Bar-bar perdagangan SETELAH tanggal sinyal (ascending)."""
        try:
            end_buffer = (pd.Timestamp(signal_date) + timedelta(days=60)).strftime("%Y-%m-%d")
            df = load_from_db(
                ticker, market or "", signal_date, end_buffer,
                self.db_path, "1d",
            )
        except Exception:  # noqa: BLE001
            return None
        if df.empty:
            return None
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df[df["timestamp"] > pd.Timestamp(signal_date)].sort_values("timestamp").reset_index(drop=True)

    # -- penyimpulan ---------------------------------------------------
    def run_age_days(self, run: Dict[str, Any]) -> float:
        started = pd.Timestamp(run["started_at"])
        return (pd.Timestamp.now(tz="UTC").tz_localize(None) - started).total_seconds() / 86400

    def conclude_run(self, run_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Tutup run: hitung akurasi & kesimpulan, simpan ke kolom summary."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM signal_runs WHERE run_id = ?", (
                    run_id if run_id is not None else
                    conn.execute("SELECT MAX(run_id) AS m FROM signal_runs").fetchone()["m"],
                )
            ).fetchone()
        if row is None:
            return None
        run = dict(row)
        if run["status"] == "concluded":
            return json.loads(run["summary"]) if run["summary"] else None

        with self._connect() as conn:
            snaps = [dict(r) for r in conn.execute(
                "SELECT * FROM signal_snapshots WHERE run_id = ?", (run["run_id"],)
            ).fetchall()]

        summary = self._build_summary(run, snaps)

        with self._connect() as conn:
            conn.execute(
                "UPDATE signal_runs SET status = 'concluded', concluded_at = ?, summary = ? "
                "WHERE run_id = ?",
                (_now_iso(), json.dumps(summary, ensure_ascii=False), run["run_id"]),
            )
        logger.info("Validator: run %s disimpulkan. %s", run["run_id"], summary["conclusion_text"])
        return summary

    def _build_summary(self, run: Dict[str, Any], snaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        directional = [s for s in snaps if s["signal_type"] in ("BUY", "SELL")]
        holds = [s for s in snaps if s["signal_type"] == "HOLD"]

        by_horizon: Dict[str, Any] = {}
        for n in self.horizons:
            out_col = f"day{n}_outcome_pct"
            evaluated = [s for s in directional if s[out_col] is not None]
            hits = [s for s in evaluated if _hit(s[out_col])]
            outcomes = [s[out_col] for s in evaluated]
            by_horizon[f"{n}d"] = {
                "evaluated": len(evaluated),
                "pending": len(directional) - len(evaluated),
                "hits": len(hits),
                "accuracy_pct": round(len(hits) / len(evaluated) * 100, 1) if evaluated else None,
                "avg_outcome_pct": round(sum(outcomes) / len(outcomes), 2) if outcomes else None,
            }

        # Breakdown per kategori rating & per ticker (per horizon terpanjang yg tersedia)
        deepest = f"day{max(self.horizons)}_outcome_pct"
        by_category: Dict[str, Any] = {}
        for cat in sorted({s["rating_category"] or "?" for s in directional}):
            grp = [s for s in directional if (s["rating_category"] or "?") == cat]
            ev = [s for s in grp if s[deepest] is not None]
            hits = [s for s in ev if _hit(s[deepest])]
            by_category[cat] = {
                "signals": len(grp),
                "evaluated": len(ev),
                "hits": len(hits),
                "accuracy_pct": round(len(hits) / len(ev) * 100, 1) if ev else None,
            }

        by_ticker: Dict[str, Any] = {}
        for tk in sorted({s["ticker"] for s in directional}):
            grp = [s for s in directional if s["ticker"] == tk]
            ev = [s for s in grp if s[deepest] is not None]
            hits = [s for s in ev if _hit(s[deepest])]
            by_ticker[tk] = {
                "signals": len(grp),
                "accuracy_pct": round(len(hits) / len(ev) * 100, 1) if ev else None,
            }

        # Kalimat kesimpulan
        total_eval = by_horizon[f"{max(self.horizons)}d"]["evaluated"]
        acc = by_horizon[f"{max(self.horizons)}d"]["accuracy_pct"]
        avg = by_horizon[f"{max(self.horizons)}d"]["avg_outcome_pct"]
        pending = by_horizon[f"{max(self.horizons)}d"]["pending"]
        if total_eval:
            verdict = (
                "SINYAL TERBUKTI VALID" if acc >= 60 else
                "SINYAL KURANG VALID" if acc >= 40 else
                "SINYAL BELUM VALID"
            )
            conclusion = (
                f"Dari {len(directional)} sinyal arah ({len(holds)} HOLD) dalam {self.run_duration_days} hari, "
                f"{total_eval} sudah bisa dievaluasi pada horizon {max(self.horizons)} bar: "
                f"akurasi {acc}% (rata-rata hasil {avg:+.2f}% per sinyal). "
                f"Kesimpulan: {verdict}."
            )
            if pending:
                conclusion += f" Masih {pending} sinyal menunggu data pasar (pending)."
        else:
            conclusion = (
                f"Tidak ada sinyal arah yang bisa dievaluasi pada horizon {max(self.horizons)} bar "
                f"(data pasar belum tersedia / semua sinyal masih pending)."
            )

        return {
            "run_id": run["run_id"],
            "started_at": run["started_at"],
            "as_of_date": run["as_of_date"],
            "total_snapshots": len(snaps),
            "directional": len(directional),
            "hold": len(holds),
            "by_horizon": by_horizon,
            "by_category": by_category,
            "by_ticker": by_ticker,
            "conclusion_text": conclusion,
        }

    # -- siklus harian utama --------------------------------------------
    def run_daily_cycle(self, signals: List[Dict[str, Any]], as_of_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Satu panggilan per hari dari orchestrator:
            1. Pastikan ada run aktif (buat baru jika belum ada / sudah menua).
            2. Rekam snapshot sinyal hari ini (merge, first-directional-wins).
            3. Update evaluasi snapshot lama dgn data pasar terbaru.
            4. Jika usia run >= 7 hari -> simpulkan run, run baru dibuka di cycle berikutnya.
        """
        # Tanggal acuan run = tanggal bar pasar terbaru dari snapshot yang direkam.
        if as_of_date is None:
            dates = [s.get("signal_date") for s in signals if s.get("signal_date")]
            as_of_date = max(dates) if dates else _date_str(datetime.now())

        action = "noop"
        run = self.get_active_run()
        if run is None or self.run_age_days(run) >= self.run_duration_days:
            if run is not None:
                self.conclude_run(run["run_id"])
            run_id = self.start_new_run(as_of_date)
            action = "started_new"
        else:
            run_id = run["run_id"]

        merge_stats = self.record_signals(run_id, signals)
        updated = self.update_validation()

        run = self.get_active_run()
        return {
            "action": action,
            "run_id": run_id,
            "run_age_days": round(self.run_age_days(run), 1) if run else None,
            "merge": merge_stats,
            "snapshots_updated": updated,
            "days_remaining": (
                round(self.run_duration_days - self.run_age_days(run), 1) if run else None
            ),
        }

    # -- utilitas baca utk dashboard -------------------------------------
    def get_run_snapshots(self, run_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM signal_snapshots WHERE run_id = ? ORDER BY ticker", (run_id,)
            ).fetchall()]

    def get_all_runs(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM signal_runs ORDER BY run_id DESC"
            ).fetchall()]


class _ValidatorConnection:
    """Context manager koneksi SQLite kecil (commit/rollback/close)."""

    def __init__(self, db_path: str):
        import sqlite3
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.conn.close()
        return False

    def execute(self, sql: str, params: tuple = ()):
        return self.conn.execute(sql, params)
