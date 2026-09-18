"""
risk_manager.py
================

Modul manajemen risiko (Risk Management) yang TERPISAH sepenuhnya dari
signal engine. Modul ini tidak menghasilkan sinyal trading (BUY/SELL) —
tugasnya hanya menerima sinyal dari luar (signal engine) lalu menghitung
stop loss, take profit, ukuran posisi, dan memutuskan apakah trade boleh
dieksekusi berdasarkan aturan risiko yang berlaku.

Prinsip desain:
- Semua parameter risiko dapat dikonfigurasi lewat constructor (tidak ada
  angka hardcoded di dalam logic).
- Setiap method independen dan bisa dipanggil satu per satu, atau lewat
  `evaluate_trade_plan()` sebagai orchestrator.
- Tidak bergantung pada implementasi signal engine apa pun.

Dependensi: pandas (untuk parameter `portfolio_history`), tidak ada
dependensi ke modul sinyal.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

BUY = "BUY"
SELL = "SELL"
_VALID_DIRECTIONS = {BUY, SELL}


class RiskManagerError(ValueError):
    """Exception untuk input yang tidak valid ke RiskManager."""


@dataclass
class RiskManager:
    """
    Konfigurasi & logic manajemen risiko trading.

    Parameters
    ----------
    max_position_size_pct : float
        Batas maksimal alokasi capital per posisi (mis. 0.05 = 5% dari capital).
    atr_multiplier_stoploss : float
        Pengali ATR untuk menentukan jarak stop loss dari harga entry.
    min_risk_reward_ratio : float
        Rasio minimum take_profit_distance / stop_loss_distance yang wajib
        dipenuhi agar sebuah trade layak diambil (mis. 2.0 = TP minimal 2x SL).
    max_daily_drawdown_pct : float
        Circuit breaker harian: jika drawdown harian melebihi ini, trading
        dihentikan untuk hari itu.
    max_weekly_drawdown_pct : float
        Circuit breaker mingguan: jika drawdown mingguan melebihi ini, trading
        dihentikan untuk minggu itu.
    risk_per_trade_pct : float
        Parameter tambahan (dibuat configurable, bukan hardcoded) untuk
        risk-based position sizing: persentase maksimal capital yang boleh
        hilang dalam satu trade apabila stop loss tersentuh
        (mis. 0.01 = maksimal rugi 1% capital per trade).
    position_precision : int
        Jumlah desimal untuk pembulatan (floor) ukuran posisi/unit, berguna
        untuk instrumen fraksional seperti crypto.
    capital : float, optional
        Modal yang tersedia untuk trading. Dipakai oleh method `evaluate()`
        (adapter untuk run_daily_simulation_job). Bisa di-set setelah
        instansiasi, mis. `rm.capital = 100_000_000`.
    """

    max_position_size_pct: float = 0.05
    atr_multiplier_stoploss: float = 2.0
    min_risk_reward_ratio: float = 2.0
    max_daily_drawdown_pct: float = 0.03
    max_weekly_drawdown_pct: float = 0.08
    risk_per_trade_pct: float = 0.01
    position_precision: int = 6
    capital: Optional[float] = None

    def __post_init__(self) -> None:
        self._validate_config()

    # ------------------------------------------------------------------ #
    # Validasi konfigurasi
    # ------------------------------------------------------------------ #
    def _validate_config(self) -> None:
        checks = {
            "max_position_size_pct": self.max_position_size_pct,
            "atr_multiplier_stoploss": self.atr_multiplier_stoploss,
            "min_risk_reward_ratio": self.min_risk_reward_ratio,
            "max_daily_drawdown_pct": self.max_daily_drawdown_pct,
            "max_weekly_drawdown_pct": self.max_weekly_drawdown_pct,
            "risk_per_trade_pct": self.risk_per_trade_pct,
        }
        for name, value in checks.items():
            if value is None or value <= 0:
                raise RiskManagerError(f"{name} harus bernilai positif, dapat: {value}")

        if not (0 < self.max_position_size_pct <= 1):
            raise RiskManagerError("max_position_size_pct harus dalam rentang (0, 1]")
        if not (0 < self.risk_per_trade_pct <= 1):
            raise RiskManagerError("risk_per_trade_pct harus dalam rentang (0, 1]")

    @staticmethod
    def _validate_direction(direction: str) -> str:
        if not isinstance(direction, str):
            raise RiskManagerError(f"direction harus string, dapat: {type(direction)}")
        direction_upper = direction.upper()
        if direction_upper not in _VALID_DIRECTIONS:
            raise RiskManagerError(
                f"direction tidak valid: '{direction}'. Harus salah satu dari {_VALID_DIRECTIONS}"
            )
        return direction_upper

    # ------------------------------------------------------------------ #
    # 1. Stop Loss
    # ------------------------------------------------------------------ #
    def calculate_stop_loss(self, entry_price: float, atr: float, direction: str) -> float:
        """
        Hitung stop loss berbasis ATR.

        BUY  -> stop_loss = entry_price - (atr_multiplier_stoploss * atr)
        SELL -> stop_loss = entry_price + (atr_multiplier_stoploss * atr)
        """
        if entry_price <= 0:
            raise RiskManagerError(f"entry_price harus positif, dapat: {entry_price}")
        if atr <= 0:
            raise RiskManagerError(f"atr harus positif, dapat: {atr}")

        direction = self._validate_direction(direction)
        distance = self.atr_multiplier_stoploss * atr

        if direction == BUY:
            stop_loss = entry_price - distance
        else:  # SELL
            stop_loss = entry_price + distance

        if stop_loss <= 0:
            raise RiskManagerError(
                f"Hasil stop_loss <= 0 ({stop_loss}). Cek nilai entry_price/atr/multiplier."
            )

        logger.debug("calculate_stop_loss: %s entry=%.6f atr=%.6f -> sl=%.6f",
                     direction, entry_price, atr, stop_loss)
        return stop_loss

    # ------------------------------------------------------------------ #
    # 2. Take Profit
    # ------------------------------------------------------------------ #
    def calculate_take_profit(self, entry_price: float, stop_loss: float, direction: str) -> float:
        """
        Hitung take profit berdasarkan min_risk_reward_ratio.

        risk = abs(entry_price - stop_loss)
        BUY  -> take_profit = entry_price + (risk * min_risk_reward_ratio)
        SELL -> take_profit = entry_price - (risk * min_risk_reward_ratio)
        """
        if entry_price <= 0:
            raise RiskManagerError(f"entry_price harus positif, dapat: {entry_price}")
        if stop_loss <= 0:
            raise RiskManagerError(f"stop_loss harus positif, dapat: {stop_loss}")

        direction = self._validate_direction(direction)
        risk = abs(entry_price - stop_loss)
        if risk == 0:
            raise RiskManagerError("Jarak stop_loss ke entry_price = 0, tidak bisa hitung TP.")

        reward = risk * self.min_risk_reward_ratio

        if direction == BUY:
            take_profit = entry_price + reward
        else:  # SELL
            take_profit = entry_price - reward

        logger.debug("calculate_take_profit: %s entry=%.6f sl=%.6f -> tp=%.6f",
                     direction, entry_price, stop_loss, take_profit)
        return take_profit

    # ------------------------------------------------------------------ #
    # 3. Position Sizing
    # ------------------------------------------------------------------ #
    def calculate_position_size(
        self, capital: float, entry_price: float, stop_loss: float
    ) -> Dict[str, float]:
        """
        Hitung ukuran posisi maksimal dengan MENGGABUNGKAN dua batasan:

        1. Batasan alokasi capital: nilai posisi tidak boleh melebihi
           `max_position_size_pct` dari capital.
        2. Batasan risk-based sizing: kerugian saat stop loss tersentuh
           tidak boleh melebihi `risk_per_trade_pct` dari capital.

        max_units diambil dari yang PALING KECIL (paling konservatif)
        di antara keduanya.

        Returns
        -------
        dict: {"max_units", "capital_allocated", "risk_amount"}
        """
        if capital <= 0:
            raise RiskManagerError(f"capital harus positif, dapat: {capital}")
        if entry_price <= 0:
            raise RiskManagerError(f"entry_price harus positif, dapat: {entry_price}")
        if stop_loss <= 0:
            raise RiskManagerError(f"stop_loss harus positif, dapat: {stop_loss}")

        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            raise RiskManagerError("risk_per_unit = 0 (entry_price == stop_loss), tidak bisa sizing.")

        # Batasan 1: berdasarkan alokasi capital maksimal per posisi
        max_capital_for_position = capital * self.max_position_size_pct
        units_by_capital_cap = max_capital_for_position / entry_price

        # Batasan 2: berdasarkan risiko maksimal per trade
        max_capital_at_risk = capital * self.risk_per_trade_pct
        units_by_risk_cap = max_capital_at_risk / risk_per_unit

        max_units = min(units_by_capital_cap, units_by_risk_cap)

        # Floor ke presisi tertentu supaya tidak "overshoot" limit karena pembulatan
        factor = 10 ** self.position_precision
        max_units = math.floor(max_units * factor) / factor
        max_units = max(max_units, 0.0)

        capital_allocated = max_units * entry_price
        risk_amount = max_units * risk_per_unit

        result = {
            "max_units": max_units,
            "capital_allocated": capital_allocated,
            "risk_amount": risk_amount,
        }
        logger.debug("calculate_position_size: capital=%.2f entry=%.6f sl=%.6f -> %s",
                     capital, entry_price, stop_loss, result)
        return result

    # ------------------------------------------------------------------ #
    # 4. Circuit Breaker (drawdown harian & mingguan)
    # ------------------------------------------------------------------ #
    def check_circuit_breaker(self, portfolio_history: pd.DataFrame) -> Dict[str, Any]:
        """
        Cek drawdown harian & mingguan dari histori portfolio.

        Parameters
        ----------
        portfolio_history : pd.DataFrame
            Wajib memiliki kolom "date" (datetime-like) dan "portfolio_value" (numeric),
            diurutkan atau tidak (akan diurutkan otomatis).

        Returns
        -------
        dict: {
            "trading_allowed": bool,
            "reason": str atau None,
            "daily_drawdown_pct": float,
            "weekly_drawdown_pct": float,
        }
        """
        required_cols = {"date", "portfolio_value"}
        if portfolio_history is None or not isinstance(portfolio_history, pd.DataFrame):
            raise RiskManagerError("portfolio_history harus berupa pandas.DataFrame")
        if not required_cols.issubset(portfolio_history.columns):
            raise RiskManagerError(
                f"portfolio_history harus memiliki kolom {required_cols}, "
                f"dapat: {set(portfolio_history.columns)}"
            )
        if portfolio_history.empty:
            raise RiskManagerError("portfolio_history kosong")

        df = portfolio_history.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        current_value = float(df["portfolio_value"].iloc[-1])
        latest_date = df["date"].iloc[-1].normalize()

        # --- Drawdown harian: dibandingkan terhadap nilai portfolio di awal hari ini
        day_mask = df["date"].dt.normalize() == latest_date
        day_data = df.loc[day_mask, "portfolio_value"]
        start_of_day_value = float(day_data.iloc[0])
        daily_drawdown_pct = self._safe_drawdown(start_of_day_value, current_value)

        # --- Drawdown mingguan: dibandingkan terhadap nilai portfolio di awal minggu ini
        # (minggu dimulai hari Senin, mengikuti ISO week)
        week_start_date = latest_date - timedelta(days=latest_date.weekday())
        week_mask = df["date"].dt.normalize() >= week_start_date
        week_data = df.loc[week_mask, "portfolio_value"]
        start_of_week_value = float(week_data.iloc[0])
        weekly_drawdown_pct = self._safe_drawdown(start_of_week_value, current_value)

        trading_allowed = True
        reasons = []

        if daily_drawdown_pct >= self.max_daily_drawdown_pct:
            trading_allowed = False
            reasons.append(
                f"Daily drawdown {daily_drawdown_pct:.2%} >= batas "
                f"{self.max_daily_drawdown_pct:.2%} (circuit breaker harian aktif)"
            )

        if weekly_drawdown_pct >= self.max_weekly_drawdown_pct:
            trading_allowed = False
            reasons.append(
                f"Weekly drawdown {weekly_drawdown_pct:.2%} >= batas "
                f"{self.max_weekly_drawdown_pct:.2%} (circuit breaker mingguan aktif)"
            )

        result = {
            "trading_allowed": trading_allowed,
            "reason": " | ".join(reasons) if reasons else None,
            "daily_drawdown_pct": daily_drawdown_pct,
            "weekly_drawdown_pct": weekly_drawdown_pct,
        }
        logger.debug("check_circuit_breaker: %s", result)
        return result

    @staticmethod
    def _safe_drawdown(start_value: float, current_value: float) -> float:
        """Drawdown positif berarti rugi. Nilai <= 0 dianggap tidak ada drawdown (0.0)."""
        if start_value <= 0:
            return 0.0
        drawdown = (start_value - current_value) / start_value
        return max(drawdown, 0.0)

    # ------------------------------------------------------------------ #
    # 5. Orchestrator: evaluate_trade_plan
    # ------------------------------------------------------------------ #
    def evaluate_trade_plan(
        self,
        signal: Dict[str, Any],
        current_price: float,
        atr: float,
        capital: float,
        portfolio_history: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrator: menggabungkan semua method di atas menjadi satu trade plan.

        Parameters
        ----------
        signal : dict
            Sinyal dari signal engine (bukan bagian dari modul ini). Wajib
            berisi key "direction" bernilai "BUY" atau "SELL". Boleh berisi
            key opsional "entry_price" — jika tidak ada, `current_price`
            akan dipakai sebagai entry_price.
        current_price : float
            Harga pasar saat ini.
        atr : float
            Nilai ATR terkini (dari price data, dihitung di luar modul ini).
        capital : float
            Modal yang tersedia untuk trading.
        portfolio_history : pd.DataFrame, optional
            Jika diberikan, `check_circuit_breaker` akan dijalankan untuk
            menentukan apakah trading masih diizinkan.

        Returns
        -------
        dict: {
            "entry_price", "stop_loss", "take_profit", "position_size",
            "risk_amount", "risk_reward_ratio", "approved", "rejection_reason"
        }
        """
        rejection_reasons = []

        # --- Validasi dasar sinyal
        direction = signal.get("direction") if isinstance(signal, dict) else None
        try:
            direction = self._validate_direction(direction)
        except RiskManagerError as exc:
            return {
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "position_size": None,
                "risk_amount": None,
                "risk_reward_ratio": None,
                "approved": False,
                "rejection_reason": f"Sinyal tidak valid: {exc}",
            }

        entry_price = signal.get("entry_price", current_price) if isinstance(signal, dict) else current_price

        # --- Circuit breaker (jika histori portfolio disediakan)
        if portfolio_history is not None:
            try:
                cb_result = self.check_circuit_breaker(portfolio_history)
            except RiskManagerError as exc:
                cb_result = {"trading_allowed": False, "reason": f"Circuit breaker error: {exc}"}
            if not cb_result["trading_allowed"]:
                rejection_reasons.append(cb_result["reason"])

        # --- Hitung stop loss & take profit
        try:
            stop_loss = self.calculate_stop_loss(entry_price, atr, direction)
            take_profit = self.calculate_take_profit(entry_price, stop_loss, direction)
        except RiskManagerError as exc:
            return {
                "entry_price": entry_price,
                "stop_loss": None,
                "take_profit": None,
                "position_size": None,
                "risk_amount": None,
                "risk_reward_ratio": None,
                "approved": False,
                "rejection_reason": f"Gagal hitung SL/TP: {exc}",
            }

        risk_distance = abs(entry_price - stop_loss)
        reward_distance = abs(take_profit - entry_price)
        risk_reward_ratio = reward_distance / risk_distance if risk_distance > 0 else 0.0

        if risk_reward_ratio < self.min_risk_reward_ratio:
            rejection_reasons.append(
                f"Risk/reward ratio {risk_reward_ratio:.2f} < minimum "
                f"{self.min_risk_reward_ratio:.2f}"
            )

        # --- Position sizing
        try:
            sizing = self.calculate_position_size(capital, entry_price, stop_loss)
        except RiskManagerError as exc:
            return {
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size": None,
                "risk_amount": None,
                "risk_reward_ratio": risk_reward_ratio,
                "approved": False,
                "rejection_reason": f"Gagal hitung position size: {exc}",
            }

        if sizing["max_units"] <= 0:
            rejection_reasons.append(
                "Position size = 0 (capital terlalu kecil relatif terhadap risiko per unit)"
            )

        approved = len(rejection_reasons) == 0
        rejection_reason = " | ".join(rejection_reasons) if rejection_reasons else None

        trade_plan = {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": sizing["max_units"],
            "risk_amount": sizing["risk_amount"],
            "risk_reward_ratio": risk_reward_ratio,
            "approved": approved,
            "rejection_reason": rejection_reason,
        }
        logger.info("evaluate_trade_plan: direction=%s -> approved=%s", direction, approved)
        return trade_plan

    # ------------------------------------------------------------------ #
    # 6. Adapter: evaluate (cocok dengan Protocol RiskManager di
    #    paper_trading_simulator.py -> run_daily_simulation_job)
    # ------------------------------------------------------------------ #
    def evaluate(self, signal: Dict[str, Any], indicators: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Adapter agar cocok dengan Protocol RiskManager yang diharapkan
        run_daily_simulation_job() di paper_trading_simulator.py.

        Membutuhkan atribut `self.capital` sudah di-set, mis.:
            rm = RiskManager(...)
            rm.capital = 100_000_000

        Parameters
        ----------
        signal : dict
            Wajib berisi "direction" ("BUY"/"SELL"). Boleh berisi "entry_price".
        indicators : dict
            Wajib berisi harga saat ini di key "close", dan ATR di key
            "atr_14" atau "atr".

        Returns
        -------
        dict atau None
            None jika trade ditolak risk manager. Jika disetujui, dict
            dengan key: entry_price, stop_loss_price, take_profit_price,
            position_size (siap dipakai PaperTradingSimulator.open_trade).
        """
        current_price = indicators.get("close")
        if current_price is None:
            current_price = signal.get("entry_price")

        atr = indicators.get("atr_14")
        if atr is None:
            atr = indicators.get("atr")

        if self.capital is None:
            raise RiskManagerError(
                "RiskManager.capital belum di-set. Contoh: rm.capital = 100_000_000"
            )

        plan = self.evaluate_trade_plan(
            signal=signal,
            current_price=current_price,
            atr=atr,
            capital=self.capital,
        )
        if not plan["approved"]:
            return None

        return {
            "entry_price": plan["entry_price"],
            "stop_loss_price": plan["stop_loss"],
            "take_profit_price": plan["take_profit"],
            "position_size": plan["position_size"],
        }


# ---------------------------------------------------------------------- #
# Contoh pemakaian mandiri (tidak dijalankan saat di-import sebagai modul)
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rm = RiskManager(
        max_position_size_pct=0.05,
        atr_multiplier_stoploss=2.0,
        min_risk_reward_ratio=2.0,
        max_daily_drawdown_pct=0.03,
        max_weekly_drawdown_pct=0.08,
        risk_per_trade_pct=0.01,
    )

    # Sinyal contoh, biasanya datang dari signal engine terpisah
    dummy_signal = {"direction": "BUY", "confidence": 0.8}

    # Histori portfolio contoh (untuk circuit breaker)
    history = pd.DataFrame(
        {
            "date": pd.date_range("2026-09-01", periods=3, freq="D"),
            "portfolio_value": [100_000, 99_000, 97_500],
        }
    )

    plan = rm.evaluate_trade_plan(
        signal=dummy_signal,
        current_price=50_000,
        atr=800,
        capital=100_000,
        portfolio_history=history,
    )
    print(plan)

    # Contoh pemakaian adapter evaluate() (untuk run_daily_simulation_job)
    rm.capital = 100_000
    adapter_result = rm.evaluate(
        signal=dummy_signal,
        indicators={"close": 50_000, "atr_14": 800},
    )
    print(adapter_result)