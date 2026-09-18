"""
technical_indicators.py
========================

Modul untuk menghitung indikator teknikal dari data OHLCV (Open, High, Low,
Close, Volume) menggunakan library `ta` (Technical Analysis Library in
Python: https://github.com/bukosabino/ta).

Input yang diharapkan
----------------------
DataFrame `df` dengan kolom minimal (case-sensitive, huruf kecil):
    - timestamp : datetime / str, urut ascending (data lama -> baru)
    - open      : float
    - high      : float
    - low       : float
    - close     : float
    - volume    : float / int

Catatan penting:
    - Data HARUS sudah terurut ascending berdasarkan timestamp, karena
      semua indikator berbasis rolling window (SMA, EMA, ATR, dsb).
    - Setiap fungsi mengembalikan DataFrame baru (copy), tidak mengubah
      df asli secara in-place (menghindari SettingWithCopyWarning &
      efek samping yang tidak diinginkan).

Instalasi dependency:
    pip install ta pandas
"""

from __future__ import annotations

import pandas as pd
from ta.trend import (
    SMAIndicator,
    EMAIndicator,
    MACD,
    ADXIndicator,
    CCIIndicator,
    IchimokuIndicator,
    WMAIndicator,
)
from ta.momentum import (
    RSIIndicator,
    StochasticOscillator,
    StochRSIIndicator,
    WilliamsRIndicator,
    AwesomeOscillatorIndicator,
    ROCIndicator,
    UltimateOscillator,
)
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, ChaikinMoneyFlowIndicator

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _validate_ohlcv(df: pd.DataFrame) -> None:
    """Validasi minimal: kolom wajib tersedia dan timestamp terurut ascending."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    ts = pd.to_datetime(df["timestamp"])
    if not ts.is_monotonic_increasing:
        raise ValueError(
            "Kolom 'timestamp' harus terurut ascending (data lama -> baru) "
            "sebelum menghitung indikator berbasis rolling window."
        )


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan indikator tren ke DataFrame OHLCV.

    Kolom baru yang ditambahkan
    ----------------------------
    sma_20, sma_50, sma_200 : Simple Moving Average
        SMA(n) = (1/n) * sum(close[i-n+1 : i])
        Rata-rata harga penutupan sederhana selama n periode terakhir.

    ema_20, ema_50 : Exponential Moving Average
        EMA(t) = alpha * close(t) + (1 - alpha) * EMA(t-1),  alpha = 2 / (n + 1)
        Mirip SMA tapi memberi bobot lebih besar pada harga terbaru,
        sehingga lebih responsif terhadap perubahan harga terkini.

    macd, macd_signal, macd_hist : Moving Average Convergence Divergence
        macd        = EMA_12(close) - EMA_26(close)
        macd_signal = EMA_9(macd)
        macd_hist   = macd - macd_signal
        Mengukur momentum tren: MACD di atas signal line -> momentum naik,
        di bawah -> momentum turun. macd_hist menunjukkan kekuatan/percepatan
        selisih tersebut.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame OHLCV, terurut ascending by timestamp.

    Returns
    -------
    pd.DataFrame
        Copy dari df dengan kolom-kolom di atas ditambahkan.
    """
    _validate_ohlcv(df)
    out = df.copy()

    out["sma_20"] = SMAIndicator(close=out["close"], window=20).sma_indicator()
    out["sma_50"] = SMAIndicator(close=out["close"], window=50).sma_indicator()
    out["sma_200"] = SMAIndicator(close=out["close"], window=200).sma_indicator()

    out["ema_20"] = EMAIndicator(close=out["close"], window=20).ema_indicator()
    out["ema_50"] = EMAIndicator(close=out["close"], window=50).ema_indicator()

    macd_ind = MACD(
        close=out["close"],
        window_slow=26,
        window_fast=12,
        window_sign=9,
    )
    out["macd"] = macd_ind.macd()
    out["macd_signal"] = macd_ind.macd_signal()
    out["macd_hist"] = macd_ind.macd_diff()

    return out


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan indikator momentum ke DataFrame OHLCV.

    Kolom baru yang ditambahkan
    ----------------------------
    rsi_14 : Relative Strength Index (periode 14)
        RS  = avg_gain(14) / avg_loss(14)
        RSI = 100 - (100 / (1 + RS))
        Mengukur kecepatan & besaran perubahan harga. RSI > 70 umumnya
        dianggap overbought, RSI < 30 dianggap oversold.

    stoch_k, stoch_d : Stochastic Oscillator
        %K = 100 * (close - low_14) / (high_14 - low_14)
        %D = SMA_3(%K)
        di mana low_14/high_14 adalah nilai terendah/tertinggi selama
        14 periode terakhir. Mengukur posisi harga penutupan relatif
        terhadap rentang harga (high-low) dalam periode tersebut.
        %K > 80 -> overbought, %K < 20 -> oversold. %D adalah versi
        smoothed dari %K, dipakai sebagai signal line.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame OHLCV, terurut ascending by timestamp.

    Returns
    -------
    pd.DataFrame
        Copy dari df dengan kolom-kolom di atas ditambahkan.
    """
    _validate_ohlcv(df)
    out = df.copy()

    out["rsi_14"] = RSIIndicator(close=out["close"], window=14).rsi()

    stoch = StochasticOscillator(
        high=out["high"],
        low=out["low"],
        close=out["close"],
        window=14,
        smooth_window=3,
    )
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()

    return out


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan indikator volatilitas ke DataFrame OHLCV.

    Kolom baru yang ditambahkan
    ----------------------------
    atr_14 : Average True Range (periode 14)  [PENTING - dipakai untuk stop-loss]
        TR = max(
            high - low,
            abs(high - close_prev),
            abs(low  - close_prev)
        )
        ATR_14 = EMA/Wilder-smoothed rata-rata TR selama 14 periode.
        Mengukur besaran pergerakan harga (volatilitas) tanpa memandang
        arah. Umum dipakai untuk menentukan jarak stop-loss dinamis,
        misal: stop_loss = entry_price - (k * atr_14).

    bb_upper, bb_middle, bb_lower : Bollinger Bands (20 periode, 2 std dev)
        bb_middle = SMA_20(close)
        std       = rolling_std_20(close)
        bb_upper  = bb_middle + 2 * std
        bb_lower  = bb_middle - 2 * std
        Mengukur volatilitas relatif terhadap rata-rata bergerak; band
        yang melebar menandakan volatilitas meningkat, band yang
        menyempit (squeeze) menandakan volatilitas rendah.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame OHLCV, terurut ascending by timestamp.

    Returns
    -------
    pd.DataFrame
        Copy dari df dengan kolom-kolom di atas ditambahkan.
    """
    _validate_ohlcv(df)
    out = df.copy()

    atr_ind = AverageTrueRange(
        high=out["high"],
        low=out["low"],
        close=out["close"],
        window=14,
    )
    out["atr_14"] = atr_ind.average_true_range()

    bb = BollingerBands(close=out["close"], window=20, window_dev=2)
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_middle"] = bb.bollinger_mavg()
    out["bb_lower"] = bb.bollinger_lband()

    return out


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan indikator volume ke DataFrame OHLCV.

    Kolom baru yang ditambahkan
    ----------------------------
    obv : On-Balance Volume
        Jika close(t) > close(t-1): OBV(t) = OBV(t-1) + volume(t)
        Jika close(t) < close(t-1): OBV(t) = OBV(t-1) - volume(t)
        Jika close(t) == close(t-1): OBV(t) = OBV(t-1)
        Kumulatif volume yang "mengonfirmasi" arah harga; dipakai untuk
        melihat apakah tren harga didukung oleh tekanan beli/jual yang
        sesuai (divergence antara harga dan OBV bisa jadi sinyal
        pembalikan).

    volume_sma_20 : Simple Moving Average dari volume, periode 20
        volume_sma_20(t) = (1/20) * sum(volume[t-19 : t])
        Rata-rata volume perdagangan 20 periode terakhir, jadi baseline
        "volume normal".

    volume_spike_ratio : Rasio volume hari ini terhadap volume_sma_20
        volume_spike_ratio(t) = volume(t) / volume_sma_20(t)
        Nilai > 1 berarti volume hari ini di atas rata-rata (potensi
        lonjakan minat/aktivitas); semakin tinggi rasio, semakin
        signifikan lonjakannya.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame OHLCV, terurut ascending by timestamp.

    Returns
    -------
    pd.DataFrame
        Copy dari df dengan kolom-kolom di atas ditambahkan.
    """
    _validate_ohlcv(df)
    out = df.copy()

    out["obv"] = OnBalanceVolumeIndicator(
        close=out["close"], volume=out["volume"]
    ).on_balance_volume()

    out["volume_sma_20"] = out["volume"].rolling(window=20, min_periods=20).mean()
    out["volume_spike_ratio"] = out["volume"] / out["volume_sma_20"]

    return out


# ---------------------------------------------------------------------------
# EXTENDED INDICATOR SET (untuk voting/rating ala TradingView & Investing.com)
# ---------------------------------------------------------------------------
# Indikator di bawah ini adalah superset dari indikator dasar di atas dan
# dipakai oleh signal_engine (mode voting) untuk mereplikasi metodologi
# "Technical Ratings" ala TradingView / gauges ala Investing.com.


def add_extended_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan moving average tambahan + indikator tren standar yang dipakai
    rating teknikal publik (TradingView/Investing.com).

    Kolom baru:
        sma_10, sma_30, sma_100   : SMA tambahan (TV memakai SMA 10/20/30/50/100/200)
        ema_10, ema_100, ema_13   : EMA tambahan (ema_13 = baseline Bull Bear Power)
        hull_9                    : Hull MA(9)
            HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n)), n = 9
            MA yang mengurangi lag dibanding SMA/EMA biasa.
        vwma_20                   : Volume Weighted MA(20)
            VWMA = sum(close*vol, 20) / sum(vol, 20)
            Seperti SMA20 tapi memberi bobot lebih pada hari bervolume besar.
        ichimoku_conversion/base/span_a/span_b : Ichimoku Cloud (9, 26, 52)
            conversion = (high9 + low9)/2, base = (high26 + low26)/2
            span_a = (conversion + base)/2, span_b = (high52 + low52)/2
        adx, adx_pos, adx_neg     : ADX(14) + +DI/-DI (kekuatan & arah tren)
        cci_20                    : Commodity Channel Index(20)
    """
    _validate_ohlcv(df)
    out = df.copy()
    c, h, l, v = out["close"], out["high"], out["low"], out["volume"]

    out["sma_10"] = SMAIndicator(close=c, window=10).sma_indicator()
    out["sma_30"] = SMAIndicator(close=c, window=30).sma_indicator()
    out["sma_100"] = SMAIndicator(close=c, window=100).sma_indicator()

    out["ema_10"] = EMAIndicator(close=c, window=10).ema_indicator()
    out["ema_13"] = EMAIndicator(close=c, window=13).ema_indicator()
    out["ema_100"] = EMAIndicator(close=c, window=100).ema_indicator()

    # Hull MA(9): WMA(2*WMA(4) - WMA(9), 3)
    wma_half = WMAIndicator(close=c, window=4).wma()
    wma_full = WMAIndicator(close=c, window=9).wma()
    out["hull_9"] = WMAIndicator(close=2 * wma_half - wma_full, window=3).wma()

    # Volume Weighted MA(20)
    out["vwma_20"] = (c * v).rolling(20).sum() / v.rolling(20).sum()

    ichi = IchimokuIndicator(high=h, low=l, window1=9, window2=26, window3=52)
    out["ichimoku_conversion"] = ichi.ichimoku_conversion_line()
    out["ichimoku_base"] = ichi.ichimoku_base_line()
    out["ichimoku_span_a"] = ichi.ichimoku_a()
    out["ichimoku_span_b"] = ichi.ichimoku_b()

    adx_ind = ADXIndicator(high=h, low=l, close=c, window=14)
    out["adx"] = adx_ind.adx()
    out["adx_pos"] = adx_ind.adx_pos()
    out["adx_neg"] = adx_ind.adx_neg()

    out["cci_20"] = CCIIndicator(high=h, low=l, close=c, window=20).cci()

    return out


def add_extended_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan osilator tambahan yang dipakai rating teknikal publik.

    Kolom baru:
        stochrsi_k, stochrsi_d    : Stochastic RSI (3,3,14,14) — RSI yang
                                    di-stochastic-kan; 0-100, >80 overbought.
        williams_r                : Williams %R(14), rentang -100..0;
                                    > -20 overbought, < -80 oversold.
        awesome_oscillator        : Awesome Oscillator = SMA5(median) - SMA34(median)
        momentum_10               : Momentum(10) = close - close[10 periode lalu]
        ultimate_oscillator       : Ultimate Oscillator (7, 14, 28), 0-100
        roc_9                     : Rate of Change(9), dalam persen
        bull_power, bear_power    : Bull/Bear Power(13) = high/low - EMA13
    """
    _validate_ohlcv(df)
    out = df.copy()
    c, h, l = out["close"], out["high"], out["low"]

    stoch_rsi = StochRSIIndicator(close=c, window=14, smooth1=3, smooth2=3)
    out["stochrsi_k"] = stoch_rsi.stochrsi_k()
    out["stochrsi_d"] = stoch_rsi.stochrsi_d()

    out["williams_r"] = WilliamsRIndicator(high=h, low=l, close=c, lbp=14).williams_r()

    out["awesome_oscillator"] = AwesomeOscillatorIndicator(
        high=h, low=l, window1=5, window2=34
    ).awesome_oscillator()

    out["momentum_10"] = c - c.shift(10)

    out["ultimate_oscillator"] = UltimateOscillator(
        high=h, low=l, close=c, window1=7, window2=14, window3=28
    ).ultimate_oscillator()

    out["roc_9"] = ROCIndicator(close=c, window=9).roc()

    ema_13 = out["ema_13"] if "ema_13" in out.columns else EMAIndicator(close=c, window=13).ema_indicator()
    out["bull_power"] = h - ema_13
    out["bear_power"] = l - ema_13

    return out


def add_extended_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan indikator volume tambahan.

    Kolom baru:
        mfi_14 : Money Flow Index(14) — "RSI-nya volume", 0-100;
                 < 20 oversold, > 80 overbought.
        cmf_20 : Chaikin Money Flow(20) — akumulasi/distribusi berbasis
                 jarak close dari high-low; > 0 akumulasi, < 0 distribusi.
    """
    _validate_ohlcv(df)
    out = df.copy()
    h, l, c, v = out["high"], out["low"], out["close"], out["volume"]

    out["mfi_14"] = MFIIndicator(high=h, low=l, close=c, volume=v, window=14).money_flow_index()
    out["cmf_20"] = ChaikinMoneyFlowIndicator(high=h, low=l, close=c, volume=v, window=20).chaikin_money_flow()

    return out


def add_extended_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Jalankan seluruh fungsi indikator extended secara berurutan."""
    out = add_trend_indicators(df)
    out = add_momentum_indicators(out)
    out = add_volatility_indicators(out)
    out = add_volume_indicators(out)
    out = add_extended_trend_indicators(out)
    out = add_extended_momentum_indicators(out)
    out = add_extended_volume_indicators(out)
    return out


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper yang menjalankan seluruh fungsi indikator secara berurutan
    (trend -> momentum -> volatility -> volume), lalu membuang baris-baris
    awal yang masih mengandung NaN akibat rolling window (misal sma_200
    baru punya nilai valid mulai baris ke-200).

    Urutan pemanggilan:
        1. add_trend_indicators
        2. add_momentum_indicators
        3. add_volatility_indicators
        4. add_volume_indicators
        5. add_extended_trend_indicators
        6. add_extended_momentum_indicators
        7. add_extended_volume_indicators
        8. dropna() pada baris-baris awal

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame OHLCV mentah dengan kolom minimal:
        timestamp, open, high, low, close, volume (terurut ascending).

    Returns
    -------
    pd.DataFrame
        DataFrame lengkap dengan seluruh kolom indikator, tanpa baris
        yang mengandung NaN dari warm-up period rolling window.
        Index di-reset (0, 1, 2, ...).
    """
    _validate_ohlcv(df)

    out = df.copy()
    out = add_trend_indicators(out)
    out = add_momentum_indicators(out)
    out = add_volatility_indicators(out)
    out = add_volume_indicators(out)
    out = add_extended_trend_indicators(out)
    out = add_extended_momentum_indicators(out)
    out = add_extended_volume_indicators(out)

    out = out.dropna().reset_index(drop=True)

    return out


if __name__ == "__main__":
    # Contoh penggunaan singkat dengan data dummy
    import numpy as np

    n = 300
    rng = pd.date_range("2024-01-01", periods=n, freq="D")
    rs = np.random.default_rng(42)
    close = 100 + np.cumsum(rs.normal(0, 1, n))
    high = close + rs.uniform(0, 2, n)
    low = close - rs.uniform(0, 2, n)
    open_ = close + rs.uniform(-1, 1, n)
    volume = rs.uniform(1000, 5000, n)

    dummy_df = pd.DataFrame(
        {
            "timestamp": rng,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    features = build_feature_set(dummy_df)
    print(features.tail())
    print(f"\nJumlah baris setelah dropna: {len(features)} (dari {len(dummy_df)})")