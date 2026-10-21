"""
IRRBB parameters.

Sources
-------
* BCBS, Interest rate risk in the banking book (April 2016), standardised
  framework (Tables 1-4) and Annex 2; shock sizes recalibrated in BCBS,
  Recalibration of shocks for interest rate risk in the banking book (July
  2024), SRP31.90 Table 2.
* EU: EBA final draft RTS on supervisory outlier tests (EBA/RTS/2022/10),
  adopted as Commission Delegated Regulation (EU) 2024/856, with the large
  NII decline threshold raised to 5% of Tier 1 (EBA Opinion, 27 April 2023);
  CRD Article 98(5) (EVE outlier test at 15% of Tier 1); EBA/GL/2022/14 on
  IRRBB and CSRBB.
"""
from __future__ import annotations

import numpy as np

# 19 time buckets and midpoints (BCBS 2016, Table 1)
BUCKET_EDGES = [0.0, 1 / 365, 1 / 12, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 15.0, 20.0, np.inf]
MIDPOINTS = np.array([0.0028, 0.0417, 0.1667, 0.375, 0.625, 0.875, 1.25, 1.75, 2.5, 3.5, 4.5, 5.5,
                      6.5, 7.5, 8.5, 9.5, 12.5, 17.5, 25.0])
BUCKET_LABELS = ["O/N", "O/N–1M", "1M–3M", "3M–6M", "6M–9M", "9M–1Y", "1Y–1.5Y", "1.5Y–2Y", "2Y–3Y",
                 "3Y–4Y", "4Y–5Y", "5Y–6Y", "6Y–7Y", "7Y–8Y", "8Y–9Y", "9Y–10Y", "10Y–15Y", "15Y–20Y", ">20Y"]

SCENARIOS = ["Parallel up", "Parallel down", "Steepener", "Flattener", "Short rates up", "Short rates down"]
NII_SCENARIOS = ["Parallel up", "Parallel down"]

# Shock sizes in basis points (parallel, short, long)
SHOCKS_BCBS_2024 = {  # SRP31.90 Table 2, July 2024 recalibration
    "ARS": (400, 500, 300), "AUD": (350, 425, 300), "BRL": (400, 500, 300), "CAD": (200, 275, 175),
    "CHF": (175, 250, 200), "CNY": (225, 300, 150), "EUR": (225, 350, 200), "GBP": (275, 425, 250),
    "HKD": (225, 375, 200), "IDR": (400, 500, 300), "INR": (325, 475, 225), "JPY": (100, 100, 100),
    "KRW": (225, 350, 225), "MXN": (400, 500, 200), "RUB": (400, 500, 300), "SAR": (275, 375, 250),
    "SEK": (275, 425, 200), "SGD": (175, 250, 225), "TRY": (400, 500, 300), "USD": (200, 300, 225),
    "ZAR": (325, 500, 225),
}
SHOCKS_EU_RTS = {  # EBA/RTS/2022/10 Annex I (Delegated Regulation (EU) 2024/856)
    "ARS": (400, 500, 300), "AUD": (300, 450, 200), "BGN": (250, 350, 150), "BRL": (400, 500, 300),
    "CAD": (200, 300, 150), "CHF": (100, 150, 100), "CNY": (250, 300, 150), "CZK": (200, 250, 100),
    "DKK": (200, 250, 150), "EUR": (200, 250, 100), "GBP": (250, 300, 150), "HKD": (200, 250, 100),
    "HRK": (250, 400, 200), "HUF": (300, 450, 200), "IDR": (400, 500, 350), "INR": (400, 500, 300),
    "JPY": (100, 100, 100), "KRW": (300, 400, 200), "MXN": (400, 500, 300), "PLN": (250, 350, 150),
    "RON": (350, 500, 250), "RUB": (400, 500, 300), "SAR": (200, 300, 150), "SEK": (200, 300, 150),
    "SGD": (150, 200, 100), "TRY": (400, 500, 300), "USD": (200, 300, 150), "ZAR": (400, 500, 300),
}

# Behavioural scalars (BCBS 2016 Tables 3 and 4)
CPR_MULT = dict(zip(SCENARIOS, [0.8, 1.2, 0.8, 1.2, 0.8, 1.2]))
TDRR_MULT = dict(zip(SCENARIOS, [1.2, 0.8, 0.8, 1.2, 1.2, 0.8]))

# NMD caps (BCBS 2016 Table 2): (cap on core share, cap on average maturity of core in years)
NMD_CAPS = {"Retail transactional": (0.90, 5.0), "Retail non-transactional": (0.70, 4.5), "Wholesale": (0.50, 4.0)}

KAO_VOL_UPLIFT = 1.25            # +25% relative implied volatility for automatic options
EVE_SOT_THRESHOLD = 0.15         # BCBS SRP31 / CRD Art. 98(5)(a)
NII_SOT_THRESHOLD_EU = 0.05      # Delegated Regulation (EU) 2024/856, Art. 6 (as amended)
EU_GAIN_WEIGHT = 0.50            # RTS Art. 4(l): positive changes weighted 50%


def eu_floor(t):
    """RTS Art. 4(k): -150 bp at zero maturity rising 3 bp a year to 0% at 50 years."""
    t = np.asarray(t, dtype=float)
    return np.minimum(-0.015 + 0.0003 * t, 0.0)


def shock_vector(scenario: str, shocks_bp: tuple, t=MIDPOINTS) -> np.ndarray:
    """Annex 2 / SRP31.91 scenario shapes, in decimal, at maturities t."""
    par, sh, lo = (x / 1e4 for x in shocks_bp)
    s_short = np.exp(-np.asarray(t) / 4.0)
    d_short, d_long = sh * s_short, lo * (1 - s_short)
    return {
        "Parallel up": np.full_like(s_short, par),
        "Parallel down": np.full_like(s_short, -par),
        "Steepener": -0.65 * np.abs(d_short) + 0.9 * np.abs(d_long),
        "Flattener": 0.8 * np.abs(d_short) - 0.6 * np.abs(d_long),
        "Short rates up": d_short,
        "Short rates down": -d_short,
    }[scenario]
