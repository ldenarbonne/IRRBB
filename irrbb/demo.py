"""Demonstration IRRBB book of a mid-sized bank (EUR m equivalent, fictitious)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import POS_COLUMNS, Curve

TENORS = [1 / 365, 1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]


def demo_curves() -> dict:
    return {
        "EUR": Curve(TENORS, [0.0193, 0.0195, 0.0198, 0.0200, 0.0203, 0.0212, 0.0222, 0.0240, 0.0255,
                              0.0268, 0.0282, 0.0286, 0.0280]),
        "USD": Curve(TENORS, [0.0375, 0.0372, 0.0370, 0.0362, 0.0350, 0.0338, 0.0335, 0.0340, 0.0350,
                              0.0360, 0.0375, 0.0383, 0.0380]),
    }


N = np.nan
ROWS = [
    # id, name, ccy, side, type, notional, rate%, spread%, maturity, reset, freq, cpr%, tdrr%, nmd_segment,
    # core%, core_mat, beta%, accounting, rating, strike%, vol_bp, option_side
    ("A1", "Cash and central bank reserves", "EUR", "Asset", "Overnight", 4000, 1.93, 0.0, 0.003, N, N, N, N, "", N, N, N, "AC", "", N, N, ""),
    ("A2", "Sovereign bonds (FVOCI)", "EUR", "Asset", "Fixed bullet", 4500, 2.60, 0.25, 6, N, 1, N, N, "", N, N, N, "FVOCI", "AAA-AA", N, N, ""),
    ("A3", "Corporate and covered bonds (FVOCI)", "EUR", "Asset", "Fixed bullet", 2000, 3.10, 0.80, 4, N, 1, N, N, "", N, N, N, "FVOCI", "A", N, N, ""),
    ("A4", "Interbank loans and reverse repos", "EUR", "Asset", "Floating", 3000, 2.05, 0.05, 0.5, 0.0833, N, N, N, "", N, N, N, "AC", "", N, N, ""),
    ("A5", "Corporate loans – floating (3M Euribor + margin)", "EUR", "Asset", "Floating", 6000, 3.80, 1.80, 5, 0.25, N, N, N, "", N, N, N, "AC", "", N, N, ""),
    ("A6", "Corporate loans – fixed bullet", "EUR", "Asset", "Fixed bullet", 3000, 4.20, 1.80, 4, N, 1, N, N, "", N, N, N, "AC", "", N, N, ""),
    ("A7", "Residential mortgages – fixed, prepayable", "EUR", "Asset", "Fixed amortising", 14000, 3.20, 0.90, 20, N, N, 5, N, "", N, N, N, "AC", "", N, N, ""),
    ("A8", "Consumer and SME loans – fixed amortising", "EUR", "Asset", "Fixed amortising", 5000, 6.50, 4.20, 5, N, N, 8, N, "", N, N, N, "AC", "", N, N, ""),
    ("A9", "USD corporate loans – floating", "USD", "Asset", "Floating", 2500, 5.40, 1.70, 4, 0.25, N, N, N, "", N, N, N, "AC", "", N, N, ""),
    ("A10", "USD bonds (amortised cost)", "USD", "Asset", "Fixed bullet", 1000, 4.10, 0.75, 3, N, 1, N, N, "", N, N, N, "AC", "A", N, N, ""),
    ("L1", "Retail sight deposits (transactional)", "EUR", "Liability", "NMD", 12000, 0.10, N, N, N, N, N, N, "Retail transactional", 85, 6.0, 20, "AC", "", N, N, ""),
    ("L2", "Retail savings accounts (non-transactional)", "EUR", "Liability", "NMD", 6000, 1.25, N, N, N, N, N, N, "Retail non-transactional", 75, 4.0, 50, "AC", "", N, N, ""),
    ("L3", "Corporate sight deposits (wholesale)", "EUR", "Liability", "NMD", 5000, 1.00, N, N, N, N, N, N, "Wholesale", 45, 3.0, 70, "AC", "", N, N, ""),
    ("L4", "Retail term deposits (redeemable)", "EUR", "Liability", "Term deposit", 5000, 2.60, 0.40, 2, N, 1, N, 5, "", N, N, N, "AC", "", N, N, ""),
    ("L5", "Senior unsecured bonds", "EUR", "Liability", "Fixed bullet", 6000, 3.40, 1.00, 5, N, 1, N, N, "", N, N, N, "AC", "", N, N, ""),
    ("L6", "Covered bonds issued", "EUR", "Liability", "Fixed bullet", 3000, 2.90, 0.35, 7, N, 1, N, N, "", N, N, N, "AC", "", N, N, ""),
    ("L7", "Interbank and repo funding", "EUR", "Liability", "Floating", 2500, 2.00, 0.05, 0.25, 0.0833, N, N, N, "", N, N, N, "AC", "", N, N, ""),
    ("L8", "Central bank funding", "EUR", "Liability", "Fixed bullet", 1000, 2.00, 0.0, 1, N, 1, N, N, "", N, N, N, "AC", "", N, N, ""),
    ("L9", "USD wholesale funding – floating", "USD", "Liability", "Floating", 2000, 3.90, 0.20, 1, 0.25, N, N, N, "", N, N, N, "AC", "", N, N, ""),
    ("L10", "USD senior bonds – fixed", "USD", "Liability", "Fixed bullet", 1500, 4.50, 1.10, 5, N, 1, N, N, "", N, N, N, "AC", "", N, N, ""),
    ("D1", "Payer swaps (hedge of mortgages)", "EUR", "Asset", "Payer swap", 6000, 2.55, N, 7, 0.5, 1, N, N, "", N, N, N, "FVTPL", "", N, N, ""),
    ("D2", "Receiver swaps (hedge of sight deposits)", "EUR", "Asset", "Receiver swap", 2000, 2.45, N, 5, 0.5, 1, N, N, "", N, N, N, "FVTPL", "", N, N, ""),
    ("O1", "Floor at 0% embedded in floating corporate loans", "EUR", "Asset", "Floor", 6000, N, N, 5, N, N, N, N, "", N, N, N, "AC", "", 0.0, 90, "Sold"),
    ("O2", "Bought cap on USD funding", "USD", "Asset", "Cap", 1000, N, N, 3, N, N, N, N, "", N, N, N, "FVTPL", "", 5.0, 110, "Bought"),
]

TIER1 = 3500.0


def demo_book() -> pd.DataFrame:
    df = pd.DataFrame(ROWS, columns=POS_COLUMNS)
    return df
