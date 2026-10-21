"""
IRRBB engine: notional repricing cash flows, EVE and NII under the
supervisory shock scenarios, automatic options add-on, outlier tests,
CSRBB and the Pillar 3 IRRBB1 table.

Positions are expressed in reporting-currency equivalent (EUR m); the
``ccy`` field selects the yield curve and the shock calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import params as P

POS_COLUMNS = ["id", "name", "ccy", "side", "type", "notional", "rate", "spread", "maturity", "reset", "freq",
               "cpr", "tdrr", "nmd_segment", "core_share", "core_maturity", "beta", "accounting", "rating",
               "strike", "vol_bp", "option_side"]
TYPES = ["Overnight", "Fixed bullet", "Fixed amortising", "Floating", "NMD", "Term deposit",
         "Payer swap", "Receiver swap", "Floor", "Cap"]


# ==========================================================================
# Curves and regime
# ==========================================================================


@dataclass
class Regime:
    name: str = "BCBS"
    floor_zero: bool = False          # BCBS national discretion: floor post-shock rates at 0%
    nmd_mode: str = "Standardised caps"   # or "Internal model"

    @property
    def eu(self) -> bool:
        return self.name.startswith("EU")

    def shocks(self, ccy: str) -> tuple:
        table = P.SHOCKS_EU_RTS if self.eu else P.SHOCKS_BCBS_2024
        return table.get(ccy, (200, 300, 150))


@dataclass
class Curve:
    tenors: list
    zeros: list                       # continuously compounded, decimal

    def z(self, t):
        return np.interp(np.asarray(t, dtype=float), self.tenors, self.zeros)


def scenario_rate(curve: Curve, t, scenario: str | None, regime: Regime, ccy: str):
    """Post-shock zero rate at maturities t (scenario None = baseline)."""
    t = np.asarray(t, dtype=float)
    r0 = curve.z(t)
    if scenario is None:
        return r0
    r = r0 + P.shock_vector(scenario, regime.shocks(ccy), t)
    if regime.eu:
        floor = np.minimum(P.eu_floor(t), r0)     # observed rates below the floor are kept
        r = np.maximum(r, floor)
    elif regime.floor_zero:
        r = np.maximum(r, np.minimum(0.0, r0))
    return r


# ==========================================================================
# Cash flows
# ==========================================================================


def _f(x, default=0.0):
    try:
        v = float(x)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def nmd_parameters(pos: pd.Series, regime: Regime) -> tuple[float, float]:
    core, avg = _f(pos["core_share"]) / 100, _f(pos["core_maturity"], 1.0)
    if regime.nmd_mode == "Standardised caps":
        cap_share, cap_mat = P.NMD_CAPS.get(pos["nmd_segment"], (0.5, 4.0))
        core, avg = min(core, cap_share), min(avg, cap_mat)
    return core, avg


def cash_flows(pos: pd.Series, scenario: str | None, regime: Regime) -> tuple[np.ndarray, np.ndarray]:
    """Notional repricing cash flows (times, amounts) from the bank's point of view.
    Assets positive, liabilities negative; swaps net (receive positive)."""
    typ = pos["type"]
    N = _f(pos["notional"])
    r = _f(pos["rate"]) / 100
    T = max(_f(pos["maturity"], 1.0), 1 / 365)
    sign = 1.0 if pos["side"] == "Asset" else -1.0
    times, amts = [], []

    def add(t, a):
        times.append(t)
        amts.append(a)

    if typ == "Overnight":
        add(1 / 365, N * (1 + r / 365))
    elif typ in ("Fixed bullet", "Term deposit"):
        freq = _f(pos["freq"], 1.0) or 1.0
        redeemed = 0.0
        if typ == "Term deposit":
            base = _f(pos["tdrr"]) / 100
            tdrr = min(1.0, base * (P.TDRR_MULT[scenario] if scenario else 1.0))
            redeemed = tdrr
            add(1 / 365, N * tdrr)                          # early redemption slotted overnight
        live = N * (1 - redeemed)
        ts = np.arange(freq, T + 1e-9, freq)
        if len(ts) == 0 or abs(ts[-1] - T) > 1e-9:
            ts = np.append(ts, T)
        prev = 0.0
        for t in ts:
            add(t, live * r * (t - prev))
            prev = t
        add(T, live)
    elif typ == "Fixed amortising":
        cpr = min(1.0, _f(pos["cpr"]) / 100 * (P.CPR_MULT[scenario] if scenario else 1.0))
        smm = 1 - (1 - cpr) ** (1 / 12)
        n = int(round(T * 12))
        bal = N
        for m in range(1, n + 1):
            rem = n - m + 1
            i_m = r / 12
            pay = bal * i_m / (1 - (1 + i_m) ** -rem) if i_m > 0 else bal / rem
            interest = bal * i_m
            sched = pay - interest
            prepay = smm * (bal - sched)
            add(m / 12, interest + sched + prepay)
            bal -= sched + prepay
    elif typ == "Floating":
        reset = _f(pos["reset"], 0.25) or 0.25
        nxt = min(reset, T)
        spread = _f(pos["spread"]) / 100
        add(nxt, N * (1 + r * nxt))                         # par at next reset + current coupon
        ts = np.arange(nxt + reset, T + 1e-9, reset)        # non-repricing spread component
        for t in ts:
            add(t, N * spread * reset)
    elif typ == "NMD":
        core, avg = nmd_parameters(pos, regime)
        add(1 / 365, N * (1 - core))                        # non-core: overnight
        H = 2 * avg                                         # linear run-off with average maturity avg
        months = max(int(round(H * 12)), 1)
        per = N * core / months
        bal = N * core
        for m in range(1, months + 1):
            add(m / 12, per + bal * r / 12)
            bal -= per
    elif typ in ("Payer swap", "Receiver swap"):
        reset = _f(pos["reset"], 0.5) or 0.5
        freq = _f(pos["freq"], 1.0) or 1.0
        fixed_t = np.arange(freq, T + 1e-9, freq)
        fixed = [(t, N * r * freq) for t in fixed_t] + [(T, N)]
        floating = [(min(reset, T), N)]                     # floating leg reprices to par
        s = 1.0 if typ == "Payer swap" else -1.0            # payer receives floating
        for t, a in floating:
            add(t, s * a)
        for t, a in fixed:
            add(t, -s * a)
        sign = 1.0
    else:  # options handled in the KAO add-on
        return np.array([]), np.array([])
    return np.asarray(times), sign * np.asarray(amts)


def slot(times: np.ndarray, amounts: np.ndarray) -> np.ndarray:
    """Split cash flows between adjacent bucket midpoints (BCBS 2016 para 101(ii))."""
    out = np.zeros(len(P.MIDPOINTS))
    mp = P.MIDPOINTS
    for t, a in zip(times, amounts):
        if t <= mp[0]:
            out[0] += a
        elif t >= mp[-1]:
            out[-1] += a
        else:
            j = int(np.searchsorted(mp, t))
            w = (t - mp[j - 1]) / (mp[j] - mp[j - 1])
            out[j - 1] += a * (1 - w)
            out[j] += a * w
    return out


# ==========================================================================
# Automatic options (Bachelier caplet / floorlet strips)
# ==========================================================================


def option_value(pos: pd.Series, curve: Curve, scenario: str | None, regime: Regime, vol_mult: float) -> float:
    N, K, T = _f(pos["notional"]), _f(pos["strike"]) / 100, _f(pos["maturity"], 1.0)
    sig = _f(pos["vol_bp"], 80.0) / 1e4 * vol_mult
    tau = 0.25
    starts = np.arange(tau, T - 1e-9, tau)
    if len(starts) == 0:
        return 0.0
    ends = starts + tau
    z_s = scenario_rate(curve, starts, scenario, regime, pos["ccy"])
    z_e = scenario_rate(curve, ends, scenario, regime, pos["ccy"])
    df_s, df_e = np.exp(-z_s * starts), np.exp(-z_e * ends)
    fwd = (df_s / df_e - 1) / tau
    sd = sig * np.sqrt(starts)
    d = (fwd - K) / sd
    if pos["type"] == "Cap":
        pay = (fwd - K) * norm.cdf(d) + sd * norm.pdf(d)
    else:
        pay = (K - fwd) * norm.cdf(-d) + sd * norm.pdf(d)
    return float(N * np.sum(tau * df_e * pay))


def kao(book: pd.DataFrame, curves: dict, scenario: str, regime: Regime, ccy: str) -> float:
    """BCBS 2016 paras 130-131: sum over sold options minus sum over bought options of
    V(scenario, vol x 1.25) - V(current curve, current vol)."""
    total = 0.0
    for _, p in book[(book["ccy"] == ccy) & book["type"].isin(["Floor", "Cap"])].iterrows():
        dv = (option_value(p, curves[ccy], scenario, regime, P.KAO_VOL_UPLIFT)
              - option_value(p, curves[ccy], None, regime, 1.0))
        total += dv if p["option_side"] == "Sold" else -dv
    return total


# ==========================================================================
# EVE
# ==========================================================================


@dataclass
class EVEResult:
    by_ccy: pd.DataFrame          # scenario x ccy -> delta EVE (loss positive)
    aggregate: pd.Series          # scenario -> aggregated delta EVE (regime rule)
    measure: float                # max over scenarios
    worst: str
    base_eve: dict
    gaps: dict                    # ccy -> DataFrame of slotted cash flows by scenario
    kao: pd.DataFrame


def compute_eve(book: pd.DataFrame, curves: dict, regime: Regime) -> EVEResult:
    ccys = sorted(book["ccy"].unique())
    rows, base, gaps, kaos = [], {}, {}, []
    for c in ccys:
        sub = book[book["ccy"] == c]
        flows = {}
        for scn in [None] + P.SCENARIOS:
            cf = np.zeros(len(P.MIDPOINTS))
            for _, p in sub.iterrows():
                t, a = cash_flows(p, scn, regime)
                if len(t):
                    cf += slot(t, a)
            flows[scn or "Base"] = cf
        gaps[c] = pd.DataFrame(flows, index=P.BUCKET_LABELS)
        r0 = scenario_rate(curves[c], P.MIDPOINTS, None, regime, c)
        eve0 = float(np.sum(flows["Base"] * np.exp(-r0 * P.MIDPOINTS)))
        base[c] = eve0
        for scn in P.SCENARIOS:
            ri = scenario_rate(curves[c], P.MIDPOINTS, scn, regime, c)
            evei = float(np.sum(flows[scn] * np.exp(-ri * P.MIDPOINTS)))
            k = kao(book, curves, scn, regime, c)
            kaos.append({"ccy": c, "scenario": scn, "KAO": k})
            rows.append({"scenario": scn, "ccy": c, "dEVE": eve0 - evei + k})
    df = pd.DataFrame(rows).pivot(index="scenario", columns="ccy", values="dEVE").reindex(P.SCENARIOS)
    if regime.eu:   # RTS Art. 4(l): gains weighted 50% (simplified: no ERM II specifics)
        agg = df.apply(lambda r: r[r > 0].sum() + P.EU_GAIN_WEIGHT * r[r < 0].sum(), axis=1)
    else:           # BCBS 2016 para 132: losses only, floored at zero
        agg = df.apply(lambda r: max(0.0, r[r > 0].sum()), axis=1)
    measure = float(max(agg.max(), 0.0))
    return EVEResult(df, agg, measure, str(agg.idxmax()), base, gaps, pd.DataFrame(kaos))


# ==========================================================================
# NII (one year, constant balance sheet)
# ==========================================================================


def _repricing_profile(pos: pd.Series, scenario: str | None, regime: Regime) -> np.ndarray:
    """Cumulative share of the balance repriced at the end of each month m = 1..12."""
    typ = pos["type"]
    m = np.arange(1, 13) / 12
    T = max(_f(pos["maturity"], 1.0), 1 / 365)
    if typ == "Overnight":
        return np.ones(12)
    if typ in ("Floating", "Payer swap", "Receiver swap"):
        reset = _f(pos["reset"], 0.25) or 0.25
        return (m >= min(reset, T) - 1e-9).astype(float)
    if typ in ("Fixed bullet",):
        return (m >= T - 1e-9).astype(float)
    if typ == "Term deposit":
        base = _f(pos["tdrr"]) / 100
        tdrr = min(1.0, base * (P.TDRR_MULT[scenario] if scenario else 1.0))
        return np.where(m >= T - 1e-9, 1.0, tdrr)
    if typ == "Fixed amortising":
        t, a = cash_flows(pos, scenario, regime)
        N = abs(_f(pos["notional"]))
        r = _f(pos["rate"]) / 100
        # principal share of each monthly cash flow (approximation: interest on outstanding balance)
        bal, out, cum = N, [], 0.0
        for k in range(min(12, len(a))):
            interest = bal * r / 12
            principal = abs(a[k]) - interest
            cum += principal
            bal -= principal
            out.append(cum / N)
        out += [out[-1] if out else 0.0] * (12 - len(out))
        return np.minimum(np.array(out), 1.0)
    return np.zeros(12)


def position_nii(pos: pd.Series, curves: dict, scenario: str | None, regime: Regime) -> float:
    typ = pos["type"]
    if typ in ("Floor", "Cap"):
        return 0.0
    c = pos["ccy"]
    N = _f(pos["notional"])
    r = _f(pos["rate"]) / 100
    sign = 1.0 if pos["side"] == "Asset" else -1.0
    T = max(_f(pos["maturity"], 1.0), 1 / 365)
    if typ == "NMD":
        on = scenario_rate(curves[c], [1 / 365], scenario, regime, c)[0] - curves[c].z([1 / 365])[0]
        rate = max(0.0, r + _f(pos["beta"]) / 100 * on) if r >= 0 else r
        return sign * N * rate
    if typ in ("Payer swap", "Receiver swap"):
        reset = _f(pos["reset"], 0.5) or 0.5
        f0 = curves[c].z([reset])[0]
        f1 = scenario_rate(curves[c], [reset], scenario, regime, c)[0]
        prof = _repricing_profile(pos, scenario, regime)
        flt = np.mean((1 - prof) * f0 + prof * f1)
        s = 1.0 if typ == "Payer swap" else -1.0
        return s * N * (flt - r)
    if typ == "Overnight":
        tenor = 1 / 365
    elif typ == "Floating":
        tenor = _f(pos["reset"], 0.25) or 0.25
    else:
        tenor = max(T, 1 / 12)
    spread = pos["spread"]
    spread = _f(spread, np.nan) / 100
    if np.isnan(spread):
        spread = r - float(curves[c].z([tenor])[0])
    new_rate = float(scenario_rate(curves[c], [tenor], scenario, regime, c)[0]) + spread
    prof = _repricing_profile(pos, scenario, regime)
    monthly = (1 - prof) * r + prof * new_rate
    return sign * N * float(np.mean(monthly))


@dataclass
class NIIResult:
    base: float
    by_scenario: pd.DataFrame     # scenario x ccy -> delta NII
    total: pd.Series
    positions: pd.DataFrame


def compute_nii(book: pd.DataFrame, curves: dict, regime: Regime, scenarios=None) -> NIIResult:
    scenarios = scenarios or P.SCENARIOS
    rows, prow = [], []
    base_total = 0.0
    for _, p in book.iterrows():
        b = position_nii(p, curves, None, regime)
        base_total += b
        pr = {"id": p["id"], "name": p["name"], "ccy": p["ccy"], "base NII": b}
        for s in scenarios:
            v = position_nii(p, curves, s, regime)
            pr[s] = v - b
            rows.append({"scenario": s, "ccy": p["ccy"], "dNII": v - b})
        prow.append(pr)
    df = pd.DataFrame(rows).groupby(["scenario", "ccy"])["dNII"].sum().unstack().reindex(scenarios)
    return NIIResult(base_total, df, df.sum(axis=1), pd.DataFrame(prow))


# ==========================================================================
# Outlier tests and Pillar 3
# ==========================================================================


def outlier_tests(eve: EVEResult, nii: NIIResult, tier1: float, regime: Regime) -> pd.DataFrame:
    rows = [{"test": "EVE (six scenarios)", "metric": eve.measure / tier1, "threshold": P.EVE_SOT_THRESHOLD,
             "worst scenario": eve.worst, "breach": eve.measure / tier1 > P.EVE_SOT_THRESHOLD}]
    if regime.eu:
        par = nii.total.reindex(P.NII_SCENARIOS)
        decline = float(max(-par.min(), 0.0))
        rows.append({"test": "NII (parallel up / down)", "metric": decline / tier1,
                     "threshold": P.NII_SOT_THRESHOLD_EU, "worst scenario": str(par.idxmin()),
                     "breach": decline / tier1 > P.NII_SOT_THRESHOLD_EU})
    return pd.DataFrame(rows)


def pillar3_irrbb1(eve: EVEResult, nii: NIIResult, tier1: float) -> pd.DataFrame:
    rows = []
    for s in P.SCENARIOS:
        rows.append({"Supervisory shock scenario": s, "ΔEVE": -eve.aggregate[s],
                     "ΔNII": nii.total.get(s, np.nan) if s in P.NII_SCENARIOS else np.nan})
    rows.append({"Supervisory shock scenario": "Maximum", "ΔEVE": -eve.measure,
                 "ΔNII": float(nii.total.reindex(P.NII_SCENARIOS).min())})
    rows.append({"Supervisory shock scenario": "Tier 1 capital", "ΔEVE": tier1, "ΔNII": np.nan})
    return pd.DataFrame(rows).set_index("Supervisory shock scenario")


# ==========================================================================
# CSRBB (simplified, EBA/GL/2022/14)
# ==========================================================================

DEFAULT_SPREAD_SHOCKS = {"AAA-AA": 50, "A": 100, "BBB": 175, "BB and below": 350}


def compute_csrbb(book: pd.DataFrame, curves: dict, shocks_bp: dict, regime: Regime) -> pd.DataFrame:
    rows = []
    sec = book[book["rating"].fillna("").astype(str).str.len() > 0]
    for _, p in sec.iterrows():
        if p["type"] in ("NMD", "Floor", "Cap", "Payer swap", "Receiver swap"):
            continue
        t, a = cash_flows(p, None, regime)
        c = p["ccy"]
        z = curves[c].z(t)
        s0 = max(_f(p["spread"]) / 100, 0.0)
        ds = shocks_bp.get(str(p["rating"]), 100) / 1e4
        pv0 = float(np.sum(a * np.exp(-(z + s0) * t)))
        pv1 = float(np.sum(a * np.exp(-(z + s0 + ds) * t)))
        fv = str(p["accounting"]) in ("FVOCI", "FVTPL")
        rows.append({"id": p["id"], "name": p["name"], "rating": p["rating"], "accounting": p["accounting"],
                     "notional": _f(p["notional"]), "spread shock (bp)": ds * 1e4, "ΔEV": pv1 - pv0,
                     "OCI / P&L impact": (pv1 - pv0) if fv else 0.0})
    return pd.DataFrame(rows)


# ==========================================================================
# Input helpers
# ==========================================================================


def curves_table(curves: dict) -> pd.DataFrame:
    ccys = sorted(curves)
    t = curves[ccys[0]].tenors
    df = pd.DataFrame({"tenor (years)": t})
    for c in ccys:
        df[f"{c} zero rate (%)"] = np.round(np.asarray(curves[c].zeros) * 100, 4)
    return df


def table_to_curves(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["tenor (years)"]).sort_values("tenor (years)")
    out = {}
    for col in df.columns:
        if col.endswith("zero rate (%)"):
            c = col.split()[0].upper()
            sub = df[["tenor (years)", col]].dropna()
            out[c] = Curve(sub["tenor (years)"].astype(float).tolist(), (sub[col].astype(float) / 100).tolist())
    return out


def validate_book(book: pd.DataFrame, curves: dict) -> list[str]:
    """Return a list of blocking errors (empty when the book can be processed)."""
    errs = []
    missing = [c for c in POS_COLUMNS if c not in book.columns]
    if missing:
        return [f"Missing columns: {', '.join(missing)}"]
    for i, p in book.iterrows():
        tag = f"Row {i + 1} ({p['id']})"
        if p["type"] not in TYPES:
            errs.append(f"{tag}: unknown type '{p['type']}'")
        if p["side"] not in ("Asset", "Liability"):
            errs.append(f"{tag}: side must be Asset or Liability")
        if p["ccy"] not in curves:
            errs.append(f"{tag}: no yield curve for currency '{p['ccy']}'")
        if _f(p["notional"]) < 0:
            errs.append(f"{tag}: notional must be positive (use the side column)")
        if p["type"] == "NMD" and p["nmd_segment"] not in P.NMD_CAPS:
            errs.append(f"{tag}: NMD segment must be one of {list(P.NMD_CAPS)}")
        if p["type"] in ("Floor", "Cap") and p["option_side"] not in ("Sold", "Bought"):
            errs.append(f"{tag}: option_side must be Sold or Bought")
        if p["type"] not in ("Overnight", "NMD") and _f(p["maturity"]) <= 0:
            errs.append(f"{tag}: maturity (years) must be positive")
    return errs


def normalise_book(book: pd.DataFrame) -> pd.DataFrame:
    df = book.copy()
    for c in POS_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[POS_COLUMNS]
    for c in ["id", "name", "ccy", "side", "type", "nmd_segment", "accounting", "rating", "option_side"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["ccy"] = df["ccy"].str.upper()
    num = ["notional", "rate", "spread", "maturity", "reset", "freq", "cpr", "tdrr", "core_share",
           "core_maturity", "beta", "strike", "vol_bp"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["type"] != ""].reset_index(drop=True)
    return df


def behavioural_table(book: pd.DataFrame) -> pd.DataFrame:
    """Effective CPR and TDRR by scenario (BCBS 2016 Tables 3 and 4)."""
    rows = []
    for _, p in book.iterrows():
        if p["type"] == "Fixed amortising" and _f(p["cpr"]) > 0:
            base, mult, lab = _f(p["cpr"]), P.CPR_MULT, "CPR"
        elif p["type"] == "Term deposit" and _f(p["tdrr"]) > 0:
            base, mult, lab = _f(p["tdrr"]), P.TDRR_MULT, "TDRR"
        else:
            continue
        r = {"id": p["id"], "name": p["name"], "parameter": lab, "Base (%)": base}
        for s in P.SCENARIOS:
            r[s] = min(100.0, base * mult[s])
        rows.append(r)
    return pd.DataFrame(rows)


def nmd_table(book: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, p in book[book["type"] == "NMD"].iterrows():
        cap_s, cap_m = P.NMD_CAPS.get(p["nmd_segment"], (0.5, 4.0))
        core, avg = _f(p["core_share"]) / 100, _f(p["core_maturity"], 1.0)
        rows.append({"id": p["id"], "name": p["name"], "segment": p["nmd_segment"], "balance": _f(p["notional"]),
                     "internal core share": core, "cap": cap_s, "applied core share": min(core, cap_s),
                     "internal avg maturity (y)": avg, "cap (y)": cap_m, "applied avg maturity (y)": min(avg, cap_m),
                     "pass-through beta": _f(p["beta"]) / 100})
    return pd.DataFrame(rows)
