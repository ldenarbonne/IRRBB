"""IRRBB engine tests with hand-checked values."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from irrbb import demo
from irrbb import engine as E
from irrbb import params as P

FLAT = {"EUR": E.Curve([0.0, 30.0], [0.02, 0.02])}
BCBS, EU = E.Regime("BCBS"), E.Regime("EU")


def pos(**kw):
    row = {c: np.nan for c in E.POS_COLUMNS}
    row.update({"id": "X", "name": "x", "ccy": "EUR", "side": "Asset", "nmd_segment": "", "accounting": "AC",
                "rating": "", "option_side": ""})
    row.update(kw)
    return row


def book(*rows):
    return E.normalise_book(pd.DataFrame(list(rows), columns=E.POS_COLUMNS))


# ---------------------------------------------------------------- shocks
def test_shock_shapes_bcbs_2024_eur():
    sh = P.SHOCKS_BCBS_2024["EUR"]
    t = np.array([0.0, 1e6])
    assert np.allclose(P.shock_vector("Parallel up", sh, t), 0.0225)
    assert P.shock_vector("Short rates up", sh, t)[0] == pytest.approx(0.035)
    assert P.shock_vector("Short rates up", sh, t)[1] == pytest.approx(0.0)
    # steepener: -0.65 x 350 at t=0, +0.9 x 200 at long end
    st = P.shock_vector("Steepener", sh, t)
    assert st[0] == pytest.approx(-0.65 * 0.035) and st[1] == pytest.approx(0.9 * 0.02)
    fl = P.shock_vector("Flattener", sh, t)
    assert fl[0] == pytest.approx(0.8 * 0.035) and fl[1] == pytest.approx(-0.6 * 0.02)


def test_eu_lower_bound():
    assert P.eu_floor(0) == pytest.approx(-0.015)
    assert P.eu_floor(10) == pytest.approx(-0.012)
    assert P.eu_floor(60) == 0.0
    # 2% flat curve, EU parallel down 200 bp at t=1 -> 0%, above the -1.47% floor
    assert E.scenario_rate(FLAT["EUR"], [1.0], "Parallel down", EU, "EUR")[0] == pytest.approx(0.0)
    low = {"EUR": E.Curve([0, 30], [0.002, 0.002])}
    r = E.scenario_rate(low["EUR"], [0.0], "Parallel down", EU, "EUR")[0]
    assert r == pytest.approx(-0.015)                      # floored
    r_bcbs = E.scenario_rate(low["EUR"], [0.0], "Parallel down", BCBS, "EUR")[0]
    assert r_bcbs == pytest.approx(0.002 - 0.0225)         # no floor under BCBS by default


def test_slotting_between_midpoints():
    out = E.slot(np.array([0.5]), np.array([100.0]))      # midway between 0.375 and 0.625
    assert out[3] == pytest.approx(50) and out[4] == pytest.approx(50)
    assert E.slot(np.array([40.0]), np.array([10.0]))[-1] == 10


# ---------------------------------------------------------------- EVE
def test_zero_coupon_eve_matches_closed_form():
    b = book(pos(type="Fixed bullet", notional=100, rate=0, maturity=2.5, freq=2.5))
    res = E.compute_eve(b, FLAT, BCBS)
    expected = 100 * (np.exp(-0.02 * 2.5) - np.exp(-0.0425 * 2.5))
    assert res.by_ccy.loc["Parallel up", "EUR"] == pytest.approx(expected, rel=1e-9)
    assert res.base_eve["EUR"] == pytest.approx(100 * np.exp(-0.05))


def test_bcbs_vs_eu_aggregation():
    df = pd.DataFrame({"EUR": [10.0, -20.0], "USD": [-5.0, 8.0]}, index=["Parallel up", "Parallel down"])
    bcbs = df.apply(lambda r: max(0.0, r[r > 0].sum()), axis=1)
    eu = df.apply(lambda r: r[r > 0].sum() + 0.5 * r[r < 0].sum(), axis=1)
    assert list(bcbs) == [10.0, 8.0] and list(eu) == [7.5, -2.0]
    # same rule inside the engine on a two-currency book with offsetting positions
    curves = {"EUR": FLAT["EUR"], "USD": E.Curve([0, 30], [0.03, 0.03])}
    b = book(pos(type="Fixed bullet", notional=100, rate=0, maturity=5, freq=5),
             pos(id="Y", ccy="USD", side="Liability", type="Fixed bullet", notional=100, rate=0, maturity=5, freq=5))
    rb, re = E.compute_eve(b, curves, BCBS), E.compute_eve(b, curves, EU)
    up_b = rb.by_ccy.loc["Parallel up"]
    assert rb.aggregate["Parallel up"] == pytest.approx(up_b["EUR"])          # USD gain ignored
    up_e = re.by_ccy.loc["Parallel up"]
    assert re.aggregate["Parallel up"] == pytest.approx(up_e["EUR"] + 0.5 * up_e["USD"])


def test_payer_swap_gains_when_rates_rise():
    b = book(pos(type="Payer swap", notional=100, rate=2.0, maturity=5, reset=0.5, freq=1))
    r = E.compute_eve(b, FLAT, BCBS).by_ccy["EUR"]
    assert r["Parallel up"] < 0 < r["Parallel down"]


# ---------------------------------------------------------------- behavioural
def test_nmd_caps():
    p = pd.Series(pos(type="NMD", side="Liability", nmd_segment="Retail transactional", core_share=95,
                      core_maturity=7))
    assert E.nmd_parameters(p, E.Regime(nmd_mode="Standardised caps")) == (0.9, 5.0)
    assert E.nmd_parameters(p, E.Regime(nmd_mode="Internal model")) == (0.95, 7.0)
    p["notional"], p["rate"] = 1000, 0.0
    t, a = E.cash_flows(p, None, E.Regime(nmd_mode="Standardised caps"))
    assert a[0] == pytest.approx(-100)                       # non-core 10% overnight
    assert a.sum() == pytest.approx(-1000)
    assert t.max() == pytest.approx(10.0)                    # linear run-off over 2 x 5 years


def test_cpr_and_tdrr_multipliers():
    bt = E.behavioural_table(book(pos(type="Fixed amortising", notional=100, rate=3, maturity=10, cpr=10),
                                  pos(id="T", side="Liability", type="Term deposit", notional=100, rate=2,
                                      maturity=2, freq=1, tdrr=90)))
    cpr, tdrr = bt.iloc[0], bt.iloc[1]
    assert cpr["Parallel up"] == pytest.approx(8) and cpr["Parallel down"] == pytest.approx(12)
    assert tdrr["Parallel up"] == 100 and tdrr["Parallel down"] == pytest.approx(72)   # capped at 100%
    # redeemed amount slotted overnight
    t, a = E.cash_flows(book(pos(type="Term deposit", side="Liability", notional=100, rate=0, maturity=2,
                                 freq=1, tdrr=10)).iloc[0], "Flattener", BCBS)
    assert t[0] == pytest.approx(1 / 365) and a[0] == pytest.approx(-12)


def test_prepayments_shorten_mortgage_cash_flows():
    p = book(pos(type="Fixed amortising", notional=100, rate=3, maturity=10, cpr=10)).iloc[0]
    t_up, a_up = E.cash_flows(p, "Parallel up", BCBS)
    t_dn, a_dn = E.cash_flows(p, "Parallel down", BCBS)
    wal = lambda t, a: np.sum(t * a) / np.sum(a)  # noqa: E731
    assert wal(t_dn, a_dn) < wal(t_up, a_up)


# ---------------------------------------------------------------- options
def test_kao_sign_bought_cap_and_sold_floor():
    cap = book(pos(type="Cap", notional=100, maturity=5, strike=3.0, vol_bp=100, option_side="Bought"))
    floor = book(pos(type="Floor", notional=100, maturity=5, strike=1.0, vol_bp=100, option_side="Sold"))
    assert E.kao(cap, FLAT, "Parallel up", BCBS, "EUR") < 0       # bought cap gains -> reduces loss
    assert E.kao(floor, FLAT, "Parallel down", BCBS, "EUR") > 0   # sold floor loses when rates fall


def test_bachelier_atm_caplet():
    # single ATM caplet: value = N tau DF sigma sqrt(T) / sqrt(2 pi)
    p = pd.Series(pos(type="Cap", notional=100, maturity=0.5, strike=2.0, vol_bp=100))
    zero = {"EUR": E.Curve([0, 30], [0.0, 0.0])}
    p["strike"] = 0.0
    v = E.option_value(p, zero["EUR"], None, BCBS, 1.0)
    assert v == pytest.approx(100 * 0.25 * 0.01 * np.sqrt(0.25) / np.sqrt(2 * np.pi), rel=1e-9)


# ---------------------------------------------------------------- NII and SOT
def test_overnight_nii_parallel_up():
    b = book(pos(type="Overnight", notional=100, rate=2.0, spread=0.0, maturity=1 / 365))
    nii = E.compute_nii(b, FLAT, BCBS)
    assert nii.base == pytest.approx(2.0)
    assert nii.total["Parallel up"] == pytest.approx(2.25)


def test_nmd_beta_and_zero_floor():
    b = book(pos(type="NMD", side="Liability", notional=100, rate=0.5, nmd_segment="Wholesale", core_share=50,
                 core_maturity=2, beta=40))
    nii = E.compute_nii(b, FLAT, BCBS)
    assert nii.total["Parallel up"] == pytest.approx(-100 * 0.4 * 0.0225)   # deposit cost rises
    assert nii.total["Parallel down"] == pytest.approx(0.5)                 # rate floored at 0%


def test_fixed_bullet_does_not_reprice_within_year():
    b = book(pos(type="Fixed bullet", notional=100, rate=3, spread=1, maturity=3, freq=1))
    assert E.compute_nii(b, FLAT, BCBS).total.abs().max() == pytest.approx(0)


def test_outlier_tests_and_pillar3():
    b, c = demo.demo_book(), demo.demo_curves()
    eve, nii = E.compute_eve(b, c, EU), E.compute_nii(b, c, EU)
    sot = E.outlier_tests(eve, nii, demo.TIER1, EU)
    assert list(sot["threshold"]) == [0.15, 0.05]
    assert not sot["breach"].any()
    tight = E.outlier_tests(eve, nii, 1000.0, EU)
    assert tight["breach"].all()
    p3 = E.pillar3_irrbb1(eve, nii, demo.TIER1)
    assert p3.loc["Maximum", "ΔEVE"] == pytest.approx(-eve.measure)
    assert p3.loc["Steepener", "ΔNII"] != p3.loc["Steepener", "ΔNII"]       # NaN: parallel only


def test_demo_results_stable():
    b, c = demo.demo_book(), demo.demo_curves()
    eve_b = E.compute_eve(b, c, BCBS)
    eve_e = E.compute_eve(b, c, EU)
    assert eve_b.worst == "Steepener" and eve_b.measure == pytest.approx(273, abs=2)
    assert eve_e.measure == pytest.approx(181, abs=2)
    assert E.validate_book(b, c) == []


def test_csrbb_only_fair_value_hits_oci():
    cs = E.compute_csrbb(demo.demo_book(), demo.demo_curves(), E.DEFAULT_SPREAD_SHOCKS, BCBS)
    assert (cs["ΔEV"] < 0).all()
    assert cs.loc[cs["accounting"] == "AC", "OCI / P&L impact"].eq(0).all()


def test_irrbb_page_runs():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=300).run()
    assert not at.exception and not at.error
    assert at.metric[1].value == "7.8%"
    next(r for r in at.radio if r.label == "Regulatory regime").set_value("EU (RTS 2024/856, GL 2022/14)").run()
    assert not at.exception and not at.error
    next(r for r in at.radio if r.label == "NMD treatment").set_value("Internal model").run()
    assert not at.exception and not at.error
