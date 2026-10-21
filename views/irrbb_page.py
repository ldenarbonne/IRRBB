"""
IRRBB page: EVE and NII under the six supervisory shock scenarios, outlier
tests, NMD standardised vs internal comparison, behavioural options,
automatic options, CSRBB and the Pillar 3 IRRBB1 table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from irrbb import demo
from irrbb import engine as E
from irrbb import params as P

C = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#d6a12a", "#d03b3b"]
SC_COLOR = dict(zip(P.SCENARIOS, C))
GOOD, WARN, CRIT = "#0ca30c", "#d6a12a", "#d03b3b"


def m(x: float) -> str:
    return f"{x:,.0f} m"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("📈 IRRBB")
st.sidebar.caption("EVE · NII · outlier tests · CSRBB · Pillar 3")
regime_label = st.sidebar.radio("Regulatory regime",
                                ["BCBS (SRP31, 2024 shocks)", "EU (RTS 2024/856, GL 2022/14)"])
is_eu = regime_label.startswith("EU")
floor_zero = False
if not is_eu:
    floor_zero = st.sidebar.checkbox("Floor post-shock rates at 0% (national discretion)", value=False)
nmd_mode = st.sidebar.radio("NMD treatment", ["Standardised caps", "Internal model"],
                            help="Standardised: core share and average maturity capped by segment "
                                 "(BCBS 2016 Table 2). Internal: the bank's own estimates.")
tier1 = st.sidebar.number_input("Tier 1 capital (EUR m)", min_value=1.0, value=demo.TIER1, step=100.0)
regime = E.Regime("EU" if is_eu else "BCBS", floor_zero=floor_zero, nmd_mode=nmd_mode)

st.sidebar.subheader("Banking book")
src = st.sidebar.radio("Input", ["Demo bank (editable)", "Upload CSV"], key="irrbb_src")
uploaded = None
if src == "Upload CSV":
    up = st.sidebar.file_uploader("Positions (CSV, one row per position)", type=["csv", "txt"], key="irrbb_up")
    if up is not None:
        try:
            uploaded = pd.read_csv(up, sep=None, engine="python")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Could not read the file: {exc}")
    else:
        st.sidebar.info("Download the template, fill it and upload it.")
st.sidebar.download_button("Download template (demo book)", demo.demo_book().to_csv(index=False).encode(),
                           "irrbb_positions_demo.csv", "text/csv")

st.title("Interest Rate Risk in the Banking Book")
st.caption(f"{regime_label} · NMD: {nmd_mode} · amounts in EUR million equivalent · "
           f"Tier 1 = {tier1:,.0f} m")

tabs = st.tabs(["Positions & curves", "Overview", "EVE", "NII", "NMD comparison", "Options",
                "CSRBB", "Pillar 3 IRRBB1", "Methodology"])

# --------------------------------------------------------------------------
# Positions & curves (inputs first so that results use the edited data)
# --------------------------------------------------------------------------
with tabs[0]:
    st.markdown("Each row is a position. **Rates, spreads, CPR, TDRR, core share and beta are in %**, "
                "maturities and reset periods in years. Options are entered as separate rows "
                "(`Floor` / `Cap`, strike in %, normal volatility in bp).")
    base_book = E.normalise_book(uploaded) if uploaded is not None else demo.demo_book()
    book = st.data_editor(
        base_book, num_rows="dynamic", width="stretch", hide_index=True, key="irrbb_book",
        column_config={
            "type": st.column_config.SelectboxColumn("type", options=E.TYPES, required=True),
            "side": st.column_config.SelectboxColumn("side", options=["Asset", "Liability"], required=True),
            "nmd_segment": st.column_config.SelectboxColumn("nmd_segment", options=[""] + list(P.NMD_CAPS)),
            "accounting": st.column_config.SelectboxColumn("accounting", options=["AC", "FVOCI", "FVTPL"]),
            "rating": st.column_config.SelectboxColumn("rating", options=[""] + list(E.DEFAULT_SPREAD_SHOCKS)),
            "option_side": st.column_config.SelectboxColumn("option_side", options=["", "Sold", "Bought"]),
        })
    st.markdown("**Risk-free zero curves** (continuously compounded). Add a column "
                "`XXX zero rate (%)` to support another currency.")
    curve_tbl = st.data_editor(E.curves_table(demo.demo_curves()), num_rows="dynamic", hide_index=True,
                               key="irrbb_curves")
    curves = E.table_to_curves(curve_tbl)
    book = E.normalise_book(book)
    errors = E.validate_book(book, curves)
    with st.expander("Field reference"):
        st.markdown("""
| Field | Use |
|---|---|
| `type` | Overnight, Fixed bullet, Fixed amortising, Floating, NMD, Term deposit, Payer swap, Receiver swap, Floor, Cap |
| `rate` | Current client rate (%), fixed coupon for swaps |
| `spread` | Commercial margin over the risk-free rate (%). Used for NII rollover and CSRBB |
| `reset` / `freq` | Repricing period of floating legs / coupon frequency (years) |
| `cpr` / `tdrr` | Base prepayment / early redemption rates (% per year), scaled by scenario |
| `core_share`, `core_maturity` | Internal NMD estimates (% of balance, years); capped in standardised mode |
| `beta` | NMD pass-through (%) of the overnight shock to the client rate, floored at 0% |
| `accounting`, `rating` | CSRBB: FVOCI / FVTPL positions hit OCI or P&L |
""")

if errors:
    for e in errors:
        st.error(e)
    st.stop()


@st.cache_data(show_spinner="Computing EVE and NII…")
def run(book: pd.DataFrame, curve_tbl: pd.DataFrame, reg_name: str, floor0: bool, nmd: str):
    cv = E.table_to_curves(curve_tbl)
    reg = E.Regime(reg_name, floor_zero=floor0, nmd_mode=nmd)
    return E.compute_eve(book, cv, reg), E.compute_nii(book, cv, reg)


eve, nii = run(book, curve_tbl, regime.name, floor_zero, nmd_mode)
sot = E.outlier_tests(eve, nii, tier1, regime)

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tabs[1]:
    eve_ratio = eve.measure / tier1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EVE risk measure", m(eve.measure), help="Maximum aggregated ΔEVE loss over the six scenarios")
    c2.metric("ΔEVE / Tier 1", f"{eve_ratio:.1%}", f"threshold {P.EVE_SOT_THRESHOLD:.0%}", delta_color="off")
    c3.metric("Worst EVE scenario", eve.worst)
    worst_nii = float(nii.total.reindex(P.NII_SCENARIOS).min())
    c4.metric("Worst ΔNII (parallel)", m(worst_nii), f"{worst_nii / tier1:.1%} of Tier 1", delta_color="off")

    for _, r in sot.iterrows():
        colour = CRIT if r["breach"] else (WARN if r["metric"] > 0.8 * r["threshold"] else GOOD)
        status = "OUTLIER" if r["breach"] else "within threshold"
        st.markdown(f"<span style='color:{colour};font-weight:600'>● {r['test']}: {r['metric']:.2%} of Tier 1 "
                    f"vs {r['threshold']:.0%} – {status}</span> (worst: {r['worst scenario']})",
                    unsafe_allow_html=True)

    a, b = st.columns(2)
    fig = go.Figure(go.Bar(x=P.SCENARIOS, y=[eve.aggregate[s] for s in P.SCENARIOS],
                           marker_color=[SC_COLOR[s] for s in P.SCENARIOS]))
    fig.add_hline(y=P.EVE_SOT_THRESHOLD * tier1, line_dash="dash", line_color=CRIT,
                  annotation_text="SOT 15% Tier 1")
    fig.update_layout(title="Aggregated ΔEVE loss by scenario (positive = loss)", height=380,
                      margin=dict(t=50, b=10), yaxis_title="EUR m")
    a.plotly_chart(fig, width="stretch")
    fig = go.Figure(go.Bar(x=P.SCENARIOS, y=[nii.total[s] for s in P.SCENARIOS],
                           marker_color=[SC_COLOR[s] for s in P.SCENARIOS]))
    if is_eu:
        fig.add_hline(y=-P.NII_SOT_THRESHOLD_EU * tier1, line_dash="dash", line_color=CRIT,
                      annotation_text="SOT −5% Tier 1")
    fig.update_layout(title="ΔNII over 12 months, constant balance sheet", height=380,
                      margin=dict(t=50, b=10), yaxis_title="EUR m")
    b.plotly_chart(fig, width="stretch")

    st.markdown(f"**Base-case NII (12 months):** {m(nii.base)} · **Base EVE by currency:** "
                + ", ".join(f"{k} {m(v)}" for k, v in eve.base_eve.items()))
    st.caption("Base EVE excludes equity and non-interest-bearing items; CET1 is not a cash flow "
               "(SRP31). ΔEVE = EVE(base) − EVE(shock) + KAO, so a positive value is a loss.")

# --------------------------------------------------------------------------
# EVE
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Shocked risk-free curves")
    ccy = st.selectbox("Currency", list(curves), key="eve_ccy")
    tt = np.linspace(1 / 365, 30, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tt, y=E.scenario_rate(curves[ccy], tt, None, regime, ccy) * 100, name="Base",
                             line=dict(color="black", width=3)))
    for s in P.SCENARIOS:
        fig.add_trace(go.Scatter(x=tt, y=E.scenario_rate(curves[ccy], tt, s, regime, ccy) * 100, name=s,
                                 line=dict(color=SC_COLOR[s])))
    if is_eu:
        fig.add_trace(go.Scatter(x=tt, y=P.eu_floor(tt) * 100, name="EU lower bound",
                                 line=dict(color="grey", dash="dot")))
    sh = regime.shocks(ccy)
    fig.update_layout(height=400, xaxis_title="Maturity (years)", yaxis_title="Zero rate (%)",
                      title=f"{ccy} – shocks (parallel, short, long) = {sh} bp", margin=dict(t=50))
    st.plotly_chart(fig, width="stretch")

    st.subheader("ΔEVE by currency and scenario")
    t = eve.by_ccy.copy()
    t["Aggregated (regime rule)"] = eve.aggregate
    t["% Tier 1"] = eve.aggregate / tier1
    st.dataframe(t.style.format({**{c: "{:,.0f}" for c in t.columns if c != "% Tier 1"}, "% Tier 1": "{:.2%}"}),
                 width="stretch")
    st.caption("BCBS: only currencies with a loss are summed, and the result is floored at zero. "
               "EU: losses plus 50% of gains are summed across currencies.")

    st.subheader("Repricing cash-flow gap (slotted on bucket midpoints)")
    g = eve.gaps[ccy]
    scen = st.selectbox("Cash flows under", ["Base"] + P.SCENARIOS, key="gap_scn")
    fig = go.Figure(go.Bar(x=g.index, y=g[scen], marker_color=np.where(g[scen] >= 0, C[0], C[1])))
    fig.add_trace(go.Scatter(x=g.index, y=g[scen].cumsum(), name="Cumulative gap", yaxis="y",
                             line=dict(color="black")))
    fig.update_layout(height=380, yaxis_title="EUR m", showlegend=False, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.download_button("Download cash-flow gaps (CSV)", g.to_csv().encode(), f"gaps_{ccy}.csv", "text/csv")

# --------------------------------------------------------------------------
# NII
# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("ΔNII by currency (12 months, constant balance sheet)")
    t = nii.by_scenario.copy()
    t["Total"] = nii.total
    t["% Tier 1"] = nii.total / tier1
    st.dataframe(t.style.format({**{c: "{:,.1f}" for c in t.columns if c != "% Tier 1"}, "% Tier 1": "{:.2%}"}),
                 width="stretch")
    st.subheader("Contribution by position")
    pp = nii.positions.set_index(["id", "name", "ccy"])
    st.dataframe(pp.style.format("{:,.1f}").background_gradient(cmap="RdYlGn", subset=P.SCENARIOS, axis=None),
                 width="stretch")
    st.caption("Maturing and repricing positions are replaced by identical positions at the shocked market "
               "rate plus their current commercial margin. NMD rates move by beta × overnight shock and are "
               "floored at 0%. Prepayments and early redemptions follow the scenario multipliers.")

# --------------------------------------------------------------------------
# NMD comparison
# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Non-maturity deposits: standardised caps vs internal model")
    st.dataframe(E.nmd_table(book).style.format({"balance": "{:,.0f}", "internal core share": "{:.0%}",
                                                 "cap": "{:.0%}", "applied core share": "{:.0%}",
                                                 "pass-through beta": "{:.0%}"}, precision=1),
                 width="stretch", hide_index=True)
    other = "Internal model" if nmd_mode == "Standardised caps" else "Standardised caps"
    eve_o, nii_o = run(book, curve_tbl, regime.name, floor_zero, other)
    res = {nmd_mode: eve, other: eve_o}
    cmp = pd.DataFrame({k: v.aggregate for k, v in res.items()})
    cmp.loc["EVE risk measure"] = [v.measure for v in res.values()]
    cmp.loc["% Tier 1"] = [v.measure / tier1 for v in res.values()]
    st.dataframe(cmp.style.format("{:,.0f}").format("{:.2%}", subset=pd.IndexSlice[["% Tier 1"], :]),
                 width="stretch")
    fig = go.Figure()
    for i, (k, v) in enumerate(res.items()):
        fig.add_trace(go.Bar(x=P.SCENARIOS, y=[v.aggregate[s] for s in P.SCENARIOS], name=k, marker_color=C[i]))
    fig.update_layout(barmode="group", height=360, yaxis_title="ΔEVE loss, EUR m", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.caption("Caps (BCBS 2016 Table 2): retail transactional 90% / 5 years, retail non-transactional 70% / "
               "4.5 years, wholesale 50% / 4 years. The non-core part is slotted overnight; the core part runs "
               "off linearly with the stated average maturity. Lengthening deposit duration raises the value of "
               "liabilities' 'hedge' against fixed-rate assets: the sign of the effect depends on the bank's gap.")

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Behavioural options: prepayments and early redemptions")
    bt = E.behavioural_table(book)
    if bt.empty:
        st.info("No position with a CPR or TDRR.")
    else:
        st.dataframe(bt.style.format(precision=1), width="stretch", hide_index=True)
    st.caption("CPR multipliers γ = 0.8 / 1.2 / 0.8 / 1.2 / 0.8 / 1.2 and TDRR multipliers u = 1.2 / 0.8 / 0.8 / "
               "1.2 / 1.2 / 0.8 for parallel up / down, steepener, flattener, short up / down "
               "(BCBS 2016 Tables 3 and 4). Redeemed term deposits are slotted overnight.")
    st.subheader("Automatic interest rate options (KAO add-on)")
    st.dataframe(eve.kao.pivot(index="scenario", columns="ccy", values="KAO").reindex(P.SCENARIOS)
                 .style.format("{:,.1f}"), width="stretch")
    st.caption("KAO = Σ sold ΔV − Σ bought ΔV, where ΔV is the value under the shocked curve with implied "
               "volatility × 1.25 minus the current value (Bachelier caplets / floorlets, quarterly). "
               "Added to ΔEVE of each currency.")

# --------------------------------------------------------------------------
# CSRBB
# --------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Credit spread risk in the banking book (simplified)")
    st.markdown("Instantaneous widening of credit spreads by rating on positions with a rating "
                "(securities). Shocks are illustrative and should be calibrated internally "
                "(EBA/GL/2022/14, section 4.6).")
    shocks = st.data_editor(pd.DataFrame({"rating": list(E.DEFAULT_SPREAD_SHOCKS),
                                          "spread shock (bp)": list(E.DEFAULT_SPREAD_SHOCKS.values())}),
                            hide_index=True, key="csrbb_shocks")
    cs = E.compute_csrbb(book, curves, dict(zip(shocks["rating"], shocks["spread shock (bp)"])), regime)
    if cs.empty:
        st.info("No rated position in the book.")
    else:
        k1, k2 = st.columns(2)
        k1.metric("ΔEV (all rated positions)", m(cs["ΔEV"].sum()))
        k2.metric("OCI / P&L impact (fair-value positions)", m(cs["OCI / P&L impact"].sum()),
                  f"{cs['OCI / P&L impact'].sum() / tier1:.1%} of Tier 1", delta_color="off")
        st.dataframe(cs.style.format(precision=1), width="stretch", hide_index=True)

# --------------------------------------------------------------------------
# Pillar 3
# --------------------------------------------------------------------------
with tabs[7]:
    title = "EU IRRBB1" if is_eu else "IRRBB1"
    st.subheader(f"Template {title} – quantitative information on IRRBB (EUR m)")
    p3 = E.pillar3_irrbb1(eve, nii, tier1)
    p3.columns = pd.MultiIndex.from_tuples([("ΔEVE", "T"), ("ΔNII", "T")])
    st.dataframe(p3.style.format("{:,.0f}", na_rep=""), width="stretch")
    st.caption("Signs follow the disclosure: a negative ΔEVE / ΔNII is a decrease. "
               + "ΔNII is disclosed for the parallel up and down scenarios only. "
               + "Column T-1 (previous period) is left to the user. "
               + "ΔEVE rows show the aggregated measure per scenario.")
    st.download_button("Download IRRBB1 (CSV)", E.pillar3_irrbb1(eve, nii, tier1).to_csv().encode(),
                       "irrbb1.csv", "text/csv")

# --------------------------------------------------------------------------
# Methodology
# --------------------------------------------------------------------------
with tabs[8]:
    st.markdown(r"""
### Standardised framework (BCBS 2016, SRP31)
1. **Cash flows** – notional repricing cash flows (principal and interest up to repricing) are generated
   monthly for each position and split linearly between the two adjacent midpoints of the 19 time buckets.
   Floating instruments reprice to par at the next reset; the commercial spread beyond the reset is kept.
2. **Scenarios** – parallel up/down, steepener, flattener, short up/down with
   $S_{short}(t)=e^{-t/4}$: steepener $=-0.65|\Delta R_{short}|+0.9|\Delta R_{long}|$,
   flattener $=+0.8|\Delta R_{short}|-0.6|\Delta R_{long}|$.
   Shock sizes: BCBS July 2024 recalibration (EUR 225 / 350 / 200 bp) or EU RTS Annex I (EUR 200 / 250 / 100 bp).
3. **Discounting** – continuously compounded risk-free zero rates; ΔEVE$_{i,c}$ = EVE$_0$ − EVE$_i$ + KAO$_{i,c}$.
4. **Aggregation** – BCBS: $\max_i \max(0, \sum_{c} \max(0,\Delta EVE_{i,c}))$; EU: losses plus 50% of gains.
5. **NMD** – core / non-core split; caps by segment in standardised mode; side-by-side with the internal model.
6. **Behavioural options** – CPR and TDRR scaled by scenario multipliers.
7. **Automatic options** – KAO with volatility × 1.25.

### EU specificities
* Post-shock lower bound of −150 bp at zero maturity rising by 3 bp a year (0% at 50 years); if the observed
  rate is already lower, the observed rate is kept.
* **Outlier tests** (CRD Art. 98(5)): ΔEVE > 15% of Tier 1; large decline in NII > 5% of Tier 1 under parallel
  up / down over one year, constant balance sheet, including commercial margins.
* **CSRBB** to be monitored and assessed (EBA/GL/2022/14).

### Limitations
* Positions in EUR equivalent; no FX conversion of results.
* Simplified products (bullet, amortising annuity, plain floaters, vanilla swaps, cap/floor strips).
* NII: the commercial margin is held constant; no dynamic balance sheet; NMD pricing via a single beta.
* CSRBB is a single spread-shock sensitivity, not a full CSRBB framework.
* Educational tool: it does not replace regulatory reporting.
""")

st.divider()
st.caption("Built by [Laurent Motais de Narbonne](https://www.linkedin.com/in/laurent-motais-de-narbonne/) · "
           "Sources: BCBS d368 (2016), d578 (2024), SRP31; EBA/RTS/2022/10; EBA/GL/2022/14.")
