# ALM & Liquidity Suite

An interactive Streamlit application covering the two pillars of balance-sheet risk management:

* **Interest rate risk in the banking book (IRRBB)**: ΔEVE and ΔNII under the six supervisory shock scenarios, supervisory outlier tests (SOT), standardised vs internal treatment of non-maturity deposits, prepayments and early redemptions, automatic options, a simplified CSRBB module and the Pillar 3 **IRRBB1** table.
* **Liquidity**: LCR, NSFR and survival horizon under stress.

Both modules implement the Basel Committee standards with an **EU switch**.

**Live app:** `https://<your-app-name>.streamlit.app` *(to be completed after deployment)*

![tests](https://github.com/ldenarbonne/alm-liquidity-suite/actions/workflows/tests.yml/badge.svg)

---

## 1. Pages

| Page | Content |
|---|---|
| **IRRBB – EVE, NII, SOT** | Positions & curves (editable or CSV), Overview, EVE, NII, NMD comparison, Options, CSRBB, Pillar 3 IRRBB1, Methodology |
| **Liquidity – LCR, NSFR, survival** | Overview, LCR, NSFR, Survival horizon, Methodology |

## 2. IRRBB module

### 2.1 Features

| Tab | Content |
|---|---|
| **Positions & curves** | Position-level banking book (24 demo positions in EUR and USD), editable risk-free zero curves, CSV import and template |
| **Overview** | EVE risk measure, ΔEVE / Tier 1, worst scenario, NII outlier test (EU), charts by scenario |
| **EVE** | Shocked curves, ΔEVE by currency and scenario with BCBS or EU aggregation, repricing gap on the 19 time buckets |
| **NII** | 12-month ΔNII under a constant balance sheet, by currency and by position |
| **NMD comparison** | Standardised caps vs internal model, side by side |
| **Options** | Effective CPR and TDRR by scenario, KAO add-on for caps and floors |
| **CSRBB** | Spread widening by rating, impact on economic value and on OCI / P&L |
| **Pillar 3 IRRBB1** | Disclosure table with CSV export |

### 2.2 Method

* **Cash flows**: notional repricing cash flows are generated monthly and split linearly between the two adjacent midpoints of the 19 time buckets. Floating instruments reprice to par at the next reset date.
* **Scenarios**: parallel up/down, steepener, flattener, short rates up/down (SRP31 / Annex 2 of BCBS d368).
* **Shock sizes**: BCBS July 2024 recalibration (EUR 225 / 350 / 200 bp) or EU RTS Annex I (EUR 200 / 250 / 100 bp).
* **ΔEVE**: ΔEVE = EVE(base) − EVE(shock) + KAO; equity is excluded.
* **NMD**: core and non-core split. Caps of 90% / 5 years, 70% / 4.5 years and 50% / 4 years apply by segment.
* **Behavioural options**: CPR multipliers 0.8 / 1.2 and TDRR multipliers 1.2 / 0.8 by scenario. Redeemed deposits are slotted overnight.
* **Automatic options**: Bachelier pricing; the shocked value uses implied volatility × 1.25.

### 2.3 BCBS vs EU

| Item | BCBS (SRP31) | EU (Delegated Regulation 2024/856, CRD Art. 98(5)) |
|---|---|---|
| Shock sizes | 2024 recalibration | RTS Annex I |
| Post-shock lower bound | none (0% floor as national discretion) | −150 bp at zero maturity, +3 bp a year, 0% at 50 years |
| Cross-currency aggregation | losses only, floored at zero | losses + 50% of gains |
| EVE outlier test | 15% of Tier 1 | 15% of Tier 1 |
| NII outlier test | – | large decline > 5% of Tier 1 (parallel up / down, 1 year, constant balance sheet) |
| CSRBB | – | to be monitored (EBA/GL/2022/14) |

### 2.4 Illustrative results (demo bank, Tier 1 = EUR 3,500 m)

| | BCBS, standardised NMD | EU, standardised NMD | BCBS, internal NMD | EU, internal NMD |
|---|---|---|---|---|
| EVE risk measure | 273 | 181 | 373 | 338 |
| % Tier 1 | 7.8% | 5.2% | 10.7% | 9.6% |
| Worst scenario | Steepener | Short rates down | Parallel down | Parallel down |
| ΔNII parallel down | −170 | −145 | −170 | −145 |

* The NII outlier test (EU) stands at 4.2% of Tier 1, close to the 5% threshold.
* CSRBB: −199 m on FVOCI securities.
* Lengthening the internal deposit maturity beyond the caps raises the falling-rate exposure. This is why supervisors monitor the gap between internal and capped assumptions.

## 3. Liquidity module

### 3.1 Features

| Tab | Content |
|---|---|
| **Overview** | LCR, NSFR and survival horizons, liquidity surplus under the three scenarios, BCBS vs EU comparison |
| **LCR** | HQLA by level with haircuts and 15% / 40% caps (EU: 70% cap on Level 1 covered bonds), 30-day outflows and inflows, 75% inflow cap, waterfall chart |
| **NSFR** | ASF and RSF by category and maturity bucket, derivative add-ons, charts by source and use |
| **Survival horizon** | 90-day daily ladder under idiosyncratic, market-wide and combined stress, counterbalancing capacity, outflows by source, CSV export |
| **Methodology** | Formulas, EU deviations, limitations |

**Inputs** come from two sources:
* a demo bank (EUR 50 bn balance sheet) that can be edited row by row;
* a CSV file (`code, d30, m6, m12, y1`).

A pre-filled template and a reference table of the 41 categories, with their treatments in both regimes, are in `templates/`. The stress assumptions (run-off, rollover and draw-down rates, haircuts, monetisation lags, central bank access) can be edited in the app.

### 3.2 EU deviations implemented

| Item | BCBS | EU |
|---|---|---|
| Extremely high-quality covered bonds | Level 2A, 15% haircut, RSF 15% | Level 1, 7% haircut, 70% cap, RSF 7% |
| Non-RMBS securitisations | not HQLA | Level 2B, 35% haircut, RSF 35% |
| Level 2B RMBS – RSF | 50% | 25% |
| Reverse repos with financials < 6m, Level 1 collateral | RSF 10% | RSF 0% |
| Reverse repos with financials < 6m, other collateral | RSF 15% | RSF 5% |
| Unsecured loans to financials < 6m | RSF 15% | RSF 10% |
| Cover-pool assets encumbered for covered bonds | RSF 100% | RSF 85% |
| Gross derivative liabilities | RSF 20% | RSF 5% |
| Operational deposits held at other banks (LCR inflow) | 0% | 5% |
| Inflow cap | 75% | 75% (90% for specialised institutions with approval) |

The deviations are documented in the BCBS RCAP assessments of the EU (LCR 2017, NSFR 2022). The reduced RSF factors for short-term transactions with financial customers were kept in 2025.

### 3.3 Illustrative results (demo bank, EUR m)

| | BCBS | EU |
|---|---|---|
| Stock of HQLA | 10,150 | 10,465 |
| Net cash outflows | 6,565 | 6,550 |
| **LCR** | **154.6%** | **159.8%** |
| Available stable funding | 37,285 | 37,285 |
| Required stable funding | 27,072 | 25,677 |
| **NSFR** | **137.7%** | **145.2%** |

| Scenario | Survival horizon | Minimum surplus over 90 days |
|---|---|---|
| Idiosyncratic | 85 days | −461 |
| Market-wide | > 90 days | +1,951 |
| Combined | 71 days | −2,358 |

Both regulatory ratios are comfortable. Even so, under the combined scenario the bank runs out of liquidity before 90 days. This is the gap the survival horizon is designed to reveal: the LCR only covers 30 days and assumes stylised outflows.

## 4. Project structure

```
alm-liquidity-suite/
├── app.py                 # Router (st.navigation)
├── views/
│   ├── irrbb_page.py      # IRRBB page
│   └── liquidity_page.py  # Liquidity page
├── irrbb/
│   ├── params.py          # Buckets, scenario shapes, BCBS 2024 and EU shocks, multipliers, caps, thresholds
│   ├── engine.py          # Cash flows, EVE, NII, KAO, SOT, CSRBB, IRRBB1, input validation
│   └── demo.py            # Demo banking book and curves
├── liquidity/             # LCR, NSFR, survival horizon (params, ratios, stress, io, demo)
├── templates/             # CSV templates (positions, balance sheet, category reference)
├── tests/                 # 36 pytest tests (hand-checked values, both pages)
└── .github/workflows/     # CI on Python 3.11 and 3.12
```

## 5. Run locally

```bash
git clone https://github.com/ldenarbonne/alm-liquidity-suite.git
cd alm-liquidity-suite
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q
streamlit run app.py
```

## 6. Deploy on Streamlit Community Cloud

1. Push the repository to GitHub (public).
2. On [share.streamlit.io](https://share.streamlit.io), click **Create app**, choose the repository, branch `main` and main file `app.py`; select Python 3.12.
3. Paste the app URL at the top of this README.

## 7. Limitations

* IRRBB positions are entered in EUR equivalent; products are simplified (bullets, annuities, plain floaters, vanilla swaps, cap and floor strips).
* NII assumes a constant balance sheet with constant commercial margins. A single beta drives NMD pricing.
* CSRBB is a spread-shock sensitivity, not a full CSRBB framework. The demo shocks are illustrative.
* The liquidity balance sheet is entered by category and maturity bucket, and the EU 70% cap is implemented in a simplified form.
* Stress and behavioural assumptions are illustrative and should be calibrated within the ICAAP / ILAAP.
* Educational tool: it does not replace regulatory reporting.

## 8. References

### IRRBB

* BCBS (2016), [*Interest rate risk in the banking book*](https://www.bis.org/bcbs/publ/d368.pdf) (d368).
* BCBS (2024), [*Recalibration of shocks for interest rate risk in the banking book*](https://www.bis.org/bcbs/publ/d578.pdf) (d578).
* Basel Framework SRP31 and DIS70.
* Commission Delegated Regulation (EU) 2024/856 (supervisory outlier tests, EBA/RTS/2022/10).
* EBA/GL/2022/14 on IRRBB and CSRBB.

### Liquidity

* Basel Framework LIQ30 and LIQ40.
* Commission Delegated Regulation (EU) 2015/61.
* Regulation (EU) No 575/2013, Part Six.
* BCBS RCAP assessments of the EU (LCR 2017, NSFR 2022).

---

**Author:** [Laurent Motais de Narbonne](https://www.linkedin.com/in/laurent-motais-de-narbonne/), Senior Risk Manager. MIT licence.
