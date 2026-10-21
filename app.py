"""
ALM & Liquidity Suite – entry point.

Liquidity (LCR, NSFR, survival horizon) and interest rate risk in the
banking book (EVE, NII, outlier tests, CSRBB, Pillar 3 IRRBB1).

Author: Laurent Motais de Narbonne
"""
import streamlit as st

st.set_page_config(page_title="ALM & Liquidity Suite", page_icon="🏦", layout="wide")

pages = [
    st.Page("views/irrbb_page.py", title="IRRBB – EVE, NII, SOT", icon="📈", url_path="irrbb", default=True),
    st.Page("views/liquidity_page.py", title="Liquidity – LCR, NSFR, survival", icon="💧", url_path="liquidity"),
]
st.navigation(pages).run()
