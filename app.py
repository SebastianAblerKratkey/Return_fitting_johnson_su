import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import scipy.stats as stats
from io import BytesIO
import streamlit as st
import yfinance as yf

st.header("Johnson SU Distribution Fitting")
st.write("Fits a Johnson SU distribution to daily log returns of any Yahoo Finance ticker.")

# ── Inputs ────────────────────────────────────────────────────────────────────
ticker = st.text_input("Enter a Yahoo Finance ticker:", help="Example: ^GSPC")

if not ticker:
    st.stop()

ticker = ticker.strip().upper()

# ── Download & clean ──────────────────────────────────────────────────────────
st.write(f"Fetching daily price data for **{ticker}**...")

try:
    raw = yf.download([ticker], period="max", auto_adjust=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    if raw.empty:
        st.error(f"No data found for ticker '{ticker}'. Please check the ticker symbol.")
        st.stop()

    min_date = raw.index.min().date()
    max_date = raw.index.max().date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

    adj_close = raw["Adj Close"].astype(float)
    adj_close = adj_close[(adj_close.index.date >= start_date) & (adj_close.index.date <= end_date)]

except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# ── Log returns ───────────────────────────────────────────────────────────────
log_returns = np.log(adj_close / adj_close.shift(1)).dropna()
return_data = np.sort(log_returns.values)  # clean numpy array

if len(return_data) < 30:
    st.error("Not enough data to fit a distribution. Please expand the date range.")
    st.stop()

trading_days_per_year = 252

# ── Distribution fitting ──────────────────────────────────────────────────────
# Johnson SU
su_a, su_b, su_loc, su_scale = stats.johnsonsu.fit(return_data)
cdf_jsu  = stats.johnsonsu.cdf(return_data, a=su_a, b=su_b, loc=su_loc, scale=su_scale)
pdf_jsu  = stats.johnsonsu.pdf(return_data, a=su_a, b=su_b, loc=su_loc, scale=su_scale)

# Normal
cdf_norm = stats.norm.cdf(return_data, loc=return_data.mean(), scale=return_data.std())
pdf_norm = stats.norm.pdf(return_data, loc=return_data.mean(), scale=return_data.std())

# Empirical distribution function
edf = np.arange(1, len(return_data) + 1) / len(return_data)

# KS goodness-of-fit (Kolmogorov-Smirnov approximation)
supremum_jsu  = np.max(np.abs(edf - cdf_jsu))
p_value_jsu   = np.exp(-supremum_jsu**2 * len(return_data))
supremum_norm = np.max(np.abs(edf - cdf_norm))
p_value_norm  = np.exp(-supremum_norm**2 * len(return_data))

# Distribution moments
mean_d     = pd.Series(return_data).mean()
std_d      = pd.Series(return_data).std()
skewness_d = pd.Series(return_data).skew()
kurtosis_d = pd.Series(return_data).kurtosis()

# Annualised figures
ann_mean = mean_d * trading_days_per_year
ann_vol  = std_d  * np.sqrt(trading_days_per_year)

# ── Summary metrics ───────────────────────────────────────────────────────────
st.subheader("Summary statistics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Annualised mean return", f"{ann_mean:.2%}")
c2.metric("Annualised volatility",  f"{ann_vol:.2%}")
c3.metric("Skewness",               f"{skewness_d:.2f}")
c4.metric("Excess kurtosis",        f"{kurtosis_d:.2f}")

st.markdown(
    f"""
    **Goodness-of-fit — Kolmogorov-Smirnov test:**
    - P-value Johnson SU distribution: **{p_value_jsu:.2%}**
    - P-value Normal distribution: **{p_value_norm:.2%}**

    **Distribution moments of daily log-returns:**
    - Mean: {mean_d:.4%}
    - Standard deviation: {std_d:.4%}
    - Skewness: {skewness_d:.4f}
    - Excess kurtosis: {kurtosis_d:.4f}

    **Fitted Johnson SU parameters:**
    - a (shape): {su_a:.4f} &nbsp;|&nbsp; b (shape): {su_b:.4f}
    - loc: {su_loc:.6f} &nbsp;|&nbsp; scale: {su_scale:.6f}
    """
)

# ── Plots ─────────────────────────────────────────────────────────────────────
st.subheader("Distribution fitting")

color1 = "cornflowerblue"
color2 = "darkmagenta"
color3 = "royalblue"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: histogram + PDFs
ax1.hist(return_data, bins=100, density=True, alpha=0.6,
         color=color1, label="Histogram of daily log-returns")
ax1.plot(return_data, pdf_jsu,  color=color2, lw=1.5, label="PDF — Johnson SU")
ax1.plot(return_data, pdf_norm, color=color3, lw=1.5, label="PDF — Normal")
ax1.set_xlabel("Daily log-returns")
ax1.set_ylabel("Density")
ax1.grid(True, ls="--")
ax1.legend(fontsize=8)
ax1.xaxis.set_major_formatter(plt.FuncFormatter("{:,.1%}".format))

# Right: CDFs
ax2.plot(return_data, edf,      color=color1, lw=1.5, label="Empirical CDF")
ax2.plot(return_data, cdf_jsu,  color=color2, lw=1.5, label="CDF — Johnson SU")
ax2.plot(return_data, cdf_norm, color=color3, lw=1.5, label="CDF — Normal")
ax2.set_xlabel("Daily log-returns")
ax2.set_ylabel("Cumulative probability")
ax2.grid(True, ls="--")
ax2.legend(fontsize=8)
ax2.xaxis.set_major_formatter(plt.FuncFormatter("{:,.1%}".format))

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── Price history chart ───────────────────────────────────────────────────────
st.subheader("Price history")

fig2, ax = plt.subplots(figsize=(12, 4))
ax.plot(adj_close.index, adj_close.values, color=color1, lw=1)
ax.set_ylabel("Adjusted close price")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(MaxNLocator())
ax.grid(True, ls="--")
plt.xticks(rotation=0)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# ── Rolling volatility chart ──────────────────────────────────────────────────
st.subheader("Rolling volatility")

rolling_window = st.number_input(
    "Rolling window (days)", min_value=5, max_value=252, value=30, step=5
)

rolling_vol = log_returns.rolling(rolling_window).std() * np.sqrt(trading_days_per_year)
long_run_vol = log_returns.std() * np.sqrt(trading_days_per_year)

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(rolling_vol.index, rolling_vol.values,
         color=color1, lw=1.25,
         label=f"Rolling volatility ({rolling_window}d)")
ax3.axhline(long_run_vol, color=color2, lw=1.25, ls="--",
            label=f"Long-run avg volatility ({long_run_vol:.1%})")
ax3.yaxis.set_major_formatter(plt.FuncFormatter("{:,.0%}".format))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax3.xaxis.set_major_locator(MaxNLocator())
ax3.grid(True, ls="--")
ax3.legend(fontsize=9)
plt.xticks(rotation=0)
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

# ── Export return data ────────────────────────────────────────────────────────
st.subheader("Export")

export_df = pd.DataFrame({
    "Date":        log_returns.index,
    "Adj Close":   adj_close.reindex(log_returns.index).values,
    "Log Return":  log_returns.values,
})

towrite = BytesIO()
export_df.to_excel(towrite, index=False)
towrite.seek(0)
import base64
b64 = base64.b64encode(towrite.read()).decode()
st.markdown(
    f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" '
    f'download="{ticker}_log_returns.xlsx">Download log returns as Excel</a>',
    unsafe_allow_html=True
)
