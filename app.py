import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import scipy.stats as stats
from io import BytesIO
import streamlit as st
import yfinance as yf
import pandas_market_calendars as mcal
import os
import matplotlib.font_manager as fm
import base64


font_path = os.path.join(os.path.dirname(__file__), "ARIAL.TTF")
font_path_bold = os.path.join(os.path.dirname(__file__), "ARIALBD.TTF")
fm.fontManager.addfont(font_path)
fm.fontManager.addfont(font_path_bold)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8

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

# ── Chart A: Histogram + PDFs ─────────────────────────────────────────────────
fig_a, ax_a = plt.subplots(figsize=(15.52/2.54, 12.02/2.54))

ax_a.hist(return_data, bins=100, density=True,
          color="#D1D1D1", label="Histogram of daily log-returns")
ax_a.plot(return_data, pdf_jsu,  color="#001836", lw=1.5, label="PDF Johnson SU distribution")
ax_a.plot(return_data, pdf_norm, color="#8497B0",  lw=1.5, label="PDF Normal distribution")

ax_a.set_ylabel("Density")
ax_a.set_xlabel("Daily log-returns")
ax_a.xaxis.set_major_formatter(plt.FuncFormatter("{:.0%}".format))
ax_a.grid(False)
ax_a.legend(fontsize=8, frameon=False, loc="upper left")
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.tick_params(axis="both", length=0)

for label in ax_a.get_yticklabels():
    label.set_fontsize(8)
for label in ax_a.get_xticklabels():
    label.set_fontsize(8)

plt.tight_layout()
fig_a.text(0.0, -0.02,
           f"Kolmogorov-Smirnov test p-value — Johnson SU distribution: {p_value_jsu:.1%}, Normal distribution: {p_value_norm:.1%}",
           ha="center", va="top", fontsize=7)
st.pyplot(fig_a)
plt.close(fig_a)

# ── Chart B: Empirical vs fitted CDFs ────────────────────────────────────────
fig_b, ax_b = plt.subplots(figsize=(15.52/2.54, 12.02/2.54))

ax_b.plot(return_data, edf,      color="#D1D1D1", lw=1.5, label="Empirical distribution function")
ax_b.plot(return_data, cdf_jsu,  color="#001836", lw=1.5, label="CDF Johnson SU distribution")
ax_b.plot(return_data, cdf_norm, color="#8497B0",  lw=1.5, label="CDF Normal distribution")

ax_b.set_ylabel("Cumulative probability")
ax_b.set_xlabel("Daily log-returns")
ax_b.xaxis.set_major_formatter(plt.FuncFormatter("{:.0%}".format))
ax_b.grid(False)
ax_b.legend(fontsize=8, frameon=False, loc="upper left")
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.tick_params(axis="both", length=0)

for label in ax_b.get_yticklabels():
    label.set_fontsize(8)
for label in ax_b.get_xticklabels():
    label.set_fontsize(8)

plt.tight_layout()
st.pyplot(fig_b)
plt.close(fig_b)

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

# ── Monte Carlo simulation ────────────────────────────────────────────────────
st.subheader("Monte Carlo simulation")

current_price = float(adj_close.iloc[-1])

exchanges = mcal.get_calendar_names()
default_exchange = exchanges.index("NYSE")
exchange = st.selectbox("Select exchange", exchanges, index=default_exchange)

col1, col2 = st.columns(2)
with col1:
    sim_runs = st.number_input("Number of simulation runs", value=10000, min_value=100, step=1000)
with col2:
    end_of_year = pd.Timestamp(pd.Timestamp.today().year, 12, 31).date()
    sim_end_date = st.date_input("Simulation end date", value=end_of_year)

cal = mcal.get_calendar(exchange)
trading_days = mcal.date_range(
    cal.schedule(start_date=adj_close.index[-1].date(), end_date=sim_end_date), 
    frequency='1D'
)
sim_trading_days = len(trading_days) - 1

st.write(f"Number of trading days to simulate: **{sim_trading_days:,}**")
if st.button("Run simulation"):
    days_to_add = sim_trading_days / 20
    
    # Draw random returns from fitted Johnson SU distribution
    cdf_for_sim = stats.johnsonsu.cdf(return_data, a=su_a, b=su_b, loc=su_loc, scale=su_scale)
    returns_sim = stats.johnsonsu.ppf(
        np.random.uniform(cdf_for_sim.min(), cdf_for_sim.max(), size=(sim_trading_days, sim_runs)),
        a=su_a, b=su_b, loc=su_loc, scale=su_scale
    )

    # Compound into price paths
    growth_factors = np.exp(returns_sim)
    growth_t0 = np.ones((1, growth_factors.shape[1])) * current_price
    growth_paths = np.vstack((growth_t0, growth_factors))
    price_paths = np.cumprod(growth_paths, axis=0)

    # Summary statistics at end of horizon
    final_prices = price_paths[-1, :]
    mean_final   = float(np.mean(final_prices))
    median_final = float(np.median(final_prices))
    pct_5        = float(np.percentile(final_prices, 5))
    pct_25       = float(np.percentile(final_prices, 25))
    pct_75       = float(np.percentile(final_prices, 75))
    pct_95       = float(np.percentile(final_prices, 95))

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean price", f"{mean_final:,.2f}")
    c2.metric("Median price", f"{median_final:,.2f}")
    c3.metric("Current price", f"{current_price:,.2f}")

    c4, c5, c6, c7 = st.columns(4)
    c4.metric("5th percentile",  f"{pct_5:,.2f}")
    c5.metric("25th percentile", f"{pct_25:,.2f}")
    c6.metric("75th percentile", f"{pct_75:,.2f}")
    c7.metric("95th percentile", f"{pct_95:,.2f}")

    # Build date index for x-axis
    last_date = adj_close.index[-1]
    sim_dates = pd.bdate_range(start=last_date, periods=sim_trading_days + 1)

    # Plot
    fig4, ax4 = plt.subplots(figsize=(15.52/2.54, 12.02/2.54))

    # Historic price from start of current year to last available date
    start_of_year = pd.Timestamp(pd.Timestamp.today().year, 1, 1)
    historic_ytd = adj_close[adj_close.index >= start_of_year]
    ax4.plot(historic_ytd.index, historic_ytd.values,
             color="#001836", lw=1.5, label="Price development (YTD)", zorder=4)
    
    # Simulated paths — lowest layer (single entry for legend)
    sim_lines = ax4.plot(sim_dates, price_paths[:, :300], color="#D1D1D1", alpha=0.5, lw=1.5, zorder=1)
    sim_lines[0].set_label("Simulated price paths (YTG)")
    for line in sim_lines[1:]:
        line.set_label("_nolegend_")
    
    # Percentile bands — above simulation lines
    ax4.fill_between(sim_dates,
                     np.percentile(price_paths, 5, axis=1),
                     np.percentile(price_paths, 95, axis=1),
                     color="cornflowerblue", alpha=0.15, label="5th–95th percentile", zorder=2)
    ax4.fill_between(sim_dates,
                     np.percentile(price_paths, 25, axis=1),
                     np.percentile(price_paths, 75, axis=1),
                     color="cornflowerblue", alpha=0.3, label="25th–75th percentile", zorder=2)
    
    # Mean path — top layer
    mean_path = price_paths.mean(axis=1)
    ax4.plot(sim_dates, mean_path, color="#8497B0", lw=1.5, label="Mean path", zorder=3)
    
    # Current price annotation — just above end of historic line, slight left offset
    ax4.annotate(f"{current_price:,.0f}",
                 xy=(adj_close.index[-1], current_price),
                 xytext=(adj_close.index[-1] - pd.Timedelta(days=days_to_add*0.8), current_price * 1.02),
                 color="#001836", fontsize=8, fontweight="bold",
                 verticalalignment="bottom", horizontalalignment="right",
                 zorder=5)
    
    # Annotations at end of horizon
    for val, color in [
        (mean_final, "#8497B0"),
        (pct_95,     "cornflowerblue"),
        (pct_75,     "cornflowerblue"),
        (pct_25,     "cornflowerblue"),
        (pct_5,      "cornflowerblue"),
    ]:
        ax4.text(sim_dates[-1] + pd.Timedelta(days=days_to_add),
                 val, f"{val:,.0f}", color=color,
                 verticalalignment="center", fontsize=8, fontweight="bold", zorder=5)
    
    ax4.set_ylabel("")
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax4.figure.autofmt_xdate(rotation=0, ha="center")
    ax4.grid(False)
    ax4.legend(fontsize=8, frameon=False)
    ax4.set_xlim(left=pd.Timestamp(pd.Timestamp.today().year, 1, 1), right=sim_dates[-1])
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.tick_params(axis="both", length=0)
    
    for label in ax4.get_yticklabels():
        label.set_fontsize(8)
    for label in ax4.get_xticklabels():
        label.set_fontsize(8)
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    # Return distribution at horizon
    fig5, ax5 = plt.subplots(figsize=(15.52/2.54, 12.02/2.54))
    
    ax5.hist(final_prices, bins=100, color="#D1D1D1",
             label="Simulated prices at end of horizon")
    ax5.axvline(mean_final,   color="#001836", lw=1.5, label=f"Mean: {mean_final:,.0f}")
    ax5.axvline(median_final, color="#8497B0", lw=1.5, label=f"Median: {median_final:,.0f}")
    
    ax5.set_xlabel(f"Simulated price at end of horizon ({sim_end_date.strftime('%d-%b-%y')})")
    ax5.set_ylabel("Frequency")
    ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax5.grid(False)
    ax5.legend(fontsize=8, frameon=False, loc="best")
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.tick_params(axis="both", length=0)
    
    for label in ax5.get_yticklabels():
        label.set_fontsize(8)
    for label in ax5.get_xticklabels():
        label.set_fontsize(8)
    
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    st.markdown(
        f"""
        Based on **{int(sim_runs):,}** simulated price paths until **{sim_end_date}** using returns drawn 
        from the fitted Johnson SU distribution:
        - Mean final price: **{mean_final:,.2f}**
        - Median final price: **{median_final:,.2f}**
        - 90% of outcomes fall between **{pct_5:,.2f}** and **{pct_95:,.2f}**
        """
    )

    # ── Simulated vs historical moments comparison ────────────────────────────────
    st.markdown("**Distribution moments — simulated vs historical daily log-returns:**")
    
    sim_daily_returns = returns_sim.flatten()
    
    moments_df = pd.DataFrame({
        "Moment": ["Mean", "Standard deviation", "Skewness", "Excess kurtosis"],
        "Historical": [
            f"{mean_d:.4%}",
            f"{std_d:.4%}",
            f"{skewness_d:.4f}",
            f"{kurtosis_d:.4f}"
        ],
        "Simulated": [
            f"{float(np.mean(sim_daily_returns)):.4%}",
            f"{float(np.std(sim_daily_returns)):.4%}",
            f"{float(pd.Series(sim_daily_returns).skew()):.4f}",
            f"{float(pd.Series(sim_daily_returns).kurtosis()):.4f}"
        ]
    })
    
    st.dataframe(moments_df, hide_index=True)

    moments_export_df = pd.DataFrame({
    "Moment": ["Mean", "Standard deviation", "Skewness", "Excess kurtosis"],
    "Historical": [mean_d, std_d, skewness_d, kurtosis_d],
    "Simulated": [
        float(np.mean(sim_daily_returns)),
        float(np.std(sim_daily_returns)),
        float(pd.Series(sim_daily_returns).skew()),
        float(pd.Series(sim_daily_returns).kurtosis())
        ]
    })
    
    towrite = BytesIO()
    moments_export_df.to_excel(towrite, index=False)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    st.markdown(
        f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" '
        f'download="distribution_moments_{ticker}.xlsx">Download distribution moments as Excel</a>',
        unsafe_allow_html=True
    )


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
b64 = base64.b64encode(towrite.read()).decode()
st.markdown(
    f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" '
    f'download="{ticker}_log_returns.xlsx">Download log returns as Excel</a>',
    unsafe_allow_html=True
)
