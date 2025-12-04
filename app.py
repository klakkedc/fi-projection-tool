import math
import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Conditional import so it works locally & on Streamlit Cloud
try:
    from streamlit_js_eval import streamlit_js_eval
except ModuleNotFoundError:
    # Fallback stub for local development (no browser localStorage)
    def streamlit_js_eval(*args, **kwargs):
        return None


# ---------- Session State Setup ----------

if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {}  # name -> dict of parameters

if "active_scenario" not in st.session_state:
    st.session_state["active_scenario"] = None

# Default portfolio table (used if nothing loaded from localStorage)
if "portfolio_df" not in st.session_state:
    st.session_state["portfolio_df"] = pd.DataFrame(
        [
            {"Ticker": "VWCE.DE", "Shares": 10.0, "Price": 0.0, "Value": 0.0, "Class": "Stocks"},
            {"Ticker": "AGGH.DE", "Shares": 10.0, "Price": 0.0, "Value": 0.0, "Class": "Bonds"},
        ]
    )


# ---------- Data Models ----------

@dataclass
class OneTimeInjection:
    year_offset: int  # 1 = end of first year, 2 = end of second year, etc.
    amount: float


@dataclass
class ProjectionInput:
    name: str
    current_age: int
    retirement_age: int
    current_portfolio: float
    monthly_invest: float
    annual_bonus_invest: float
    one_time_injections: List[OneTimeInjection]
    annual_return: float  # e.g. 0.065 for 6.5%
    target_monthly_spend: float  # for FI calculation (today's €)
    safe_withdrawal_rate: float = 0.04  # 4% rule by default
    inflation_rate: float = 0.0         # optional inflation per year
    crash_year_offset: Optional[int] = None  # 1-based: year of crash
    crash_drawdown: float = 0.0               # e.g. 0.37 for -37%


@dataclass
class YearResult:
    age: int
    year_index: int
    portfolio_end: float
    fi_number_this_year: float


@dataclass
class ProjectionResult:
    params: ProjectionInput
    fi_number_today: float
    fi_age: Optional[int]
    fi_year_index: Optional[int]
    years: List[YearResult]
    coast_fi_age: Optional[int]


# ---------- Core Deterministic Projection ----------

def run_projection(params: ProjectionInput) -> ProjectionResult:
    """
    Runs a yearly projection with monthly contributions, annual bonus,
    optional one-time injections, optional inflation-adjusted FI,
    and optional crash in a given year (applied after that year's contributions).
    """
    current_portfolio = params.current_portfolio
    num_years = params.retirement_age - params.current_age

    # Map one-time injections by year index (1-based)
    injections_by_year = {}
    for inj in params.one_time_injections:
        injections_by_year.setdefault(inj.year_offset, 0.0)
        injections_by_year[inj.year_offset] += inj.amount

    years: List[YearResult] = []

    annual_spend_today = params.target_monthly_spend * 12
    fi_number_today = annual_spend_today / params.safe_withdrawal_rate

    for year_idx in range(1, num_years + 1):
        annual_rate_this_year = params.annual_return
        monthly_rate = (1 + annual_rate_this_year) ** (1 / 12) - 1

        # 12 months of growth + monthly investing
        for _ in range(12):
            current_portfolio = current_portfolio * (1 + monthly_rate) + params.monthly_invest

        # Add annual bonus at year end
        current_portfolio += params.annual_bonus_invest

        # Add any one-time injections scheduled at this year
        if year_idx in injections_by_year:
            current_portfolio += injections_by_year[year_idx]

        # Apply crash drawdown AFTER contributions in that year
        if params.crash_year_offset is not None and params.crash_year_offset == year_idx:
            if params.crash_drawdown > 0:
                current_portfolio *= (1.0 - params.crash_drawdown)

        # FI number in nominal euros for this year (inflation adjustment)
        fi_number_this_year = fi_number_today * ((1 + params.inflation_rate) ** year_idx)

        age = params.current_age + year_idx
        years.append(
            YearResult(
                age=age,
                year_index=year_idx,
                portfolio_end=current_portfolio,
                fi_number_this_year=fi_number_this_year,
            )
        )

    # Determine FI age (first year where portfolio >= FI number that year)
    fi_age: Optional[int] = None
    fi_year_index: Optional[int] = None
    for yr in years:
        if yr.portfolio_end >= yr.fi_number_this_year:
            fi_age = yr.age
            fi_year_index = yr.year_index
            break

    # Coast FI age: first age where, if you stop investing completely and
    # just let the portfolio grow until retirement, you'll still hit FI at retirement.
    coast_fi_age: Optional[int] = None
    annual_rate = params.annual_return
    num_years_total = num_years
    fi_number_at_retirement = fi_number_today * ((1 + params.inflation_rate) ** num_years_total)

    for yr in years:
        years_remaining = params.retirement_age - yr.age
        future_portfolio = yr.portfolio_end * ((1 + annual_rate) ** years_remaining)
        if future_portfolio >= fi_number_at_retirement:
            coast_fi_age = yr.age
            break

    return ProjectionResult(
        params=params,
        fi_number_today=fi_number_today,
        fi_age=fi_age,
        fi_year_index=fi_year_index,
        years=years,
        coast_fi_age=coast_fi_age,
    )


# ---------- Monte Carlo Simulation ----------

def monte_carlo_simulation(
    current_portfolio: float,
    monthly_invest: float,
    annual_bonus: float,
    one_time_injections: List[OneTimeInjection],
    current_age: int,
    retirement_age: int,
    annual_return: float,
    annual_volatility: float,
    crash_year_offset: Optional[int] = None,
    crash_drawdown: float = 0.0,
    runs: int = 1000,
):
    """
    Monte Carlo simulation of future portfolio values.
    annual_return and annual_volatility are decimals (0.065, 0.15).
    """
    years = retirement_age - current_age
    ages = [current_age + i for i in range(1, years + 1)]

    all_paths = np.zeros((runs, years))

    # Pre-index injections for speed
    injections_by_year = {}
    for inj in one_time_injections:
        injections_by_year.setdefault(inj.year_offset, 0.0)
        injections_by_year[inj.year_offset] += inj.amount

    for r in range(runs):
        value = current_portfolio
        for y in range(years):
            yearly_return = np.random.normal(annual_return, annual_volatility)
            monthly_rate = (1 + yearly_return) ** (1 / 12) - 1

            for _ in range(12):
                value = value * (1 + monthly_rate) + monthly_invest

            # Annual bonus
            value += annual_bonus

            # One-time injections at end of year y+1
            if (y + 1) in injections_by_year:
                value += injections_by_year[y + 1]

            # If this is the crash year, apply drawdown AFTER contributions
            if crash_year_offset is not None and crash_year_offset == (y + 1):
                if crash_drawdown > 0:
                    value *= (1.0 - crash_drawdown)

            all_paths[r, y] = value

    median_curve = np.median(all_paths, axis=0)
    p10_curve = np.percentile(all_paths, 10, axis=0)
    p90_curve = np.percentile(all_paths, 90, axis=0)

    return {
        "ages": ages,
        "median": median_curve,
        "p10": p10_curve,
        "p90": p90_curve,
        "all_paths": all_paths,
    }


# ---------- Asset Allocation Engine ----------

def compute_portfolio_from_allocation(weights_pct: dict):
    """
    weights_pct: dict with keys 'stocks','bonds','reit','gold','cash' in %
    Returns (expected_return, volatility)
    """
    labels = ["stocks", "bonds", "reit", "gold", "cash"]
    w = np.array([weights_pct.get(k, 0.0) for k in labels], dtype=float)
    total = w.sum()
    if total <= 0:
        return 0.0, 0.0
    w = w / total

    # Reasonable long-term assumptions
    mu = np.array([
        0.07,  # stocks
        0.02,  # bonds
        0.06,  # REIT
        0.03,  # gold
        0.01,  # cash
    ])

    sigma = np.array([
        0.18,  # stocks
        0.06,  # bonds
        0.16,  # REIT
        0.15,  # gold
        0.01,  # cash
    ])

    corr = np.array([
        [1.00,  0.10, 0.70, 0.20, 0.00],
        [0.10,  1.00, 0.20, 0.00, 0.10],
        [0.70,  0.20, 1.00, 0.25, 0.00],
        [0.20,  0.00, 0.25, 1.00, 0.00],
        [0.00,  0.10, 0.00, 0.00, 1.00],
    ])

    cov = np.outer(sigma, sigma) * corr

    expected_return = float(w @ mu)
    variance = float(w @ cov @ w)
    volatility = math.sqrt(variance)

    return expected_return, volatility


# ---------- Portfolio Helpers ----------

ASSET_CLASSES = ["Stocks", "Bonds", "REIT", "Gold", "Cash"]
LOCALSTORAGE_KEY = "fi_portfolio_v1"


def fetch_prices_for_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple price fetch: 1d history per ticker.
    Shows warnings if something fails but keeps app usable.
    """
    df = df.copy()

    prices = []
    values = []

    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).strip()
        shares = float(row.get("Shares", 0.0) or 0.0)

        price = 0.0

        if ticker:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
                else:
                    st.warning(f"⚠️ No price data returned for {ticker}.")
            except Exception as e:
                st.warning(f"⚠️ Could not fetch price for {ticker}: {e}")

        value = price * shares
        prices.append(price)
        values.append(value)

    df["Price"] = prices
    df["Value"] = values

    total_value = df["Value"].sum()
    if total_value > 0:
        df["Weight"] = df["Value"] / total_value
    else:
        df["Weight"] = 0.0

    if "Class" not in df.columns:
        df["Class"] = "Stocks"
    df["Class"] = df["Class"].where(df["Class"].isin(ASSET_CLASSES), "Stocks")

    return df


def allocation_from_portfolio(df: pd.DataFrame) -> dict:
    """
    Aggregate portfolio by asset class -> % weights dict.
    """
    if df.empty or df["Value"].sum() <= 0:
        return {"stocks": 0, "bonds": 0, "reit": 0, "gold": 0, "cash": 0}

    grp = df.groupby("Class")["Value"].sum()
    total = grp.sum()
    weights = grp / total

    return {
        "stocks": float(weights.get("Stocks", 0.0) * 100),
        "bonds": float(weights.get("Bonds", 0.0) * 100),
        "reit": float(weights.get("REIT", 0.0) * 100),
        "gold": float(weights.get("Gold", 0.0) * 100),
        "cash": float(weights.get("Cash", 0.0) * 100),
    }


# ---------- Portfolio History (daily performance) ----------

def compute_portfolio_history(df: pd.DataFrame, period: str = "6mo") -> pd.DataFrame:
    """
    Build a daily history of total portfolio value over a given period
    based on the CURRENT tickers + shares in the table.

    period examples: '3mo', '6mo', '1y', '2y'
    """
    df = df.copy()

    # Only tickers with positive shares
    if "Shares" not in df.columns or "Ticker" not in df.columns:
        return pd.DataFrame()

    df = df[df["Shares"].astype(float) > 0]
    if df.empty:
        return pd.DataFrame()

    tickers = df["Ticker"].astype(str).str.strip().tolist()
    shares_map = dict(zip(df["Ticker"].astype(str).str.strip(), df["Shares"].astype(float)))

    try:
        price_data = yf.download(
            tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        st.error(f"Could not download history for portfolio: {e}")
        return pd.DataFrame()

    # Handle single vs multiple tickers structure
    if isinstance(price_data, pd.DataFrame) and "Adj Close" in price_data.columns:
        close = price_data["Adj Close"]
    else:
        close = price_data

    # If single ticker, make it a DataFrame
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    # Align columns with tickers present in shares_map
    cols = [c for c in close.columns if c in shares_map]
    if not cols:
        return pd.DataFrame()

    close = close[cols]

    # Multiply each ticker's price by its shares
    for ticker in cols:
        close[ticker] = close[ticker] * shares_map[ticker]

    # Sum across tickers -> total portfolio value per day
    portfolio_series = close.sum(axis=1)
    hist_df = pd.DataFrame({"Date": portfolio_series.index, "PortfolioValue": portfolio_series.values})
    hist_df.set_index("Date", inplace=True)

    return hist_df


# ---------- LocalStorage Helpers (browser persistence) ----------

def load_portfolio_from_localstorage():
    """
    Reads portfolio JSON from browser localStorage (if present)
    and loads it into st.session_state["portfolio_df"].
    """
    stored = streamlit_js_eval(
        js_expressions="window.localStorage.getItem('" + LOCALSTORAGE_KEY + "')",
        key="load_portfolio",
        want_output=True,
    )
    if stored:
        try:
            df = pd.read_json(stored)
            if not df.empty:
                st.session_state["portfolio_df"] = df
        except Exception as e:
            st.warning(f"Could not decode stored portfolio from localStorage: {e}")


def save_portfolio_to_localstorage(df: pd.DataFrame):
    """
    Saves the current portfolio dataframe to browser localStorage
    as a JSON string.
    """
    try:
        json_str = df.to_json()
        js_code = f"window.localStorage.setItem('{LOCALSTORAGE_KEY}', {json.dumps(json_str)});"

        # Use unique key each time to avoid Streamlit duplicate-key errors
        unique_key = f"save_portfolio_{np.random.randint(0, 1_000_000)}"
        streamlit_js_eval(js_expressions=js_code, key=unique_key)

    except Exception as e:
        st.warning(f"Could not save portfolio to localStorage: {e}")


# ---------- Streamlit UI ----------

def main():
    st.set_page_config(page_title="FI & Portfolio Projection", layout="wide")
    st.title("📈 Financial Independence & Portfolio Projection Tool")

    # Only load from localStorage once per session
    if "portfolio_initialized" not in st.session_state:
        try:
            load_portfolio_from_localstorage()
        except Exception as e:
            st.warning(f"Could not load portfolio from local storage: {e}")

        # If still nothing set, keep or create default
        if "portfolio_df" not in st.session_state or st.session_state["portfolio_df"].empty:
            st.session_state["portfolio_df"] = pd.DataFrame(
                [
                    {"Ticker": "VWCE.DE", "Shares": 10.0, "Price": 0.0, "Value": 0.0, "Class": "Stocks"},
                    {"Ticker": "AGGH.DE", "Shares": 10.0, "Price": 0.0, "Value": 0.0, "Class": "Bonds"},
                ]
            )

        st.session_state["portfolio_initialized"] = True


    st.markdown(
        """
        This tool projects your portfolio year by year, calculates your **FI number**,  
        estimates your **FI age**, **Coast FI age**, and runs a **Monte Carlo simulation**  
        including an optional **historical crash scenario** and an **asset allocation engine**  
        that can derive returns from your **real portfolio holdings**.
        """
    )

    # --- Portfolio section (top) ---

    st.subheader("📥 Your Current Portfolio (Tickers + Shares)")

    st.markdown(
        "Enter your tickers and number of shares. "
        "Click **Fetch live prices** to update values. "
        "Choose the asset class per row to feed the allocation engine.\n\n"
        "_Your portfolio is automatically saved in this browser and will be restored next time._"
    )

    with st.expander("Edit portfolio holdings", expanded=True):
        edited_df = st.data_editor(
            st.session_state["portfolio_df"],
            num_rows="dynamic",
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Shares": st.column_config.NumberColumn(
                    "Shares",
                    step=0.01,
                    format="%.4f",
                ),
                "Class": st.column_config.SelectboxColumn("Class", options=ASSET_CLASSES),
            },
            use_container_width=True,
        )

        # Update session state and save to browser storage
        st.session_state["portfolio_df"] = edited_df
        save_portfolio_to_localstorage(st.session_state["portfolio_df"])

        if st.button("🔄 Fetch live prices"):
            st.session_state["portfolio_df"] = fetch_prices_for_portfolio(st.session_state["portfolio_df"])
            save_portfolio_to_localstorage(st.session_state["portfolio_df"])

    portfolio_df = st.session_state["portfolio_df"]

    # Ensure Price/Value/Weight columns exist
    for col in ["Price", "Value", "Weight"]:
        if col not in portfolio_df.columns:
            portfolio_df[col] = 0.0
    st.session_state["portfolio_df"] = portfolio_df

    total_portfolio_value = portfolio_df["Value"].sum()
    st.markdown(f"**Total portfolio value (from table):** €{total_portfolio_value:,.2f}")

    if len(portfolio_df) > 0:
        st.dataframe(
            portfolio_df.style.format(
                {"Price": "€{:,.2f}", "Value": "€{:,.2f}", "Weight": "{:.2%}"}
            ),
            use_container_width=True,
        )

    # --- NEW: Historical performance chart ---
    with st.expander("📈 Historical performance of this portfolio (based on past prices)", expanded=False):
        period = st.selectbox(
            "Period",
            options=["3mo", "6mo", "1y", "2y"],
            index=1,  # default 6mo
            key="hist_period",
        )

        if st.button("Build history from prices", key="build_history"):
            hist_df = compute_portfolio_history(portfolio_df, period=period)
            if hist_df.empty:
                st.warning("No historical data could be built for this portfolio (check tickers).")
            else:
                st.line_chart(hist_df["PortfolioValue"])
                start_val = hist_df["PortfolioValue"].iloc[0]
                end_val = hist_df["PortfolioValue"].iloc[-1]
                change_pct = (end_val / start_val - 1) * 100 if start_val > 0 else 0.0
                st.markdown(
                    f"Start: **€{start_val:,.0f}** → End: **€{end_val:,.0f}** "
                    f"({change_pct:+.1f}%) over **{period}**."
                )

    # Derive allocation from portfolio (for use-my-portfolio mode)
    portfolio_alloc = allocation_from_portfolio(portfolio_df)

    # --- Sidebar Inputs ---

    st.sidebar.header("🔧 Basic Inputs")

    col1, col2 = st.sidebar.columns(2)
    current_age = col1.number_input(
        "Current Age", min_value=16, max_value=80, value=27, step=1, key="current_age"
    )
    retirement_age = col2.number_input(
        "Retirement Age",
        min_value=st.session_state.current_age + 1,
        max_value=80,
        value=67,
        step=1,
        key="retirement_age",
    )

    current_portfolio_input = st.sidebar.number_input(
        "Current Portfolio for Projection (€)",
        min_value=0.0,
        step=1000.0,
        value=float(total_portfolio_value or 44327.73),
        key="current_portfolio",
    )

    monthly_invest = st.sidebar.number_input(
        "Monthly Investment (€)", min_value=0.0, step=50.0, value=1000.0, key="monthly_invest"
    )

    annual_bonus_invest = st.sidebar.number_input(
        "Annual Bonus Invested (€)", min_value=0.0, step=500.0, value=5000.0, key="annual_bonus"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 FI Target")

    annual_return_pct_manual = st.sidebar.number_input(
        "Expected Annual Return (manual, %)",
        min_value=0.0,
        max_value=20.0,
        value=6.5,
        step=0.1,
        key="annual_return_pct",
    )

    target_monthly_spend = st.sidebar.number_input(
        "Target Monthly Spend at FI (€)",
        min_value=500.0,
        step=100.0,
        value=3000.0,
        key="target_monthly_spend",
    )

    safe_withdrawal_rate_pct = st.sidebar.number_input(
        "Safe Withdrawal Rate (%)",
        min_value=2.0,
        max_value=6.0,
        value=4.0,
        step=0.25,
        key="swr_pct",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧯 Inflation")

    use_inflation = st.sidebar.checkbox("Adjust FI number for inflation", value=True, key="use_inflation")
    inflation_rate_pct = st.sidebar.number_input(
        "Inflation Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.25,
        key="inflation_pct",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Asset Allocation Engine")

    alloc_source = st.sidebar.radio(
        "Allocation source",
        options=["Use manual sliders", "Use my portfolio"],
        index=1 if total_portfolio_value > 0 else 0,
        key="alloc_source",
    )

    if alloc_source == "Use manual sliders":
        aa_stocks = st.sidebar.slider("Stocks (%)", 0, 100, 80, step=5, key="aa_stocks")
        aa_bonds = st.sidebar.slider("Bonds (%)", 0, 100, 10, step=5, key="aa_bonds")
        aa_reit = st.sidebar.slider("REITs (%)", 0, 100, 5, step=5, key="aa_reit")
        aa_gold = st.sidebar.slider("Gold (%)", 0, 100, 5, step=5, key="aa_gold")
        aa_cash = st.sidebar.slider("Cash (%)", 0, 100, 0, step=5, key="aa_cash")
    else:
        aa_stocks = int(round(portfolio_alloc["stocks"]))
        aa_bonds = int(round(portfolio_alloc["bonds"]))
        aa_reit = int(round(portfolio_alloc["reit"]))
        aa_gold = int(round(portfolio_alloc["gold"]))
        aa_cash = int(round(portfolio_alloc["cash"]))

        st.sidebar.write("Derived from your portfolio:")
        st.sidebar.write(f"- Stocks: **{aa_stocks}%**")
        st.sidebar.write(f"- Bonds: **{aa_bonds}%**")
        st.sidebar.write(f"- REIT: **{aa_reit}%**")
        st.sidebar.write(f"- Gold: **{aa_gold}%**")
        st.sidebar.write(f"- Cash: **{aa_cash}%**")

    alloc_total = aa_stocks + aa_bonds + aa_reit + aa_gold + aa_cash
    st.sidebar.caption(f"Total allocation: **{alloc_total}%** (normalized internally).")

    alloc_dict = {
        "stocks": aa_stocks,
        "bonds": aa_bonds,
        "reit": aa_reit,
        "gold": aa_gold,
        "cash": aa_cash,
    }

    alloc_return, alloc_vol = compute_portfolio_from_allocation(alloc_dict)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧨 Historical Crash Scenario")

    simulate_crash = st.sidebar.checkbox("Enable crash scenario", value=False, key="simulate_crash")
    crash_year_offset = None
    crash_drawdown = 0.0
    if simulate_crash:
        max_year_offset = max(1, st.session_state.retirement_age - st.session_state.current_age)
        crash_year_offset = st.sidebar.number_input(
            "Crash year offset (1 = first projected year)",
            min_value=1,
            max_value=max_year_offset,
            value=5,
            step=1,
            key="crash_year_offset",
        )
        crash_drawdown_pct = st.sidebar.number_input(
            "Crash drawdown (%)",
            min_value=5.0,
            max_value=90.0,
            value=37.0,
            step=1.0,
            key="crash_drawdown_pct",
        )
        crash_drawdown = crash_drawdown_pct / 100.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎁 One-Time Extra Injection")

    extra_5k_enabled = st.sidebar.checkbox("Include one-time €5k next year", value=True, key="extra_5k")
    one_time_injections: List[OneTimeInjection] = []
    if st.session_state.extra_5k:
        one_time_injections.append(OneTimeInjection(year_offset=1, amount=5000.0))

    with st.sidebar.expander("➕ Add Custom Injection"):
        custom_amount = st.number_input(
            "Custom Injection Amount (€)", min_value=0.0, step=500.0, value=0.0, key="custom_inj_amount"
        )
        custom_year_offset = st.number_input(
            "Year Offset (1 = end of first projected year)",
            min_value=1,
            max_value=max(1, st.session_state.retirement_age - st.session_state.current_age),
            value=2,
            step=1,
            key="custom_inj_year",
        )
        if custom_amount > 0:
            one_time_injections.append(
                OneTimeInjection(year_offset=int(st.session_state.custom_inj_year), amount=custom_amount)
            )

    early_retirement_age = st.sidebar.slider(
        "Early Retirement Age (for analysis)",
        min_value=st.session_state.current_age + 1,
        max_value=st.session_state.retirement_age,
        value=47,
        key="early_ret_age",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎲 Monte Carlo Settings")
    mc_runs = st.sidebar.number_input(
        "Number of simulations", 100, 5000, 1000, step=100, key="mc_runs"
    )

    volatility_pct_manual = st.sidebar.number_input(
        "Annual Volatility (manual, %)",
        1.0,
        50.0,
        15.0,
        step=0.5,
        key="volatility_pct",
    )

    # Decide whether to use allocation-derived or manual return/vol
    use_allocation = (alloc_source == "Use my portfolio")

    if use_allocation:
        annual_return = alloc_return
        annual_return_pct_display = annual_return * 100
        annual_volatility = alloc_vol
        volatility_pct_display = annual_volatility * 100
    else:
        annual_return = annual_return_pct_manual / 100.0
        annual_return_pct_display = annual_return_pct_manual
        annual_volatility = volatility_pct_manual / 100.0
        volatility_pct_display = volatility_pct_manual

    st.sidebar.metric(
        "Effective Annual Return (%)",
        f"{annual_return_pct_display:.2f}",
    )
    st.sidebar.metric(
        "Effective Volatility (%)",
        f"{volatility_pct_display:.2f}",
    )

    # --- Save/Load Scenarios (parameters only, portfolio separate) ---

    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Save / Load Scenarios")

    scenario_name = st.sidebar.text_input(
        "Scenario name", value=st.session_state.get("active_scenario") or ""
    )

    saved_names = list(st.session_state["scenarios"].keys())
    load_choice = st.sidebar.selectbox(
        "Load existing scenario",
        options=["(none)"] + saved_names,
        index=0,
    )

    col_save, col_delete = st.sidebar.columns(2)

    if col_save.button("Save scenario"):
        if scenario_name.strip():
            st.session_state["scenarios"][scenario_name.strip()] = {
                "current_age": st.session_state.current_age,
                "retirement_age": st.session_state.retirement_age,
                "current_portfolio": st.session_state.current_portfolio,
                "monthly_invest": st.session_state.monthly_invest,
                "annual_bonus": st.session_state.annual_bonus,
                "annual_return_pct": st.session_state.annual_return_pct,
                "target_monthly_spend": st.session_state.target_monthly_spend,
                "swr_pct": st.session_state.swr_pct,
                "use_inflation": st.session_state.use_inflation,
                "inflation_pct": st.session_state.inflation_pct,
                "extra_5k": st.session_state.extra_5k,
                "custom_inj_amount": st.session_state.custom_inj_amount,
                "custom_inj_year": st.session_state.custom_inj_year,
                "early_ret_age": st.session_state.early_ret_age,
                "mc_runs": st.session_state.mc_runs,
                "volatility_pct": st.session_state.volatility_pct,
                "simulate_crash": st.session_state.simulate_crash,
                "crash_year_offset": st.session_state.get("crash_year_offset"),
                "crash_drawdown_pct": st.session_state.get("crash_drawdown_pct", 37.0),
                "alloc_source": st.session_state.alloc_source,
                "aa_stocks": aa_stocks,
                "aa_bonds": aa_bonds,
                "aa_reit": aa_reit,
                "aa_gold": aa_gold,
                "aa_cash": aa_cash,
            }
            st.session_state["active_scenario"] = scenario_name.strip()
            st.sidebar.success(f"Saved scenario '{scenario_name.strip()}'")
        else:
            st.sidebar.error("Please enter a scenario name.")

    if col_delete.button("Delete scenario"):
        if load_choice != "(none)" and load_choice in st.session_state["scenarios"]:
            del st.session_state["scenarios"][load_choice]
            if st.session_state.get("active_scenario") == load_choice:
                st.session_state["active_scenario"] = None
            st.sidebar.success(f"Deleted scenario '{load_choice}'")
        else:
            st.sidebar.warning("Select a scenario to delete.")

    if load_choice != "(none)" and load_choice in st.session_state["scenarios"]:
        if st.sidebar.button("Load selected scenario"):
            data = st.session_state["scenarios"][load_choice]
            for key, val in data.items():
                st.session_state[key] = val
            st.session_state["active_scenario"] = load_choice
            st.experimental_rerun()

    # --- Pull values from session_state into local variables ---

    current_age = st.session_state.current_age
    retirement_age = st.session_state.retirement_age
    current_portfolio_for_projection = st.session_state.current_portfolio
    monthly_invest = st.session_state.monthly_invest
    annual_bonus_invest = st.session_state.annual_bonus
    target_monthly_spend = st.session_state.target_monthly_spend
    safe_withdrawal_rate = st.session_state.swr_pct / 100.0
    inflation_rate = (st.session_state.inflation_pct / 100.0) if st.session_state.use_inflation else 0.0
    early_retirement_age = st.session_state.early_ret_age
    mc_runs = int(st.session_state.mc_runs)
    simulate_crash = st.session_state.simulate_crash

    crash_year_offset = st.session_state.get("crash_year_offset") if simulate_crash else None
    crash_drawdown = (st.session_state.get("crash_drawdown_pct", 37.0) / 100.0) if simulate_crash else 0.0

    # --- Build scenarios ---

    base_params = ProjectionInput(
        name="Base",
        current_age=current_age,
        retirement_age=retirement_age,
        current_portfolio=current_portfolio_for_projection,
        monthly_invest=monthly_invest,
        annual_bonus_invest=annual_bonus_invest,
        one_time_injections=one_time_injections,
        annual_return=annual_return,
        target_monthly_spend=target_monthly_spend,
        safe_withdrawal_rate=safe_withdrawal_rate,
        inflation_rate=inflation_rate,
        crash_year_offset=crash_year_offset,
        crash_drawdown=crash_drawdown,
    )

    cons_params = ProjectionInput(
        name="Conservative",
        current_age=current_age,
        retirement_age=retirement_age,
        current_portfolio=current_portfolio_for_projection,
        monthly_invest=monthly_invest,
        annual_bonus_invest=annual_bonus_invest,
        one_time_injections=one_time_injections,
        annual_return=max(0.0, annual_return - 0.02),
        target_monthly_spend=target_monthly_spend,
        safe_withdrawal_rate=safe_withdrawal_rate,
        inflation_rate=inflation_rate,
        crash_year_offset=crash_year_offset,
        crash_drawdown=crash_drawdown,
    )

    aggr_params = ProjectionInput(
        name="Aggressive",
        current_age=current_age,
        retirement_age=retirement_age,
        current_portfolio=current_portfolio_for_projection,
        monthly_invest=monthly_invest,
        annual_bonus_invest=annual_bonus_invest,
        one_time_injections=one_time_injections,
        annual_return=annual_return + 0.02,
        target_monthly_spend=target_monthly_spend,
        safe_withdrawal_rate=safe_withdrawal_rate,
        inflation_rate=inflation_rate,
        crash_year_offset=crash_year_offset,
        crash_drawdown=crash_drawdown,
    )

    scenarios = [
        run_projection(base_params),
        run_projection(cons_params),
        run_projection(aggr_params),
    ]

    base_result = scenarios[0]

    # --- Layout ---

    col_main, col_side = st.columns([2.2, 0.8])

    # Sidebar results
    with col_side:
        st.subheader("🎯 FI Summary (Base)")

        st.metric(
            "FI Number Today (in today's €)",
            f"€{base_result.fi_number_today:,.0f}",
            help="Annual spending / Safe Withdrawal Rate",
        )

        if base_result.fi_age is not None:
            st.success(f"✅ FI reached at age **{base_result.fi_age}**")
        else:
            st.error("FI not reached before retirement age in base scenario.")

        if base_result.coast_fi_age is not None:
            st.info(
                f"🏖 Coast FI at age **{base_result.coast_fi_age}** "
                f"(you could stop investing then and still retire FI at {retirement_age})."
            )

        st.markdown("---")
        st.subheader("💶 Passive Income at Retirement (Base)")

        if base_result.years:
            final_portfolio = base_result.years[-1].portfolio_end
            passive_income = final_portfolio * safe_withdrawal_rate
            st.metric("Portfolio at Retirement", f"€{final_portfolio:,.0f}")
            st.metric("Annual Safe Income", f"€{passive_income:,.0f}")
            st.metric("Monthly Safe Income", f"€{passive_income / 12:,.0f}")

        st.markdown("---")
        st.subheader("🧓 Early Retirement Check")

        early_row = next((yr for yr in base_result.years if yr.age == early_retirement_age), None)
        if early_row:
            st.metric(
                f"Portfolio at age {early_retirement_age}",
                f"€{early_row.portfolio_end:,.0f}",
            )
            if early_row.portfolio_end >= early_row.fi_number_this_year:
                st.success("You are FI by this age ✅")
            else:
                st.warning("Not FI yet at this age ❌")

    # Main section
    with col_main:
        st.subheader("📊 Year-by-Year Projection (Base Scenario)")

        df_base = pd.DataFrame(
            {
                "Year Index": [y.year_index for y in base_result.years],
                "Age": [y.age for y in base_result.years],
                "Portfolio (€)": [y.portfolio_end for y in base_result.years],
                "FI Number (that year, €)": [y.fi_number_this_year for y in base_result.years],
            }
        )

        st.dataframe(
            df_base.style.format(
                {"Portfolio (€)": "€{:,.0f}", "FI Number (that year, €)": "€{:,.0f}"}
            ),
            use_container_width=True,
        )

        st.subheader("📈 Scenario Comparison")

        chart_data = {}
        ages_for_chart = [y.age for y in base_result.years]
        chart_data["Age"] = ages_for_chart

        for res in scenarios:
            chart_data[f"{res.params.name} Portfolio"] = [y.portfolio_end for y in res.years]

        chart_df = pd.DataFrame(chart_data).set_index("Age")
        st.line_chart(chart_df)

        st.markdown(
            "- **Base**: your chosen (or allocation-derived) return.\n"
            "- **Conservative**: return - 2 percentage points.\n"
            "- **Aggressive**: return + 2 percentage points.\n"
            "- Crash (if enabled) is applied in the chosen year **after** contributions."
        )

        if base_result.fi_age is not None:
            st.markdown(
                f"✅ In the **base scenario**, you reach **Financial Independence** at age "
                f"**{base_result.fi_age}**, when your portfolio first crosses the (inflation-adjusted) "
                f"FI number for that year."
            )
        else:
            st.markdown(
                "❌ In the **base scenario**, you do **not** reach FI before your chosen retirement age. "
                "Try increasing monthly investing, bonuses, or expected return, or lowering target spending."
            )

        # --- Monte Carlo section ---
        st.subheader("🎲 Monte Carlo Simulation (Base Scenario)")

        mc = monte_carlo_simulation(
            current_portfolio=current_portfolio_for_projection,
            monthly_invest=monthly_invest,
            annual_bonus=annual_bonus_invest,
            one_time_injections=one_time_injections,
            current_age=current_age,
            retirement_age=retirement_age,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            crash_year_offset=crash_year_offset,
            crash_drawdown=crash_drawdown,
            runs=mc_runs,
        )

        mc_df = pd.DataFrame(
            {
                "Age": mc["ages"],
                "Median": mc["median"],
                "10th Percentile": mc["p10"],
                "90th Percentile": mc["p90"],
            }
        ).set_index("Age")

        st.line_chart(mc_df)

        st.markdown(
            """
            - **Median** → typical path  
            - **10th percentile** → bad markets  
            - **90th percentile** → great markets  
            - If crash is enabled, **all paths** include that crash in the chosen year.
            """
        )

        # Probability of reaching FI by retirement
        st.subheader("📊 Probability of Reaching FI by Retirement")

        years_to_retirement = retirement_age - current_age
        fi_number_at_retirement = base_result.fi_number_today * ((1 + inflation_rate) ** years_to_retirement)

        final_values = mc["all_paths"][:, -1]  # portfolio at retirement for each simulation
        prob_fi = (final_values >= fi_number_at_retirement).mean() * 100.0

        st.metric("Probability of Reaching FI by Retirement", f"{prob_fi:.1f}%")

    st.markdown("---")
    st.caption("Built for Klaas' FI obsession 🧮. Portfolio is remembered in your browser, performance charted from real prices.")


if __name__ == "__main__":
    main()
