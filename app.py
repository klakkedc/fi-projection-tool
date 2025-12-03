import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st


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
    optional one-time injections, and (optionally) inflation-adjusted FI.
    """
    monthly_rate = (1 + params.annual_return) ** (1 / 12) - 1
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
        # 12 months of growth + monthly investing
        for _ in range(12):
            current_portfolio = current_portfolio * (1 + monthly_rate) + params.monthly_invest

        # Add annual bonus at year end
        current_portfolio += params.annual_bonus_invest

        # Add any one-time injections scheduled at this year
        if year_idx in injections_by_year:
            current_portfolio += injections_by_year[year_idx]

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
    # FI number at retirement in nominal terms
    fi_number_at_retirement = fi_number_today * ((1 + params.inflation_rate) ** num_years)

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
    runs: int = 1000,
):
    """
    Monte Carlo simulation of future portfolio values.
    annual_return and annual_volatility are decimals (0.065, 0.15).
    Returns:
        {
            "ages": [age1, age2, ...],
            "median": np.array,
            "p10": np.array,
            "p90": np.array,
            "all_paths": np.array shape (runs, years)
        }
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


# ---------- Streamlit UI ----------

def main():
    st.set_page_config(page_title="FI & Portfolio Projection", layout="wide")
    st.title("📈 Financial Independence & Portfolio Projection Tool")

    st.markdown(
        """
        This tool projects your portfolio year by year, calculates your **FI number**,  
        estimates your **FI age**, **Coast FI age**, and runs a **Monte Carlo simulation**  
        to show how market randomness affects your future.
        """
    )

    # --- Inputs ---

    st.sidebar.header("🔧 Inputs")

    col1, col2 = st.sidebar.columns(2)
    current_age = col1.number_input("Current Age", min_value=16, max_value=80, value=27, step=1)
    retirement_age = col2.number_input("Retirement Age", min_value=current_age + 1, max_value=80, value=67, step=1)

    current_portfolio = st.sidebar.number_input(
        "Current Portfolio (€)", min_value=0.0, step=1000.0, value=40000.00
    )

    monthly_invest = st.sidebar.number_input(
        "Monthly Investment (€)", min_value=0.0, step=50.0, value=1000.0
    )

    annual_bonus_invest = st.sidebar.number_input(
        "Annual Bonus Invested (€)", min_value=0.0, step=500.0, value=5000.0
    )

    st.sidebar.markdown("---")

    annual_return_pct = st.sidebar.number_input(
        "Expected Annual Return (%)", min_value=0.0, max_value=20.0, value=6.5, step=0.1
    )
    annual_return = annual_return_pct / 100.0

    target_monthly_spend = st.sidebar.number_input(
        "Target Monthly Spend at FI (€)", min_value=500.0, step=100.0, value=3000.0
    )

    safe_withdrawal_rate_pct = st.sidebar.number_input(
        "Safe Withdrawal Rate (%)", min_value=2.0, max_value=6.0, value=4.0, step=0.25
    )
    safe_withdrawal_rate = safe_withdrawal_rate_pct / 100.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧯 Inflation")

    use_inflation = st.sidebar.checkbox("Adjust FI number for inflation", value=True)
    inflation_rate_pct = st.sidebar.number_input(
        "Inflation Rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.25
    )
    inflation_rate = (inflation_rate_pct / 100.0) if use_inflation else 0.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎁 One-Time Extra Injection")

    extra_5k_enabled = st.sidebar.checkbox("Include one-time €5k next year", value=True)
    one_time_injections: List[OneTimeInjection] = []
    if extra_5k_enabled:
        # year_offset = 1 means: at the end of the first projected year
        one_time_injections.append(OneTimeInjection(year_offset=1, amount=5000.0))

    with st.sidebar.expander("➕ Add Custom Injection"):
        custom_amount = st.number_input("Custom Injection Amount (€)", min_value=0.0, step=500.0, value=0.0)
        custom_year_offset = st.number_input(
            "Year Offset (1 = end of first projected year)",
            min_value=1,
            max_value=max(1, retirement_age - current_age),
            value=2,
            step=1,
        )
        if custom_amount > 0:
            one_time_injections.append(
                OneTimeInjection(year_offset=int(custom_year_offset), amount=custom_amount)
            )

    # Early retirement slider
    early_retirement_age = st.sidebar.slider(
        "Early Retirement Age (for analysis)", min_value=current_age + 1, max_value=retirement_age, value=47
    )

    # Monte Carlo controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎲 Monte Carlo Settings")
    mc_runs = st.sidebar.number_input("Number of simulations", 100, 5000, 1000, step=100)
    volatility_pct = st.sidebar.number_input("Annual Volatility (%)", 5.0, 40.0, 15.0, step=0.5)
    annual_volatility = volatility_pct / 100.0

    # --- Build scenarios ---

    # Base scenario
    base_params = ProjectionInput(
        name="Base",
        current_age=current_age,
        retirement_age=retirement_age,
        current_portfolio=current_portfolio,
        monthly_invest=monthly_invest,
        annual_bonus_invest=annual_bonus_invest,
        one_time_injections=one_time_injections,
        annual_return=annual_return,
        target_monthly_spend=target_monthly_spend,
        safe_withdrawal_rate=safe_withdrawal_rate,
        inflation_rate=inflation_rate,
    )

    # Conservative scenario: lower return
    cons_params = ProjectionInput(
        name="Conservative",
        current_age=current_age,
        retirement_age=retirement_age,
        current_portfolio=current_portfolio,
        monthly_invest=monthly_invest,
        annual_bonus_invest=annual_bonus_invest,
        one_time_injections=one_time_injections,
        annual_return=max(0.0, annual_return - 0.02),
        target_monthly_spend=target_monthly_spend,
        safe_withdrawal_rate=safe_withdrawal_rate,
        inflation_rate=inflation_rate,
    )

    # Aggressive scenario: higher return
    aggr_params = ProjectionInput(
        name="Aggressive",
        current_age=current_age,
        retirement_age=retirement_age,
        current_portfolio=current_portfolio,
        monthly_invest=monthly_invest,
        annual_bonus_invest=annual_bonus_invest,
        one_time_injections=one_time_injections,
        annual_return=annual_return + 0.02,
        target_monthly_spend=target_monthly_spend,
        safe_withdrawal_rate=safe_withdrawal_rate,
        inflation_rate=inflation_rate,
    )

    scenarios = [
        run_projection(base_params),
        run_projection(cons_params),
        run_projection(aggr_params),
    ]

    base_result = scenarios[0]

    # --- Outputs ---

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
            "- **Base**: your chosen return.\n"
            "- **Conservative**: return - 2 percentage points.\n"
            "- **Aggressive**: return + 2 percentage points.\n"
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
            current_portfolio=current_portfolio,
            monthly_invest=monthly_invest,
            annual_bonus=annual_bonus_invest,
            one_time_injections=one_time_injections,
            current_age=current_age,
            retirement_age=retirement_age,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
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
            """
        )

        # Probability of reaching FI by retirement
        st.subheader("📊 Probability of Reaching FI by Retirement")

        # FI number at retirement (inflation adjusted if enabled)
        years_to_retirement = retirement_age - current_age
        fi_number_at_retirement = base_result.fi_number_today * ((1 + inflation_rate) ** years_to_retirement)

        final_values = mc["all_paths"][:, -1]  # portfolio at retirement for each simulation
        prob_fi = (final_values >= fi_number_at_retirement).mean() * 100.0

        st.metric("Probability of Reaching FI by Retirement", f"{prob_fi:.1f}%")

    st.markdown("---")
    st.caption("Built for Klaas' FI obsession 🧮. Change inputs on the left and watch your future move.")


if __name__ == "__main__":
    main()
