import math
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import streamlit as st


# ---------- Data Models ----------

@dataclass
class OneTimeInjection:
    year_offset: int  # 1 = end of first year, 2 = end of second year, etc.
    amount: float


@dataclass
class ProjectionInput:
    current_age: int
    retirement_age: int
    current_portfolio: float
    monthly_invest: float
    annual_bonus_invest: float
    one_time_injections: List[OneTimeInjection]
    annual_return: float  # e.g. 0.065 for 6.5%
    target_monthly_spend: float  # for FI calculation
    safe_withdrawal_rate: float = 0.04  # 4% rule by default


@dataclass
class YearResult:
    age: int
    year_index: int
    portfolio_end: float


@dataclass
class ProjectionResult:
    fi_number: float
    fi_age: Optional[int]
    fi_year_index: Optional[int]
    years: List[YearResult]


# ---------- Core Logic ----------

def run_projection(params: ProjectionInput) -> ProjectionResult:
    """
    Runs a yearly projection with monthly contributions, annual bonus,
    and optional one-time injections.
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

    for year_idx in range(1, num_years + 1):
        # 12 months of growth + monthly investing
        for _ in range(12):
            current_portfolio = current_portfolio * (1 + monthly_rate) + params.monthly_invest

        # Add annual bonus at year end
        current_portfolio += params.annual_bonus_invest

        # Add any one-time injections scheduled at this year
        if year_idx in injections_by_year:
            current_portfolio += injections_by_year[year_idx]

        age = params.current_age + year_idx
        years.append(YearResult(age=age, year_index=year_idx, portfolio_end=current_portfolio))

    # FI calculations
    annual_spend = params.target_monthly_spend * 12
    fi_number = annual_spend / params.safe_withdrawal_rate

    fi_age: Optional[int] = None
    fi_year_index: Optional[int] = None

    for yr in years:
        if yr.portfolio_end >= fi_number:
            fi_age = yr.age
            fi_year_index = yr.year_index
            break

    return ProjectionResult(
        fi_number=fi_number,
        fi_age=fi_age,
        fi_year_index=fi_year_index,
        years=years
    )


# ---------- Streamlit UI ----------

def main():
    st.set_page_config(page_title="FI & Portfolio Projection", layout="wide")
    st.title("📈 Financial Independence & Portfolio Projection Tool")

    st.markdown(
        """
        This tool projects your portfolio year by year, calculates your **FI number**  
        and estimates the age at which you reach **Financial Independence (FI)**  
        based on your inputs (monthly investing, bonus, returns, etc.).
        """
    )

    # --- Inputs ---

    st.sidebar.header("🔧 Inputs")

    col1, col2 = st.sidebar.columns(2)
    current_age = col1.number_input("Current Age", min_value=16, max_value=80, value=27, step=1)
    retirement_age = col2.number_input("Retirement Age", min_value=current_age + 1, max_value=80, value=67, step=1)

    current_portfolio = st.sidebar.number_input(
        "Current Portfolio (€)", min_value=0.0, step=1000.0, value=44327.73
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
    st.sidebar.subheader("🎁 One-Time Extra Injection")

    extra_5k_enabled = st.sidebar.checkbox("Include one-time €5k next year", value=True)
    one_time_injections: List[OneTimeInjection] = []
    if extra_5k_enabled:
        # year_offset = 1 means: at the end of the first projected year
        one_time_injections.append(OneTimeInjection(year_offset=1, amount=5000.0))

    # Advanced: custom injection
    with st.sidebar.expander("➕ Add Custom Injection"):
        custom_amount = st.number_input("Custom Injection Amount (€)", min_value=0.0, step=500.0, value=0.0)
        custom_year_offset = st.number_input(
            "Year Offset (1 = end of this year)", min_value=1, max_value=max(1, retirement_age - current_age),
            value=2, step=1
        )
        if custom_amount > 0:
            one_time_injections.append(OneTimeInjection(year_offset=int(custom_year_offset), amount=custom_amount))

    # --- Run Projection ---

    params = ProjectionInput(
        current_age=current_age,
        retirement_age=retirement_age,
        current_portfolio=current_portfolio,
        monthly_invest=monthly_invest,
        annual_bonus_invest=annual_bonus_invest,
        one_time_injections=one_time_injections,
        annual_return=annual_return,
        target_monthly_spend=target_monthly_spend,
        safe_withdrawal_rate=safe_withdrawal_rate
    )

    result = run_projection(params)

    # --- Outputs ---

    col_main, col_side = st.columns([2, 1])

    with col_side:
        st.subheader("🎯 FI Summary")

        st.metric("FI Number (Portfolio Needed)",
                  f"€{result.fi_number:,.0f}",
                  help="Annual spending / Safe Withdrawal Rate")

        if result.fi_age is not None:
            st.success(f"✅ FI reached at age **{result.fi_age}**")
        else:
            st.error("FI not reached before retirement age with current inputs.")

        st.markdown("---")
        st.subheader("💶 Passive Income at Retirement")

        if result.years:
            final_portfolio = result.years[-1].portfolio_end
            passive_income = final_portfolio * safe_withdrawal_rate
            st.metric("Portfolio at Retirement", f"€{final_portfolio:,.0f}")
            st.metric("Annual Safe Income", f"€{passive_income:,.0f}")
            st.metric("Monthly Safe Income", f"€{passive_income / 12:,.0f}")

    with col_main:
        st.subheader("📊 Year-by-Year Projection")

        df = pd.DataFrame(
            {
                "Year Index": [y.year_index for y in result.years],
                "Age": [y.age for y in result.years],
                "Portfolio (€)": [y.portfolio_end for y in result.years],
            }
        )

        st.dataframe(df.style.format({"Portfolio (€)": "€{:,.0f}"}), use_container_width=True)

        st.subheader("📈 Growth Chart")
        st.line_chart(df.set_index("Age")["Portfolio (€)"])

        if result.fi_age is not None:
            st.markdown(
                f"✅ You reach **Financial Independence** at age **{result.fi_age}** "
                f"when your portfolio first crosses **€{result.fi_number:,.0f}**."
            )
        else:
            st.markdown(
                "❌ With the current parameters, you do **not** reach FI before retirement age. "
                "Try increasing monthly investing, bonuses, or expected return."
            )

    st.markdown("---")
    st.caption("Built for Klaas' FI obsession. Change inputs on the left and watch your future move.")


if __name__ == "__main__":
    main()
