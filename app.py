import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import norm

st.set_page_config(page_title="Investment Portfolio Analysis Toolkit", layout="wide")

# =============================================================================
# CORE FINANCIAL CALCULATIONS
# =============================================================================


def annualized_return(prices: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate the annualized return of a price series."""
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    num_years = len(prices) / periods_per_year
    annualized = (1 + total_return) ** (1 / num_years) - 1
    return annualized

def price_to_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily returns for a given price series."""
    returns = prices.pct_change().dropna()
    return returns

def daily_price_to_monthly_price(prices: pd.Series) -> pd.Series:
    """Convert daily prices to monthly prices by resampling."""
    monthly_prices = prices.resample("ME").last()
    return monthly_prices

def daily_price_to_monthly_returns(prices: pd.Series) -> pd.Series:
    """Calculate monthly returns for a given price series."""
    monthly_prices = daily_price_to_monthly_price(prices)
    monthly_returns = monthly_prices.pct_change().dropna()
    return monthly_returns

def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate the annualized volatility of a series of returns."""
    return returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(
    returns: pd.Series, annualized_risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Calculate the Sharpe Ratio of a series of returns."""
    excess_returns = returns - annualized_risk_free_rate / periods_per_year
    # Annualize the excess returns mean and divide by annualized volatility
    annualized_excess_return = excess_returns.mean() * periods_per_year
    annualized_vol = annualized_volatility(excess_returns, periods_per_year)
    return annualized_excess_return / annualized_vol

def wealth_index(returns: pd.Series) -> pd.Series:
    """
    Calculate the wealth index from a series of returns.

    Args:
        returns (pd.Series): A series of periodic returns.

    Returns:
        pd.Series: A series representing the cumulative wealth index.
    """
    return (1 + returns).cumprod()

def drawdown(returns: pd.Series) -> pd.DataFrame:
    """
    Calculate the drawdown of a series of returns.

    Args:
        returns (pd.Series): A series of periodic returns.

    Returns:
        pd.DataFrame: A DataFrame with columns for:
                      - 'Drawdown': The percentage drawdown from the previous peak.
                      - 'Wealth Index': The cumulative product of returns (representing wealth growth).
                      - 'Peak': The historical high point of the Wealth Index.
                      - 'Max Drawdown': The maximum drawdown percentage experienced (a single value).
    """
    wi = wealth_index(returns)
    peak = wi.cummax()
    drawdown_series = (peak - wi) / peak
    drawdown_df = pd.DataFrame(
        {
            "Drawdown": drawdown_series,
            "Wealth Index": wi,
            "Peak": peak,
        }
    )
    drawdown_df['Max Drawdown'] = drawdown_series.max() # This will broadcast the single max value to all rows
    return drawdown_df

def skewness(returns: pd.Series) -> float:
    """Calculate the skewness of a series of returns."""
    return returns.skew()

def kurtosis(returns: pd.Series) -> float:
    """Calculate the kurtosis of a series of returns."""
    return returns.kurtosis()

def cornish_fisher_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Calculate the Cornish-Fisher Value at Risk (VaR) for a series of returns."""
    # Calculate statistics
    mean_return = returns.mean()
    std_return = returns.std()
    skew = returns.skew()
    excess_kurtosis = returns.kurtosis()  # pandas kurtosis() already returns excess kurtosis
    z = norm.ppf(alpha)
    z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * excess_kurtosis / 24
    var = mean_return + z_cf * std_return
    return float(var)

# =============================================================================
# DATA ACQUISITION
# =============================================================================


@st.cache_data()
def download_data(ticker, start_date, end_date):
    """Download historical stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================


def create_interactive_plot(x_data, y_data, title="Chart", x_label="X", y_label="Y"):
    """Create a simple interactive line plot with hover functionality."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="lines",
            hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, height=400)
    return fig


# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

st.title("Investment Portfolio Analysis Toolkit")
start_date = st.date_input(
    "Start Date:",
    value=pd.to_datetime("today") - pd.Timedelta(days=365 * 10),
    max_value=pd.to_datetime("today"),
    min_value=pd.to_datetime("1998-01-01"),
    help="Select the start date for historical data. Default is 10 years ago.",
)
end_date = st.date_input(
    "End Date:",
    value=pd.to_datetime("today"),
    max_value=pd.to_datetime("today"),
    min_value=pd.to_datetime("1998-01-01"),
    help="Select the end date for historical data. Default is today.",
)

tabs = st.tabs(["Analyze stock", "Compare stocks", "Portfolio optimization"])

# =============================================================================
# TAB 1: SINGLE STOCK ANALYSIS
# =============================================================================

with tabs[0]:
    cols = st.columns(3)
    with cols[0]:
        ticker = st.text_input("Select stock or ETF:", "GOOG")
    if st.button("Download Data"):
        data = download_data(ticker, start_date, end_date)
        st.success("Data downloaded successfully!")

        # Calculate key metrics
        close_prices = data["Close"][ticker]
        daily_returns = price_to_returns(close_prices)
        monthly_returns = daily_price_to_monthly_returns(close_prices)

        ann_return = annualized_return(close_prices) * 100
        volatility = annualized_volatility(daily_returns, periods_per_year=252) * 100
        sharpe = sharpe_ratio(daily_returns, annualized_risk_free_rate=0.0, periods_per_year=252)
        max_dd = drawdown(daily_returns)["Max Drawdown"].iloc[0] * 100
        max_dd_date = drawdown(daily_returns)["Drawdown"].idxmax()

        wi = wealth_index(monthly_returns)

        # Show stats in prominent metric boxes
        st.subheader("📊 Key Performance Metrics")

        metric_cols = st.columns(5)
        with metric_cols[0]:
            st.metric(
                label="Annualized Return", value=f"{ann_return:+.2f}%", delta=None
            )

        with metric_cols[1]:
            st.metric(
                label="Volatility (Annual)", value=f"{volatility:.2f}%", delta=None
            )

        with metric_cols[2]:
            st.metric(label="Sharpe Ratio", value=f"{sharpe:.2f}", delta=None)

        with metric_cols[3]:
            st.metric(
                label="Max Drawdown",
                value=f"-{max_dd:.2f}%",
                delta=f"on {max_dd_date.strftime('%Y-%m-%d')}" if max_dd_date else None,
                help="Maximum drawdown is the largest observed loss from a peak to a trough of a portfolio, before a new peak is achieved.",
            )

        with metric_cols[4]:
            # CFVaR
            st.metric(
                label="CFVaR",
                value=f"{100 * cornish_fisher_var(monthly_returns, alpha=0.05):.2f}%",
                delta=None,
                help="Cornish-Fisher Value at Risk (VaR) is a risk measure that accounts for skewness and kurtosis in the return distribution.",
            )

        # Charts section
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Stock Price Chart")
            fig = create_interactive_plot(
                x_data=data.index,
                y_data=data["Close"][ticker],
                title=f"{ticker} Stock Price",
                x_label="Date",
                y_label="Price ($)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with cols[1]:
            st.subheader("Monthly Returns")
            monthly_returns = daily_price_to_monthly_returns(data["Close"][ticker])
            mean_return = monthly_returns.mean()
            std_dev = monthly_returns.std()
            # Plot mean and standard deviation of monthly returns as well
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=monthly_returns.index,
                    y=monthly_returns,
                    name="Monthly Returns",
                    hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=monthly_returns.index,
                    y=[mean_return] * len(monthly_returns),
                    mode="lines",
                    name="Mean Return",
                    line=dict(dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=monthly_returns.index,
                    y=[mean_return + std_dev] * len(monthly_returns),
                    mode="lines",
                    name="Mean + Std Dev",
                    line=dict(dash="dash", color="red"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=monthly_returns.index,
                    y=[mean_return - std_dev] * len(monthly_returns),
                    mode="lines",
                    name="Mean - Std Dev",
                    line=dict(dash="dash", color="red"),
                )
            )
            fig.update_layout(
                title=f"{ticker} Monthly Returns",
                xaxis_title="Date",
                yaxis_title="Return (%)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with cols[0]:
            st.subheader("Drawdown (monthly returns)")
            drawdowns = drawdown(monthly_returns)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=drawdowns.index,
                    y=drawdowns["Drawdown"],
                    mode="lines",
                    name="Drawdown",
                    line=dict(color="red"),
                )
            )
            fig.update_layout(
                title=f"{ticker} Drawdown",
                xaxis_title="Date",
                yaxis_title="Value",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True, height=400)

        with cols[1]:
            st.subheader("Wealth Index (Cumulative Returns)")
            wi = wealth_index(monthly_returns)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=wi.index,
                    y=wi,
                    mode="lines",
                    name="Wealth Index",
                    hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
                )
            )
            fig.update_layout(
                title=f"{ticker} Wealth Index",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2: STOCK COMPARISON
# =============================================================================

with tabs[1]:
    cols = st.columns(2)
    with cols[0]:
        stock_A = st.text_input(
            label="Stock A Ticker:",
            value="SPY",
            help="Enter the ticker symbol for Stock A (e.g., SPY for S&P 500 ETF).",
        )
    with cols[1]:
        stock_B = st.text_input(
            label="Stock B Ticker:",
            value="VWO",
            help="Enter the ticker symbol for Stock B (e.g., VWO for Vanguard FTSE Emerging Markets ETF).",
        )
    compare_stocks_button = st.button(
        "Compare Stocks",
        help="Click to compare the performance of Stock A and Stock B over the selected date range.",
    )
    if compare_stocks_button:
        data_A = download_data(stock_A, start_date, end_date)
        data_B = download_data(stock_B, start_date, end_date)
        
        # Calculate returns
        returns_A = price_to_returns(data_A["Close"][stock_A])
        returns_B = price_to_returns(data_B["Close"][stock_B])

        # Calculate metric values
        ann_return_A = annualized_return(data_A["Close"][stock_A]) * 100
        ann_return_B = annualized_return(data_B["Close"][stock_B]) * 100
        volatility_A = returns_A.std() * np.sqrt(252) * 100
        volatility_B = returns_B.std() * np.sqrt(252) * 100
        sharpe_A = sharpe_ratio(returns_A)
        sharpe_B = sharpe_ratio(returns_B)
        max_dd_A = drawdown(returns_A)["Max Drawdown"].iloc[0] * 100
        max_dd_B = drawdown(returns_B)["Max Drawdown"].iloc[0] * 100
        max_dd_date_A = drawdown(returns_A)["Drawdown"].idxmax()
        max_dd_date_B = drawdown(returns_B)["Drawdown"].idxmax()

        cumulative_returns_A = (
            1 + daily_price_to_monthly_returns(data_A["Close"][stock_A])
        ).cumprod() * 100
        cumulative_returns_B = (
            1 + daily_price_to_monthly_returns(data_B["Close"][stock_B])
        ).cumprod() * 100

        cols = st.columns(2)
        with cols[0]:
            st.metric(
                label=f"{stock_A} Annualized Return",
                value=f"{ann_return_A:+.2f}%",
                delta=None,
            )

            st.metric(
                label=f"{stock_A} Volatility (Annual)",
                value=f"{volatility_A:.2f}%",
                delta=None,
            )

            st.metric(
                label=f"{stock_A} Sharpe Ratio", value=f"{sharpe_A:.2f}", delta=None
            )

        with cols[1]:
            st.metric(
                label=f"{stock_B} Annualized Return",
                value=f"{ann_return_B:+.2f}%",
                delta=None,
            )

            st.metric(
                label=f"{stock_B} Volatility (Annual)",
                value=f"{volatility_B:.2f}%",
                delta=None,
            )

            st.metric(
                label=f"{stock_B} Sharpe Ratio", value=f"{sharpe_B:.2f}", delta=None
            )

        with cols[0]:
            fig_A = create_interactive_plot(
                x_data=data_A.index,
                y_data=data_A["Close"][stock_A],
                title=f"{stock_A} Stock Price",
                x_label="Date",
                y_label="Price ($)",
            )
            st.subheader(f"{stock_A} Stock Price Chart")
            st.plotly_chart(fig_A, use_container_width=True)

        with cols[1]:
            fig_B = create_interactive_plot(
                x_data=data_B.index,
                y_data=data_B["Close"][stock_B],
                title=f"{stock_B} Stock Price",
                x_label="Date",
                y_label="Price ($)",
            )
            st.subheader(f"{stock_B} Stock Price Chart")
            st.plotly_chart(fig_B, use_container_width=True)

        with cols[0]:
            st.subheader(f"{stock_A} Monthly Returns")
            monthly_returns_A = daily_price_to_monthly_returns(data_A["Close"][stock_A])
            fig_A_returns = go.Figure()
            fig_A_returns.add_trace(
                go.Bar(
                    x=monthly_returns_A.index,
                    y=monthly_returns_A,
                    name="Monthly Returns",
                    hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
                )
            )
            fig_A_returns.update_layout(
                title=f"{stock_A} Monthly Returns",
                xaxis_title="Date",
                yaxis_title="Return (%)",
                height=400,
            )
            st.plotly_chart(fig_A_returns, use_container_width=True)

        with cols[1]:
            st.subheader(f"{stock_B} Monthly Returns")
            monthly_returns_B = daily_price_to_monthly_returns(data_B["Close"][stock_B])
            fig_B_returns = go.Figure()
            fig_B_returns.add_trace(
                go.Bar(
                    x=monthly_returns_B.index,
                    y=monthly_returns_B,
                    name="Monthly Returns",
                    hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
                )
            )
            fig_B_returns.update_layout(
                title=f"{stock_B} Monthly Returns",
                xaxis_title="Date",
                yaxis_title="Return (%)",
                height=400,
            )
            st.plotly_chart(fig_B_returns, use_container_width=True)
        with cols[0]:
            st.subheader(f"{stock_A} Drawdown (monthly returns)")
            drawdowns_A = drawdown(monthly_returns_A)
            fig_A_drawdown = go.Figure()
            fig_A_drawdown.add_trace(
                go.Scatter(
                    x=drawdowns_A.index,
                    y=drawdowns_A["Drawdown"],
                    mode="lines",
                    name="Drawdown",
                    line=dict(color="red"),
                )
            )
            fig_A_drawdown.update_layout(
                title=f"{stock_A} Drawdown",
                xaxis_title="Date",
                yaxis_title="Value",
                height=400,
            )
            st.plotly_chart(fig_A_drawdown, use_container_width=True)

        with cols[1]:
            st.subheader(f"{stock_B} Drawdown (monthly returns)")
            drawdowns_B = drawdown(monthly_returns_B)
            fig_B_drawdown = go.Figure()
            fig_B_drawdown.add_trace(
                go.Scatter(
                    x=drawdowns_B.index,
                    y=drawdowns_B["Drawdown"],
                    mode="lines",
                    name="Drawdown",
                    line=dict(color="red"),
                )
            )
            fig_B_drawdown.update_layout(
                title=f"{stock_B} Drawdown",
                xaxis_title="Date",
                yaxis_title="Value",
                height=400,
            )
            st.plotly_chart(fig_B_drawdown, use_container_width=True)

        with cols[0]:
            st.subheader(f"{stock_A} Cumulative Returns")
            cumulative_returns_A = (1 + monthly_returns_A).cumprod() * 100
            fig_A_cumulative = go.Figure()
            fig_A_cumulative.add_trace(
                go.Scatter(
                    x=cumulative_returns_A.index,
                    y=cumulative_returns_A,
                    mode="lines",
                    name="Cumulative Returns",
                    hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
                )
            )
            fig_A_cumulative.update_layout(
                title=f"{stock_A} Cumulative Returns",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=400,
            )
            st.plotly_chart(fig_A_cumulative, use_container_width=True)

        with cols[1]:
            st.subheader(f"{stock_B} Cumulative Returns")
            cumulative_returns_B = (1 + monthly_returns_B).cumprod() * 100
            fig_B_cumulative = go.Figure()
            fig_B_cumulative.add_trace(
                go.Scatter(
                    x=cumulative_returns_B.index,
                    y=cumulative_returns_B,
                    mode="lines",
                    name="Cumulative Returns",
                    hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
                )
            )
            fig_B_cumulative.update_layout(
                title=f"{stock_B} Cumulative Returns",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=400,
            )
            st.plotly_chart(fig_B_cumulative, use_container_width=True)

with tabs[2]:
    st.subheader("Portfolio Optimization")
    st.write(
        "This section will allow you to optimize a portfolio of multiple assets based on historical data."
    )
    st.write(
        "Currently, this feature is under development. Please check back later for updates."
    )