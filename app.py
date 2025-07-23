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


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily returns for a given price series."""
    returns = prices.pct_change().dropna()
    return returns


def calculate_monthly_returns(prices: pd.Series) -> pd.Series:
    """Calculate monthly returns for a given price series."""
    monthly_prices = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()
    return monthly_returns


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate the annualized volatility of a series of returns."""
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Calculate the Sharpe Ratio of a series of returns."""
    excess_returns = returns - risk_free_rate / periods_per_year
    return excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)


def drawdown(returns: pd.Series) -> pd.DataFrame:
    """Calculate the drawdown of a series of returns. Returns a DataFrame with columns for drawdown, cumulative returns, and peak."""
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown_series = (peak - cumulative_returns) / peak
    return pd.DataFrame(
        {
            "Drawdown": drawdown_series,
            "Cumulative Returns": cumulative_returns,
            "Peak": peak,
            "Max Drawdown": drawdown_series.max(),
        }
    )


# =============================================================================
# DATA ACQUISITION
# =============================================================================


@st.cache_data()
def download_data(ticker, start_date, end_date):
    """Download historical stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


# =============================================================================
# PORTFOLIO OPTIMIZATION
# =============================================================================
def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.02):
    """Find the tangency portfolio (maximum Sharpe ratio)."""
    n_assets = len(mean_returns)
    
    def negative_sharpe(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_vol
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = n_assets * [1. / n_assets]
    
    result = minimize(negative_sharpe, initial_guess, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x

def generate_efficient_frontier(mean_returns, cov_matrix, num_portfolios=5000):
    """Generate efficient frontier using Monte Carlo simulation."""
    n_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_vol
        results[2, i] = portfolio_return / portfolio_vol  # Sharpe ratio
    
    return results, weights_record

def calculate_cml(tangency_return, tangency_vol, risk_free_rate=0.02, max_vol=None):
    """Calculate Capital Market Line."""
    if max_vol is None:
        max_vol = tangency_vol * 2
    
    cml_vol = np.linspace(0, max_vol, 100)
    cml_return = risk_free_rate + (tangency_return - risk_free_rate) / tangency_vol * cml_vol
    
    return cml_vol, cml_return

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

tabs = st.tabs(["Analyze stock", "Compare stocks", "Optimize portfolio"])

# =============================================================================
# TAB 1: SINGLE STOCK ANALYSIS
# =============================================================================

with tabs[0]:
    with st.sidebar:
        st.header("Settings")
        st.write("Configure your analysis parameters here.")
        ticker = st.text_input("Select stock or ETF:", "GOOG")
        # Set start date to 10 years ago
        start_date = st.date_input(
            "Start Date:",
            value=pd.to_datetime("today") - pd.Timedelta(days=10 * 365),
            max_value=pd.to_datetime("today"),
            min_value=pd.to_datetime("1998-01-01"),
        )
        end_date = st.date_input(
            "End Date:",
            value=pd.to_datetime("today"),
            max_value=pd.to_datetime("today"),
            min_value=pd.to_datetime("1998-01-01"),
        )

    if st.button("Download Data"):
        data = download_data(ticker, start_date, end_date)
        st.success("Data downloaded successfully!")

        # Calculate key metrics
        close_prices = data["Close"][ticker]
        returns = calculate_returns(close_prices)
        monthly_returns = calculate_monthly_returns(close_prices)

        ann_return = annualized_return(close_prices) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe = sharpe_ratio(returns)
        max_dd = drawdown(returns)["Max Drawdown"].iloc[0] * 100
        max_dd_date = drawdown(returns)["Drawdown"].idxmax()

        cumulative_returns = (1 + monthly_returns).cumprod() * 100

        # Show stats in prominent metric boxes
        st.subheader("📊 Key Performance Metrics")

        metric_cols = st.columns(4)
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

        st.write(
            f"\$100 invested in {ticker} on {start_date.strftime('%Y-%m-%d')} would be worth ${cumulative_returns.iloc[-1]:.2f} on {end_date.strftime('%Y-%m-%d')}."
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
            monthly_returns = calculate_monthly_returns(data["Close"][ticker])
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
            st.subheader("Cumulative Returns")
            cumulative_returns = (1 + monthly_returns).cumprod() * 100
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns,
                    mode="lines",
                    name="Cumulative Returns",
                    hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
                )
            )
            fig.update_layout(
                title=f"{ticker} Cumulative Returns",
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

    if st.button("Compare Stocks"):
        data_A = download_data(stock_A, start_date, end_date)
        data_B = download_data(stock_B, start_date, end_date)
        st.success("Data downloaded successfully!")

        # Calculate returns
        returns_A = calculate_returns(data_A["Close"][stock_A])
        returns_B = calculate_returns(data_B["Close"][stock_B])

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
            1 + calculate_monthly_returns(data_A["Close"][stock_A])
        ).cumprod() * 100
        cumulative_returns_B = (
            1 + calculate_monthly_returns(data_B["Close"][stock_B])
        ).cumprod() * 100

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
            monthly_returns_A = calculate_monthly_returns(data_A["Close"][stock_A])
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
            monthly_returns_B = calculate_monthly_returns(data_B["Close"][stock_B])
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

# =============================================================================
# TAB 3: PORTFOLIO OPTIMIZATION
# =============================================================================

with tabs[2]:
    st.header("Portfolio Optimization: Efficient Frontier & CML")
    
    # Stock selection
    stocks = st.multiselect(
        "Select two stocks for Efficient Frontier Analysis",
        ['VOO', 'GLD', 'VWO', 'VEA', 'AGG'],
        default=['VOO', 'GLD']
    )
    
    # Risk-free rate input
    risk_free_rate = st.slider("Risk-free rate (%)", 0.0, 5.0, 2.0) / 100
    
    if len(stocks) != 2:
        st.warning("Please select exactly two stocks to proceed.")
    else:
        if st.button("Generate Efficient Frontier"):
            try:
                # Download data
                end_date = pd.Timestamp.now()
                start_date = end_date - pd.DateOffset(years=5)
                
                with st.spinner("Downloading data and calculating..."):
                    data = yf.download(stocks, start=start_date, end=end_date)['Close']
                    
                    # Calculate returns and statistics
                    returns = data.pct_change().dropna()
                    mean_returns = returns.mean() * 252  # Annualized
                    cov_matrix = returns.cov() * 252     # Annualized
                    
                    # Generate efficient frontier
                    results, weights_record = generate_efficient_frontier(mean_returns, cov_matrix)
                    
                    # Find tangency portfolio
                    tangency_weights = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)
                    tangency_return = np.sum(mean_returns * tangency_weights)
                    tangency_vol = np.sqrt(np.dot(tangency_weights.T, np.dot(cov_matrix, tangency_weights)))
                    tangency_sharpe = (tangency_return - risk_free_rate) / tangency_vol
                    
                    # Calculate CML
                    cml_vol, cml_return = calculate_cml(tangency_return, tangency_vol, risk_free_rate, 
                                                       max_vol=results[1, :].max())
                
                # Create the plot
                fig = go.Figure()
                
                # Efficient frontier scatter plot
                fig.add_trace(go.Scatter(
                    x=results[1, :],
                    y=results[0, :],
                    mode='markers',
                    marker=dict(
                        color=results[2, :],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Sharpe Ratio'),
                        size=4,
                        opacity=0.6
                    ),
                    name='Portfolios',
                    hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                ))
                
                # Tangency portfolio
                fig.add_trace(go.Scatter(
                    x=[tangency_vol],
                    y=[tangency_return],
                    mode='markers',
                    marker=dict(color='red', size=15, symbol='star'),
                    name='Tangency Portfolio',
                    hovertemplate='Tangency Portfolio<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                ))
                
                # Individual stocks
                stock_vols = np.sqrt(np.diag(cov_matrix))
                stock_returns = mean_returns.values
                
                fig.add_trace(go.Scatter(
                    x=stock_vols,
                    y=stock_returns,
                    mode='markers+text',
                    marker=dict(color='black', size=12, symbol='diamond'),
                    text=stocks,
                    textposition='top center',
                    name='Individual Stocks',
                    hovertemplate='%{text}<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                ))
                
                # Capital Market Line
                fig.add_trace(go.Scatter(
                    x=cml_vol,
                    y=cml_return,
                    mode='lines',
                    line=dict(color='blue', dash='dash', width=2),
                    name='Capital Market Line (CML)',
                    hovertemplate='CML<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                ))
                
                # Risk-free asset
                fig.add_trace(go.Scatter(
                    x=[0],
                    y=[risk_free_rate],
                    mode='markers',
                    marker=dict(color='green', size=10, symbol='circle'),
                    name='Risk-free Asset',
                    hovertemplate='Risk-free Asset<br>Return: %{y:.2%}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f'Efficient Frontier & CML for {stocks[0]} and {stocks[1]}',
                    xaxis_title='Volatility (Standard Deviation)',
                    yaxis_title='Expected Return',
                    showlegend=True,
                    height=600,
                    xaxis=dict(tickformat='.1%'),
                    yaxis=dict(tickformat='.1%')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Tangency Portfolio (Optimal Risky Portfolio)")
                    st.metric("Expected Return", f"{tangency_return:.2%}")
                    st.metric("Volatility", f"{tangency_vol:.2%}")
                    st.metric("Sharpe Ratio", f"{tangency_sharpe:.3f}")
                    
                    st.write("**Portfolio Weights:**")
                    for i, stock in enumerate(stocks):
                        st.write(f"• {stock}: {tangency_weights[i]:.1%}")
                
                with col2:
                    st.subheader("📊 Individual Stock Statistics")
                    for i, stock in enumerate(stocks):
                        st.write(f"**{stock}:**")
                        st.write(f"• Expected Return: {stock_returns[i]:.2%}")
                        st.write(f"• Volatility: {stock_vols[i]:.2%}")
                        st.write(f"• Sharpe Ratio: {(stock_returns[i] - risk_free_rate) / stock_vols[i]:.3f}")
                        st.write("")
                
                # Additional metrics
                st.subheader("📋 Portfolio Analysis Summary")
                
                # Correlation
                correlation = returns.corr().iloc[0, 1]
                st.write(f"**Correlation between {stocks[0]} and {stocks[1]}:** {correlation:.3f}")
                
                # Diversification benefit
                equal_weight_vol = np.sqrt(0.25 * (stock_vols[0]**2 + stock_vols[1]**2 + 2 * 0.5 * 0.5 * correlation * stock_vols[0] * stock_vols[1]))
                diversification_benefit = (np.mean(stock_vols) - equal_weight_vol) / np.mean(stock_vols)
                st.write(f"**Diversification Benefit (Equal Weights):** {diversification_benefit:.1%}")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.warning("Please make sure the selected stocks have sufficient historical data.")