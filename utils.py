import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import norm
import streamlit as st
import yfinance as yf
import logging
from typing import Union, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_data()
def download_data(ticker: Union[str, List[str]], start_date, end_date):
    """Download historical stock data from Yahoo Finance with error handling and logging."""
    try:
        if isinstance(ticker, list):
            ticker_str = ", ".join(ticker)
        else:
            ticker_str = ticker
        
        logger.info(f"Downloading data for {ticker_str} from {start_date} to {end_date}")
        
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        
        if data.empty:
            error_msg = f"No data found for ticker(s): {ticker_str}"
            logger.error(error_msg)
            st.error(error_msg)
            return pd.DataFrame()
        
        # Check for missing data
        if isinstance(ticker, list) and len(ticker) > 1:
            missing_tickers = []
            for t in ticker:
                if t not in data.columns.get_level_values(1):
                    missing_tickers.append(t)
            
            if missing_tickers:
                warning_msg = f"No data found for: {', '.join(missing_tickers)}"
                logger.warning(warning_msg)
                st.warning(warning_msg)
        
        logger.info(f"Successfully downloaded {len(data)} rows of data for {ticker_str}")
        return data
        
    except Exception as e:
        error_msg = f"Error downloading data for {ticker_str}: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        return pd.DataFrame()

def annualized_return_from_prices(prices: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate the annualized return of a price series."""
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    num_years = (len(prices) - 1) / periods_per_year
    return (1 + total_return) ** (1 / num_years) - 1

def annualized_return_from_prices_df(prices: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """Calculate the annualized return for each column in a DataFrame of prices."""
    return prices.apply(lambda col: annualized_return_from_prices(col, periods_per_year))

def annualized_return(returns, periods_per_year: int = 252) -> float:
    """Calculate the annualized return from a series of returns."""
    total_return = (1 + returns).prod() - 1
    num_years = len(returns) / periods_per_year
    return (1 + total_return) ** (1 / num_years) - 1

def daily_returns_to_monthly_returns(daily_returns: pd.Series) -> pd.Series:
    """Convert daily returns to monthly returns."""
    monthly_returns = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    return monthly_returns.dropna()

def price_to_returns(prices):
    """Calculate daily returns for a given price series or DataFrame."""
    return prices.pct_change().dropna()

def daily_price_to_monthly_returns(prices: pd.Series) -> pd.Series:
    """Calculate monthly returns for a given price series."""
    monthly_prices = prices.resample("ME").last()
    return monthly_prices.pct_change().dropna()

def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate the annualized volatility of a series of returns."""
    return returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free_rate: float = 0.0) -> float:
    """Calculate the Sharpe Ratio of a series of returns."""
    ann_return = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    return (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

def annualized_semideviation(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate the annualized semideviation (downside volatility) of a series of returns."""
    downside_returns = returns[returns < 0]
    return downside_returns.std() * np.sqrt(periods_per_year) if not downside_returns.empty else 0.0

def sortino_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free_rate: float = 0.0) -> float:
    """Calculate the Sortino Ratio of a series of returns."""
    downside_returns = returns[returns < 0]
    ann_return = annualized_return(returns, periods_per_year)
    ann_downside_vol = downside_returns.std() * np.sqrt(periods_per_year) if not downside_returns.empty else 0.0
    return (ann_return - risk_free_rate) / ann_downside_vol if ann_downside_vol > 0 else 0.0

def wealth_index(returns: pd.Series) -> pd.Series:
    """Calculate the wealth index from a series of returns."""
    return (1 + returns).cumprod()

def drawdown(returns: pd.Series) -> pd.DataFrame:
    """Calculate the drawdown of a series of returns."""
    wi = wealth_index(returns)
    peak = wi.cummax()
    drawdown_series = (wi - peak) / peak
    return pd.DataFrame({"Drawdown": drawdown_series, "Wealth Index": wi, "Peak": peak})

def cornish_fisher_var(returns: pd.Series, alpha: float = 0.01) -> float:
    """Calculate the Cornish-Fisher Value at Risk (VaR) for a series of returns."""
    mean_return = returns.mean()
    std_return = returns.std()
    skew = returns.skew()
    excess_kurtosis = returns.kurtosis()
    z = norm.ppf(alpha)
    z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * excess_kurtosis / 24
    return float(mean_return + z_cf * std_return)

def portfolio_volatility(returns: pd.DataFrame, weights: np.ndarray, periods_per_year: int = 252) -> float:
    """Calculate the portfolio volatility."""
    cov_matrix = returns.cov() * periods_per_year
    portfolio_variance = weights.T @ cov_matrix @ weights
    return np.sqrt(portfolio_variance)

def min_volatility_portfolio(daily_returns: pd.DataFrame, target_annual_return: float, periods_per_year: int = 252, asset_weight_constraints: dict = None, risk_free_rate: float = 0.0):
    """Minimize portfolio volatility for a given target return."""
    number_of_assets = daily_returns.shape[1]
    # Use compound method for expected returns
    expected_annual_returns = pd.Series(dtype=float)
    for asset_ticker in daily_returns.columns:
        cumulative_wealth_index = (1 + daily_returns[asset_ticker]).cumprod()
        expected_annual_returns[asset_ticker] = annualized_return_from_prices(cumulative_wealth_index, periods_per_year)
    
    if target_annual_return < expected_annual_returns.min() or target_annual_return > expected_annual_returns.max():
        return None
    
    def portfolio_volatility_objective(portfolio_weights):
        return portfolio_volatility(daily_returns, portfolio_weights, periods_per_year)

    optimization_constraints = [
        {'type': 'eq', 'fun': lambda portfolio_weights: np.sum(portfolio_weights) - 1},
        {'type': 'eq', 'fun': lambda portfolio_weights: np.dot(portfolio_weights, expected_annual_returns) - target_annual_return}
    ]
    
    # Set weight bounds using constraints if provided
    if asset_weight_constraints:
        weight_bounds = []
        for asset_ticker in daily_returns.columns:
            if asset_ticker in asset_weight_constraints:
                min_weight = asset_weight_constraints[asset_ticker]['min']
                max_weight = asset_weight_constraints[asset_ticker]['max']
                weight_bounds.append((min_weight, max_weight))
            else:
                weight_bounds.append((0.0, 1.0))
        weight_bounds = tuple(weight_bounds)
    else:
        weight_bounds = tuple((0.0, 1.0) for _ in range(number_of_assets))
    
    initial_weight_guess = [1.0 / number_of_assets] * number_of_assets
    
    optimization_result = minimize(portfolio_volatility_objective, initial_weight_guess, method='SLSQP', bounds=weight_bounds, constraints=optimization_constraints)
    
    if optimization_result.success:
        optimal_weights = np.maximum(0, optimization_result.x)
        optimal_weights /= np.sum(optimal_weights)
        return optimal_weights.tolist()
    return None

def calculate_efficient_frontier(asset_prices: pd.DataFrame, periods_per_year: int = 252, number_of_points: int = 50, asset_weight_constraints: dict = None, risk_free_rate: float = 0.0):
    """Calculate the efficient frontier."""
    if asset_prices.empty:
        return pd.DataFrame()
    
    # Calculate returns from prices
    daily_returns = price_to_returns(asset_prices)
    
    # Calculate expected returns using compound method
    expected_annual_returns = annualized_return_from_prices_df(asset_prices, periods_per_year)
    
    annual_covariance_matrix = daily_returns.cov() * periods_per_year
    number_of_assets = len(daily_returns.columns)

    def portfolio_variance_function(portfolio_weights):
        return portfolio_weights.T @ annual_covariance_matrix @ portfolio_weights

    # Set weight bounds using constraints if provided
    if asset_weight_constraints:
        weight_bounds = []
        for asset_ticker in daily_returns.columns:
            if asset_ticker in asset_weight_constraints:
                min_weight = asset_weight_constraints[asset_ticker]['min']
                max_weight = asset_weight_constraints[asset_ticker]['max']
                weight_bounds.append((min_weight, max_weight))
            else:
                weight_bounds.append((0.0, 1.0))
        weight_bounds = tuple(weight_bounds)
    else:
        weight_bounds = tuple((0.0, 1.0) for _ in range(number_of_assets))
    
    weights_sum_constraint = {'type': 'eq', 'fun': lambda portfolio_weights: np.sum(portfolio_weights) - 1}
    initial_weight_guess = np.array([1.0 / number_of_assets] * number_of_assets)

    # Min variance portfolio
    min_variance_optimization_result = minimize(portfolio_variance_function, initial_weight_guess, method='SLSQP',
                              bounds=weight_bounds, constraints=[weights_sum_constraint])
    
    if not min_variance_optimization_result.success:
        st.error("Failed to calculate minimum variance portfolio.")
        return pd.DataFrame()

    min_variance_weights = min_variance_optimization_result.x
    min_variance_weights /= np.sum(min_variance_weights)
    min_variance_return = np.dot(min_variance_weights, expected_annual_returns)
    min_variance_volatility = np.sqrt(portfolio_variance_function(min_variance_weights))

    maximum_expected_return = expected_annual_returns.max()
    target_return_range = np.linspace(min_variance_return, maximum_expected_return, number_of_points)
    efficient_frontier_portfolio_data = []

    # Add min vol portfolio
    min_volatility_sharpe_ratio = (min_variance_return - risk_free_rate) / min_variance_volatility if min_variance_volatility > 0 else 0
    min_volatility_portfolio_row = {'Actual Return': min_variance_return, 'Volatility': min_variance_volatility, 'Sharpe Ratio': min_volatility_sharpe_ratio}
    for asset_index, asset_ticker in enumerate(daily_returns.columns):
        min_volatility_portfolio_row[asset_ticker] = min_variance_weights[asset_index]
    efficient_frontier_portfolio_data.append(min_volatility_portfolio_row)
    
    # Calculate other points
    for target_return_level in target_return_range:
        if np.isclose(target_return_level, min_variance_return, atol=1e-2):
            continue

        portfolio_constraints = [
            weights_sum_constraint,
            {'type': 'eq', 'fun': lambda portfolio_weights: np.dot(portfolio_weights, expected_annual_returns) - target_return_level}
        ]

        optimization_result = minimize(portfolio_variance_function, initial_weight_guess, method='SLSQP',
                          bounds=weight_bounds, constraints=portfolio_constraints)
        if not optimization_result.success:
            continue

        if optimization_result.success and optimization_result.fun >= 0:  # Ensure positive variance
            optimal_weights = np.maximum(0, optimization_result.x)
            optimal_weights /= np.sum(optimal_weights)
            portfolio_expected_return = np.dot(optimal_weights, expected_annual_returns)
            portfolio_volatility_value = np.sqrt(portfolio_variance_function(optimal_weights))
            
            # Validate calculations
            if portfolio_volatility_value > 0 and not np.isnan(portfolio_expected_return) and not np.isnan(portfolio_volatility_value):
                portfolio_sharpe_ratio = (portfolio_expected_return - risk_free_rate) / portfolio_volatility_value

                portfolio_row_data = {'Actual Return': portfolio_expected_return, 'Volatility': portfolio_volatility_value, 'Sharpe Ratio': portfolio_sharpe_ratio}
                for asset_index, asset_ticker in enumerate(daily_returns.columns):
                    portfolio_row_data[asset_ticker] = optimal_weights[asset_index]
                efficient_frontier_portfolio_data.append(portfolio_row_data)

    efficient_frontier_dataframe = pd.DataFrame(efficient_frontier_portfolio_data)
    if not efficient_frontier_dataframe.empty:
        efficient_frontier_dataframe = efficient_frontier_dataframe.dropna().sort_values(by='Volatility').reset_index(drop=True)
        # Remove any duplicate or invalid points
        efficient_frontier_dataframe = efficient_frontier_dataframe[(efficient_frontier_dataframe['Volatility'] > 0) & (efficient_frontier_dataframe['Actual Return'].notna())].drop_duplicates(subset=['Volatility'], keep='first')
    return efficient_frontier_dataframe

def create_interactive_plot(x_data, y_data, title="Chart", x_label="X", y_label="Y"):
    """Create a simple interactive line plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode="lines"))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, height=400)
    return fig

def plot_efficient_frontier(efficient_frontier_dataframe, daily_returns_data=None, custom_portfolio_data=None):
    """Plot the efficient frontier."""
    plotly_figure = go.Figure()

    if efficient_frontier_dataframe.empty:
        plotly_figure.add_annotation(text="No efficient frontier data available", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return plotly_figure

    sorted_efficient_frontier = efficient_frontier_dataframe.sort_values('Volatility').reset_index(drop=True)
    
    # Plot efficient frontier
    plotly_figure.add_trace(go.Scatter(x=sorted_efficient_frontier['Volatility'], y=sorted_efficient_frontier['Actual Return'],
                            mode='lines+markers', name='Efficient Frontier',
                            line=dict(color='blue', width=3), marker=dict(size=4)))

    # Highlight special portfolios
    if not sorted_efficient_frontier.empty and 'Sharpe Ratio' in sorted_efficient_frontier.columns:
        maximum_sharpe_index = sorted_efficient_frontier['Sharpe Ratio'].idxmax()
        minimum_volatility_index = sorted_efficient_frontier['Volatility'].idxmin();
        
        plotly_figure.add_trace(go.Scatter(x=[sorted_efficient_frontier.loc[maximum_sharpe_index, 'Volatility']], 
                                y=[sorted_efficient_frontier.loc[maximum_sharpe_index, 'Actual Return']],
                                mode='markers', name='Max Sharpe', 
                                marker=dict(size=15, color='red', symbol='star')))
        
        plotly_figure.add_trace(go.Scatter(x=[sorted_efficient_frontier.loc[minimum_volatility_index, 'Volatility']], 
                                y=[sorted_efficient_frontier.loc[minimum_volatility_index, 'Actual Return']],
                                mode='markers', name='Min Vol', 
                                marker=dict(size=15, color='green', symbol='diamond')))
    
    # Custom portfolio
    if custom_portfolio_data:
        plotly_figure.add_trace(go.Scatter(x=[custom_portfolio_data['volatility']], y=[custom_portfolio_data['return']],
                                mode='markers', name='Custom Portfolio',
                                marker=dict(size=15, color='orange', symbol='circle')))

    # Individual assets
    if daily_returns_data is not None and not daily_returns_data.empty:
        # Calculate individual asset metrics
        individual_asset_annual_returns = {}
        individual_asset_volatilities = {}
        for asset_ticker in daily_returns_data.columns:
            # Use compound method for individual assets
            asset_wealth_index = (1 + daily_returns_data[asset_ticker]).cumprod()
            individual_asset_annual_returns[asset_ticker] = annualized_return_from_prices(asset_wealth_index, 252)
            individual_asset_volatilities[asset_ticker] = annualized_volatility(daily_returns_data[asset_ticker])
            
            plotly_figure.add_trace(go.Scatter(x=[individual_asset_volatilities[asset_ticker]], y=[individual_asset_annual_returns[asset_ticker]],
                                    mode='markers', name=asset_ticker, 
                                    marker=dict(size=10, symbol='circle', opacity=0.8)))

    plotly_figure.update_layout(title="Efficient Frontier", xaxis_title="Volatility (Annual)",
                     yaxis_title="Expected Return (Annual)", height=600,
                     xaxis=dict(tickformat='.1%'), yaxis=dict(tickformat='.1%'))
    return plotly_figure

def create_portfolio_allocation_chart(asset_tickers, portfolio_weights, chart_title="Asset Allocation", bar_color='lightblue'):
    """Create a bar chart for portfolio asset allocation."""
    sorted_allocation_data = sorted(zip(asset_tickers, portfolio_weights))
    sorted_asset_tickers, sorted_portfolio_weights = zip(*sorted_allocation_data)
    
    allocation_figure = go.Figure(data=[go.Bar(x=sorted_asset_tickers, y=sorted_portfolio_weights, marker_color=bar_color,
                                text=[f"{weight:.1%}" for weight in sorted_portfolio_weights], textposition='auto')])
    allocation_figure.update_layout(title=chart_title, xaxis_title="Assets", yaxis_title="Weight",
                     yaxis=dict(tickformat='.1%'), height=300, showlegend=False)
    return allocation_figure

