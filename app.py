import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import utils

st.set_page_config(page_title="Investment Portfolio Analysis", layout="wide")
st.title("Investment Portfolio Analysis Toolkit")

# Date inputs
start_date = st.date_input(
    "Start Date:", value=pd.to_datetime("today") - pd.Timedelta(days=365 * 10)
)
end_date = st.date_input("End Date:", value=pd.to_datetime("today"))

tabs = st.tabs(["Analyze Stock", "Compare Stocks", "Portfolio Optimization"])

# Tab 1: Single Stock Analysis
with tabs[0]:
    ticker = st.text_input("Stock or ETF Ticker:", "GOOG")
    data = utils.download_data(ticker, start_date, end_date)

    close_prices = data["Close"][ticker]
    daily_returns = utils.price_to_returns(close_prices)
    monthly_returns = utils.daily_price_to_monthly_returns(close_prices)

    # Key metrics - use compound method for returns consistently
    ann_return = utils.annualized_return_from_prices(close_prices) * 100
    volatility = utils.annualized_volatility(daily_returns) * 100
    sharpe = utils.sharpe_ratio(daily_returns)
    max_dd = utils.drawdown(daily_returns)["Drawdown"].min() * 100
    cfvar = utils.cornish_fisher_var(monthly_returns, alpha=0.01) * 100

    # Display metrics
    cols = st.columns(5)
    with cols[0]:
        st.metric("Annualized Return", f"{ann_return:+.2f}%")
    with cols[1]:
        st.metric("Volatility", f"{volatility:.2f}%")
    with cols[2]:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with cols[3]:
        st.metric("Max Drawdown", f"{max_dd:.2f}%")
    with cols[4]:
        st.metric("CFVaR (99%)", f"{cfvar:.2f}%")

    # Charts
    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(
            utils.create_interactive_plot(
                data.index, close_prices, f"{ticker} Price", "Date", "Price ($)"
            ),
            use_container_width=True,
        )

        drawdowns = utils.drawdown(monthly_returns)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=drawdowns.index,
                y=drawdowns["Drawdown"],
                mode="lines",
                line=dict(color="red"),
            )
        )
        fig.update_layout(
            title=f"{ticker} Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with cols[1]:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=monthly_returns.index, y=monthly_returns, name="Monthly Returns")
        )
        fig.update_layout(
            title=f"{ticker} Monthly Returns",
            xaxis_title="Date",
            yaxis_title="Return",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        wi = utils.wealth_index(monthly_returns)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wi.index, y=wi, mode="lines"))
        fig.update_layout(
            title=f"{ticker} Wealth Index",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Stock Comparison
with tabs[1]:
    input_columns = st.columns(2)
    with input_columns[0]:
        first_stock_ticker = st.text_input("Stock A:", "SPY")
    with input_columns[1]:
        second_stock_ticker = st.text_input("Stock B:", "VWO")

    if st.button("Compare Stocks"):
        stock_tickers_to_compare = [first_stock_ticker, second_stock_ticker]
        stock_analysis_data = {}
        
        for current_stock_ticker in stock_tickers_to_compare:
            stock_price_data = utils.download_data(current_stock_ticker, start_date, end_date)
            stock_closing_prices = stock_price_data["Close"][current_stock_ticker]
            stock_daily_returns = utils.price_to_returns(stock_closing_prices)
            stock_monthly_returns = utils.daily_price_to_monthly_returns(stock_closing_prices)
            
            stock_analysis_data[current_stock_ticker] = {
                'return': utils.annualized_return_from_prices(stock_closing_prices) * 100,
                'volatility': utils.annualized_volatility(stock_daily_returns) * 100,
                'sharpe': utils.sharpe_ratio(stock_daily_returns),
                'max_drawdown': utils.drawdown(stock_daily_returns)["Drawdown"].min() * 100,
                'cfvar': utils.cornish_fisher_var(stock_monthly_returns, alpha=0.01) * 100,
                'data': stock_price_data,
                'close_prices': stock_closing_prices,
                'daily_returns': stock_daily_returns,
                'monthly_returns': stock_monthly_returns
            }

        # Display comparison metrics
        comparison_columns = st.columns(2)
        for stock_index, current_stock_ticker in enumerate(stock_tickers_to_compare):
            with comparison_columns[stock_index]:
                st.subheader(f"{current_stock_ticker}")
                current_stock_metrics = stock_analysis_data[current_stock_ticker]
                
                # Display key metrics
                metrics_columns = st.columns(3)
                with metrics_columns[0]:
                    st.metric("Return", f"{current_stock_metrics['return']:.2f}%")
                    st.metric("Volatility", f"{current_stock_metrics['volatility']:.2f}%")
                with metrics_columns[1]:
                    st.metric("Sharpe", f"{current_stock_metrics['sharpe']:.2f}")
                    st.metric("Max Drawdown", f"{current_stock_metrics['max_drawdown']:.2f}%")
                with metrics_columns[2]:
                    st.metric("CFVaR (99%)", f"{current_stock_metrics['cfvar']:.2f}%")

        # Display charts in rows
        st.subheader("Price Comparison")
        price_chart_columns = st.columns(2)
        for stock_index, current_stock_ticker in enumerate(stock_tickers_to_compare):
            with price_chart_columns[stock_index]:
                st.plotly_chart(
                    utils.create_interactive_plot(
                        stock_analysis_data[current_stock_ticker]['data'].index,
                        stock_analysis_data[current_stock_ticker]['close_prices'],
                        f"{current_stock_ticker} Price",
                        "Date",
                        "Price ($)",
                    ),
                    use_container_width=True,
                )

        st.subheader("Drawdown Comparison")
        drawdown_chart_columns = st.columns(2)
        for stock_index, current_stock_ticker in enumerate(stock_tickers_to_compare):
            with drawdown_chart_columns[stock_index]:
                stock_drawdowns = utils.drawdown(stock_analysis_data[current_stock_ticker]['monthly_returns'])
                drawdown_figure = go.Figure()
                drawdown_figure.add_trace(
                    go.Scatter(
                        x=stock_drawdowns.index,
                        y=stock_drawdowns["Drawdown"],
                        mode="lines",
                        line=dict(color="red"),
                    )
                )
                drawdown_figure.update_layout(
                    title=f"{current_stock_ticker} Drawdown",
                    xaxis_title="Date",
                    yaxis_title="Drawdown",
                    height=400,
                )
                st.plotly_chart(drawdown_figure, use_container_width=True)

        st.subheader("Monthly Returns Comparison")
        monthly_returns_chart_columns = st.columns(2)
        for stock_index, current_stock_ticker in enumerate(stock_tickers_to_compare):
            with monthly_returns_chart_columns[stock_index]:
                stock_monthly_returns = stock_analysis_data[current_stock_ticker]['monthly_returns']
                monthly_returns_figure = go.Figure()
                monthly_returns_figure.add_trace(
                    go.Bar(x=stock_monthly_returns.index, y=stock_monthly_returns, name="Monthly Returns")
                )
                monthly_returns_figure.update_layout(
                    title=f"{current_stock_ticker} Monthly Returns",
                    xaxis_title="Date",
                    yaxis_title="Return",
                    height=400,
                )
                st.plotly_chart(monthly_returns_figure, use_container_width=True)

        st.subheader("Wealth Index Comparison")
        wealth_index_chart_columns = st.columns(2)
        for stock_index, current_stock_ticker in enumerate(stock_tickers_to_compare):
            with wealth_index_chart_columns[stock_index]:
                stock_wealth_index = utils.wealth_index(stock_analysis_data[current_stock_ticker]['monthly_returns'])
                wealth_index_figure = go.Figure()
                wealth_index_figure.add_trace(go.Scatter(x=stock_wealth_index.index, y=stock_wealth_index, mode="lines"))
                wealth_index_figure.update_layout(
                    title=f"{current_stock_ticker} Wealth Index",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    height=400,
                )
                st.plotly_chart(wealth_index_figure, use_container_width=True)

# Tab 3: Portfolio Optimization
with tabs[2]:
    asset_input_columns = st.columns(5)
    selected_asset_tickers = []
    for asset_input_index in range(5):
        with asset_input_columns[asset_input_index]:
            asset_ticker_input = st.text_input(
                f"Asset {asset_input_index+1}", value=["MSFT", "QQQ", "VOO", "VXUS", "GLD"][asset_input_index]
            )
            if asset_ticker_input.strip():
                selected_asset_tickers.append(asset_ticker_input.strip())

    portfolio_price_data = utils.download_data(selected_asset_tickers, start_date, end_date)["Close"]
    portfolio_daily_returns = utils.price_to_returns(portfolio_price_data)
    portfolio_annualized_returns = utils.annualized_return_from_prices_df(portfolio_price_data)

    efficient_frontier_data = utils.calculate_efficient_frontier(portfolio_price_data)


    # Find optimal portfolios
    optimal_portfolios = {
        'min_vol': efficient_frontier_data.iloc[efficient_frontier_data["Volatility"].idxmin()],
        'max_sharpe': efficient_frontier_data.iloc[efficient_frontier_data["Sharpe Ratio"].idxmax()]
    }

    # Custom target return slider
    target_return_percentage = st.slider(
        "Target Annual Return (%):",
        min_value=float(portfolio_annualized_returns.min() * 100),
        max_value=float(portfolio_annualized_returns.max() * 100),
        value=float(portfolio_annualized_returns.mean() * 100),
    )

    custom_portfolio_weights = utils.min_volatility_portfolio(portfolio_daily_returns, target_return_percentage / 100)
    custom_portfolio_metrics = None

    if custom_portfolio_weights:
        custom_portfolio_volatility = utils.portfolio_volatility(portfolio_daily_returns, np.array(custom_portfolio_weights))
        custom_portfolio_return = np.dot(custom_portfolio_weights, portfolio_annualized_returns)
        custom_portfolio_metrics = {
            "volatility": custom_portfolio_volatility,
            "return": custom_portfolio_return,
            "sharpe": custom_portfolio_return / custom_portfolio_volatility if custom_portfolio_volatility > 0 else 0,
        }

    # Plot efficient frontier
    efficient_frontier_figure = utils.plot_efficient_frontier(efficient_frontier_data, portfolio_daily_returns, custom_portfolio_metrics)
    st.plotly_chart(efficient_frontier_figure, use_container_width=True)

    # Display portfolios
    portfolio_display_columns = st.columns(3)
    portfolio_display_configurations = [
        (0, "🟢 Min Volatility", optimal_portfolios['min_vol'], "lightgreen"),
        (1, "⭐ Max Sharpe", optimal_portfolios['max_sharpe'], "gold"),
        (2, "🎯 Custom Target", None, "lightblue")
    ]

    for column_index, portfolio_title, portfolio_data, chart_color in portfolio_display_configurations:
        with portfolio_display_columns[column_index]:
            st.subheader(portfolio_title)
            
            if column_index < 2:  # Min Vol and Max Sharpe
                st.metric("Return", f"{portfolio_data['Actual Return']:.2%}")
                st.metric("Volatility", f"{portfolio_data['Volatility']:.2%}")
                st.metric("Sharpe", f"{portfolio_data['Sharpe Ratio']:.3f}")
                portfolio_allocation_weights = [portfolio_data[asset_ticker] for asset_ticker in portfolio_daily_returns.columns]
            else:  # Custom Target
                if custom_portfolio_weights and custom_portfolio_metrics:
                    st.metric("Return", f"{custom_portfolio_metrics['return']:.2%}")
                    st.metric("Volatility", f"{custom_portfolio_metrics['volatility']:.2%}")
                    st.metric("Sharpe", f"{custom_portfolio_metrics['sharpe']:.3f}")
                    portfolio_allocation_weights = custom_portfolio_weights
                else:
                    st.error("Unable to optimize for target return")
                    continue
            
            st.plotly_chart(
                utils.create_portfolio_allocation_chart(
                    portfolio_daily_returns.columns.tolist(), portfolio_allocation_weights, "Allocation", chart_color
                ),
                use_container_width=True,
            )
