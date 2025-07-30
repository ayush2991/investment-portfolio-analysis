import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import utils

st.set_page_config(page_title="Investment Portfolio Analysis", layout="wide")
st.title("Investment Portfolio Analysis Toolkit")

# Sidebar for global configuration
with st.sidebar:
    st.header("📊 Global Settings")
    
    # Date inputs
    start_date = st.date_input(
        "Start Date:", 
        value=pd.to_datetime("today") - pd.Timedelta(days=365 * 10),
        help="Analysis start date for all tabs"
    )
    end_date = st.date_input(
        "End Date:", 
        value=pd.to_datetime("today"),
        help="Analysis end date for all tabs"
    )
    
    # Risk-free rate input
    risk_free_rate = st.number_input(
        "Risk-Free Rate (%):",
        min_value=0.0,
        max_value=20.0,
        value=4.0,
        step=0.1,
        help="Annual risk-free rate for Sharpe ratio calculations"
    ) / 100.0
    

tabs = st.tabs(["Analyze Stock", "Compare Stocks", "Compare Portfolios", "Portfolio Optimization", "Top Winners"])

# Tab 1: Single Stock Analysis
with tabs[0]:
    ticker = st.text_input("Stock or ETF Ticker:", "GOOG")
    
    try:
        data = utils.download_data(ticker, start_date, end_date)
        
        if data.empty:
            st.error("No data available. Please check the ticker symbol and date range.")
        else:
            close_prices = data["Close"][ticker]
            daily_returns = utils.price_to_returns(close_prices)
            monthly_returns = utils.daily_price_to_monthly_returns(close_prices)

            # Key metrics - use compound method for returns consistently
            ann_return = utils.annualized_return_from_prices(close_prices) * 100
            volatility = utils.annualized_volatility(daily_returns) * 100
            semideviation = utils.annualized_semideviation(daily_returns) * 100
            sharpe = utils.sharpe_ratio(daily_returns, risk_free_rate=risk_free_rate)
            sortino = utils.sortino_ratio(daily_returns, risk_free_rate=risk_free_rate)
            max_dd = utils.drawdown(daily_returns)["Drawdown"].min() * 100
            cfvar = utils.cornish_fisher_var(monthly_returns, alpha=0.01) * 100

            # Display metrics
            cols = st.columns(7)
            with cols[0]:
                st.metric("Annualized Return", f"{ann_return:+.2f}%")
            with cols[1]:
                st.metric("Volatility", f"{volatility:.2f}%")
            with cols[2]:
                st.metric("Semideviation", f"{semideviation:.2f}%")
            with cols[3]:
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            with cols[4]:
                st.metric("Max Drawdown", f"{max_dd:.2f}%")
            with cols[5]:
                st.metric("CFVaR (99%)", f"{cfvar:.2f}%")
            with cols[6]:
                st.metric("Sortino Ratio", f"{sortino:.2f}")

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

    except Exception as e:
        st.error(f"An error occurred while analyzing {ticker}: {str(e)}")

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
        
        try:
            for current_stock_ticker in stock_tickers_to_compare:
                stock_price_data = utils.download_data(current_stock_ticker, start_date, end_date)
                
                if stock_price_data.empty:
                    st.error(f"No data available for {current_stock_ticker}. Skipping...")
                    continue
                
                stock_closing_prices = stock_price_data["Close"][current_stock_ticker]
                stock_daily_returns = utils.price_to_returns(stock_closing_prices)
                stock_monthly_returns = utils.daily_price_to_monthly_returns(stock_closing_prices)
                
                stock_analysis_data[current_stock_ticker] = {
                    'return': utils.annualized_return_from_prices(stock_closing_prices) * 100,
                    'volatility': utils.annualized_volatility(stock_daily_returns) * 100,
                    'semideviation': utils.annualized_semideviation(stock_daily_returns) * 100,
                    'sharpe': utils.sharpe_ratio(stock_daily_returns, risk_free_rate=risk_free_rate),
                    'sortino': utils.sortino_ratio(stock_daily_returns, risk_free_rate=risk_free_rate),
                    'max_drawdown': utils.drawdown(stock_daily_returns)["Drawdown"].min() * 100,
                    'cfvar': utils.cornish_fisher_var(stock_monthly_returns, alpha=0.01) * 100,
                    'data': stock_price_data,
                    'close_prices': stock_closing_prices,
                    'daily_returns': stock_daily_returns,
                    'monthly_returns': stock_monthly_returns
                }

            if len(stock_analysis_data) < 2:
                st.error("Need data for at least 2 stocks to compare.")
            else:
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
                            st.metric("Semideviation", f"{current_stock_metrics['semideviation']:.2f}%")
                        with metrics_columns[1]:
                            st.metric("Sharpe", f"{current_stock_metrics['sharpe']:.2f}")
                            st.metric("Max Drawdown", f"{current_stock_metrics['max_drawdown']:.2f}%")
                        with metrics_columns[2]:
                            st.metric("Sortino", f"{current_stock_metrics['sortino']:.2f}")
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
                            key=f"price_chart_{current_stock_ticker}_{stock_index}"
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
                        st.plotly_chart(drawdown_figure, use_container_width=True, key=f"drawdown_chart_{current_stock_ticker}_{stock_index}")

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
                        st.plotly_chart(monthly_returns_figure, use_container_width=True, key=f"monthly_returns_chart_{current_stock_ticker}_{stock_index}")

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
                        st.plotly_chart(wealth_index_figure, use_container_width=True, key=f"wealth_index_chart_{current_stock_ticker}_{stock_index}")

        except Exception as e:
            st.error(f"An error occurred during stock comparison: {str(e)}")

# Tab 2: Portfolio Comparison
with tabs[2]:
    portfolio_comparison_columns = st.columns(2)
    
    # Portfolio A specification
    with portfolio_comparison_columns[0]:
        st.subheader("Portfolio A")
        portfolio_a_assets = []
        portfolio_a_weights = []
        
        for asset_index in range(5):
            asset_columns = st.columns([3, 1])
            with asset_columns[0]:
                asset_ticker = st.text_input(
                    f"Asset {asset_index + 1}",
                    value=["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"][asset_index] if asset_index < 5 else "",
                    key=f"portfolio_a_asset_{asset_index}"
                )
            with asset_columns[1]:
                asset_weight = st.number_input(
                    "Weight",
                    min_value=0.0,
                    max_value=100.0,
                    value=20.0 if asset_index < 5 and asset_ticker.strip() else 0.0,
                    step=1.0,
                    key=f"portfolio_a_weight_{asset_index}"
                ) / 100.0
            
            if asset_ticker.strip():
                portfolio_a_assets.append(asset_ticker.strip())
                portfolio_a_weights.append(asset_weight)
            elif asset_weight > 0:
                # Clear weight if asset ticker is empty but weight is set
                st.rerun()
        
        # Normalize weights for Portfolio A
        if portfolio_a_weights and sum(portfolio_a_weights) > 0:
            portfolio_a_weights = [w / sum(portfolio_a_weights) for w in portfolio_a_weights]
    
    # Portfolio B specification  
    with portfolio_comparison_columns[1]:
        st.subheader("Portfolio B")
        portfolio_b_assets = []
        portfolio_b_weights = []
        
        for asset_index in range(5):
            asset_columns = st.columns([3, 1])
            with asset_columns[0]:
                asset_ticker = st.text_input(
                    f"Asset {asset_index + 1}",
                    value=["SPY", "VWO", "VEA", "BND", "GLD"][asset_index] if asset_index < 5 else "",
                    key=f"portfolio_b_asset_{asset_index}"
                )
            with asset_columns[1]:
                asset_weight = st.number_input(
                    "Weight",
                    min_value=0.0,
                    max_value=100.0,
                    value=20.0 if asset_index < 5 and asset_ticker.strip() else 0.0,
                    step=1.0,
                    key=f"portfolio_b_weight_{asset_index}"
                ) / 100.0
            
            if asset_ticker.strip():
                portfolio_b_assets.append(asset_ticker.strip())
                portfolio_b_weights.append(asset_weight)
            elif asset_weight > 0:
                # Clear weight if asset ticker is empty but weight is set
                st.rerun()
        
        # Normalize weights for Portfolio B
        if portfolio_b_weights and sum(portfolio_b_weights) > 0:
            portfolio_b_weights = [w / sum(portfolio_b_weights) for w in portfolio_b_weights]
            
    # Calculate and display portfolio metrics
    if st.button("Compare Portfolios") and portfolio_a_assets and portfolio_b_assets:
        try:
            # Download data for both portfolios
            all_assets = list(set(portfolio_a_assets + portfolio_b_assets))
            price_data = utils.download_data(all_assets, start_date, end_date)
            
            if price_data.empty:
                st.error("No data available for the specified assets.")
            else:
                # Check if all required assets have data
                available_assets = price_data["Close"].columns.tolist()
                missing_a = [asset for asset in portfolio_a_assets if asset not in available_assets]
                missing_b = [asset for asset in portfolio_b_assets if asset not in available_assets]
                
                if missing_a or missing_b:
                    if missing_a:
                        st.error(f"Missing data for Portfolio A assets: {', '.join(missing_a)}")
                    if missing_b:
                        st.error(f"Missing data for Portfolio B assets: {', '.join(missing_b)}")
                else:
                    daily_returns_data = utils.price_to_returns(price_data["Close"])
                    
                    # Calculate Portfolio A metrics
                    portfolio_a_daily_returns = daily_returns_data[portfolio_a_assets]
                    portfolio_a_portfolio_returns = (portfolio_a_daily_returns * portfolio_a_weights).sum(axis=1)
                    portfolio_a_annualized_return = utils.annualized_return(portfolio_a_portfolio_returns) * 100
                    portfolio_a_annualized_volatility = utils.annualized_volatility(portfolio_a_portfolio_returns) * 100
                    portfolio_a_sharpe_ratio = utils.sharpe_ratio(portfolio_a_portfolio_returns, risk_free_rate=risk_free_rate)
                    
                    # Calculate Portfolio B metrics
                    portfolio_b_daily_returns = daily_returns_data[portfolio_b_assets]
                    portfolio_b_portfolio_returns = (portfolio_b_daily_returns * portfolio_b_weights).sum(axis=1)
                    portfolio_b_annualized_return = utils.annualized_return(portfolio_b_portfolio_returns) * 100
                    portfolio_b_annualized_volatility = utils.annualized_volatility(portfolio_b_portfolio_returns) * 100
                    portfolio_b_sharpe_ratio = utils.sharpe_ratio(portfolio_b_portfolio_returns, risk_free_rate=risk_free_rate)
                    
                    # Display comparison results
                    st.subheader("Portfolio Comparison Results")
                    comparison_results_columns = st.columns(2)
                    
                    with comparison_results_columns[0]:
                        st.subheader("Portfolio A Performance")
                        st.metric("Annualized Return", f"{portfolio_a_annualized_return:.2f}%")
                        st.metric("Annualized Volatility", f"{portfolio_a_annualized_volatility:.2f}%")
                        st.metric("Sharpe Ratio", f"{portfolio_a_sharpe_ratio:.3f}")
                        
                        # Portfolio A allocation chart
                        st.plotly_chart(
                            utils.create_portfolio_allocation_chart(
                                portfolio_a_assets, portfolio_a_weights, "Portfolio A Allocation", "lightblue"
                            ),
                            use_container_width=True,
                        )
                    
                    with comparison_results_columns[1]:
                        st.subheader("Portfolio B Performance")
                        st.metric("Annualized Return", f"{portfolio_b_annualized_return:.2f}%")
                        st.metric("Annualized Volatility", f"{portfolio_b_annualized_volatility:.2f}%")
                        st.metric("Sharpe Ratio", f"{portfolio_b_sharpe_ratio:.3f}")
                        
                        # Portfolio B allocation chart
                        st.plotly_chart(
                            utils.create_portfolio_allocation_chart(
                                portfolio_b_assets, portfolio_b_weights, "Portfolio B Allocation", "lightcoral"
                            ),
                            use_container_width=True,
                        )
                    
                    # Performance comparison charts
                    st.subheader("Performance Comparison Charts")
                    
                    # Wealth index comparison
                    portfolio_a_wealth_index = utils.wealth_index(portfolio_a_portfolio_returns)
                    portfolio_b_wealth_index = utils.wealth_index(portfolio_b_portfolio_returns)
                    
                    wealth_comparison_figure = go.Figure()
                    wealth_comparison_figure.add_trace(go.Scatter(
                        x=portfolio_a_wealth_index.index, 
                        y=portfolio_a_wealth_index, 
                        mode="lines", 
                        name="Portfolio A",
                        line=dict(color="blue")
                    ))
                    wealth_comparison_figure.add_trace(go.Scatter(
                        x=portfolio_b_wealth_index.index, 
                        y=portfolio_b_wealth_index, 
                        mode="lines", 
                        name="Portfolio B",
                        line=dict(color="red")
                    ))
                    wealth_comparison_figure.update_layout(
                        title="Portfolio Wealth Index Comparison",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return",
                        height=500
                    )
                    st.plotly_chart(wealth_comparison_figure, use_container_width=True)
                    
        except Exception as error_message:
            st.error(f"Error calculating portfolio metrics: {str(error_message)}")

# Tab 4: Portfolio Optimization
with tabs[3]:
    asset_input_columns = st.columns(5)
    selected_asset_tickers = []
    asset_weight_constraints = {}
    
    for asset_input_index in range(5):
        with asset_input_columns[asset_input_index]:
            asset_ticker_input = st.text_input(
                f"Asset {asset_input_index+1}", value=["MSFT", "QQQ", "VOO", "VXUS", "GLD"][asset_input_index]
            )
            if asset_ticker_input.strip():
                selected_asset_tickers.append(asset_ticker_input.strip())
                
                # Add weight constraint inputs side by side
                weight_constraint_columns = st.columns(2)
                with weight_constraint_columns[0]:
                    min_weight = st.number_input(
                        f"Min %", 
                        min_value=0.0, 
                        max_value=100.0, 
                        value=0.0, 
                        step=1.0,
                        key=f"min_weight_{asset_input_index}"
                    ) / 100.0
                
                with weight_constraint_columns[1]:
                    max_weight = st.number_input(
                        f"Max %", 
                        min_value=0.0, 
                        max_value=100.0, 
                        value=100.0, 
                        step=1.0,
                        key=f"max_weight_{asset_input_index}"
                    ) / 100.0
                
                # Ensure min <= max
                if min_weight > max_weight:
                    st.error(f"Min weight cannot exceed max weight for {asset_ticker_input.strip()}")
                    max_weight = min_weight
                
                asset_weight_constraints[asset_ticker_input.strip()] = {
                    'min': min_weight,
                    'max': max_weight
                }

    if selected_asset_tickers:
        try:
            portfolio_price_data = utils.download_data(selected_asset_tickers, start_date, end_date)
            
            if portfolio_price_data.empty:
                st.error("No data available for the specified assets.")
            else:
                # Check if all assets have data
                if len(selected_asset_tickers) == 1:
                    available_assets = [selected_asset_tickers[0]] if selected_asset_tickers[0] in portfolio_price_data.columns else []
                else:
                    available_assets = portfolio_price_data["Close"].columns.tolist()
                
                missing_assets = [asset for asset in selected_asset_tickers if asset not in available_assets]
                
                if missing_assets:
                    st.error(f"Missing data for assets: {', '.join(missing_assets)}")
                    st.info("Please check the ticker symbols and try again.")
                else:
                    if len(selected_asset_tickers) == 1:
                        portfolio_price_data = portfolio_price_data[selected_asset_tickers[0]].to_frame()
                        portfolio_price_data.columns = selected_asset_tickers
                    else:
                        portfolio_price_data = portfolio_price_data["Close"]
                    
                    portfolio_daily_returns = utils.price_to_returns(portfolio_price_data)
                    portfolio_annualized_returns = utils.annualized_return_from_prices_df(portfolio_price_data)

                    efficient_frontier_data = utils.calculate_efficient_frontier(
                        portfolio_price_data, 
                        asset_weight_constraints=asset_weight_constraints,
                        risk_free_rate=risk_free_rate
                    )

                    if not efficient_frontier_data.empty:
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

                        custom_portfolio_weights = utils.min_volatility_portfolio(
                            portfolio_daily_returns, 
                            target_return_percentage / 100, 
                            asset_weight_constraints=asset_weight_constraints,
                            risk_free_rate=risk_free_rate
                        )
                        custom_portfolio_metrics = None

                        if custom_portfolio_weights:
                            custom_portfolio_volatility = utils.portfolio_volatility(portfolio_daily_returns, np.array(custom_portfolio_weights))
                            custom_portfolio_return = np.dot(custom_portfolio_weights, portfolio_annualized_returns)
                            custom_portfolio_sharpe = (custom_portfolio_return - risk_free_rate) / custom_portfolio_volatility if custom_portfolio_volatility > 0 else 0
                            custom_portfolio_metrics = {
                                "volatility": custom_portfolio_volatility,
                                "return": custom_portfolio_return,
                                "sharpe": custom_portfolio_sharpe,
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
                    else:
                        st.error("Failed to calculate efficient frontier")
        except Exception as e:
            st.error(f"An error occurred during portfolio optimization: {str(e)}")
    else:
        st.info("Enter at least one asset symbol")
        st.info("Enter at least one asset symbol")

# Tab 5: Top Winners
with tabs[4]:
    st.header("S&P 500 Top 50 Winners")
    control_columns = st.columns(4)
    with control_columns[0]:
        top_n_display = st.slider(
            "Top N Winners to Display:",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="Number of top winners to display"
        )
        analyze_button = st.button("Analyze Top Winners", type="primary")
    
    if analyze_button:
        try:
            # Get top 50 S&P 500 tickers and calculate performance
            top_tickers = utils.get_sp500_top_tickers(50)
            performance_data = utils.calculate_wealth_performance(top_tickers, start_date, end_date)
            
            if not performance_data:
                st.error("No performance data could be calculated.")
            else:
                # Get S&P 500 reference data if requested
                sp500_wealth_index = None
                try:
                    sp500_data = utils.download_data('SPY', start_date, end_date)
                    if not sp500_data.empty:
                        sp500_prices = sp500_data['Close']['SPY'] if isinstance(sp500_data.columns, pd.MultiIndex) else sp500_data['Close']
                        sp500_returns = utils.price_to_returns(sp500_prices)
                        sp500_wealth_index = utils.wealth_index(sp500_returns)
                except Exception as e:
                    st.warning(f"Could not load S&P 500 reference data: {str(e)}")
                
                # Sort performers by wealth index
                sorted_performers = sorted(performance_data.items(), 
                                         key=lambda x: x[1]['final_value'], reverse=True)
                
                # Display performance table
                table_data = []
                for i, (ticker, data) in enumerate(sorted_performers[:20]):
                    table_data.append({
                        'Rank': i + 1,
                        'Ticker': ticker,
                        'Final Wealth Index': f"{data['final_value']:.2f}",
                        'Total Return': f"{data['total_return']:.1f}%",
                        'Annualized Return': f"{data['annualized_return']:.1f}%"
                    })
                
                performance_df = pd.DataFrame(table_data)
                st.dataframe(performance_df, use_container_width=True, hide_index=True)
                
                # Plot wealth index chart
                wealth_fig = utils.plot_top_performers_wealth_index(
                    performance_data, 
                    top_n=top_n_display, 
                    sp500_data=sp500_wealth_index
                )
                st.plotly_chart(wealth_fig, use_container_width=True)
                
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
                # Create and display performance table                