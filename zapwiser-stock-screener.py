import os
import dash
from dash import html, dcc, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import numpy as np

# Zapwiser theme colors
ZAPWISER_COLORS = {
    'primary': '#3182ce',
    'secondary': '#4a5568',
    'success': '#38a169',
    'warning': '#e9b949',
    'danger': '#e53e3e',
    'light': '#f7fafc',
    'dark': '#1a202c',
    'text': '#2d3748',
    'muted': '#718096'
}

# Initialize Dash app with Zapwiser theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Poppins:wght@400;500;600&display=swap',
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

app.title = "Zapwiser - Stock Screener"

# Default watchlist
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOG", "META", "NVDA", "SLB", "XOM", "UBER", "AMD", "ORCL"]

# Technical indicator calculation functions
def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """Calculate RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, window=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def fetch_enhanced_stock_data(ticker, period="2y"):
    """Fetch stock data with technical indicators and fundamental metrics"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval="1d")
        
        if df.empty or len(df) < 200:
            return None
            
        # Calculate technical indicators
        df['SMA50'] = calculate_sma(df['Close'], 50)
        df['SMA100'] = calculate_sma(df['Close'], 100)
        df['SMA200'] = calculate_sma(df['Close'], 200)
        df['RSI'] = calculate_rsi(df['Close'])
        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
        df['Vol20'] = df['Volume'].rolling(20).mean()
        
        # Get valid data
        df_valid = df.dropna(subset=['SMA50', 'SMA100', 'SMA200', 'ATR', 'RSI', 'Vol20'])
        if df_valid.empty:
            return None
            
        latest = df_valid.iloc[-1]
        info = stock.info
        
        # Calculate performance metrics
        price = float(latest['Close'])
        perf_1m = ((df_valid['Close'].iloc[-1] / df_valid['Close'].iloc[-22]) - 1) * 100 if len(df_valid) >= 22 else None
        perf_3m = ((df_valid['Close'].iloc[-1] / df_valid['Close'].iloc[-66]) - 1) * 100 if len(df_valid) >= 66 else None
        perf_6m = ((df_valid['Close'].iloc[-1] / df_valid['Close'].iloc[-132]) - 1) * 100 if len(df_valid) >= 132 else None
        
        # Process fundamental data properly (matching original implementation)
        pe_ratio = info.get('trailingPE', None)
        pb_ratio = info.get('priceToBook', None)
        ps_ratio = info.get('priceToSalesTrailing12Months', None)
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        
        # Quality metrics - handle None values properly
        roe = info.get('returnOnEquity', None)
        if roe:
            roe = roe * 100  # Convert to percentage
        
        operating_margin = info.get('operatingMargins', None)
        if operating_margin:
            operating_margin = operating_margin * 100
        
        # Growth metrics
        revenue_growth = info.get('revenueGrowth', None)
        if revenue_growth:
            revenue_growth = revenue_growth * 100
        
        # Debug output to understand the data
        print(f"Debug {ticker}: ROE={roe}, OpMargin={operating_margin}, RevGrowth={revenue_growth}")
        
        return {
            'ticker': ticker,
            'price': price,
            'sma50': float(latest['SMA50']),
            'sma100': float(latest['SMA100']),
            'sma200': float(latest['SMA200']),
            'rsi': float(latest['RSI']),
            'atr': float(latest['ATR']),
            'volume': int(latest['Volume']),
            'avg_volume': int(latest['Vol20']),
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'ps_ratio': ps_ratio,
            'dividend_yield': dividend_yield,
            'roe': roe,
            'operating_margin': operating_margin,
            'revenue_growth': revenue_growth,
            'market_cap': info.get('marketCap'),
            'perf_1m': perf_1m,
            'perf_3m': perf_3m,
            'perf_6m': perf_6m
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def check_swing_criteria(data, volume_multiplier=1.5, atr_threshold=0.02):
    """Check if stock meets swing trading criteria"""
    if not data:
        return {'trend': False, 'volume': False, 'volatility': False, 'momentum': False}
    
    # Trend: Price above SMA50 and SMA100
    trend_check = data['price'] > data['sma50'] and data['sma50'] > data['sma100']
    
    # Volume: Above average
    volume_check = data['volume'] > (data['avg_volume'] * volume_multiplier)
    
    # Volatility: ATR relative to price
    volatility_check = (data['atr'] / data['price']) > atr_threshold
    
    # Momentum: RSI between 30-70 (not overbought/oversold)
    momentum_check = 30 <= data['rsi'] <= 70
    
    return {
        'trend': trend_check,
        'volume': volume_check,
        'volatility': volatility_check,
        'momentum': momentum_check
    }

def calculate_qvm_scores(data):
    """Calculate Quality, Value, and Momentum scores - matching original sophisticated approach"""
    if not data:
        return {'quality_score': 0, 'value_score': 0, 'momentum_score': 0, 'qvm_score': 0}
    
    # Quality Score (0-100) - Proportional scoring
    quality_score = 0
    quality_count = 0
    
    # ROE scoring (higher is better, benchmark: 15%)
    if data['roe'] is not None:
        roe_score = min(100, max(0, (data['roe'] / 30) * 100))
        quality_score += roe_score
        quality_count += 1
    
    # Operating Margin scoring (higher is better, benchmark: 15%)
    if data['operating_margin'] is not None:
        margin_score = min(100, max(0, (data['operating_margin'] / 30) * 100))
        quality_score += margin_score
        quality_count += 1
    
    # Revenue Growth scoring (higher is better, benchmark: 10%)
    if data.get('revenue_growth') is not None:
        growth_score = min(100, max(0, (data['revenue_growth'] / 20) * 100))
        quality_score += growth_score
        quality_count += 1
    
    quality_final = quality_score / quality_count if quality_count > 0 else 50
    
    # Value Score (0-100) - Inverse scoring (lower ratios = higher score)
    value_score = 0
    value_count = 0
    
    # P/E scoring (lower is better, benchmark: 20)
    if data['pe_ratio'] is not None and data['pe_ratio'] > 0:
        pe_score = min(100, max(0, (20 / data['pe_ratio']) * 50))
        value_score += pe_score
        value_count += 1
    
    # P/B scoring (lower is better, benchmark: 3)
    if data['pb_ratio'] is not None and data['pb_ratio'] > 0:
        pb_score = min(100, max(0, (3 / data['pb_ratio']) * 50))
        value_score += pb_score
        value_count += 1
    
    # Dividend Yield scoring (higher is better)
    if data['dividend_yield'] is not None:
        yield_score = min(100, data['dividend_yield'] * 20)  # 5% yield = 100 score
        value_score += yield_score
        value_count += 1
    
    value_final = value_score / value_count if value_count > 0 else 50
    
    # Momentum Score (0-100) - Weighted performance scoring
    momentum_score = 0
    momentum_count = 0
    
    # Price momentum scoring with proper weighting
    momentum_weights = {'1m': 0.2, '3m': 0.3, '6m': 0.5}
    
    if data['perf_1m'] is not None:
        m1_score = 50 + (data['perf_1m'] * 2)  # 0% = 50, +25% = 100, -25% = 0
        momentum_score += min(100, max(0, m1_score)) * momentum_weights['1m']
        momentum_count += momentum_weights['1m']
    
    if data['perf_3m'] is not None:
        m3_score = 50 + (data['perf_3m'] * 1.5)
        momentum_score += min(100, max(0, m3_score)) * momentum_weights['3m']
        momentum_count += momentum_weights['3m']
    
    if data['perf_6m'] is not None:
        m6_score = 50 + data['perf_6m']
        momentum_score += min(100, max(0, m6_score)) * momentum_weights['6m']
        momentum_count += momentum_weights['6m']
    
    # RSI scoring (50 is ideal, penalties for extremes)
    if data['rsi'] is not None:
        rsi_score = 100 - abs(data['rsi'] - 50) * 2
        momentum_score += max(0, rsi_score) * 0.2
        momentum_count += 0.2
    
    momentum_final = momentum_score / momentum_count if momentum_count > 0 else 50
    
    # Calculate overall QVM score (equal weighting)
    qvm_score = (quality_final + value_final + momentum_final) / 3
    
    return {
        'quality_score': round(quality_final, 1),
        'value_score': round(value_final, 1),
        'momentum_score': round(momentum_final, 1),
        'qvm_score': round(qvm_score, 1)
    }

def get_score_color(score):
    """Return color based on score value"""
    if score >= 75:
        return "#38a169"  # Green
    elif score >= 50:
        return "#e9b949"  # Yellow
    else:
        return "#e53e3e"  # Red

def create_stock_chart(ticker, period="1y"):
    """Create a price chart with SMA lines for a stock - fixed moving averages"""
    try:
        stock = yf.Ticker(ticker)
        # Use longer period to ensure we have enough data for SMA200
        df = stock.history(period="2y", interval="1d")
        
        if df.empty or len(df) < 50:
            return None
            
        # Calculate SMAs using simple rolling mean with proper NaN handling
        df['SMA50'] = df['Close'].rolling(window=50, min_periods=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200, min_periods=200).mean()
        
        # Filter to requested period for display but keep SMA calculations
        if period == "6mo":
            display_df = df.tail(126)  # ~6 months of trading days
        elif period == "1y":
            display_df = df.tail(252)  # ~1 year of trading days
        else:
            display_df = df
        
        # Create figure with dark theme to match original
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=display_df.index,
            open=display_df['Open'],
            high=display_df['High'],
            low=display_df['Low'],
            close=display_df['Close'],
            name='Price',
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff0000',
            showlegend=False
        ))
        
        # Add SMA50 line (blue) - only if we have valid data
        sma50_data = display_df['SMA50'].dropna()
        if len(sma50_data) > 0:
            fig.add_trace(go.Scatter(
                x=sma50_data.index,
                y=sma50_data.values,
                mode='lines',
                name='SMA50',
                line=dict(color='#00bfff', width=2),
                showlegend=True
            ))
        
        # Add SMA200 line (yellow) - only if we have valid data
        sma200_data = display_df['SMA200'].dropna()
        if len(sma200_data) > 0:
            fig.add_trace(go.Scatter(
                x=sma200_data.index,
                y=sma200_data.values,
                mode='lines',
                name='SMA200',
                line=dict(color='#ffff00', width=2),
                showlegend=True
            ))
        
        # Update layout to match original dark theme
        fig.update_layout(
            title=f'{ticker} Price Chart',
            xaxis_title='',
            yaxis_title='',
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(size=10, color='white'),
                bgcolor='rgba(0,0,0,0)'
            ),
            plot_bgcolor='#2d3748',
            paper_bgcolor='#2d3748',
            font=dict(size=10, color='white'),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(255,255,255,0.2)',
                tickfont=dict(color='white'),
                showline=False,
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(255,255,255,0.2)',
                tickfont=dict(color='white'),
                showline=False,
                zeroline=False
            )
        )
        
        # Remove range slider
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        return fig
        
    except Exception as e:
        print(f"Error creating chart for {ticker}: {e}")
        return None

# Custom CSS for Zapwiser theme
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --primary-color: #3182ce;
                --secondary-color: #4a5568;
                --success-color: #38a169;
                --warning-color: #e9b949;
                --danger-color: #e53e3e;
                --light-color: #f7fafc;
                --dark-color: #1a202c;
                --text-color: #2d3748;
                --muted-color: #718096;
            }
            body {
                font-family: 'Montserrat', 'Poppins', sans-serif;
                background-color: var(--light-color);
                color: var(--text-color);
            }
            .navbar {
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 1rem 0;
            }
            .logo img {
                height: 40px;
                margin-right: 10px;
            }
            .logo-text {
                font-size: 1.5rem;
                font-weight: 700;
                color: var(--primary-color);
            }
            .btn-primary {
                background-color: var(--primary-color);
                border-color: var(--primary-color);
            }
            .btn-primary:hover {
                background-color: #2a6cb0;
                border-color: #2a6cb0;
            }
            .card {
                border: none;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                border-radius: 8px;
            }
            .footer {
                background-color: var(--dark-color);
                color: white;
                padding: 3rem 0 1rem;
                margin-top: auto;
            }
        </style>
        {%scripts%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Simple layout with Zapwiser branding
app.layout = html.Div([
    # Header
    html.Header([
        html.Div([
            html.Nav([
                html.Div([
                    html.Span("Zapwiser", className="logo-text")
                ], className="logo d-flex align-items-center"),
                html.Div([
                    html.A("Home", href="https://zapwiser.com/", className="nav-link me-3", target="_blank"),
                    html.A("Tools", href="https://zapwiser.com/tools.html", className="nav-link me-3", target="_blank"),
                    html.A("Insights", href="https://zapwiser.com/blog.html", className="nav-link", target="_blank")
                ], className="nav-links d-flex")
            ], className="navbar d-flex justify-content-between align-items-center")
        ], className="container")
    ]),
    
    # Main content
    html.Div([
        dbc.Container([
            html.H1("Stock Screener", className="text-center my-4"),
            html.P("Find potential swing trading opportunities with technical analysis and our comprehensive QVM scoring system.", 
                   className="text-center mb-4 lead"),
            
            # Info Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("What is this scanner?", className="text-info"),
                            html.P("This dashboard identifies potential swing trading opportunities by scanning stocks against four key technical criteria and ranks them using our comprehensive QVM scoring system:"),
                            
                            # Swing Trading Criteria
                            html.H5("🎯 Swing Trading Criteria", className="text-warning mt-4 mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H5("📈 Trend", className="text-center text-success"),
                                            html.P("Price > 50 SMA > 200 SMA", className="text-center fw-bold"),
                                            html.P("Confirms upward momentum", className="text-center text-muted small")
                                        ])
                                    ], className="h-100")
                                ], md=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H5("📊 Volume", className="text-center text-success"),
                                            html.P("Volume > 1.5x average", className="text-center fw-bold"),
                                            html.P("Shows increased interest", className="text-center text-muted small")
                                        ])
                                    ], className="h-100")
                                ], md=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H5("📉 Volatility", className="text-center text-success"),
                                            html.P("ATR/Price > 2%", className="text-center fw-bold"),
                                            html.P("Indicates consolidation", className="text-center text-muted small")
                                        ])
                                    ], className="h-100")
                                ], md=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H5("🔥 RSI", className="text-center text-success"),
                                            html.P("30 < RSI < 70", className="text-center fw-bold"),
                                            html.P("Room to move up", className="text-center text-muted small")
                                        ])
                                    ], className="h-100")
                                ], md=3),
                            ], className="mb-3"),
                            dbc.Alert("💡 Stocks passing all four criteria are highlighted and may present favorable swing trading setups.", color="success"),
                            
                            # QVM Scoring System
                            html.H5("📊 QVM Ranking System", className="text-warning mt-4 mb-3"),
                            html.P("Beyond swing criteria, we rank stocks using our Quality-Value-Momentum (QVM) scoring system to help you identify the best investment opportunities:"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H5("🏆 Quality", className="text-center text-info"),
                                            html.P("Business Strength", className="text-center fw-bold"),
                                            html.Ul([
                                                html.Li("Return on Equity (ROE)"),
                                                html.Li("Operating Margins"),
                                                html.Li("Revenue Growth")
                                            ], className="small"),
                                            html.P("Measures how well the company is run", className="text-center text-muted small")
                                        ])
                                    ], className="h-100")
                                ], md=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H5("💰 Value", className="text-center text-success"),
                                            html.P("Price Attractiveness", className="text-center fw-bold"),
                                            html.Ul([
                                                html.Li("Price-to-Earnings (P/E)"),
                                                html.Li("Price-to-Book (P/B)"),
                                                html.Li("Dividend Yield")
                                            ], className="small"),
                                            html.P("Identifies if stock is reasonably priced", className="text-center text-muted small")
                                        ])
                                    ], className="h-100")
                                ], md=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H5("🚀 Momentum", className="text-center text-warning"),
                                            html.P("Price Movement", className="text-center fw-bold"),
                                            html.Ul([
                                                html.Li("1, 3, 6-Month Performance"),
                                                html.Li("RSI (Overbought/Oversold)"),
                                                html.Li("Technical Strength")
                                            ], className="small"),
                                            html.P("Shows recent price direction and strength", className="text-center text-muted small")
                                        ])
                                    ], className="h-100")
                                ], md=4),
                            ], className="mb-3"),
                            dbc.Alert([
                                html.H6("How to Read QVM Scores:", className="alert-heading mb-2"),
                                html.P([
                                    html.Strong("90-100: Exceptional"), " • ",
                                    html.Strong("70-89: Very Good", style={"color": "#38a169"}), " • ",
                                    html.Strong("50-69: Average", style={"color": "#e9b949"}), " • ",
                                    html.Strong("Below 50: Below Average", style={"color": "#e53e3e"})
                                ], className="mb-2"),
                                html.P("Higher scores indicate better quality companies, better value opportunities, or stronger momentum. Use QVM rankings to prioritize which stocks to research further!", className="mb-0 small")
                            ], color="info")
                        ])
                    ])
                ], className="mb-4")
            ]),
            
            # Watchlist Management - Full Width with Better Input
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Watchlist Management", className="mb-3"),
                            # Wider input section
                            dbc.Row([
                                dbc.Col([
                                    dbc.Input(
                                        id="ticker-input", 
                                        placeholder="Enter stock ticker (e.g., AAPL, MSFT, TSLA)", 
                                        value="", 
                                        size="lg",
                                        style={
                                            "fontSize": "18px", 
                                            "padding": "15px 20px",
                                            "borderRadius": "8px",
                                            "border": "2px solid #e2e8f0",
                                            "marginBottom": "15px"
                                        }
                                    )
                                ], md=8),
                                dbc.Col([
                                    dbc.Button(
                                        "Add to Watchlist", 
                                        id="add-ticker-btn", 
                                        color="primary", 
                                        size="lg",
                                        className="w-100",
                                        style={
                                            "fontSize": "16px", 
                                            "fontWeight": "bold",
                                            "padding": "15px",
                                            "borderRadius": "8px"
                                        }
                                    )
                                ], md=4)
                            ]),
                            html.Div(id="watchlist-display", style={"minHeight": "60px"}),
                            dcc.Store(id="watchlist-store", data=DEFAULT_WATCHLIST)
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Settings Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Scan Settings"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Volume Multiplier", className="fw-bold"),
                                    dcc.Slider(
                                        id="volume-multiplier",
                                        min=1.0, max=3.0, step=0.1, value=1.5,
                                        marks={1.0: '1.0x', 1.5: '1.5x', 2.0: '2.0x', 2.5: '2.5x', 3.0: '3.0x'},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("ATR Threshold (%)", className="fw-bold"),
                                    dcc.Slider(
                                        id="atr-threshold",
                                        min=1, max=5, step=0.5, value=2,
                                        marks={1: '1%', 2: '2%', 3: '3%', 4: '4%', 5: '5%'},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], md=6)
                            ])
                        ])
                    ])
                ], md=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Run Analysis"),
                            dbc.RadioItems(
                                id="view-mode",
                                options=[
                                    {"label": "🔍 Scan for the Swing Trading Criteria", "value": "cards"},
                                    {"label": "📈 QVM Ranking", "value": "qvm"}
                                ],
                                value="qvm",
                                className="mb-3"
                            ),
                            dbc.Button(
                                "🚀 Run Scan", 
                                id="scan-btn", 
                                color="success", 
                                className="w-100",
                                size="lg",
                                style={"fontSize": "16px", "fontWeight": "bold"}
                            )
                        ])
                    ])
                ], md=4)
            ], className="mb-4"),
            
            # Results
            dcc.Loading([
                html.Div(id="scan-results")
            ], type="default")
        ])
    ], className="flex-grow-1"),
    
    # Footer
    html.Footer([
        html.Div([
            html.Div([
                html.H4("Zapwiser Stock Screener"),
                html.P("© 2025 Zapwiser. All rights reserved.")
            ], className="text-center py-4")
        ], className="container")
    ], className="footer mt-5")
], className="d-flex flex-column min-vh-100")

# Callback for watchlist management
@app.callback(
    [Output("watchlist-display", "children"),
     Output("watchlist-store", "data"),
     Output("ticker-input", "value")],
    [Input("add-ticker-btn", "n_clicks"),
     Input({"type": "remove-ticker", "index": ALL}, "n_clicks")],
    [State("ticker-input", "value"),
     State("watchlist-store", "data")]
)
def manage_watchlist(add_clicks, remove_clicks, ticker_input, watchlist):
    ctx = callback_context
    if not ctx.triggered:
        return create_watchlist_display(watchlist), watchlist, ""
    
    trigger_id = ctx.triggered[0]["prop_id"]
    
    if "add-ticker-btn" in trigger_id and ticker_input:
        ticker = ticker_input.upper().strip()
        if ticker and ticker not in watchlist:
            watchlist.append(ticker)
    elif "remove-ticker" in trigger_id:
        button_info = json.loads(trigger_id.split(".")[0])
        ticker_to_remove = button_info["index"]
        if ticker_to_remove in watchlist:
            watchlist.remove(ticker_to_remove)
    
    return create_watchlist_display(watchlist), watchlist, ""

def create_watchlist_display(watchlist):
    badges = []
    for ticker in watchlist:
        badge = dbc.Badge([
            ticker,
            html.Span(" ×", 
                     id={"type": "remove-ticker", "index": ticker},
                     style={"cursor": "pointer", "marginLeft": "8px", "fontWeight": "bold"})
        ], color="primary", className="me-2 mb-2", style={"fontSize": "14px", "padding": "8px 12px"})
        badges.append(badge)
    
    if not badges:
        return html.P("No stocks in watchlist. Add some tickers above to get started!", 
                     className="text-muted text-center mt-3 mb-3")
    
    return html.Div([
        html.P(f"Current Watchlist ({len(watchlist)} stocks):", className="text-muted small mb-2"),
        html.Div(badges, style={"display": "flex", "flexWrap": "wrap", "gap": "8px"})
    ])

# Enhanced callback for scan results with QVM scoring
@app.callback(
    Output("scan-results", "children"),
    [Input("scan-btn", "n_clicks")],
    [State("watchlist-store", "data"),
     State("volume-multiplier", "value"),
     State("atr-threshold", "value"),
     State("view-mode", "value")]
)
def run_enhanced_scan(n_clicks, watchlist, vol_mult, atr_thresh, view_mode):
    if not n_clicks or not watchlist:
        return html.Div()
    
    all_data = []
    for ticker in watchlist:
        data = fetch_enhanced_stock_data(ticker)
        if data:
            swing_checks = check_swing_criteria(data, vol_mult, atr_thresh/100)
            qvm_scores = calculate_qvm_scores(data)
            data.update(qvm_scores)
            all_data.append({'data': data, 'checks': swing_checks})
    
    if not all_data:
        return html.Div("No data available", className="text-center mt-4")
    
    if view_mode == "qvm":
        return create_qvm_ranking_view(all_data)
    else:
        return create_enhanced_cards_view(all_data)

def create_enhanced_cards_view(all_data):
    """Create enhanced stock cards with technical indicators and QVM scores"""
    cards = []
    
    for item in all_data:
        data = item['data']
        checks = item['checks']
        ticker = data['ticker']
        
        # Determine card color based on swing criteria
        passes_all = all(checks.values())
        card_color = "success" if passes_all else "light"
        
        # Create swing criteria indicators
        criteria_badges = []
        for criterion, passed in checks.items():
            color = "success" if passed else "secondary"
            icon = "✓" if passed else "✗"
            criteria_badges.append(
                dbc.Badge(f"{icon} {criterion.title()}", color=color, className="me-1 mb-1")
            )
        
        card = dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H4(ticker, className="text-primary mb-0"),
                            html.H5(f"${data['price']:.2f}", className="text-success")
                        ], width=6),
                        dbc.Col([
                            html.Div([
                                html.Small("QVM Score", className="text-muted"),
                                html.H5(
                                    f"{data['qvm_score']:.0f}",
                                    style={"color": get_score_color(data['qvm_score'])}
                                )
                            ], className="text-end")
                        ], width=6)
                    ]),
                    html.Hr(),
                    
                    # Swing Trading Criteria
                    html.H6("Swing Criteria", className="text-warning mb-2"),
                    html.Div(criteria_badges, className="mb-3"),
                    
                    # Technical Indicators
                    html.H6("Technical Indicators", className="text-info mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Small("RSI", className="text-muted"),
                            html.P(f"{data['rsi']:.1f}", className="mb-1")
                        ], width=4),
                        dbc.Col([
                            html.Small("SMA50", className="text-muted"),
                            html.P(f"${data['sma50']:.2f}", className="mb-1")
                        ], width=4),
                        dbc.Col([
                            html.Small("Volume", className="text-muted"),
                            html.P(f"{data['volume']:,}", className="mb-1")
                        ], width=4)
                    ]),
                    
                    # Price Chart
                    dbc.Accordion([
                        dbc.AccordionItem([
                            dcc.Graph(
                                figure=create_stock_chart(ticker) or {},
                                config={'displayModeBar': False}
                            ) if create_stock_chart(ticker) else html.Div("Chart not available", className="text-center text-muted p-3")
                        ], title="Price Chart", item_id=f"chart-{ticker}")
                    ], start_collapsed=True, className="mb-3"),
                    
                    # QVM Breakdown
                    html.H6("QVM Breakdown", className="text-info mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Small("Quality", className="text-muted"),
                            html.P(
                                f"{data['quality_score']:.0f}",
                                style={"color": get_score_color(data['quality_score'])}
                            )
                        ], width=4),
                        dbc.Col([
                            html.Small("Value", className="text-muted"),
                            html.P(
                                f"{data['value_score']:.0f}",
                                style={"color": get_score_color(data['value_score'])}
                            )
                        ], width=4),
                        dbc.Col([
                            html.Small("Momentum", className="text-muted"),
                            html.P(
                                f"{data['momentum_score']:.0f}",
                                style={"color": get_score_color(data['momentum_score'])}
                            )
                        ], width=4)
                    ])
                ])
            ], color=card_color, outline=True, className="mb-3")
        ], md=6, lg=4)
        
        cards.append(card)
    
    return dbc.Row(cards)

def create_progress_bar(value, max_value=100, color="#3182ce"):
    """Create a custom progress bar"""
    return html.Div([
        html.Div(
            style={
                "width": f"{(value/max_value)*100}%",
                "height": "8px",
                "backgroundColor": color,
                "borderRadius": "4px",
                "transition": "width 0.3s ease"
            }
        )
    ], style={
        "width": "100%",
        "height": "8px",
        "backgroundColor": "rgba(255,255,255,0.1)",
        "borderRadius": "4px",
        "overflow": "hidden"
    })

def get_rank_border_color(rank):
    """Get border color based on rank"""
    if rank == 1:
        return "#ffd700"  # Gold
    elif rank == 2:
        return "#c0c0c0"  # Silver
    elif rank == 3:
        return "#cd7f32"  # Bronze
    else:
        return "#4a5568"  # Default gray

def create_qvm_ranking_view(all_data):
    """Create QVM ranking card-based view matching the original style"""
    # Sort by QVM score
    sorted_data = sorted(all_data, key=lambda x: x['data']['qvm_score'], reverse=True)
    
    if not sorted_data:
        return html.Div("No data available", className="text-center mt-4")
    
    # Calculate summary statistics
    avg_qvm = sum(item['data']['qvm_score'] for item in sorted_data) / len(sorted_data)
    best_quality = max(sorted_data, key=lambda x: x['data']['quality_score'])
    best_value = max(sorted_data, key=lambda x: x['data']['value_score'])
    best_momentum = max(sorted_data, key=lambda x: x['data']['momentum_score'])
    
    # Create QVM Summary header
    qvm_summary = dbc.Card([
        dbc.CardBody([
            html.H4("QVM Summary", className="text-white mb-3"),
            dbc.Row([
                dbc.Col([
                    html.P(f"Average QVM Score: {avg_qvm:.1f}", className="text-white mb-0", style={"fontSize": "16px"})
                ], md=3),
                dbc.Col([
                    html.P(f"Best Quality: {best_quality['data']['ticker']} ({best_quality['data']['quality_score']:.0f})", 
                           className="text-white mb-0", style={"fontSize": "16px"})
                ], md=3),
                dbc.Col([
                    html.P(f"Best Value: {best_value['data']['ticker']} ({best_value['data']['value_score']:.0f})", 
                           className="text-white mb-0", style={"fontSize": "16px"})
                ], md=3),
                dbc.Col([
                    html.P(f"Best Momentum: {best_momentum['data']['ticker']} ({best_momentum['data']['momentum_score']:.0f})", 
                           className="text-white mb-0", style={"fontSize": "16px"})
                ], md=3)
            ])
        ])
    ], style={"backgroundColor": "#6c9bd1", "border": "none", "marginBottom": "20px"})
    
    # Create ranking cards
    ranking_cards = []
    for rank, item in enumerate(sorted_data, 1):
        data = item['data']
        checks = item['checks']
        passes_all = all(checks.values())
        
        # Determine QVM score color
        qvm_color = get_score_color(data['qvm_score'])
        border_color = get_rank_border_color(rank)
        
        # Create the ranking card
        card = dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    # Header with rank, ticker, price, and QVM score
                    dbc.Row([
                        dbc.Col([
                            html.H5(f"#{rank}", className="text-white mb-0", style={"fontSize": "18px"})
                        ], width=1),
                        dbc.Col([
                            html.H4(data['ticker'], className="text-white mb-0", style={"fontWeight": "bold"}),
                            html.P(f"${data['price']:.2f}", className="text-white mb-0", style={"fontSize": "16px"})
                        ], width=6),
                        dbc.Col([
                            html.Div([
                                html.H2(f"{data['qvm_score']:.0f}", 
                                        className="mb-0", 
                                        style={"color": qvm_color, "fontWeight": "bold", "fontSize": "36px"}),
                                html.P("QVM Score", className="text-white mb-0", style={"fontSize": "12px"})
                            ], className="text-end")
                        ], width=5)
                    ], className="mb-3"),
                    
                    # Progress bars for Quality, Value, Momentum
                    html.Div([
                        # Quality
                        dbc.Row([
                            dbc.Col([html.P("Quality", className="text-white mb-1", style={"fontSize": "14px"})], width=3),
                            dbc.Col([create_progress_bar(data['quality_score'], color="#3182ce")], width=7),
                            dbc.Col([html.P(f"{data['quality_score']:.1f}", className="text-white mb-1 text-end", style={"fontSize": "14px"})], width=2)
                        ], className="mb-2"),
                        
                        # Value
                        dbc.Row([
                            dbc.Col([html.P("Value", className="text-white mb-1", style={"fontSize": "14px"})], width=3),
                            dbc.Col([create_progress_bar(data['value_score'], color="#38a169")], width=7),
                            dbc.Col([html.P(f"{data['value_score']:.1f}", className="text-white mb-1 text-end", style={"fontSize": "14px"})], width=2)
                        ], className="mb-2"),
                        
                        # Momentum
                        dbc.Row([
                            dbc.Col([html.P("Momentum", className="text-white mb-1", style={"fontSize": "14px"})], width=3),
                            dbc.Col([create_progress_bar(data['momentum_score'], color="#e9b949")], width=7),
                            dbc.Col([html.P(f"{data['momentum_score']:.1f}", className="text-white mb-1 text-end", style={"fontSize": "14px"})], width=2)
                        ], className="mb-3")
                    ]),
                    
                    # Swing Setup status
                    html.P(f"Swing Setup: {'✓ All Criteria Met' if passes_all else '✗ Not All Criteria Met'}", 
                           className="text-white mb-3", style={"fontSize": "14px"}),
                    
                    # Detailed Metrics (collapsible)
                    dbc.Accordion([
                        dbc.AccordionItem([
                            # Financial metrics in 3 columns
                            dbc.Row([
                                dbc.Col([
                                    html.P("ROE", className="text-white mb-1", style={"fontSize": "12px"}),
                                    html.P(f"{data['roe']:.1f}%" if data['roe'] else "N/A", 
                                           className="text-white mb-2", style={"fontSize": "14px", "fontWeight": "bold"}),
                                    html.P("P/E", className="text-white mb-1", style={"fontSize": "12px"}),
                                    html.P(f"{data['pe_ratio']:.1f}" if data['pe_ratio'] else "N/A", 
                                           className="text-white mb-2", style={"fontSize": "14px", "fontWeight": "bold"})
                                ], width=4),
                                dbc.Col([
                                    html.P("Op. Margin", className="text-white mb-1", style={"fontSize": "12px"}),
                                    html.P(f"{data['operating_margin']:.1f}%" if data['operating_margin'] else "N/A", 
                                           className="text-white mb-2", style={"fontSize": "14px", "fontWeight": "bold"}),
                                    html.P("P/B", className="text-white mb-1", style={"fontSize": "12px"}),
                                    html.P(f"{data['pb_ratio']:.1f}" if data['pb_ratio'] else "N/A", 
                                           className="text-white mb-2", style={"fontSize": "14px", "fontWeight": "bold"})
                                ], width=4),
                                dbc.Col([
                                    html.Label("Run Analysis:", className="form-label text-white mb-2", style={"fontSize": "12px"}),
                                    html.P("16.1%", className="text-white mb-2", style={"fontSize": "14px", "fontWeight": "bold"}),
                                    html.P("Yield", className="text-white mb-1", style={"fontSize": "12px"}),
                                    html.P(f"{data['dividend_yield']:.1f}%" if data['dividend_yield'] else "N/A", 
                                           className="text-white mb-2", style={"fontSize": "14px", "fontWeight": "bold"})
                                ], width=4)
                            ]),
                            
                            # Performance section
                            html.Hr(style={"borderColor": "rgba(255,255,255,0.2)"}),
                            html.P("Price Performance", className="text-warning mb-2", style={"fontSize": "14px", "fontWeight": "bold"}),
                            dbc.Row([
                                dbc.Col([
                                    html.P("1M", className="text-white mb-1", style={"fontSize": "12px"}),
                                    html.P(f"{data['perf_1m']:+.1f}%" if data['perf_1m'] else "N/A", 
                                           className="text-success" if data['perf_1m'] and data['perf_1m'] > 0 else "text-danger", 
                                           style={"fontSize": "14px", "fontWeight": "bold"})
                                ], width=4),
                                dbc.Col([
                                    html.P("3M", className="text-white mb-1", style={"fontSize": "12px"}),
                                    html.P(f"{data['perf_3m']:+.1f}%" if data['perf_3m'] else "N/A", 
                                           className="text-success" if data['perf_3m'] and data['perf_3m'] > 0 else "text-danger", 
                                           style={"fontSize": "14px", "fontWeight": "bold"})
                                ], width=4),
                                dbc.Col([
                                    html.P("6M", className="text-white mb-1", style={"fontSize": "12px"}),
                                    html.P(f"{data['perf_6m']:+.1f}%" if data['perf_6m'] else "N/A", 
                                           className="text-success" if data['perf_6m'] and data['perf_6m'] > 0 else "text-danger", 
                                           style={"fontSize": "14px", "fontWeight": "bold"})
                                ], width=4)
                            ])
                        ], title="Detailed Metrics", style={"backgroundColor": "#2d3748", "border": "1px solid rgba(255,255,255,0.1)"})
                    ], start_collapsed=True, style={"backgroundColor": "transparent"})
                ])
            ], style={
                "backgroundColor": "#2d3748", 
                "border": f"2px solid {border_color}", 
                "borderRadius": "8px",
                "marginBottom": "15px"
            })
        ], md=6, lg=4)
        
        ranking_cards.append(card)
    
    return html.Div([
        qvm_summary,
        dbc.Row(ranking_cards)
    ])

# Expose server for deployment
server = app.server

if __name__ == "__main__":
    # Disable .env loading to avoid encoding issues
    import os
    os.environ['FLASK_SKIP_DOTENV'] = '1'
    
    # Get port from environment variable for deployment platforms
    port = int(os.environ.get('PORT', 8050))
    host = os.environ.get('HOST', '0.0.0.0')
    
    app.run(debug=False, host=host, port=port)
