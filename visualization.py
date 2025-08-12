import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

def create_html_report(backtest_results_dir, output_file=None):
    """
    Create an HTML report with interactive charts
    
    Args:
        backtest_results_dir: Directory containing backtest results
        output_file: Output HTML file path
    """
    # Set default output file if not provided
    if output_file is None:
        output_file = os.path.join(backtest_results_dir, 'backtest_report.html')
    
    # Load data
    trades_path = os.path.join(backtest_results_dir, 'trades.csv')
    equity_path = os.path.join(backtest_results_dir, 'equity_curve.csv')
    metrics_path = os.path.join(backtest_results_dir, 'metrics.json')
    
    trades_df = pd.read_csv(trades_path) if os.path.exists(trades_path) else pd.DataFrame()
    equity_df = pd.read_csv(equity_path) if os.path.exists(equity_path) else pd.DataFrame()
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f) if os.path.exists(metrics_path) else {}
    
    # Convert timestamps to datetime
    if not trades_df.empty:
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    
    if not equity_df.empty:
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    
    # Create figure with equity curve and trades
    fig = create_dashboard(trades_df, equity_df, metrics)
    
    # Write to HTML file
    with open(output_file, 'w') as f:
        f.write(fig.to_html(include_plotlyjs=True, full_html=True))
    
    print(f"HTML report created: {output_file}")
    return output_file

def create_dashboard(trades_df, equity_df, metrics):
    """Create an interactive dashboard with Plotly"""
    # Create subplot structure
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"type": "pie"}, {"type": "table"}]
        ],
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=("Equity Curve", "Trade P&L", "Win/Loss Ratio", "Trade Details")
    )
    
    # Add equity curve
    if not equity_df.empty:
        fig.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='rgba(0, 128, 255, 0.8)', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 128, 255, 0.2)'
            ),
            row=1, col=1
        )
        
        # Add position markers
        long_entries = equity_df[equity_df['position'] > 0]
        short_entries = equity_df[equity_df['position'] < 0]
        trade_exits = equity_df[equity_df['trade_pl'] != 0]
        
        # Long entry markers
        if not long_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=long_entries['timestamp'],
                    y=long_entries['equity'],
                    mode='markers',
                    name='Long Entry',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ),
                row=1, col=1
            )
        
        # Short entry markers
        if not short_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=short_entries['timestamp'],
                    y=short_entries['equity'],
                    mode='markers',
                    name='Short Entry',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ),
                row=1, col=1
            )
        
        # Trade exit markers
        if not trade_exits.empty:
            # Color based on profit/loss
            colors = ['green' if pl > 0 else 'red' for pl in trade_exits['trade_pl']]
            fig.add_trace(
                go.Scatter(
                    x=trade_exits['timestamp'],
                    y=trade_exits['equity'],
                    mode='markers',
                    name='Trade Exit',
                    marker=dict(color=colors, size=