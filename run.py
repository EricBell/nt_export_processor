import os
import typer
from typing import Optional
from backtest_engine import MESFuturesTrifectaBacktester
from visualization import create_html_report

def main(
    data_file: str = typer.Argument(..., help="Path to your 3-minute OHLCV data file (CSV format)"),
    output_dir: str = typer.Option("./backtest_results", "--output", "-o", 
                                  help="Output directory for backtest results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """
    Run MES Futures Trifecta Strategy Backtest
    
    Expected CSV format:
    - Required columns: timestamp, open, high, low, close, volume
    - Timestamp should be in datetime format
    """
    if verbose:
        typer.echo(f"Starting backtest with data file: {data_file}")
        typer.echo(f"Output directory: {output_dir}")
    
    # Check if data file exists
    if not os.path.exists(data_file):
        typer.echo(f"❌ Error: Data file '{data_file}' not found.", err=True)
        raise typer.Exit(1)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            typer.echo(f"Created output directory: {output_dir}")
    
    # Run backtest
    backtest = MESFuturesTrifectaBacktester(data_file, output_dir)
    success = backtest.run_backtest()
    
    if success:
        # Create and open HTML report
        html_report = create_html_report(output_dir)
        typer.echo(f"✅ Backtest completed successfully!")
        typer.echo(f"📊 Report saved to: {html_report}")
        typer.echo(f"📁 Results directory: {output_dir}")
    else:
        typer.echo("❌ Backtest failed. Check logs for details.", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    typer.run(main)