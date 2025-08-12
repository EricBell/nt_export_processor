import os
from backtest_engine import MESFuturesTrifectaBacktester
from visualization import create_html_report

def main():
    # Set paths
    data_path = "your_3min_data.csv"  # Replace with your 3-min data file
    output_dir = "./backtest_results"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run backtest
    backtest = MESFuturesTrifectaBacktester(data_path, output_dir)
    success = backtest.run_backtest()
    
    if success:
        # Create and open HTML report
        html_report = create_html_report(output_dir)
        print(f"Backtest completed successfully! Report saved to {html_report}")
    else:
        print("Backtest failed. Check logs for details.")

if __name__ == "__main__":
    main()