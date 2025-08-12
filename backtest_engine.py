import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MESFuturesTrifectaBacktester:
    def __init__(self, data_path, output_dir="./backtest_results"):
        """Initialize the backtester with data path and parameters"""
        self.data_path = data_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Strategy parameters (Trifecta strategy)
        self.params = {
            "name": "MES Futures Trifecta Strategy",
            "instrument": "MES",
            "timeframe": "3min",
            "initial_capital": 10000.0,
            "indicators": {
                "tradeline": {"type": "EMA", "period": 9, "slope_bars": 3},
                "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "money_flow": {"type": "CMF", "period": 20},
                "volume": {"sma_period": 20, "min_threshold": 1.25},
                "vwap": {"reset_on_session": True},
                "atr": {"period": 14}
            },
            "entry_rules": {
                "long": {
                    "tradeline_slope": "positive",
                    "macd_histogram": "positive",
                    "cmf": "positive",
                    "price_vs_vwap": "above",
                    "volume_filter": True,
                    "stagger_window": 3
                },
                "short": {
                    "tradeline_slope": "negative",
                    "macd_histogram": "negative",
                    "cmf": "negative",
                    "price_vs_vwap": "below",
                    "volume_filter": True,
                    "stagger_window": 3
                }
            },
            "risk_management": {
                "stop_loss": {"type": "ATR", "multiplier": 1.0},
                "take_profit": {"type": "ATR", "multiplier": 1.0},
                "position_sizing": {
                    "risk_per_trade": 50.0,
                    "max_contracts": 10
                }
            },
            "execution_settings": {
                "slippage_per_leg": 0.25,
                "commission_per_contract": 1.0
            }
        }
        
        # MES contract specifications
        self.contract_specs = {
            "tick_size": 0.25,  # 0.25 index points
            "tick_value": 1.25,  # $1.25 per tick
            "contract_size": 5.0  # $5 per point
        }
        
        # Initialize state variables
        self.data = None
        self.trades = []
        self.equity_curve = []
        self.current_capital = self.params["initial_capital"]
        self.metrics = {}
    
    def load_data(self):
        """Load and prepare the 3-min MES futures data"""
        print(f"Loading data from {self.data_path}...")
        
        try:
            # Load data from CSV
            df = pd.read_csv(self.data_path)
            
            # Handle different timestamp column names
            timestamp_col = None
            for col_name in ['timestamp', 'DateTime', 'datetime', 'Timestamp']:
                if col_name in df.columns:
                    timestamp_col = col_name
                    break
            
            if timestamp_col is None:
                raise ValueError("CSV must have a timestamp/DateTime column")
            
            # Rename to standard 'timestamp' column
            if timestamp_col != 'timestamp':
                df = df.rename(columns={timestamp_col: 'timestamp'})
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Handle different OHLCV column name formats
            column_mapping = {}
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            
            for required_col in required_cols:
                found_col = None
                # Check both lowercase and capitalized versions
                for col_variant in [required_col, required_col.capitalize(), required_col.upper()]:
                    if col_variant in df.columns:
                        found_col = col_variant
                        break
                
                if found_col is None:
                    raise ValueError(f"CSV must have {required_col} column (tried: {required_col}, {required_col.capitalize()}, {required_col.upper()})")
                
                # Map to lowercase standard
                if found_col != required_col:
                    column_mapping[found_col] = required_col
            
            # Rename columns to standard lowercase format
            if column_mapping:
                df = df.rename(columns=column_mapping)
            
            # Extract session date for VWAP calculations
            df['date'] = df['timestamp'].dt.date
            
            print(f"Loaded {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")
            self.data = df
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def calculate_indicators(self):
        """Calculate all indicators required for the strategy"""
        print("Calculating indicators...")
        df = self.data
        
        # Calculate EMA9 (tradeline)
        df['ema9'] = df['close'].ewm(span=self.params['indicators']['tradeline']['period'], adjust=False).mean()
        
        # Calculate EMA slope over N bars
        slope_bars = self.params['indicators']['tradeline']['slope_bars']
        df['ema9_slope'] = df['ema9'] - df['ema9'].shift(slope_bars)
        df['ema9_slope_dir'] = df['ema9_slope'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'flat'))
        
        # Calculate MACD
        fast = self.params['indicators']['macd']['fast_period']
        slow = self.params['indicators']['macd']['slow_period']
        signal = self.params['indicators']['macd']['signal_period']
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_hist_dir'] = df['macd_hist'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'flat'))
        
        # Calculate Money Flow (CMF)
        cmf_period = self.params['indicators']['money_flow']['period']
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['money_flow'] = df['typical_price'] * df['volume']
        
        # Calculate money flow volume
        df['mfv'] = df.apply(lambda row: 
                          row['money_flow'] if row['typical_price'] > row['typical_price'].shift(1) else
                          (-row['money_flow'] if row['typical_price'] < row['typical_price'].shift(1) else 0), 
                          axis=1)
        
        # Calculate CMF
        df['cmf'] = df['mfv'].rolling(window=cmf_period).sum() / df['volume'].rolling(window=cmf_period).sum()
        df['cmf_dir'] = df['cmf'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'flat'))
        
        # Calculate Volume SMA
        vol_period = self.params['indicators']['volume']['sma_period']
        vol_threshold = self.params['indicators']['volume']['min_threshold']
        df['vol_sma'] = df['volume'].rolling(window=vol_period).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma']
        df['vol_filter'] = df['vol_ratio'] >= vol_threshold
        
        # Calculate VWAP
        if self.params['indicators']['vwap']['reset_on_session']:
            # Reset VWAP each session (day)
            df['pv'] = df['typical_price'] * df['volume']
            df['cum_pv'] = df.groupby('date')['pv'].cumsum()
            df['cum_vol'] = df.groupby('date')['volume'].cumsum()
            df['vwap'] = df['cum_pv'] / df['cum_vol']
        else:
            # Continuous VWAP
            df['pv'] = df['typical_price'] * df['volume']
            df['cum_pv'] = df['pv'].cumsum()
            df['cum_vol'] = df['volume'].cumsum()
            df['vwap'] = df['cum_pv'] / df['cum_vol']
        
        df['price_vs_vwap'] = df.apply(lambda x: 'above' if x['close'] > x['vwap'] else 
                                      ('below' if x['close'] < x['vwap'] else 'equal'), axis=1)
        
        # Calculate ATR for stop loss and take profit
        atr_period = self.params['indicators']['atr']['period']
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=atr_period).mean()
        
        # Flag potential entry points based on rules
        # Long signals
        df['long_signal'] = (
            (df['ema9_slope_dir'] == 'positive') & 
            (df['macd_hist_dir'] == 'positive') & 
            (df['cmf_dir'] == 'positive') & 
            (df['price_vs_vwap'] == 'above') & 
            (df['vol_filter'] == True)
        )
        
        # Short signals
        df['short_signal'] = (
            (df['ema9_slope_dir'] == 'negative') & 
            (df['macd_hist_dir'] == 'negative') & 
            (df['cmf_dir'] == 'negative') & 
            (df['price_vs_vwap'] == 'below') & 
            (df['vol_filter'] == True)
        )
        
        # Save the updated dataframe
        self.data = df
        print("Indicators calculated successfully")
    
    def run_backtest(self):
        """Run the backtest simulation"""
        print("Running backtest...")
        
        # Load and prepare data
        if not self.load_data():
            return False
        
        # Calculate indicators
        self.calculate_indicators()
        
        # Initialize results
        self.equity_curve = [self.params['initial_capital']]
        self.current_capital = self.params['initial_capital']
        self.trades = []
        
        # Initialize position tracking
        current_position = None
        last_entry_idx = None
        
        # Get ATR multipliers for stop loss and take profit
        sl_multiplier = self.params['risk_management']['stop_loss']['multiplier']
        tp_multiplier = self.params['risk_management']['take_profit']['multiplier']
        
        # Get risk per trade in dollars
        risk_per_trade = self.params['risk_management']['position_sizing']['risk_per_trade']
        max_contracts = self.params['risk_management']['position_sizing']['max_contracts']
        
        # Simulation parameters
        slippage = self.params['execution_settings']['slippage_per_leg']
        commission = self.params['execution_settings']['commission_per_contract']
        
        # Setup tracking columns
        self.data['position'] = 0
        self.data['equity'] = self.params['initial_capital']
        self.data['trade_pl'] = 0
        
        # Warmup period
        warmup_bars = max(
            self.params['indicators']['tradeline']['period'],
            self.params['indicators']['macd']['slow_period'] + self.params['indicators']['macd']['signal_period'],
            self.params['indicators']['money_flow']['period'],
            self.params['indicators']['volume']['sma_period'],
            self.params['indicators']['atr']['period']
        )
        
        print(f"Using {warmup_bars} warmup bars for indicators")
        
        # Process each bar
        for i in range(warmup_bars, len(self.data)):
            current_bar = self.data.iloc[i]
            
            # Skip if NaN indicators
            if pd.isna(current_bar['ema9']) or pd.isna(current_bar['macd_hist']) or \
               pd.isna(current_bar['cmf']) or pd.isna(current_bar['atr']):
                continue
                
            # Process existing position
            if current_position is not None:
                self._process_position(current_position, current_bar, i)
                # Check if position was closed
                if current_position.get('closed', False):
                    current_position = None
            
            # Check for new entry if no position
            if current_position is None and (last_entry_idx is None or i - last_entry_idx > self.params['entry_rules']['long']['stagger_window']):
                if current_bar['long_signal']:
                    current_position = self._enter_position('LONG', current_bar, i, sl_multiplier, tp_multiplier, risk_per_trade, max_contracts)
                    last_entry_idx = i
                elif current_bar['short_signal']:
                    current_position = self._enter_position('SHORT', current_bar, i, sl_multiplier, tp_multiplier, risk_per_trade, max_contracts)
                    last_entry_idx = i
        
        # Close any open position at the end
        if current_position is not None:
            final_bar = self.data.iloc[-1]
            self._close_position(current_position, final_bar['close'], final_bar['timestamp'], 'End of Backtest', len(self.data)-1)
        
        # Calculate performance metrics
        self._calculate_metrics()
        
        # Save results
        self._save_results()
        
        print("Backtest completed successfully")
        return True
    
    def _process_position(self, position, current_bar, bar_idx):
        """Process an open position for stop loss and take profit"""
        slippage = self.params['execution_settings']['slippage_per_leg']
        
        # Update mark-to-market equity
        if position['direction'] == 'LONG':
            pos_value = position['contracts'] * (current_bar['close'] - position['entry_price']) * self.contract_specs['contract_size'] / self.contract_specs['tick_size'] * self.contract_specs['tick_value']
            
            # Check stop loss
            if current_bar['low'] <= position['stop_loss']:
                exit_price = max(position['stop_loss'] - slippage * self.contract_specs['tick_size'], current_bar['low'])
                self._close_position(position, exit_price, current_bar['timestamp'], 'Stop Loss', bar_idx)
                position['closed'] = True
                return
                
            # Check take profit
            if current_bar['high'] >= position['take_profit']:
                exit_price = min(position['take_profit'] + slippage * self.contract_specs['tick_size'], current_bar['high'])
                self._close_position(position, exit_price, current_bar['timestamp'], 'Take Profit', bar_idx)
                position['closed'] = True
                return
                
        else:  # SHORT
            pos_value = position['contracts'] * (position['entry_price'] - current_bar['close']) * self.contract_specs['contract_size'] / self.contract_specs['tick_size'] * self.contract_specs['tick_value']
            
            # Check stop loss
            if current_bar['high'] >= position['stop_loss']:
                exit_price = min(position['stop_loss'] + slippage * self.contract_specs['tick_size'], current_bar['high'])
                self._close_position(position, exit_price, current_bar['timestamp'], 'Stop Loss', bar_idx)
                position['closed'] = True
                return
                
            # Check take profit
            if current_bar['low'] <= position['take_profit']:
                exit_price = max(position['take_profit'] - slippage * self.contract_specs['tick_size'], current_bar['low'])
                self._close_position(position, exit_price, current_bar['timestamp'], 'Take Profit', bar_idx)
                position['closed'] = True
                return
        
        # Update equity
        self.data.iloc[bar_idx, self.data.columns.get_loc('equity')] = self.current_capital + pos_value
    
    def _enter_position(self, direction, current_bar, bar_idx, sl_multiplier, tp_multiplier, risk_per_trade, max_contracts):
        """Enter a new position"""
        slippage = self.params['execution_settings']['slippage_per_leg']
        commission = self.params['execution_settings']['commission_per_contract']
        
        # Calculate entry price with slippage
        if direction == 'LONG':
            entry_price = current_bar['close'] + slippage * self.contract_specs['tick_size']
            stop_loss = entry_price - sl_multiplier * current_bar['atr']
            take_profit = entry_price + tp_multiplier * current_bar['atr']
            
            # Calculate position size
            risk_per_contract = (entry_price - stop_loss) * self.contract_specs['contract_size'] / self.contract_specs['tick_size'] * self.contract_specs['tick_value']
        else:  # SHORT
            entry_price = current_bar['close'] - slippage * self.contract_specs['tick_size']
            stop_loss = entry_price + sl_multiplier * current_bar['atr']
            take_profit = entry_price - tp_multiplier * current_bar['atr']
            
            # Calculate position size
            risk_per_contract = (stop_loss - entry_price) * self.contract_specs['contract_size'] / self.contract_specs['tick_size'] * self.contract_specs['tick_value']
        
        # Ensure at least 1 contract, max by parameter
        contracts = min(max(1, int(risk_per_trade / risk_per_contract)), max_contracts)
        
        # Record entry
        position = {
            'direction': direction,
            'entry_time': current_bar['timestamp'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'contracts': contracts,
            'initial_risk': risk_per_trade
        }
        
        # Deduct commission
        self.current_capital -= commission * contracts
        
        # Update position in data
        self.data.iloc[bar_idx, self.data.columns.get_loc('position')] = contracts if direction == 'LONG' else -contracts
        
        # Log entry
        print(f"Enter {direction}: {current_bar['timestamp']} at {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}, {contracts} contracts")
        
        return position
    
    def _close_position(self, position, exit_price, exit_time, exit_reason, bar_idx):
        """Close a position and record the trade"""
        commission = self.params['execution_settings']['commission_per_contract']
        
        if position['direction'] == 'LONG':
            # Calculate P&L
            points_profit = exit_price - position['entry_price']
            dollar_profit = points_profit * position['contracts'] * self.contract_specs['contract_size'] / self.contract_specs['tick_size'] * self.contract_specs['tick_value']
        else:  # SHORT
            # Calculate P&L
            points_profit = position['entry_price'] - exit_price
            dollar_profit = points_profit * position['contracts'] * self.contract_specs['contract_size'] / self.contract_specs['tick_size'] * self.contract_specs['tick_value']
        
        dollar_profit -= commission * position['contracts']  # Subtract commission
        
        # Update capital
        self.current_capital += dollar_profit
        
        # Record trade
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'direction': position['direction'],
            'contracts': position['contracts'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'stop_loss': position['stop_loss'],
            'take_profit': position['take_profit'],
            'profit_loss': dollar_profit,
            'profit_loss_pct': dollar_profit / self.params['initial_capital'] * 100,
            'exit_reason': exit_reason
        }
        self.trades.append(trade)
        
        # Log position exit
        print(f"{exit_reason} - Exit {position['direction']}: {exit_time} at {exit_price:.2f}, {dollar_profit:.2f} profit, {position['contracts']} contracts")
        
        # Update equity curve
        self.equity_curve.append(self.current_capital)
        self.data.iloc[bar_idx, self.data.columns.get_loc('equity')] = self.current_capital
        self.data.iloc[bar_idx, self.data.columns.get_loc('trade_pl')] = dollar_profit
        self.data.iloc[bar_idx, self.data.columns.get_loc('position')] = 0
    
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            print("No trades executed in backtest")
            self.metrics = {"total_trades": 0, "net_profit": 0.0}
            return
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit_loss'] > 0])
        losing_trades = len(trades_df[trades_df['profit_loss'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        net_profit = sum(trades_df['profit_loss'])
        net_profit_pct = net_profit / self.params['initial_capital'] * 100
        
        # Win/loss metrics
        avg_win = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['profit_loss'] < 0]['profit_loss'].mean() if losing_trades > 0 else 0
        profit_factor = abs(trades_df[trades_df['profit_loss'] > 0]['profit_loss'].sum() / trades_df[trades_df['profit_loss'] < 0]['profit_loss'].sum()) if trades_df[trades_df['profit_loss'] < 0]['profit_loss'].sum() != 0 else float('inf')
        
        # Calculate drawdown
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.cummax()
        drawdown_series = (equity_series - running_max) / running_max * 100
        max_drawdown = abs(drawdown_series.min())
        
        # Store metrics
        self.metrics = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "net_profit": net_profit,
            "net_profit_pct": net_profit_pct,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown
        }
        
        # Print summary
        print(f"Performance Metrics:")
        print(f"Total Trades: {total_trades} (Win: {winning_trades}, Loss: {losing_trades})")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Net Profit: ${net_profit:.2f} ({net_profit_pct:.2f}%)")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
    
    def _save_results(self):
        """Save backtest results to files"""
        # Save trades to CSV
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_path = os.path.join(self.output_dir, 'trades.csv')
            trades_df.to_csv(trades_path, index=False)
            print(f"Trades saved to {trades_path}")
        
        # Save equity curve to CSV
        equity_df = pd.DataFrame({
            'timestamp': self.data['timestamp'],
            'equity': self.data['equity'],
            'position': self.data['position'],
            'trade_pl': self.data['trade_pl']
        })
        equity_path = os.path.join(self.output_dir, 'equity_curve.csv')
        equity_df.to_csv(equity_path, index=False)
        print(f"Equity curve saved to {equity_path}")
        
        # Save metrics to JSON
        if self.metrics:
            import json
            metrics_path = os.path.join(self.output_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            print(f"Metrics saved to {metrics_path}")