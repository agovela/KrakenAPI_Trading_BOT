import os
import time
import json
import hmac
import hashlib
import base64
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import logging
import ta
from dataclasses import dataclass
from enum import Enum
from PyQt5.QtWidgets import QSplitter, QHBoxLayout, QPushButton, QLabel, QTextEdit, QMessageBox
from PyQt5.QtCore import QTimer
import urllib.parse
import sys
import importlib
import importlib.util
import dynamic_strategy


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop-loss"
    TRAILING_STOP = "trailing-stop"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class TradingConfig:
    max_usd_amount: float
    base_currency: str = "USD"
    quote_currency: str = "XBT"
    trailing_stop_percentage: float = 2.0
    stop_loss_percentage: float = 5.0
    take_profit_percentage: float = 10.0
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    ma_fast_period: int = 9
    ma_slow_period: int = 21
    demo_mode: bool = False
    demo_balance: float = 10000.0  # Default demo balance
    concurrent_trades: int = 1     # Limit for concurrent trades
    max_trade_percentage: float = 15.0  # Maximum percentage of available balance to use per trade
    ecosystem: str = "USDC"  # Ecosystem for trading pairs (USDC, USD, BTC)
    strategy: str = "Breakout-Focused"  # Selected trading strategy

class DemoState:
    def __init__(self, initial_balance: float):
        self.balance = initial_balance
        self.positions: Dict[str, Dict] = {}  # pair -> {amount, entry_price}
        self.orders: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.start_time = datetime.now()
        self.initial_balance = initial_balance
        self.rlusd_balance = 0.0  # Realized USD profit

    def get_balance(self) -> Dict[str, float]:
        return {
            'ZUSD': self.balance,
            'XXBT': sum(pos['amount'] for pos in self.positions.values() if 'XBT' in pos['pair']),
            'XETH': sum(pos['amount'] for pos in self.positions.values() if 'ETH' in pos['pair']),
            'RLUSD': self.rlusd_balance
        }

    def add_trade(self, pair: str, side: str, amount: float, price: float):
        if side == 'buy':
            entry_price = price
        else:
            entry_price = self.positions[pair]['entry_price'] if pair in self.positions else price
        trade = {
            'timestamp': datetime.now(),
            'pair': pair,
            'side': side,
            'amount': amount,
            'price': price,
            'value': amount * price,
            'entry_price': entry_price
        }
        self.trade_history.append(trade)
        if side == 'buy':
            self.balance -= amount * price
            if pair in self.positions:
                self.positions[pair]['amount'] += amount
            else:
                self.positions[pair] = {'amount': amount, 'entry_price': price}
        else:
            self.balance += amount * price
            # Calculate realized profit if position is closed
            profit = 0.0
            if pair in self.positions:
                self.positions[pair]['amount'] -= amount
                if self.positions[pair]['amount'] <= 0:
                    profit = (price - entry_price) * amount
                    self.rlusd_balance += profit
                    del self.positions[pair]
            else:
                profit = (price - entry_price) * amount
                self.rlusd_balance += profit

    def get_performance_metrics(self) -> Dict:
        total_trades = len(self.trade_history)
        winning_trades = sum(
            1 for trade in self.trade_history
            if trade['side'] == 'sell' and trade['price'] > trade['entry_price']
        )
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'current_balance': self.balance,
            'profit_loss': self.balance - self.initial_balance,
            'profit_loss_percentage': ((self.balance - self.initial_balance) / self.initial_balance * 100),
            'running_time': (datetime.now() - self.start_time).total_seconds() / 3600  # in hours
        }

    def get_rlusd_balance(self) -> float:
        return self.rlusd_balance

class TechnicalAnalysis:
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate RSI for the given price series."""
        prices = pd.Series(prices)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]

    @staticmethod
    def calculate_moving_averages(prices: List[float], fast_period: int, slow_period: int) -> Tuple[float, float]:
        """Calculate fast and slow moving averages."""
        prices = pd.Series(prices)
        fast_ma = prices.rolling(window=fast_period).mean().iloc[-1]
        slow_ma = prices.rolling(window=slow_period).mean().iloc[-1]
        return fast_ma, slow_ma

    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        prices = pd.Series(prices)
        middle_band = prices.rolling(window=period).mean().iloc[-1]
        std = prices.rolling(window=period).std().iloc[-1]
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return upper_band, middle_band, lower_band

class KrakenProBot:
    def __init__(self, api_key: str, api_secret: str, config: TradingConfig, base_url: str = "https://api.kraken.com"):
        """
        Initialize the Kraken Pro trading bot.
        
        Args:
            api_key (str): Your Kraken Pro API key
            api_secret (str): Your Kraken Pro API secret
            config (TradingConfig): Trading configuration
            base_url (str): Kraken Pro API base URL
        """
        self.api_key = api_key
        self.api_secret = api_secret



        self.base_url = base_url
        self.config = config
        self.session = requests.Session()
        self.active_orders: Dict[str, Dict] = {}
        self.position_tracking: Dict[str, Dict] = {}
        self.trading_pairs = []  # Will be set dynamically
        self.ignored_pairs = set()
        self.refresh_ignored_pairs()
        
        # Initialize demo state if in demo mode
        if config.demo_mode:
            self.demo_state = DemoState(config.demo_balance)
            logger.info(f"Demo mode initialized with ${config.demo_balance:.2f}")

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        self.refresh_interval = 300  # seconds (5 minutes)
        self.refresh_seconds_left = self.refresh_interval
        self.refresh_timer = QTimer()
        #self.refresh_timer.timeout.connect(self.update_refresh_countdown)
        self.refresh_timer.start(1000)

    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        url = self.base_url + endpoint
        if data is None:
            data = {}
        data['nonce'] = str(int(1000 * time.time()))
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = endpoint.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        headers = {
            'API-Key': self.api_key,
            'API-Sign': sigdigest.decode()
        }
        try:
            if method.upper() == 'GET':
                resp = self.session.get(url, headers=headers, params=data, timeout=10)
            else:
                resp = requests.post(url, headers=headers, data=data, timeout=10)
            return resp.json()
        except Exception as e:
            logger.error(f"[API][ERROR] Exception during {method} {endpoint}: {e}")
            return {'error': [str(e)]}
        
        
    
    def _simulate_api_response(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Simulate API responses in demo mode."""
        if endpoint == '/0/private/Balance':
            return {'result': self.demo_state.get_balance()}
        elif endpoint == '/0/public/Ticker':
            pair = data.get('pair', 'XBTUSD')
            # Simulate realistic USD prices for common pairs
            usd_price_map = {
                'ADAUSDC': 0.45,
                'SOLUSDC': 140.0,
                'SUSDC': 0.0001,  # Very small price for S token
                'LTCUSDC': 85.0,
                'TRUMPUSDC': 0.00001,  # Very small price for TRUMP token
                'XBTUSD': 63000.0,
                'ETHUSD': 3100.0,
                'AAVEUSD': 173.0,
                '1INCHUSD': 0.20,
                'MATICUSD': 0.70,
                'LINKUSD': 14.0,
                'DOTUSD': 7.0,
                'UNIUSD': 7.5,
                'BCHUSD': 480.0,
                'ZRXUSD': 0.25,
                'ACHUSD': 0.03,
                'ACAUSD': 0.10,
            }
            if pair.endswith('USD') or pair.endswith('USDC'):
                base_price = usd_price_map.get(pair, 1.0)
                price_variation = np.random.normal(0, 0.01)  # 1% stddev
                current_price = base_price * (1 + price_variation)
            else:
                base_price = 0.01
                price_variation = np.random.normal(0, 0.05)
                current_price = base_price * (1 + price_variation)
            return {
                'result': {
                    pair: {
                        'c': [str(current_price), str(current_price)],
                        'v': ['1000.0', '1000.0'],
                        'p': [str(current_price * 0.99), str(current_price * 1.01)]
                    }
                }
            }
        elif endpoint == '/0/private/AddOrder':
            order_id = f"DEMO-{int(time.time())}"
            pair = data.get('pair', 'XBTUSD')
            side = data.get('side', 'buy')
            volume = float(data.get('volume', 0))
            price = float(data.get('price', 0))
            
            # Record the trade in demo state
            self.demo_state.add_trade(pair, side, volume, price)
            
            return {
                'result': {
                    'txid': [order_id],
                    'descr': {
                        'order': f"{side} {volume} {pair} @ {price}"
                    }
                }
            }
        elif endpoint == '/0/public/OHLC':
            pair = data.get('pair', 'XBTUSD')
            interval = int(data.get('interval', 1))
            now = int(time.time())
            # Generate 100 fake OHLC candles, 1 minute apart
            ohlc = []
            price = 50000.0 if 'XBT' in pair else 3000.0
            for i in range(100):
                t = now - 60 * (99 - i) * interval
                open_ = price * (1 + np.random.normal(0, 0.001))
                high = open_ * (1 + np.random.uniform(0, 0.002))
                low = open_ * (1 - np.random.uniform(0, 0.002))
                close = open_ * (1 + np.random.normal(0, 0.001))
                vwap = (high + low + close) / 3
                volume = np.random.uniform(0.5, 5)
                count = np.random.randint(1, 10)
                ohlc.append([t, open_, high, low, close, vwap, volume, count])
                price = close  # next open is this close
            return {'result': {pair: ohlc}}
        return {'error': ['Unknown endpoint']}

    def get_ohlc_data(self, pair: str, interval: int = 1) -> pd.DataFrame:
        #logger.info(f"[BOT] Fetching OHLC data for pair: {pair}, interval: {interval}")
        try:
            api_symbol = self.get_api_symbol(pair)
            # Always fetch fresh data from the API
            data = self._request('GET', '/0/public/OHLC', {'pair': api_symbol, 'interval': interval})
            # print(f"[DEBUG][get_ohlc_data] {pair} (api_symbol={api_symbol}) API response: {data}")  # Removed for cleaner logs
            # Check if the pair exists in the response
            if api_symbol not in data.get('result', {}):
                logger.warning(f"Pair {pair} not available in OHLC data")
                # Return empty DataFrame with correct columns
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
            df = pd.DataFrame(data['result'][api_symbol], 
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            return df
        except Exception as e:
            logger.error(f"Error getting OHLC data for {pair}: {str(e)}")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])

    def analyze_market(self, pair: str) -> Dict[str, Union[float, bool]]:
        """Perform technical analysis on the given trading pair using 5-minute candles and breakout rule."""
        df = self.get_ohlc_data(pair, interval=5)  # Use 5-minute timeframe
        
        # If DataFrame is empty, return default values
        if df.empty:
            logger.warning(f"No data available for {pair}, returning default values")
            return {
                'rsi': 50.0,
                'fast_ma': 0.0,
                'slow_ma': 0.0,
                'upper_bb': 0.0,
                'middle_bb': 0.0,
                'lower_bb': 0.0,
                'current_price': 0.0,
                'is_overbought': False,
                'is_oversold': False,
                'ma_crossover_buy': False,
                'ma_crossover_sell': False,
                'breakout': False
            }
            
        closes = df['close'].astype(float).tolist()
        highs = df['high'].astype(float).tolist()
        
        # Calculate indicators
        rsi = TechnicalAnalysis.calculate_rsi(closes, self.config.rsi_period)
        fast_ma, slow_ma = TechnicalAnalysis.calculate_moving_averages(
            closes, self.config.ma_fast_period, self.config.ma_slow_period)
        upper_bb, middle_bb, lower_bb = TechnicalAnalysis.calculate_bollinger_bands(closes)
        
        current_price = float(closes[-1])
        # Breakout rule: price above highest high of last 10 candles (excluding current)
        breakout = False
        if len(highs) > 10:
            recent_high = max(highs[-11:-1])
            breakout = current_price > recent_high
        
        # --- RELAXED BUY LOGIC ---
        relaxed_buy = (rsi < 60) or (fast_ma > slow_ma) or breakout
        return {
            'rsi': rsi,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'upper_bb': upper_bb,
            'middle_bb': middle_bb,
            'lower_bb': lower_bb,
            'current_price': current_price,
            'is_overbought': rsi > self.config.rsi_overbought,
            'is_oversold': rsi < 60,  # relaxed oversold
            'ma_crossover_buy': fast_ma > slow_ma or relaxed_buy,
            'ma_crossover_sell': fast_ma < slow_ma,
            'breakout': breakout
        }

    def get_order_info(self, txid):
        """Fetch order info from Kraken for a given txid."""
        data = {'txid': txid}
        response = self._request('POST', '/0/private/QueryOrders', data)
        return response

    def place_order(self, pair: str, order_type: OrderType, side: OrderSide, 
                   volume: float, price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> Dict:
        """Place a new order with advanced options."""
        # Guard: Prevent zero or negative volume orders
        if volume is None or volume <= 0:
            logger.warning(f"Attempted to place order with non-positive volume: {volume}")
            return {'error': ['Order volume must be greater than zero.']}
            
        # Patch: In demo mode, for market orders, fetch the current price
        if self.config.demo_mode and order_type == OrderType.MARKET:
            ticker = self.get_ticker(pair)
            price = float(ticker['result'][pair]['c'][0])
            
        data = {
            'pair': pair,
            'type': side.value,
            'ordertype': order_type.value,
            'volume': str(volume),
        }
        
        # Handle different order types
        if order_type == OrderType.TRAILING_STOP:
            data['trailing_stop'] = str(self.config.trailing_stop_percentage)
        elif order_type in [OrderType.LIMIT, OrderType.STOP_LOSS]:
            if price is not None:
                data['price'] = str(price)
            if stop_price is not None:
                data['stop_price'] = str(stop_price)
                
        # Patch: Always set price for demo mode market orders
        if self.config.demo_mode and order_type == OrderType.MARKET:
            data['price'] = str(price)
            
        logger.info(f"[ORDER] Placing {order_type.value.upper()} {side.value.upper()} for {volume} {pair}")
        #logger.info(f"[API] {method} {endpoint} with data {data}")  # keep this one commented for now
        response = self._request('POST', '/0/private/AddOrder', data)
        
        exec_price = None
        if 'result' in response and 'txid' in response['result']:
            order_id = response['result']['txid'][0]
            # Fetch order info to get executed price
            order_info = self.get_order_info(order_id)
            if 'result' in order_info and order_id in order_info['result']:
                exec_price = float(order_info['result'][order_id].get('price', 0))
            self.active_orders[order_id] = {
                'pair': pair,
                'type': order_type.value,
                'side': side.value,
                'volume': volume,
                'price': exec_price,  # Use executed price
                'stop_price': stop_price,
                'timestamp': datetime.now()
            }
            if order_id in self.active_orders:
                order = self.active_orders[order_id].copy()
                order['status'] = 'Closed'
                order['timestamp'] = datetime.now()
            else:
                # Fallback: try to get as much info as possible from result
                txid = None
                price = None
                if isinstance(response['result'], dict) and 'result' in response['result']:
                    txid = response['result']['result'].get('txid', [order_id])[0]
                    descr = response['result']['result'].get('descr', {})
                    # Try to parse price from descr['order'] if possible
                    if 'order' in descr and '@' in descr['order']:
                        try:
                            price = float(descr['order'].split('@')[-1].strip())
                        except Exception:
                            price = None
                order = {
                    'order_id': txid or order_id,
                    'pair': pair,
                    'type': 'market',
                    'side': 'sell',
                    'volume': volume,
                    'price': price,
                    'stop_price': None,
                    'timestamp': datetime.now(),
                    'status': 'Closed',
                    'current_price': price,
                    'bought_price': None,
                    'pl_dollar': None,
                    'pl_percent': None
                }
        return response

    def place_trailing_stop(self, pair: str, side: OrderSide, volume: float, 
                          current_price: float) -> Dict:
        """Place a true trailing stop order."""
        try:
            # For trailing stop orders, we need to use the correct price format
            data = {
                'pair': pair,
                'type': side.value,
                'ordertype': 'trailing-stop',
                'volume': str(volume),
                'price': f"+{self.config.trailing_stop_percentage}%"  # Using the correct format for trailing stop
            }
            
            logger.info(f"Placing trailing stop order with data: {data}")
            response = self._request('POST', '/0/private/AddOrder', data)
            
            # Log the full response for debugging
            logger.info(f"Trailing stop response: {json.dumps(response, indent=2)}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in place_trailing_stop: {str(e)}")
            return {'error': [str(e)]}

    def execute_trading_strategy(self, pair: str, balance: dict) -> None:
        """Execute the trading strategy based on technical analysis and breakout rule."""
        try:
            if not self.should_trade_pair(pair):
                logger.info(f"[DEBUG] {pair}: Skipping, pair is ignored or blacklisted.")
                return

            # Check if we've reached the concurrent trades limit
            active_trades = len([order for order in self.active_orders.values() 
                               if order.get('status', 'Active') == 'Active'])
            if active_trades >= self.config.concurrent_trades:
                logger.info(f"[DEBUG] {pair}: Skipping, reached concurrent trades limit ({self.config.concurrent_trades}).")
                return

            # Check if we already have an active position for this pair
            if pair in self.active_orders:
                logger.info(f"[DEBUG] {pair}: Skipping, already have an active position.")
                return

            analysis = self.analyze_market(pair)
            current_price = analysis['current_price']

            # Use the correct balance for the pair's quote currency
            if pair.endswith('XBT') or pair.endswith('BTC'):
                available_quote = float(balance.get('XXBT', 0))
                quote_currency = 'XXBT'
            elif pair.endswith('USDC'):
                available_quote = float(balance.get('USDC', 0))
                quote_currency = 'USDC'
            elif pair.endswith('USD'):
                available_quote = float(balance.get('ZUSD', 0))
                quote_currency = 'ZUSD'
            else:
                available_quote = float(balance.get('ZUSD', 0))  # fallback
                quote_currency = 'ZUSD'

            # Fetch pair info for min order size, decimals, and notional
            ordermin = 0.0
            lot_decimals = 8
            min_notional = 10.0
            try:
                assetpairs_resp = self.session.get(f"{self.base_url}/0/public/AssetPairs?pair={pair}").json()
                pair_info = assetpairs_resp['result']
                # Kraken's API may use a different key for the pair, so find the first key
                for k, v in pair_info.items():
                    if isinstance(v, dict):
                        ordermin = float(v.get('ordermin', 0))
                        lot_decimals = int(v.get('lot_decimals', 8))
                        # Try to get minimum notional value (costmin)
                        min_notional = float(v.get('costmin', 10.0))
                        break
            except Exception as e:
                logger.warning(f"[WARN] Could not fetch AssetPairs info for {pair}: {e}")
            logger.info(f"[DEBUG] {pair}: Using min_notional={min_notional}")

            # Apply a more conservative fee buffer
            fee_buffer = 0.90  # Use 90% of available balance
            max_trade_amount = min(self.config.max_usd_amount, available_quote * (self.config.max_trade_percentage / 100) * fee_buffer)
            # Position size is always in base currency (amount to buy/sell)
            position_size = max_trade_amount / current_price if current_price > 0 else 0
            # Round to allowed precision
            position_size = round(position_size, lot_decimals)

            # Calculate notional and total cost (including fee estimate)
            notional = position_size * current_price
            fee_estimate = 0.003  # 0.3% fee
            total_cost = position_size * current_price * (1 + fee_estimate)

            # Hard minimum notional (e.g., $10 in BTC) for BTC-quoted pairs
            hard_min_notional = min_notional
            if quote_currency == 'XXBT':
                try:
                    btc_usd_ticker = self.get_ticker('XBTUSDC')
                    btc_usd_price = float(btc_usd_ticker['result']['XBTUSDC']['c'][0])
                    hard_min_notional = max(min_notional, 10.0 / btc_usd_price)
                except Exception as e:
                    logger.warning(f"[WARN] Could not fetch BTC/USD price for hard_min_notional: {e}")
                    hard_min_notional = max(min_notional, 0.0002)  # fallback to ~0.0002 BTC if price fetch fails
            logger.info(f"[DEBUG] {pair}: position_size={position_size}, ordermin={ordermin}, notional={notional}, min_notional={min_notional}, hard_min_notional={hard_min_notional}, total_cost={total_cost}, available_quote={available_quote}")

            # Check all minimums and available balance
            if position_size < ordermin or notional < hard_min_notional or total_cost > available_quote:
                logger.info(f"[DEBUG] {pair}: Skipping trade, fails minimums or not enough balance for fees.")
                return

            logger.info(f"[DEBUG] {pair}: Available {quote_currency}: {available_quote}, Max trade amount: {max_trade_amount}, Position size: {position_size}, Current price: {current_price}, ordermin: {ordermin}, lot_decimals: {lot_decimals}, notional: {notional}")

            if position_size <= 0:
                logger.info(f"[DEBUG] {pair}: Skipping trade, position size is zero or negative.")
                return

            # Moderate trading logic: buy on MA crossover or breakout, sell on MA crossover sell
            if analysis['ma_crossover_buy'] or analysis['breakout']:
                logger.info(f"[DEBUG] {pair}: Buy signal triggered (ma_crossover_buy={analysis['ma_crossover_buy']}, breakout={analysis['breakout']})")
                try:
                    # Place market buy order
                    buy_response = self.place_order(
                        pair=pair,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,
                        volume=position_size
                    )
                    # Check for actual errors, not just empty error list
                    if 'error' in buy_response and buy_response['error'] and len(buy_response['error']) > 0:
                        logger.error(f"[ERROR] {pair}: Failed to place buy order: {buy_response['error']}")
                        return
                    logger.info(f"[DEBUG] {pair}: Successfully placed buy order")
                    # Try to place trailing stop, but continue even if it fails
                    try:
                        stop_response = self.place_trailing_stop(
                            pair=pair,
                            side=OrderSide.SELL,
                            volume=position_size,
                            current_price=current_price
                        )
                        if 'error' in stop_response and stop_response['error'] and len(stop_response['error']) > 0:
                            logger.error(f"[ERROR] {pair}: Failed to place trailing stop: {stop_response['error']}")
                        else:
                            logger.info(f"[DEBUG] {pair}: Successfully placed trailing stop")
                    except Exception as e:
                        logger.error(f"[ERROR] {pair}: Exception during trailing stop placement: {str(e)}")
                except Exception as e:
                    logger.error(f"[ERROR] {pair}: Exception during buy order placement: {str(e)}")
            elif analysis['ma_crossover_sell']:
                logger.info(f"[DEBUG] {pair}: Sell signal triggered (ma_crossover_sell={analysis['ma_crossover_sell']})")
                try:
                    # Place market sell order
                    sell_response = self.place_order(
                        pair=pair,
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL,
                        volume=position_size
                    )
                    # Check for actual errors, not just empty error list
                    if 'error' in sell_response and sell_response['error'] and len(sell_response['error']) > 0:
                        logger.error(f"[ERROR] {pair}: Failed to place sell order: {sell_response['error']}")
                        return
                    logger.info(f"[DEBUG] {pair}: Successfully placed sell order")
                    # Try to place trailing stop, but continue even if it fails
                    try:
                        stop_response = self.place_trailing_stop(
                            pair=pair,
                            side=OrderSide.BUY,
                            volume=position_size,
                            current_price=current_price
                        )
                        if 'error' in stop_response and stop_response['error'] and len(stop_response['error']) > 0:
                            logger.error(f"[ERROR] {pair}: Failed to place trailing stop: {stop_response['error']}")
                        else:
                            logger.info(f"[DEBUG] {pair}: Successfully placed trailing stop")
                    except Exception as e:
                        logger.error(f"[ERROR] {pair}: Exception during trailing stop placement: {str(e)}")
                except Exception as e:
                    logger.error(f"[ERROR] {pair}: Exception during sell order placement: {str(e)}")
        except Exception as e:
            logger.error(f"[ERROR] {pair}: Unexpected error in execute_trading_strategy: {str(e)}")

    def get_balance(self) -> Dict[str, float]:
        """Get account balance from Kraken."""
        response = self._request('POST', '/0/private/Balance')
        if 'result' in response:
            balances = {k: float(v) for k, v in response['result'].items()}
            # Keep USD and USDC separate, don't overwrite
            if 'USD' in balances:
                balances['ZUSD'] = balances['USD']
            if 'USDC' in balances:
                balances['USDC'] = balances['USDC']  # Keep USDC separate
            return balances
        else:
            logger.error(f"Balance API error: {response}")
            return {'ZUSD': 0.0, 'RLUSD': 0.0}
        
    def get_ticker(self, pair: str) -> Dict:
        logger.info(f"[BOT] Fetching ticker for pair: {pair}")
        api_symbol = self.get_api_symbol(pair)
        return self._request('GET', '/0/public/Ticker', {'pair': api_symbol})

    def get_performance_metrics(self) -> Dict:
        """Get trading performance metrics."""
        if self.config.demo_mode:
            return self.demo_state.get_performance_metrics()
        return {}

    def monitor_positions(self) -> None:
        """Monitor and manage open positions with stop-loss and take-profit logic."""
        for order_id, order in list(self.active_orders.items()):
            try:
                pair = order['pair']
                ticker_response = self.get_ticker(pair)
                
                # Validate ticker response
                if 'error' in ticker_response and ticker_response['error']:
                    logger.error(f"[ERROR] Failed to get ticker for {pair}: {ticker_response['error']}")
                    continue
                    
                if 'result' not in ticker_response or pair not in ticker_response['result']:
                    logger.error(f"[ERROR] Invalid ticker response for {pair}: {ticker_response}")
                    continue
                
                current_price = float(ticker_response['result'][pair]['c'][0])
                
                # Get the entry price and position size with validation
                entry_price = order.get('price')
                position_size = order.get('volume')
                
                if entry_price is None or position_size is None:
                    logger.error(f"[ERROR] Missing price or volume data for order {order_id}: price={entry_price}, volume={position_size}")
                    continue
                
                try:
                    entry_price = float(entry_price)
                    position_size = float(position_size)
                except (ValueError, TypeError) as e:
                    logger.error(f"[ERROR] Invalid price or volume format for order {order_id}: {e}")
                    continue
                
                if entry_price <= 0 or position_size <= 0:
                    logger.error(f"[ERROR] Invalid price or volume values for order {order_id}: price={entry_price}, volume={position_size}")
                    continue
                
                # Calculate price change percentage
                price_change_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Stop-loss check (default 5%)
                if price_change_pct <= -self.config.stop_loss_percentage:
                    logger.info(f"[STOP-LOSS] {pair}: Price dropped {price_change_pct:.2f}% below entry. Executing stop-loss.")
                    # Place market sell order
                    sell_response = self.place_order(
                        pair=pair,
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL,
                        volume=position_size
                    )
                    if 'error' not in sell_response or not sell_response['error']:
                        order['status'] = 'Closed'
                        logger.info(f"[STOP-LOSS] Successfully executed stop-loss for {pair}")
                    else:
                        logger.error(f"[STOP-LOSS] Failed to execute stop-loss for {pair}: {sell_response.get('error', 'Unknown error')}")
                
                # Take-profit check (default 10%)
                elif price_change_pct >= self.config.take_profit_percentage:
                    logger.info(f"[TAKE-PROFIT] {pair}: Price increased {price_change_pct:.2f}% above entry. Executing take-profit.")
                    # Place market sell order
                    sell_response = self.place_order(
                        pair=pair,
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL,
                        volume=position_size
                    )
                    if 'error' not in sell_response or not sell_response['error']:
                        order['status'] = 'Closed'
                        logger.info(f"[TAKE-PROFIT] Successfully executed take-profit for {pair}")
                    else:
                        logger.error(f"[TAKE-PROFIT] Failed to execute take-profit for {pair}: {sell_response.get('error', 'Unknown error')}")
                
                # Update the order in active_orders
                self.active_orders[order_id] = order
                
            except Exception as e:
                logger.error(f"[ERROR] Error monitoring position {order_id}: {str(e)}")
                continue

    def fetch_available_pairs(self, base_currencies=None):
        """Fetch available trading pairs from Kraken, only those ending with USD."""
        url = f"{self.base_url}/0/public/AssetPairs"
        response = self.session.get(url).json()
        pairs = []
        for pair, info in response.get('result', {}).items():
            if pair.endswith('USD'):
                pairs.append(pair)
        return pairs

    def get_display_name(self, api_symbol):
        return self.pair_display_map.get(api_symbol, api_symbol)

    def get_api_symbol(self, display_name):
        return self.display_to_api_map.get(display_name, display_name)

    def set_trading_pairs(self, pairs):
        """Set the list of trading pairs to use (from GUI or config), use as-is from GUI."""
        self.trading_pairs = pairs

    def refresh_ignored_pairs(self):
        """Fetch open orders/positions from Kraken and add those pairs to ignored_pairs."""
        try:
            # Fetch open orders
            open_orders = self._request('POST', '/0/private/OpenOrders')
            for order_id, order in open_orders.get('result', {}).get('open', {}).items():
                descr = order.get('descr', {})
                pair = descr.get('pair')
                if pair:
                    self.ignored_pairs.add(pair)
            # Optionally, fetch open positions if needed
            # open_positions = self._request('POST', '/0/private/OpenPositions')
            # for pos_id, pos in open_positions.get('result', {}).items():
            #     pair = pos.get('pair')
            #     if pair:
            #         self.ignored_pairs.add(pair)
        except Exception as e:
            logger.warning(f"Could not refresh ignored pairs: {str(e)}")

    def should_trade_pair(self, pair):
        return pair not in self.ignored_pairs

    def update_refresh_countdown(self):
        """This method is deprecated and should be handled by the GUI."""
        pass

    def force_refresh_cycle(self):
        """This method is deprecated and should be handled by the GUI."""
        pass

    def confirm_sell_all(self):
        reply = QMessageBox.question(self, 'Confirm Sell All',
                                     'Are you sure you want to sell all open positions at market?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.sell_all_at_market()

    def get_realized_pl_live(self):
        """Calculate realized P/L from closed orders in live mode."""
        response = self._request('POST', '/0/private/ClosedOrders')
        realized_pl = 0.0
        if 'result' in response and 'closed' in response['result']:
            for order_id, order in response['result']['closed'].items():
                descr = order.get('descr', {})
                pair = descr.get('pair')
                side = descr.get('type')
                price = float(order.get('price', 0))
                vol = float(order.get('vol_exec', 0))
                cost = float(order.get('cost', 0))
                # Only consider sell orders for realized P/L
                if side == 'sell':
                    # Find the corresponding buy price (simple version: use order's cost/vol)
                    buy_price = float(order.get('cost', 0)) / vol if vol > 0 else 0
                    pl = (price - buy_price) * vol
                    realized_pl += pl
        return realized_pl

    def get_unrealized_pl_live(self):
        """Calculate unrealized P/L from open positions in live mode."""
        response = self._request('POST', '/0/private/OpenPositions')
        unrealized_pl = 0.0
        if 'result' in response:
            for pos_id, pos in response['result'].items():
                entry_price = float(pos.get('cost', 0)) / float(pos.get('vol', 1)) if float(pos.get('vol', 1)) > 0 else 0
                amount = float(pos.get('vol', 0))
                pair = pos.get('pair')
                # Get current price
                ticker = self.get_ticker(pair)
                if 'result' in ticker and pair in ticker['result']:
                    current_price = float(ticker['result'][pair]['c'][0])
                    unrealized_pl += (current_price - entry_price) * amount
        return unrealized_pl

def load_strategy(strategy_name: str):
    """Load a strategy module from the strategies folder."""
    try:
        # Convert strategy name to module name
        module_name = strategy_name.replace("-", "_").lower()
        strategy_path = os.path.join(os.path.dirname(__file__), "strategies", f"{module_name}.py")
        
        if not os.path.exists(strategy_path):
            logger.error(f"Strategy file not found: {strategy_path}")
            return None
            
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, strategy_path)
        if spec is None:
            logger.error(f"Could not load strategy spec: {strategy_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Error loading strategy {strategy_name}: {str(e)}")
        return None

def main():
    # Example configuration
    config = TradingConfig(
        max_usd_amount=1000.0,  # Maximum USD amount to trade
        base_currency="USD",
        quote_currency="XBT",
        trailing_stop_percentage=2.0,
        stop_loss_percentage=5.0,
        take_profit_percentage=10.0,
        rsi_period=14,
        rsi_overbought=70.0,
        rsi_oversold=30.0,
        ma_fast_period=9,
        ma_slow_period=21,
        demo_mode=True,  # Enable demo mode
        demo_balance=10000.0,  # Start with $10,000 in demo mode
        concurrent_trades=1,  # Limit for concurrent trades
        max_trade_percentage=15.0,  # Maximum percentage of available balance to use per trade
        ecosystem="USDC",  # Ecosystem for trading pairs (USDC, USD, BTC)
        strategy="Breakout-Focused"  # Default strategy
    )

    # Load API keys from Kraken_API.json
    api_key = None
    api_secret = None
    try:
        with open(os.path.join(os.path.dirname(__file__), 'Kraken_API.json'), 'r') as f:
            api_data = json.load(f)
            api_key = api_data.get('api_key')
            api_secret = api_data.get('api_secret')
    except Exception as e:
        logger.error(f"Failed to load Kraken_API.json: {e}")
        return
    

    if not api_key or not api_secret:
        logger.error("API credentials not found in Kraken_API.json")
        return

    main_app_dir = os.path.dirname(os.path.abspath(__file__))
    if main_app_dir not in sys.path:
        sys.path.append(main_app_dir)

    bot = KrakenProBot(api_key, api_secret, config)
    # Fetch and set available pairs (default: USD, XBT, EUR)
    pairs = bot.fetch_available_pairs()
    bot.set_trading_pairs(pairs)

    try:
        # Example trading loop
        while True:
            # Load the selected strategy
            strategy_module = load_strategy(config.strategy)
            if strategy_module is None:
                logger.error(f"Failed to load strategy: {config.strategy}")
                time.sleep(90)
                continue
                
            # Fetch market data for all pairs
            market_data = {}
            print ("\n\n")
            print(f"[BOT] Fetching market data for {len(bot.trading_pairs)} pairs")
            print("Please wait...")
            print ("\n\n")
            for pair in bot.trading_pairs:
                df = bot.get_ohlc_data(pair, interval=5)
                market_data[pair] = df
                
            # Get top pairs from strategy
            top_pairs = strategy_module.select_top_pairs(market_data)
            
            # Trade only the top pairs
            for pair in top_pairs:
                balance = bot.get_balance()  # Always fetch fresh balance for each pair
                bot.execute_trading_strategy(pair, balance)
                bot.monitor_positions()
            time.sleep(90)  # Wait for 1.5 minutes before next iteration
            print(f"[BOT] Done getting market data for {len(bot.trading_pairs)} pairs")
            print ("\n\n")

    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
