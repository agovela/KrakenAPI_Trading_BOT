import sys
from datetime import datetime
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt5.QtGui import QColor
from KrakenPro_Bot import KrakenProBot, TradingConfig, OrderType, OrderSide
import importlib
import dynamic_strategy
import json
import os
from PyQt5.QtWidgets import QHBoxLayout, QPushButton

print ("\n\n")
print ("-------------------------------------------------------------------------")
print (" ⭐⭐ Loaded KrakenPro GUI UI Version 1.0.1 ⭐⭐")
print ("-------------------------------------------------------------------------")
print ("\n\n")

class AnalysisWorker(QThread):
    result_signal = pyqtSignal(dict)  # {pair: analysis_result}
    def __init__(self, bot, pairs):
        super().__init__()
        self.bot = bot
        self.pairs = pairs
        self.running = True

    def run(self):
        results = {}
        for pair in self.pairs:
            if not self.running:
                break
            results[pair] = self.bot.analyze_market(pair)
        self.result_signal.emit(results)

    def stop(self):
        self.running = False

class APIWorker(QThread):
    result_signal = pyqtSignal(str, object)  # (request_type, result)
    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self.requests = []
        self.running = True

    def run(self):
        import traceback
        while self.running:
            if self.requests:
                req = self.requests.pop(0)
                req_type, args = req
                try:
                    #print(f"[APIWorker] Starting request: {req_type} {args}")
                    if req_type == 'get_balance':
                        result = self.bot.get_balance()
                    elif req_type == 'get_ticker':
                        result = self.bot.get_ticker(*args)
                    elif req_type == 'get_ohlc':
                        result = self.bot.get_ohlc_data(*args)
                    else:
                        result = None
                    #print(f"[APIWorker] Finished request: {req_type}")
                    self.result_signal.emit(req_type, result)
                except Exception as e:
                    print(f"[APIWorker][ERROR] Exception in {req_type} {args}: {e}\n{traceback.format_exc()}")
                    self.result_signal.emit(req_type, e)
            self.msleep(10)

    def stop(self):
        self.running = False

    def add_request(self, req_type, args=()):
        self.requests.append((req_type, args))

class OrderWorker(QThread):
    result_signal = pyqtSignal(str, object)  # (request_type, result)
    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self.requests = []
        self.running = True

    def run(self):
        while self.running:
            if self.requests:
                req = self.requests.pop(0)
                req_type, args = req
                try:
                    if req_type == 'place_order':
                        result = self.bot.place_order(*args)
                    elif req_type == 'monitor_positions':
                        result = self.bot.monitor_positions()
                    else:
                        result = None
                    self.result_signal.emit(req_type, result)
                except Exception as e:
                    self.result_signal.emit(req_type, e)
            self.msleep(10)

    def stop(self):
        self.running = False

    def add_request(self, req_type, args=()):
        self.requests.append((req_type, args))

class TradingBotThread(QThread):
    update_signal = pyqtSignal(list)  # List of state dicts for all pairs
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)
    
    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self.running = True
        self.analysis_results = None
        self.analysis_worker = None
        self.analysis_done = False
        self.api_worker = APIWorker(bot)
        self.order_worker = OrderWorker(bot)
        self.api_results = {}
        self.order_results = {}
        self.api_worker.result_signal.connect(self.on_api_result)
        self.order_worker.result_signal.connect(self.on_order_result)
        self.api_worker.start()
        self.order_worker.start()
        self.cycle_count = 0
        self.cycle_interval_ms = 90000  # default 1.5 min
    
    def run(self):
        import traceback
        while self.running:
            try:
                self.cycle_count += 1
                # --- LIMIT PAIRS FOR TESTING ---
                pairs = self.bot.trading_pairs
                # pairs = pairs[:3]  # Only use the first 3 pairs for testing
                self.status_signal.emit("Starting new trading cycle...")
                self.log_signal.emit(f"[HEARTBEAT] Starting cycle #{self.cycle_count}")

                # Analysis phase - Use AnalysisWorker for OHLC fetching and technical analysis
                self.status_signal.emit("Analyzing pairs (async)...")
                self.log_signal.emit(f"[HEARTBEAT] Analyzing {len(pairs)} pairs (async)...")
                self.analysis_done = False
                self.analysis_worker = AnalysisWorker(self.bot, pairs)
                self.analysis_worker.result_signal.connect(self.on_analysis_done)
                self.analysis_worker.start()
                while not self.analysis_done and self.running:
                    self.msleep(100)
                if not self.running:
                    break
                analysis_results = self.analysis_results or {}
                self.log_signal.emit(f"[HEARTBEAT] Analysis complete for {len(analysis_results)} pairs using strategy: {self.bot.config.strategy}")
                for pair, analysis in analysis_results.items():
                    if 'rsi' in analysis:
                        self.log_signal.emit(f"[DEBUG] {pair} RSI: {analysis['rsi']}")

                # API requests phase
                self.status_signal.emit("Fetching balances and tickers...")
                self.log_signal.emit("[HEARTBEAT] Fetching market data...")
                self.api_results = {}
                # Only fetch get_balance ONCE per cycle
                self.api_worker.add_request('get_balance')
                for pair in pairs:
                    self.api_worker.add_request('get_ticker', (pair,))
                import time
                start_time = time.time()
                timeout = 60
                while len(self.api_results) < 1 + len(pairs) and self.running:
                    self.msleep(10)
                    if time.time() - start_time > timeout:
                        self.log_signal.emit(f"[ERROR] API results wait timed out! Got {len(self.api_results)}/{1+len(pairs)} results: {list(self.api_results.keys())[:5]}...")
                        break
                self.log_signal.emit(f"[HEARTBEAT] Market data fetch complete (got {len(self.api_results)}/{1+len(pairs)})")
                try:
                    keys = list(self.api_results.keys())
                    sample = self.api_results[keys[0]] if keys else None
                    self.log_signal.emit(f"[DEBUG] api_results keys: {keys[:5]}{'...' if len(keys)>5 else ''}")
                    self.log_signal.emit(f"[DEBUG] api_results sample value: {str(sample)[:200]}")
                except Exception as e:
                    self.log_signal.emit(f"[DEBUG] Error logging api_results: {e}")

                # Order monitoring phase
                self.status_signal.emit("Monitoring open positions...")
                self.log_signal.emit("[HEARTBEAT] Checking open positions...")
                self.order_worker.add_request('monitor_positions')

                # --- HOT-RELOAD STRATEGY AND SELECT TOP PAIRS ---
                # Use analysis_results from AnalysisWorker
                import importlib
                import importlib.util
                
                # Load the selected strategy
                strategy_name = self.bot.config.strategy
                module_name = strategy_name.replace("-", "_").lower()
                strategy_path = os.path.join(os.path.dirname(__file__), "strategies", f"{module_name}.py")
                
                if not os.path.exists(strategy_path):
                    self.log_signal.emit(f"[ERROR] Strategy file not found: {strategy_path}")
                    continue
                    
                # Load the module
                spec = importlib.util.spec_from_file_location(module_name, strategy_path)
                if spec is None:
                    self.log_signal.emit(f"[ERROR] Could not load strategy spec: {strategy_path}")
                    continue
                    
                strategy_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(strategy_module)
                
                # Build the real market_data dictionary
                market_data = {}
                for pair in pairs:
                    df = self.bot.get_ohlc_data(pair, interval=5)
                    market_data[pair] = df
                top_pairs = strategy_module.select_top_pairs(market_data)
                self.log_signal.emit(f"[STRATEGY] Top pairs this cycle: {top_pairs}")
                for pair in top_pairs:
                    try:
                        balance = self.api_results.get(('get_balance', pair), self.api_results.get('get_balance', {}))
                        if not self.bot.should_trade_pair(pair):
                            self.log_signal.emit(f"[DEBUG] {pair}: Skipping, pair is ignored or blacklisted.")
                            continue
                        active_trades = len([order for order in self.bot.active_orders.values() if order.get('status', 'Active') == 'Active'])
                        if active_trades >= self.bot.config.concurrent_trades:
                            self.log_signal.emit(f"[DEBUG] {pair}: Skipping, reached concurrent trades limit ({self.bot.config.concurrent_trades}).")
                            continue
                        self.bot.execute_trading_strategy(pair, balance)
                    except Exception as e:
                        self.log_signal.emit(f"[ERROR] Exception in trading logic for {pair}: {e}")
                while 'monitor_positions' not in self.order_results and self.running:
                    self.msleep(10)
                self.log_signal.emit("[HEARTBEAT] Position monitoring complete")

                # GUI update phase
                self.status_signal.emit("Updating GUI...")
                self.log_signal.emit("[HEARTBEAT] Updating interface...")
                states = []
                for pair in pairs:
                    try:
                        balance = self.api_results.get(('get_balance', pair), self.api_results.get('get_balance', {}))
                        ticker = self.api_results.get(('get_ticker', pair), {})
                        state = {
                            'pair': pair,
                            'balance': balance,
                            'ticker': ticker,
                            'active_orders': self.bot.active_orders,
                            'analysis': analysis_results.get(pair, {})
                        }
                        states.append(state)
                    except Exception as e:
                        self.error_signal.emit(f"Error preparing state for {pair}: {e}")
                self.update_signal.emit(states)
                self.log_signal.emit("[HEARTBEAT] Interface update complete")
                self.status_signal.emit("Waiting for next cycle...")
                self.log_signal.emit(f"[HEARTBEAT] Cycle #{self.cycle_count} complete. Waiting 3 minutes for next cycle...")
                self.msleep(self.cycle_interval_ms)  # 1.5 minutes in milliseconds
            except Exception as e:
                tb = traceback.format_exc()
                self.status_signal.emit(f"Error: {str(e)}")
                self.error_signal.emit(str(e))
                self.log_signal.emit(f"[ERROR] Cycle #{self.cycle_count} failed: {str(e)}")
        if self.analysis_worker:
            self.analysis_worker.stop()
            self.analysis_worker.wait()
        self.api_worker.stop()
        self.api_worker.wait()
        self.order_worker.stop()
        self.order_worker.wait()
        self.log_signal.emit("[HEARTBEAT] Trading bot stopped")
    
    def stop(self):
        self.running = False
        if self.analysis_worker:
            self.analysis_worker.stop()
            self.analysis_worker.wait()
        self.api_worker.stop()
        self.api_worker.wait()
        self.order_worker.stop()
        self.order_worker.wait()
    
    def on_analysis_done(self, results):
        self.analysis_results = results
        self.analysis_done = True
    def on_api_result(self, req_type, result):
        # Store results for batching
        if req_type == 'get_balance':
            self.api_results['get_balance'] = result
        elif req_type == 'get_ticker':
            # result should be a dict with 'result' key containing the pair
            if isinstance(result, dict) and 'result' in result:
                for pair_key in result['result']:
                    self.api_results[('get_ticker', pair_key)] = result
            else:
                # fallback for error or empty result
                self.api_results[('get_ticker', None)] = result
        else:
            self.api_results[req_type] = result
    def on_order_result(self, req_type, result):
        self.order_results[req_type] = result
    def set_cycle_interval(self, seconds):
        self.cycle_interval_ms = int(seconds * 1000)

class PairFetchThread(QThread):
    pairs_fetched = pyqtSignal(list, dict, dict)
    status_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)
    def __init__(self, bot):
        super().__init__()
        self.bot = bot
    def run(self):
        self.status_signal.emit("Loading trading pairs...")
        self.log_signal.emit("Loading trading pairs...")
        
        # Define our trading pairs (removed DOGEUSDC)
        usdc_pairs = [
            'SOLUSDC',
            'SUSDC',
            'LTCUSDC',
            'TRUMPUSDC',
            'ATOMUSDC',
            'MATICUSDC',
            'LINKUSDC',
            'SHIBUSDC'
        ]
        
        # Create display names for the pairs
        pair_display_map = {
            'SOLUSDC': 'SOL/USDC',
            'SUSDC': 'S/USDC',
            'LTCUSDC': 'LTC/USDC',
            'TRUMPUSDC': 'TRUMP/USDC',
            'ATOMUSDC': 'ATOM/USDC',
            'MATICUSDC': 'MATIC/USDC',
            'LINKUSDC': 'LINK/USDC',
            'SHIBUSDC': 'SHIB/USDC'
        }
        
        # Create reverse mapping
        display_to_api_map = {v: k for k, v in pair_display_map.items()}
        
        self.pairs_fetched.emit(usdc_pairs, pair_display_map, display_to_api_map)

class KrakenProGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.trading_paused = False
        self.closed_orders = []
        self.blacklisted_pairs = set()
        # Load the UI file
        uic.loadUi('KrakenPro.ui', self)
        
        # Initialize variables
        self.config = TradingConfig(
            max_usd_amount=1000.0,
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
            demo_mode=False,
            demo_balance=10000.0,
            concurrent_trades=self.concurrent_trades_input.value()
        )
        self.bot = None
        self.bot_thread = None
        
        # Add refresh countdown functionality
        self.refresh_interval = 180  # 3 minutes in seconds
        self.refresh_seconds_left = self.refresh_interval
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.update_refresh_countdown)
        self.refresh_timer.start(1000)
        
        # Connect signals and slots
        self.connect_signals()
        
        # Initialize UI
        self.init_ui()
        
        # Start pair fetch after window is shown
        QTimer.singleShot(0, self.start_pair_fetch)
        
    def connect_signals(self):
        # Connect all button clicks and other signals
        self.demo_mode_checkbox.stateChanged.connect(self.toggle_demo_mode)
        self.connect_button.clicked.connect(self.connect_to_kraken)
        self.start_trading_button.clicked.connect(self.start_trading)
        self.force_refresh_button.clicked.connect(self.force_refresh_cycle)
        self.sell_all_button.clicked.connect(self.confirm_sell_all)
        self.pause_trading_button.clicked.connect(self.toggle_trading_pause)
        self.pair_filter_input.textChanged.connect(self.filter_pairs)
        self.auto_select_pairs_checkbox.stateChanged.connect(self.toggle_auto_select_pairs)
        self.max_trade_input.textChanged.connect(self.update_max_trade)
        self.concurrent_trades_input.valueChanged.connect(self.update_concurrent_trades)
        self.market_pair_combo.currentIndexChanged.connect(self.update_market_data_group)
        self.orders_table.customContextMenuRequested.connect(self.show_orders_context_menu)
        self.place_order_button.clicked.connect(self.place_order)
        self.ecosystem_comboBox.currentIndexChanged.connect(self.on_ecosystem_changed)
        self.cycle_timer_input.textChanged.connect(self.update_cycle_timer)
        self.strategy_combo.currentIndexChanged.connect(self.save_settings)
        self.strategy_refresh_button.clicked.connect(self.refresh_strategy_combo)
        
    def init_ui(self):
        # Set up table columns
        print ("Initializing UI")
        self.orders_table.setColumnCount(13)
        self.orders_table.setHorizontalHeaderLabels([
            "Pair", "Type", "Side", "Amount", "Cost (USD)", "Price", "Status",
            "Current Price", "Bought Price", "Stop Loss", "P/L $", "P/L %", "Order ID"
        ])
        
        # Enable context menu for orders table
        self.orders_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.orders_table.customContextMenuRequested.connect(self.show_orders_context_menu)
        
        self.all_pairs_table.setColumnCount(2)
        self.all_pairs_table.setHorizontalHeaderLabels(["Pair", "Current Price"])
        
        # Set up order type and side combos
        for order_type in OrderType:
            self.order_type_combo.addItem(order_type.value)
        for side in OrderSide:
            self.side_combo.addItem(side.value)
            
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #181a1b; color: #e0e0e0; }
            QTabWidget::pane { border: 1px solid #333; background: #181a1b; }
            QTabBar::tab { background: #232629; color: #e0e0e0; border: 1px solid #333; padding: 6px; }
            QTabBar::tab:selected { background: #232629; color: #fff; }
            QTabBar::tab:!selected { background: #181a1b; color: #888; }
            QGroupBox { border: 1px solid #333; margin-top: 10px; }
            QLineEdit, QSpinBox, QDoubleSpinBox { background-color: #232629; color: #e0e0e0; border: 1px solid #333; }
            QPushButton { background-color: #232629; color: #e0e0e0; border-radius: 4px; padding: 6px; border: 1px solid #333; }
            QPushButton:hover { background-color: #333; }
            QHeaderView::section { background-color: #232629; color: #e0e0e0; }
            QTableWidget { background-color: #181a1b; color: #e0e0e0; }
            QTextEdit { background-color: #232629; color: #e0e0e0; }
            QLabel { color: #e0e0e0; }
            QListWidget { background-color: #232629; color: #e0e0e0; border: 1px solid #333; }
        """)
        
        # Set default for ecosystem_comboBox
        idx = self.ecosystem_comboBox.findText("USDC")
        if idx >= 0:
            self.ecosystem_comboBox.setCurrentIndex(idx)
        
        print ("Done initializing UI")
        
    def toggle_demo_mode(self, state):
        is_demo = state == Qt.Checked
        self.config.demo_mode = is_demo
        self.config.demo_balance = self.demo_balance_input.value()
        
        # Update UI elements based on demo mode
        self.api_key_input.setEnabled(not is_demo)
        self.api_secret_input.setEnabled(not is_demo)
        self.demo_balance_input.setEnabled(is_demo)
        
        if is_demo:
            self.status_label.setText("Demo Mode Active")
            self.status_label.setStyleSheet("color: blue;")
            self.log_message("Demo mode activated")
        else:
            self.status_label.setText("Not Connected")
            self.status_label.setStyleSheet("color: red;")
            self.log_message("Demo mode deactivated")

    def connect_to_kraken(self):
        self.log_message("[DEBUG] connect_to_kraken called")
        self.log_message("[INFO] Fetching trading pairs from Kraken...")
        ecosystem = self.ecosystem_comboBox.currentText()
        if ecosystem == "USDC":
            pairs = [
                'AI16ZUSDC', 'ALGOUSDC', 'APEUSDC', 'ATOMUSDC', 'AVAXUSDC', 'BCHUSDC', 'BERAUSDC', 'BNBUSDC',
                'CROUSDC', 'DOTUSDC', 'EOSUSDC', 'ETHUSDC', 'EUROPUSDC', 'EURRUSDC', 'FARTCOINUSDC', 'LINKUSDC', 'LTCUSDC',
                'MANAUSDC', 'MATICUSDC', 'MELANIAUSDC', 'PENGUUSDC', 'RLUSDUSDC', 'SHIBUSDC', 'SOLUSDC', 'SUSDC', 'TONUSDC',
                'TRUMPUSDC', 'USDGUSDC', 'USDQUSDC', 'USDRUSDC', 'USTUSDC', 'VIRTUALUSDC', 'XBTUSDC', 'XDGUSDC', 'XMRUSDC',
                'XRPUSDC', 'XTZUSDC'
            ]
        elif ecosystem == "USD":
            pairs = [
                '1INCHUSD', 'AAVEUSD', 'ACAUSD', 'ACHUSD', 'ACTUSD', 'ACXUSD', 'ADAUSD', 'ADXUSD', 'AEROUSD', 'AEVOUSD',
                'AGLDUSD', 'AI16ZUSD', 'AIRUSD', 'AIXBTUSD', 'AKTUSD', 'ALCHUSD', 'ALCXUSD', 'ALGOUSD', 'ALICEUSD',
                'ALPHAUSD', 'ALTUSD', 'ANKRUSD', 'ANLOGUSD', 'ANONUSD', 'APENFTUSD', 'APEUSD', 'API3USD', 'APTUSD',
                'APUUSD', 'ARBUSD', 'ARCUSD', 'ARKMUSD', 'ARPAUSD', 'ARUSD', 'ASTRUSD', 'ATHUSD', 'ATLASUSD', 'ATOMUSD',
                'AUCTIONUSD', 'AUDIOUSD', 'AUDUSD', 'AVAAIUSD', 'AVAXUSD', 'AXSUSD', 'B3USD', 'BABYUSD', 'BADGERUSD',
                'BALUSD', 'BANANAS31USD', 'BANDUSD', 'BATUSD', 'BCHUSD', 'BEAMUSD', 'BERAUSD', 'BICOUSD', 'BIGTIMEUSD',
                'BIOUSD', 'BITUSD', 'BLURUSD', 'BLZUSD', 'BMTUSD', 'BNBUSD', 'BNCUSD', 'BNTUSD', 'BOBAUSD', 'BODENUSD',
                'BONDUSD', 'BONKUSD', 'BRICKUSD', 'BSXUSD', 'BTTUSD', 'C98USD', 'CAKEUSD', 'CELOUSD', 'CELRUSD', 'CFGUSD',
                'CHEEMSUSD', 'CHRUSD', 'CHZUSD', 'CLANKERUSD', 'CLOUDUSD', 'CLVUSD', 'COMPUSD', 'CORNUSD', 'COTIUSD',
                'COWUSD', 'CPOOLUSD', 'CQTUSD', 'CROUSD', 'CRVUSD', 'CSMUSD', 'CTSIUSD', 'CVCUSD', 'CVXUSD', 'CXTUSD',
                'CYBERUSD', 'DAIUSD', 'DASHUSD', 'DBRUSD', 'DENTUSD', 'DOGSUSD', 'DOLOUSD', 'DOTUSD', 'DRIFTUSD',
                'DRVUSD', 'DUCKUSD', 'DYDXUSD', 'DYMUSD', 'EDGEUSD', 'EGLDUSD', 'EIGENUSD', 'ELXUSD', 'ENAUSD', 'ENJUSD',
                'ENSUSD', 'EOSUSD', 'ETHFIUSD', 'ETHPYUSD', 'ETHWUSD', 'EULUSD', 'EUROPUSD', 'EURQUSD', 'EURRUSD',
                'EURTUSD', 'EWTUSD', 'FARMUSD', 'FARTCOINUSD', 'FETUSD', 'FHEUSD', 'FIDAUSD', 'FILUSD', 'FISUSD',
                'FLOKIUSD', 'FLOWUSD', 'FLRUSD', 'FLUXUSD', 'FORTHUSD', 'FWOGUSD', 'FXSUSD', 'GALAUSD', 'GALUSD',
                'GARIUSD', 'GFIUSD', 'GHIBLIUSD', 'GHSTUSD', 'GIGAUSD', 'GLMRUSD', 'GMTUSD', 'GMXUSD', 'GNOUSD',
                'GOATUSD', 'GRASSUSD', 'GRIFFAINUSD', 'GRTUSD', 'GSTUSD', 'GTCUSD', 'GUNUSD', 'GUSD', 'HDXUSD',
                'HFTUSD', 'HMSTRUSD', 'HNTUSD', 'HONEYUSD', 'HPOS10IUSD', 'ICPUSD', 'ICXUSD', 'IDEXUSD', 'IMXUSD',
                'INITUSD', 'INJUSD', 'INTRUSD', 'IPUSD', 'JAILSTOOLUSD', 'JASMYUSD', 'JSTUSD', 'JTOUSD', 'JUNOUSD',
                'JUPUSD', 'KAITOUSD', 'KARUSD', 'KASUSD', 'KAVAUSD', 'KEEPUSD', 'KERNELUSD', 'KEYUSD', 'KILTUSD',
                'KINTUSD', 'KINUSD', 'KMNOUSD', 'KNCUSD', 'KP3RUSD', 'KSMUSD', 'KUJIUSD', 'KUSD', 'L3USD', 'LAYERUSD',
                'LCXUSD', 'LDOUSD', 'LINKUSD', 'LITUSD', 'LMWRUSD', 'LOCKINUSD', 'LPTUSD', 'LQTYUSD', 'LRCUSD',
                'LSETHUSD', 'LSKUSD', 'LUNA2USD', 'LUNAUSD', 'MANAUSD', 'MASKUSD', 'MATICUSD', 'MCUSD', 'MELANIAUSD',
                'MEMEUSD', 'METISUSD', 'MEUSD', 'MEWUSD', 'MICHIUSD', 'MINAUSD', 'MIRUSD', 'MKRUSD', 'MNGOUSD',
                'MNTUSD', 'MOGUSD', 'MOODENGUSD', 'MOONUSD', 'MORPHOUSD', 'MOVEUSD', 'MOVRUSD', 'MSOLUSD', 'MUBARAKUSD',
                'MULTIUSD', 'MVUSD', 'MXCUSD', 'NANOUSD', 'NEARUSD', 'NEIROUSD', 'NILUSD', 'NMRUSD', 'NODLUSD',
                'NOSUSD', 'NOTUSD', 'NTRNUSD', 'NYMUSD', 'OCEANUSD', 'ODOSUSD', 'OGNUSD', 'OMGUSD', 'OMNIUSD',
                'OMUSD', 'ONDOUSD', 'OPUSD', 'ORCAUSD', 'ORDERUSD', 'OSMOUSD', 'OXTUSD', 'OXYUSD', 'PAXGUSD',
                'PDAUSD', 'PENDLEUSD', 'PENGUUSD', 'PEPEUSD', 'PERPUSD', 'PHAUSD', 'PLUMEUSD', 'PNUTUSD', 'POLISUSD',
                'POLSUSD', 'POLUSD', 'PONDUSD', 'PONKEUSD', 'POPCATUSD', 'PORTALUSD', 'POWRUSD', 'PRCLUSD', 'PRIMEUSD',
                'PROMPTUSD', 'PSTAKEUSD', 'PUFFERUSD', 'PYTHUSD', 'PYUSDUSD', 'QNTUSD', 'QTUMUSD', 'RADUSD', 'RAREUSD',
                'RARIUSD', 'RAYUSD', 'RBCUSD', 'REDUSD', 'RENDERUSD', 'RENUSD', 'REPV2USD', 'REQUSD', 'REZUSD',
                'RLCUSD', 'RLUSDUSD', 'ROOKUSD', 'RPLUSD', 'RSRUSD', 'RUNEUSD', 'SAFEUSD', 'SAGAUSD', 'SAMOUSD',
                'SANDUSD', 'SBRUSD', 'SCRTUSD', 'SCUSD', 'SDNUSD', 'SEIUSD', 'SGBUSD', 'SHIBUSD', 'SIGMAUSD',
                'SKYUSD', 'SNEKUSD', 'SNXUSD', 'SOLUSD', 'SONICUSD', 'SPELLUSD', 'SPICEUSD', 'SPXUSD', 'SRMUSD',
                'SSVUSD', 'STEPUSD', 'STGUSD', 'STORJUSD', 'STRDUSD', 'STRKUSD', 'STXUSD', 'SUIUSD', 'SUNDOGUSD',
                'SUNUSD', 'SUPERUSD', 'SUSD', 'SUSHIUSD', 'SWARMSUSD', 'SWELLUSD', 'SYNUSD', 'SYRUPUSD', 'TAOUSD',
                'TBTCUSD', 'TEERUSD', 'TERMUSD', 'TIAUSD', 'TITCOINUSD', 'TLMUSD', 'TNSRUSD', 'TOKENUSD', 'TOKEUSD',
                'TONUSD', 'TOSHIUSD', 'TRACUSD', 'TREMPUSD', 'TRUMPUSD', 'TRUUSD', 'TRXUSD', 'TURBOUSD', 'TUSD',
                'TUSDUSD', 'TVKUSD', 'UFDUSD', 'UMAUSD', 'UNFIUSD', 'UNIUSD', 'USDCUSD', 'USDDUSD', 'USDGUSD',
                'USDQUSD', 'USDRUSD', 'USDSUSD', 'USDTZUSD', 'USTUSD', 'USUALUSD', 'VANRYUSD', 'VELODROMEUSD',
                'VINEUSD', 'VIRTUALUSD', 'VVVUSD', 'WALUSD', 'WAXLUSD', 'WBTCUSD', 'WCTUSD', 'WELLUSD', 'WENUSD',
                'WIFUSD', 'WINUSD', 'WLDUSD', 'WOOUSD', 'WUSD', 'XBTPYUSD', 'XCNUSD', 'XDGUSD', 'XETCZUSD',
                'XETHZUSD', 'XLTCZUSD', 'XMLNZUSD', 'XREPZUSD', 'XRPRLUSD', 'XRTUSD', 'XTZUSD', 'XXBTZUSD',
                'XXLMZUSD', 'XXMRZUSD', 'XXRPZUSD', 'XZECZUSD', 'YFIUSD', 'YGGUSD', 'ZEREBROUSD', 'ZETAUSD',
                'ZEURZUSD', 'ZEUSUSD', 'ZEXUSD', 'ZGBPZUSD', 'ZKUSD', 'ZORAUSD', 'ZROUSD', 'ZRXUSD'
            ]
        elif ecosystem == "BTC":
            pairs = [
                'AAVEXBT', 'ADAXBT', 'ALGOXBT', 'ANKRXBT', 'ATOMXBT', 'BCHXBT', 'COMPXBT', 'DOTXBT', 'FILXBT', 'GRTXBT',
                'LINKXBT', 'MANAXBT', 'MATICXBT', 'MINAXBT', 'MKRXBT', 'PAXGXBT', 'SANDXBT', 'SCXBT', 'SNXXBT', 'SOLXBT',
                'TBTCXBT', 'TRXXBT', 'UNIXBT', 'WBTCXBT', 'XETCXXBT', 'XETHXXBT', 'XLTCXXBT', 'XMLNXXBT', 'XXDGXXBT',
                'XXLMXXBT', 'XXMRXXBT', 'XXRPXXBT', 'XZECXXBT', 'ZRXXBT'
            ]
        else:
            pairs = []

        print("Loaded Pairs")
        # Load API keys from Kraken_API.json
        api_key = None
        api_secret = None
        try:
            with open(os.path.join(os.path.dirname(__file__), 'Kraken_API.json'), 'r') as f:
                api_data = json.load(f)
                api_key = api_data.get('api_key')
                api_secret = api_data.get('api_secret')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load Kraken_API.json: {e}")
            self.log_message(f"[ERROR] Failed to load Kraken_API.json: {e}")
            return

        print ("Loaded API")

        if not api_key or not api_secret:
            QtWidgets.QMessageBox.warning(self, "Error", "Please enter both API key and secret in Kraken_API.json!")
            return
        try:
            self.bot = KrakenProBot(api_key, api_secret, self.config)
            # Set up display maps for forced pairs
            print("Disaply Map...")
            pair_display_map = {
                'AI16ZUSDC': 'AI16Z/USDC',
                'ALGOUSDC': 'ALGO/USDC',
                'APEUSDC': 'APE/USDC',
                'ATOMUSDC': 'ATOM/USDC',
                'AVAXUSDC': 'AVAX/USDC',
                'BCHUSDC': 'BCH/USDC',
                'BERAUSDC': 'BERA/USDC',
                'BNBUSDC': 'BNB/USDC',
                'CROUSDC': 'CRO/USDC',
                'DOTUSDC': 'DOT/USDC',
                'EOSUSDC': 'EOS/USDC',
                'ETHUSDC': 'ETH/USDC',
                'EUROPUSDC': 'EUROP/USDC',
                'EURRUSDC': 'EURR/USDC',
                'FARTCOINUSDC': 'FARTCOIN/USDC',
                'LINKUSDC': 'LINK/USDC',
                'LTCUSDC': 'LTC/USDC',
                'MANAUSDC': 'MANA/USDC',
                'MATICUSDC': 'MATIC/USDC',
                'MELANIAUSDC': 'MELANIA/USDC',
                'PENGUUSDC': 'PENGU/USDC',
                'RLUSDUSDC': 'RLUSD/USDC',
                'SHIBUSDC': 'SHIB/USDC',
                'SOLUSDC': 'SOL/USDC',
                'SUSDC': 'S/USDC',
                'TONUSDC': 'TON/USDC',
                'TRUMPUSDC': 'TRUMP/USDC',
                'USDGUSDC': 'USDG/USDC',
                'USDQUSDC': 'USDQ/USDC',
                'USDRUSDC': 'USDR/USDC',
                'USTUSDC': 'UST/USDC',
                'VIRTUALUSDC': 'VIRTUAL/USDC',
                'XBTUSDC': 'XBT/USDC',
                'XDGUSDC': 'XDG/USDC',
                'XMRUSDC': 'XMR/USDC',
                'XRPUSDC': 'XRP/USDC',
                'XTZUSDC': 'XTZ/USDC'
            }
            display_to_api_map = {v: k for k, v in pair_display_map.items()}
            self.bot.pair_display_map = pair_display_map
            self.bot.display_to_api_map = display_to_api_map
            print("Set Trading Pairs...")
            self.bot.set_trading_pairs(pairs)
            print("Set Trading Pairs Done...")
            print("Set Bot Thread...")
            self.bot_thread = TradingBotThread(self.bot)
            print("Set Bot Thread Done...")
            print("Connect Signals...")
            self.bot_thread.update_signal.connect(self.update_gui)
            self.bot_thread.error_signal.connect(self.handle_error)
            self.bot_thread.status_signal.connect(self.update_bot_status)
            self.bot_thread.log_signal.connect(self.log_message)
            self.bot_thread.start()
            print("Start Bot Thread...")
            self.status_label.setText("Connected")
            self.status_label.setStyleSheet("color: green;")
            self.log_message("Connected to Kraken Pro (forced pairs)")
            print("Connected to Kraken Pro (forced pairs) Done...")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to connect: {str(e)}")
            self.log_message(f"Connection error: {str(e)}")
            
    def place_order(self):
        if not self.bot:
            QtWidgets.QMessageBox.warning(self, "Error", "Not connected to Kraken Pro")
            return
        try:
            display_pair = self.pair_combo.currentText()
            ecosystem = self.config.ecosystem
            # Use mapping if available
            if hasattr(self.bot, 'display_to_api_map'):
                api_pair = self.bot.display_to_api_map.get(display_pair, display_pair)
            else:
                base = display_pair.split('/')[0] if '/' in display_pair else display_pair.replace('USDC', '').replace('USD', '').replace('XBT', '')
                if ecosystem == 'USDC':
                    api_pair = f"{base}USDC"
                elif ecosystem == 'USD':
                    api_pair = f"{base}USD"
                elif ecosystem == 'BTC':
                    api_pair = f"{base}XBT"
                else:
                    api_pair = display_pair
            order_type = OrderType(self.order_type_combo.currentText())
            side = OrderSide(self.side_combo.currentText())
            amount = self.amount_input.value()
            price = self.price_input.value()

            # Disconnect previous signal to avoid duplicate connections
            try:
                self.bot_thread.order_worker.result_signal.disconnect(self.on_order_placed)
            except Exception:
                pass
            self.bot_thread.order_worker.result_signal.connect(self.on_order_placed)

            # Send order request to worker thread
            if order_type in [OrderType.LIMIT, OrderType.STOP_LOSS]:
                self.bot_thread.order_worker.add_request('place_order', (api_pair, order_type, side, amount, price))
            else:
                self.bot_thread.order_worker.add_request('place_order', (api_pair, order_type, side, amount))
            self.log_message(f"Order request sent for {api_pair} ({order_type.value}, {side.value}, {amount})...")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to place order: {str(e)}")
            self.log_message(f"Order error: {str(e)}")

    def on_order_placed(self, req_type, result):
        if req_type != 'place_order':
            return
        if isinstance(result, Exception):
            QtWidgets.QMessageBox.critical(self, "Order Error", f"Order failed: {result}")
            self.log_message(f"Order error: {result}")
            return
        if 'error' in result and result['error']:
            QtWidgets.QMessageBox.critical(self, "Order Error", f"Order failed: {result['error']}")
            self.log_message(f"Order error: {result['error']}")
        else:
            self.log_message(f"Order placed: {result}")
        # Handle context menu sell
        if hasattr(self, '_pending_sell_order') and self._pending_sell_order:
            order_id = self._pending_sell_order['order_id']
            row = self._pending_sell_order['row']
            display_pair = self._pending_sell_order['display_pair']
            api_pair = self._pending_sell_order['pair']
            amount = self._pending_sell_order['amount']
            # Always create a closed order record, even if not in active_orders
            order = None
            if order_id in self.bot.active_orders:
                order = self.bot.active_orders[order_id].copy()
                order['status'] = 'Closed'
                order['timestamp'] = datetime.now()
            else:
                # Fallback: try to get as much info as possible from result
                txid = None
                price = None
                if isinstance(result, dict) and 'result' in result:
                    txid = result['result'].get('txid', [order_id])[0]
                    descr = result['result'].get('descr', {})
                    # Try to parse price from descr['order'] if possible
                    if 'order' in descr and '@' in descr['order']:
                        try:
                            price = float(descr['order'].split('@')[-1].strip())
                        except Exception:
                            price = None
                order = {
                    'order_id': txid or order_id,
                    'pair': api_pair,
                    'type': 'market',
                    'side': 'sell',
                    'volume': amount,
                    'price': price,
                    'stop_price': None,
                    'timestamp': datetime.now(),
                    'status': 'Closed',
                    'current_price': price,
                    'bought_price': None,
                    'pl_dollar': None,
                    'pl_percent': None
                }
            self.closed_orders.append(order)
            self.log_message(f"[DEBUG] Added to closed_orders: {order}")
            if order_id in self.bot.active_orders:
                del self.bot.active_orders[order_id]
            self.update_gui_with_closed_orders()
            self.orders_table.removeRow(row)
            self.log_message(f"[INFO] Successfully sold {amount} {display_pair} at market price")
            self._pending_sell_order = None
        # Handle sell all at market
        if hasattr(self, '_pending_sell_all') and self._pending_sell_all:
            if 'pair' in result:
                pair = result['pair']
                if pair in self._pending_sell_all:
                    self._pending_sell_all.remove(pair)
            if not self._pending_sell_all:
                self.log_message("Sell All at Market: All positions closed.")
                # Instead of direct update, fetch data via APIWorker
                self._pending_post_order_update = True
                self.bot_thread.api_worker.result_signal.disconnect(self.on_api_result_post_order)
                self.bot_thread.api_worker.result_signal.connect(self.on_api_result_post_order)
                self._pending_api_results = {}
                self.bot_thread.api_worker.add_request('get_balance')
                # Use the first checked or available pair for ticker
                pair = self.pair_combo.currentText()
                self.bot_thread.api_worker.add_request('get_ticker', (pair,))

    def on_api_result_post_order(self, req_type, result):
        if not hasattr(self, '_pending_api_results'):
            self._pending_api_results = {}
        self._pending_api_results[req_type] = result
        # Wait for both balance and ticker
        if 'get_balance' in self._pending_api_results and 'get_ticker' in self._pending_api_results:
            balance = self._pending_api_results['get_balance']
            ticker = self._pending_api_results['get_ticker']
            state = {
                'pair': self.pair_combo.currentText(),
                'balance': balance,
                'ticker': ticker,
                'active_orders': self.bot.active_orders,
                'analysis': {}  # Optionally, fetch analysis async as well
            }
            self.update_gui([state])
            self._pending_post_order_update = False
            self._pending_api_results = {}
            self.bot_thread.api_worker.result_signal.disconnect(self.on_api_result_post_order)

    def save_settings(self):
        self.config.max_usd_amount = self.max_usd_input.value()
        self.config.trailing_stop_percentage = self.trailing_stop_input.value()
        self.config.stop_loss_percentage = self.stop_loss_input.value()
        self.config.take_profit_percentage = self.take_profit_input.value()
        self.config.rsi_period = self.rsi_period_input.value()
        self.config.rsi_overbought = self.rsi_overbought_input.value()
        self.config.rsi_oversold = self.rsi_oversold_input.value()
        self.config.max_trade_percentage = self.maxTradePCT_input.value()
        self.config.ecosystem = self.ecosystem_comboBox.currentText()
        self.config.strategy = self.strategy_combo.currentText()
        if self.bot:
            self.bot.config = self.config
        self.log_message("Settings saved")
        
    def update_gui(self, states):
        # states is now a list of state dicts, one per pair
        if not states:
            return
        # Blacklist for pairs that error out
        if not hasattr(self, 'blacklisted_pairs'):
            self.blacklisted_pairs = set()
        new_blacklist = set()
        # Update trading pair combo dynamically (use first state for reference)
        self.pair_combo.clear()
        for i in range(self.pair_list_widget.count()):
            item = self.pair_list_widget.item(i)
            if item.checkState() == Qt.Checked:
                self.pair_combo.addItem(item.text())
        # Update market pair combo
        pairs = [self.pair_list_widget.item(i).text() for i in range(self.pair_list_widget.count()) if self.pair_list_widget.item(i).checkState() == Qt.Checked]
        if not pairs:
            pairs = [self.pair_list_widget.item(i).text() for i in range(self.pair_list_widget.count())]
        self.market_pair_combo.blockSignals(True)
        self.market_pair_combo.clear()
        for pair in pairs:
            self.market_pair_combo.addItem(pair)
        self.market_pair_combo.blockSignals(False)
        # Set current index to the first pair in states if available
        first_state = states[0]
        if first_state['pair'] in pairs:
            self.market_pair_combo.setCurrentText(first_state['pair'])
        # Update market data group for selected pair (use first state)
        self.update_market_data_group()
        # Update all pairs table
        self.all_pairs_table.setRowCount(len(states))
        for idx, state in enumerate(states):
            pair = state['pair']
            if pair in self.blacklisted_pairs:
                continue
            try:
                api_symbol = self.bot.get_api_symbol(pair)
                ticker_data = state['ticker']
                if isinstance(ticker_data, dict) and 'result' in ticker_data and api_symbol in ticker_data['result']:
                    ticker = ticker_data['result'][api_symbol]
                    price_str = ticker['c'][0]
                elif isinstance(ticker_data, dict) and 'error' in ticker_data and ticker_data['error']:
                    raise ValueError(f"Kraken API error for {pair}: {ticker_data['error']}")
                else:
                    raise ValueError(f"Ticker data missing for {pair}: {ticker_data}")
            except Exception as e:
                price_str = '--'
                self.log_message(f"[ERROR] Could not fetch price for {pair}: {e}")
                # Only blacklist if there is a Kraken API error or ticker_data is not a dict
                if (isinstance(state['ticker'], dict) and 'error' in state['ticker'] and state['ticker']['error']) or not isinstance(state['ticker'], dict):
                    new_blacklist.add(pair)
            self.all_pairs_table.setItem(idx, 0, QtWidgets.QTableWidgetItem(pair))
            self.all_pairs_table.setItem(idx, 1, QtWidgets.QTableWidgetItem(price_str))
        # Update market data (use first state)
        try:
            api_symbol = self.bot.get_api_symbol(first_state['pair'])
            ticker_data = first_state['ticker']
            if isinstance(ticker_data, dict) and 'result' in ticker_data and api_symbol in ticker_data['result']:
                ticker = ticker_data['result'][api_symbol]
                self.price_label.setText(f"${ticker['c'][0]}")
                self.volume_label.setText(f"${float(ticker['v'][1]):.2f}")
                change = float(ticker['p'][1]) - float(ticker['p'][0])
                self.change_label.setText(f"{change:.2f}%")
            elif isinstance(ticker_data, dict) and 'error' in ticker_data and ticker_data['error']:
                raise ValueError(f"Kraken API error for {first_state['pair']}: {ticker_data['error']}")
            else:
                raise ValueError(f"Ticker data missing for {first_state['pair']}: {ticker_data}")
        except Exception as e:
            self.log_message(f"[ERROR] Could not update market data: {e}")
        # Update technical indicators (use first state)
        try:
            analysis = first_state['analysis']
            self.rsi_label.setText(f"{analysis['rsi']:.2f}")
            self.ma_fast_label.setText(f"{analysis['fast_ma']:.2f}")
            self.ma_slow_label.setText(f"{analysis['slow_ma']:.2f}")
            self.bb_upper_label.setText(f"{analysis['upper_bb']:.2f}")
            self.bb_lower_label.setText(f"{analysis['lower_bb']:.2f}")
        except Exception as e:
            self.log_message(f"[ERROR] Could not update indicators: {e}")
        # Update balance (use first state)
        try:
            balance = first_state['balance']
            self.usd_balance_label.setText(f"${float(balance.get('ZUSD', 0)):.2f}")
            self.btc_balance_label.setText(f"{float(balance.get('XXBT', 0)):.8f}")
            self.eth_balance_label.setText(f"{float(balance.get('XETH', 0)):.8f}")
        except Exception as e:
            self.log_message(f"[ERROR] Could not update balances: {e}")
        # Update orders table (aggregate all active orders from all states)
        self.orders_table.setRowCount(0)
        active_orders = first_state['active_orders']
        # --- Track closed orders ---
        if not hasattr(self, 'closed_orders'):
            self.closed_orders = []
        for order_id, order in list(active_orders.items()):
            if order['pair'] in self.blacklisted_pairs:
                continue
            row = self.orders_table.rowCount()
            self.orders_table.insertRow(row)
            
            # Pair
            self.orders_table.setItem(row, 0, QtWidgets.QTableWidgetItem(order['pair']))
            # Type
            self.orders_table.setItem(row, 1, QtWidgets.QTableWidgetItem(order['type']))
            # Side
            self.orders_table.setItem(row, 2, QtWidgets.QTableWidgetItem(order['side']))
            # Amount
            self.orders_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(order['volume'])))
            
            # Cost (USD) column
            try:
                cost = float(order['volume']) * float(order.get('price', 0))
                cost_str = f"${cost:.2f}"
            except Exception as e:
                cost_str = '--'
                self.log_message(f"[ERROR] Cost calculation failed for {order_id}: {e}")
                new_blacklist.add(order['pair'])
            self.orders_table.setItem(row, 4, QtWidgets.QTableWidgetItem(cost_str))
            
            # Format price to 8 decimals if it's a float
            price_val = order.get('price', '--')
            try:
                price_str = f"{float(price_val):.8f}"
            except Exception as e:
                price_str = str(price_val)
                self.log_message(f"[ERROR] Price formatting failed for {order_id}: {e}")
                new_blacklist.add(order['pair'])
            self.orders_table.setItem(row, 5, QtWidgets.QTableWidgetItem(price_str))
            
            # Status
            self.orders_table.setItem(row, 6, QtWidgets.QTableWidgetItem("Active"))
            
            # Current Price, Bought Price, Stop Loss, P/L $, P/L %
            current_price = bought_price = stop_loss = pl_dollar = pl_percent = "--"
            pl_dollar_val = 0.0
            debug_msgs = []
            try:
                api_symbol = self.bot.get_api_symbol(order['pair'])
                ticker = None
                try:
                    ticker = self.bot.get_ticker(api_symbol)
                    current_price_val = float(ticker['result'][api_symbol]['c'][0])
                    current_price = f"{current_price_val:.8f}"
                except Exception as e:
                    debug_msgs.append(f"[ERROR] Could not fetch current price for {order_id} ({order['pair']}): {e}")
                    current_price = '--'
                    current_price_val = None
                    new_blacklist.add(order['pair'])
                
                pos = None
                bought_price_val = None
                amount = None
                if self.bot and hasattr(self.bot, 'demo_state'):
                    pos = self.bot.demo_state.positions.get(order['pair'])
                if pos:
                    try:
                        bought_price_val = float(pos['entry_price'])
                        bought_price = f"{bought_price_val:.8f}"
                        amount = float(pos['amount'])
                        # Calculate stop loss price
                        if hasattr(self.bot, 'config') and self.bot.config.stop_loss_percentage:
                            stop_loss_price = bought_price_val * (1 - self.bot.config.stop_loss_percentage / 100)
                            stop_loss = f"{stop_loss_price:.8f}"
                    except Exception as e:
                        debug_msgs.append(f"[ERROR] Could not get bought price/amount from position for {order_id}: {e}")
                        new_blacklist.add(order['pair'])
                else:
                    # Fallback: use order price if available
                    try:
                        bought_price_val = float(order.get('price', 0))
                        bought_price = f"{bought_price_val:.8f}"
                        amount = float(order.get('volume', 0))
                        # Calculate stop loss price
                        if hasattr(self.bot, 'config') and self.bot.config.stop_loss_percentage:
                            stop_loss_price = bought_price_val * (1 - self.bot.config.stop_loss_percentage / 100)
                            stop_loss = f"{stop_loss_price:.8f}"
                        debug_msgs.append(f"[ERROR] Used order price as bought price for {order_id}")
                    except Exception as e:
                        debug_msgs.append(f"[ERROR] Could not get bought price from order for {order_id}: {e}")
                        new_blacklist.add(order['pair'])
                
                # Calculate P/L if possible
                if bought_price_val is not None and current_price_val is not None and amount is not None:
                    try:
                        pl_dollar_val = (current_price_val - bought_price_val) * amount
                        pl_percent_val = ((current_price_val - bought_price_val) / bought_price_val * 100) if bought_price_val else 0
                        pl_dollar = f"{pl_dollar_val:.2f}"
                        pl_percent = f"{pl_percent_val:.2f}%"
                    except Exception as e:
                        debug_msgs.append(f"[ERROR] P/L calculation failed for {order_id}: {e}")
                        new_blacklist.add(order['pair'])
                else:
                    debug_msgs.append(f"[ERROR] Missing values for P/L calculation for {order_id}: bought_price_val={bought_price_val}, current_price_val={current_price_val}, amount={amount}")
                    new_blacklist.add(order['pair'])
            except Exception as e:
                debug_msgs.append(f"[ERROR] General error for {order_id}: {e}")
                new_blacklist.add(order['pair'])
            
            # Set all the values
            self.orders_table.setItem(row, 7, QtWidgets.QTableWidgetItem(str(current_price)))
            self.orders_table.setItem(row, 8, QtWidgets.QTableWidgetItem(str(bought_price)))
            self.orders_table.setItem(row, 9, QtWidgets.QTableWidgetItem(str(stop_loss)))
            
            pl_dollar_item = QtWidgets.QTableWidgetItem(str(pl_dollar))
            pl_percent_item = QtWidgets.QTableWidgetItem(str(pl_percent))
            
            # Color coding for P/L columns
            if pl_dollar not in ("--", "0.00"):
                try:
                    val = float(pl_dollar)
                    if val > 0:
                        pl_dollar_item.setForeground(QColor("#2ecc40"))
                        pl_percent_item.setForeground(QColor("#2ecc40"))
                    elif val < 0:
                        pl_dollar_item.setForeground(QColor("#ff4136"))
                        pl_percent_item.setForeground(QColor("#ff4136"))
                except Exception as e:
                    debug_msgs.append(f"[ERROR] Color coding failed for {order_id}: {e}")
                    new_blacklist.add(order['pair'])
            
            self.orders_table.setItem(row, 10, pl_dollar_item)
            self.orders_table.setItem(row, 11, pl_percent_item)
            self.orders_table.setItem(row, 12, QtWidgets.QTableWidgetItem(order_id))
            
            # Emit debug logs if any
            for msg in debug_msgs:
                self.log_message(msg)
            # If order is closed, move to closed_orders
            if order.get('status', 'Active') == 'Closed':
                if order not in self.closed_orders:
                    self.closed_orders.append(order)
                del active_orders[order_id]
        # Update closed orders table
        self.update_gui_with_closed_orders()
        # Update performance metrics if in demo mode (use first state)
        if self.config.demo_mode and self.bot:
            try:
                metrics = self.bot.get_performance_metrics()
                self.total_trades_label.setText(str(metrics['total_trades']))
                self.winning_trades_label.setText(str(metrics['winning_trades']))
                self.win_rate_label.setText(f"{metrics['win_rate']:.2f}%")
                self.profit_loss_label.setText(f"${metrics['profit_loss']:.2f}")
                self.profit_loss_percentage_label.setText(f"{metrics['profit_loss_percentage']:.2f}%")
                self.running_time_label.setText(f"{metrics['running_time']:.1f} hours")
            except Exception as e:
                self.log_message(f"[ERROR] Could not update performance metrics: {e}")
        # RLUSD and Unrealized P/L (use first state)
        try:
            if self.config.demo_mode and self.bot:
                rlusd = float(balance.get('RLUSD', 0))
                self.rlusd_label.setText(f"Realized P/L (USD): ${rlusd:.2f}")
                # Calculate unrealized P/L (demo)
                unrealized_pl = 0.0
                if hasattr(self.bot, 'demo_state'):
                    for pair, pos in self.bot.demo_state.positions.items():
                        entry = pos['entry_price']
                        amount = pos['amount']
                        try:
                            api_symbol = self.bot.get_api_symbol(pair)
                            ticker_data = self.bot.get_ticker(api_symbol)
                            current = ticker_data['result'][api_symbol]['c'][0]
                            unrealized_pl += (float(current) - float(entry)) * amount
                        except Exception:
                            pass
                self.unrealized_pl_label.setText(f"Unrealized P/L (USD): ${unrealized_pl:.2f}")
            elif self.bot:
                # LIVE MODE: fetch from Kraken
                realized_pl = self.bot.get_realized_pl_live()
                unrealized_pl = self.bot.get_unrealized_pl_live()
                self.rlusd_label.setText(f"Realized P/L (USD): ${realized_pl:.2f}")
                self.unrealized_pl_label.setText(f"Unrealized P/L (USD): ${unrealized_pl:.2f}")
        except Exception as e:
            self.log_message(f"[ERROR] Could not update RLUSD/Unrealized P/L: {e}")
        # Update blacklist for next cycle
        self.blacklisted_pairs = new_blacklist

    def handle_error(self, error_msg):
        self.log_message(f"Error: {error_msg}")
        
    def log_message(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
    def closeEvent(self, event):
        if self.bot_thread:
            self.bot_thread.stop()
            self.bot_thread.wait()
        event.accept()

    def update_max_trade(self, value):
        try:
            self.config.max_usd_amount = float(value)
        except ValueError:
            pass

    def update_concurrent_trades(self, value):
        self.config.concurrent_trades = value
        if self.bot:
            self.bot.config.concurrent_trades = value

    def filter_pairs(self, text):
        for i in range(self.pair_list_widget.count()):
            item = self.pair_list_widget.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def toggle_auto_select_pairs(self, state):
        is_auto = state == Qt.Checked
        self.pair_list_widget.setDisabled(is_auto)
        self.pair_filter_input.setDisabled(is_auto)

    def update_bot_status(self, status):
        if self.bot_status_label:
            self.bot_status_label.setText(f"Bot status: {status}")
            self.log_message(f"Status: {status}")

    def start_trading(self):
        self.save_settings()
        self.log_message("[DEBUG] start_trading called")
        if not self.bot:
            QtWidgets.QMessageBox.warning(self, "Error", "Please connect to Kraken Pro before starting trading.")
            return
        selected_displays = [self.pair_list_widget.item(i).text() for i in range(self.pair_list_widget.count()) if self.pair_list_widget.item(i).checkState() == Qt.Checked]
        if not selected_displays:
            selected_displays = [self.pair_list_widget.item(i).text() for i in range(self.pair_list_widget.count())]
        selected_pairs = [self.bot.display_to_api_map.get(d, d) for d in selected_displays]
        self.log_message(f"[DEBUG] Trading started with pairs: {selected_pairs}")
        self.bot.set_trading_pairs(selected_pairs)
        # If the bot_thread is already running, stop and restart it
        if self.bot_thread and self.bot_thread.isRunning():
            self.bot_thread.stop()
            self.bot_thread.wait()
        self.bot_thread = TradingBotThread(self.bot)
        self.bot_thread.update_signal.connect(self.update_gui)
        self.bot_thread.error_signal.connect(self.handle_error)
        self.bot_thread.status_signal.connect(self.update_bot_status)
        self.bot_thread.log_signal.connect(self.log_message)
        self.bot_thread.start()

    def sell_all_at_market(self):
        # Instruct the bot to sell all open positions at market price using the worker thread
        if self.bot and hasattr(self.bot, 'demo_state'):
            self._pending_sell_all = []
            for pair, pos in list(self.bot.demo_state.positions.items()):
                amount = pos['amount']
                if amount > 0:
                    # Use worker thread for each sell
                    try:
                        self.bot_thread.order_worker.result_signal.disconnect(self.on_order_placed)
                    except Exception:
                        pass
                    self.bot_thread.order_worker.result_signal.connect(self.on_order_placed)
                    self._pending_sell_all.append(pair)
                    self.bot_thread.order_worker.add_request('place_order', (pair, OrderType.MARKET, OrderSide.SELL, amount))
            self.log_message("Sell All at Market: Sell requests sent for all positions.")

    def show_orders_context_menu(self, pos):
        index = self.orders_table.indexAt(pos)
        if not index.isValid():
            return
        row = index.row()
        menu = QtWidgets.QMenu()
        sell_action = menu.addAction("Sell at Market")
        action = menu.exec_(self.orders_table.viewport().mapToGlobal(pos))
        if action == sell_action:
            try:
                display_pair = self.orders_table.item(row, 0).text()
                amount = float(self.orders_table.item(row, 3).text())
                order_id = self.orders_table.item(row, 12).text()
                ecosystem = self.config.ecosystem
                if hasattr(self.bot, 'display_to_api_map'):
                    api_pair = self.bot.display_to_api_map.get(display_pair, display_pair)
                else:
                    base = display_pair.split('/')[0] if '/' in display_pair else display_pair.replace('USDC', '').replace('USD', '').replace('XBT', '')
                    if ecosystem == 'USDC':
                        api_pair = f"{base}USDC"
                    elif ecosystem == 'USD':
                        api_pair = f"{base}USD"
                    elif ecosystem == 'BTC':
                        api_pair = f"{base}XBT"
                    else:
                        api_pair = display_pair
                if self.bot:
                    # Use worker thread for sell
                    try:
                        self.bot_thread.order_worker.result_signal.disconnect(self.on_order_placed)
                    except Exception:
                        pass
                    self.bot_thread.order_worker.result_signal.connect(self.on_order_placed)
                    self._pending_sell_order = {'order_id': order_id, 'row': row, 'pair': api_pair, 'amount': amount, 'display_pair': display_pair}
                    self.bot_thread.order_worker.add_request('place_order', (api_pair, OrderType.MARKET, OrderSide.SELL, amount))
                    self.log_message(f"Sell at Market: Sell request sent for {display_pair}.")
            except Exception as e:
                self.log_message(f"[ERROR] Error during sell action: {str(e)}")
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to sell position: {str(e)}")

    def update_gui_with_closed_orders(self):
        # Update closed orders table
        self.closed_orders_table.setRowCount(len(self.closed_orders))
        for idx, order in enumerate(self.closed_orders):
            self.closed_orders_table.setItem(idx, 0, QtWidgets.QTableWidgetItem(order.get('order_id', '--')))
            self.closed_orders_table.setItem(idx, 1, QtWidgets.QTableWidgetItem(order.get('pair', '--')))
            self.closed_orders_table.setItem(idx, 2, QtWidgets.QTableWidgetItem(order.get('type', '--')))
            self.closed_orders_table.setItem(idx, 3, QtWidgets.QTableWidgetItem(order.get('side', '--')))
            self.closed_orders_table.setItem(idx, 4, QtWidgets.QTableWidgetItem(str(order.get('volume', '--'))))
            self.closed_orders_table.setItem(idx, 5, QtWidgets.QTableWidgetItem(str(order.get('cost', '--'))))
            self.closed_orders_table.setItem(idx, 6, QtWidgets.QTableWidgetItem(str(order.get('price', '--'))))
            self.closed_orders_table.setItem(idx, 7, QtWidgets.QTableWidgetItem(order.get('status', 'Closed')))
            self.closed_orders_table.setItem(idx, 8, QtWidgets.QTableWidgetItem(str(order.get('current_price', '--'))))
            self.closed_orders_table.setItem(idx, 9, QtWidgets.QTableWidgetItem(str(order.get('bought_price', '--'))))
            self.closed_orders_table.setItem(idx, 10, QtWidgets.QTableWidgetItem(str(order.get('pl_dollar', '--'))))
            self.closed_orders_table.setItem(idx, 11, QtWidgets.QTableWidgetItem(str(order.get('pl_percent', '--'))))

    def update_market_data_group(self):
        pair = self.market_pair_combo.currentText()
        if not pair or not self.bot:
            self.market_pair_name_label.setText('--')
            self.price_label.setText('--')
            self.volume_label.setText('--')
            self.change_label.setText('--')
            return
        try:
            ticker = self.bot.get_ticker(pair)['result'][pair]
            self.market_pair_name_label.setText(pair)
            self.price_label.setText(f"${ticker['c'][0]}")
            self.volume_label.setText(f"${float(ticker['v'][1]):.2f}")
            change = float(ticker['p'][1]) - float(ticker['p'][0])
            self.change_label.setText(f"{change:.2f}%")
        except Exception:
            self.market_pair_name_label.setText(pair)
            self.price_label.setText('--')
            self.volume_label.setText('--')
            self.change_label.setText('--')

    def start_pair_fetch(self):
        print ("Starting pair fetch...")
        self.loading_label.setText("Loading trading pairs from Kraken... Please wait.")
        self.log_message("[INFO] Fetching trading pairs from Kraken. Please wait...")
        self.connect_button.setEnabled(False)
        self.pair_list_widget.clear()
        self.pair_combo.clear()  # Clear the pair combo box
        # Fetch all available USDC pairs from Kraken
        try:
            bot_for_pairs = self.bot if self.bot else KrakenProBot("demo", "demo", self.config)
            all_pairs = bot_for_pairs.fetch_available_pairs()
            usdc_pairs = [p for p in all_pairs if p.endswith('USDC')]
            if not usdc_pairs:
                raise ValueError("No USDC pairs found from Kraken API.")
        except Exception as e:
            self.log_message(f"[ERROR] Failed to fetch pairs: {e}")
            # Updated fallback list with all confirmed working pairs
            usdc_pairs = [
                'AI16ZUSDC', 'ALGOUSDC', 'APEUSDC', 'ATOMUSDC', 'AVAXUSDC', 'BCHUSDC', 'BERAUSDC', 'BNBUSDC',
                'CROUSDC', 'DOTUSDC', 'EOSUSDC', 'ETHUSDC', 'EUROPUSDC', 'EURRUSDC', 'FARTCOINUSDC', 'LINKUSDC', 'LTCUSDC',
                'MANAUSDC', 'MATICUSDC', 'MELANIAUSDC', 'PENGUUSDC', 'RLUSDUSDC', 'SHIBUSDC', 'SOLUSDC', 'SUSDC', 'TONUSDC',
                'TRUMPUSDC', 'USDGUSDC', 'USDQUSDC', 'USDRUSDC', 'USTUSDC', 'VIRTUALUSDC', 'XBTUSDC', 'XDGUSDC', 'XMRUSDC',
                'XRPUSDC', 'XTZUSDC'
            ]
        
        # Create display names for the pairs (keep the original format)
        pair_display_map = {
            'AI16ZUSDC': 'AI16Z/USDC',
            'ALGOUSDC': 'ALGO/USDC',
            'APEUSDC': 'APE/USDC',
            'ATOMUSDC': 'ATOM/USDC',
            'AVAXUSDC': 'AVAX/USDC',
            'BCHUSDC': 'BCH/USDC',
            'BERAUSDC': 'BERA/USDC',
            'BNBUSDC': 'BNB/USDC',
            'CROUSDC': 'CRO/USDC',
            'DOTUSDC': 'DOT/USDC',
            'EOSUSDC': 'EOS/USDC',
            'ETHUSDC': 'ETH/USDC',
            'EUROPUSDC': 'EUROP/USDC',
            'EURRUSDC': 'EURR/USDC',
            'FARTCOINUSDC': 'FARTCOIN/USDC',
            'LINKUSDC': 'LINK/USDC',
            'LTCUSDC': 'LTC/USDC',
            'MANAUSDC': 'MANA/USDC',
            'MATICUSDC': 'MATIC/USDC',
            'MELANIAUSDC': 'MELANIA/USDC',
            'PENGUUSDC': 'PENGU/USDC',
            'RLUSDUSDC': 'RLUSD/USDC',
            'SHIBUSDC': 'SHIB/USDC',
            'SOLUSDC': 'SOL/USDC',
            'SUSDC': 'S/USDC',
            'TONUSDC': 'TON/USDC',
            'TRUMPUSDC': 'TRUMP/USDC',
            'USDGUSDC': 'USDG/USDC',
            'USDQUSDC': 'USDQ/USDC',
            'USDRUSDC': 'USDR/USDC',
            'USTUSDC': 'UST/USDC',
            'VIRTUALUSDC': 'VIRTUAL/USDC',
            'XBTUSDC': 'XBT/USDC',
            'XDGUSDC': 'XDG/USDC',
            'XMRUSDC': 'XMR/USDC',
            'XRPUSDC': 'XRP/USDC',
            'XTZUSDC': 'XTZ/USDC'
        }
        
        # Create reverse mapping
        display_to_api_map = {v: k for k, v in pair_display_map.items()}
        
        self.fetched_pairs = usdc_pairs
        self.fetched_pair_display_map = pair_display_map
        self.fetched_display_to_api_map = display_to_api_map
        self.pair_list_widget.clear()
        for pair in usdc_pairs:
            display = pair_display_map.get(pair, pair)
            item = QtWidgets.QListWidgetItem(display)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.pair_list_widget.addItem(item)
            self.pair_combo.addItem(display)  # Add to pair combo box
        # If no pairs are checked, check all by default
        any_checked = any(self.pair_list_widget.item(i).checkState() == Qt.Checked for i in range(self.pair_list_widget.count()))
        if not any_checked:
            for i in range(self.pair_list_widget.count()):
                self.pair_list_widget.item(i).setCheckState(Qt.Checked)
        self.loading_label.setText("")
        self.connect_button.setEnabled(True)
        self.log_message(f"[INFO] Trading pairs loaded: {usdc_pairs}")

    def update_refresh_countdown(self):
        self.refresh_seconds_left -= 1
        if self.refresh_seconds_left <= 0:
            self.refresh_seconds_left = self.refresh_interval
        self.refresh_countdown_label.setText(f"Next refresh in: {self.refresh_seconds_left}s")
        self.cycle_countdown_label.setText(f"Next cycle in: {self.refresh_seconds_left}s")

    def force_refresh_cycle(self):
        if self.bot_thread and self.bot_thread.isRunning():
            self.log_message("[DEBUG] Force refresh cycle triggered.")
            self.refresh_seconds_left = 0  # Set to 0 to trigger immediate refresh
            self.cycle_countdown_label.setText("Starting new cycle...")
            self.bot_thread.msleep(1)  # Wake up the thread (if sleeping)

    def confirm_sell_all(self):
        reply = QtWidgets.QMessageBox.question(self, 'Confirm Sell All',
                                   'Are you sure you want to sell all open positions at market?',
                                   QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.sell_all_at_market()

    

    def update_pairs_for_ecosystem(self):
        ecosystem = self.ecosystem_comboBox.currentText()
        if ecosystem == "USDC":
            pairs = [
                'ADAUSDC','AI16ZUSDC', 'ALGOUSDC', 'APEUSDC', 'ATOMUSDC', 'AVAXUSDC', 'BCHUSDC', 'BERAUSDC', 'BNBUSDC',
                'CROUSDC', 'DOTUSDC', 'EOSUSDC', 'ETHUSDC', 'EUROPUSDC', 'EURRUSDC', 'FARTCOINUSDC', 'LINKUSDC', 'LTCUSDC',
                'MANAUSDC', 'MATICUSDC', 'MELANIAUSDC', 'PENGUUSDC', 'RLUSDUSDC', 'SHIBUSDC', 'SOLUSDC', 'SUSDC', 'TONUSDC',
                'TRUMPUSDC', 'USDGUSDC', 'USDQUSDC', 'USDRUSDC', 'USTUSDC', 'VIRTUALUSDC', 'XBTUSDC', 'XDGUSDC', 'XMRUSDC',
                'XRPUSDC', 'XTZUSDC'
            ]
        elif ecosystem == "BTC":
            pairs = [
                'AAVEXBT', 'ADAXBT', 'ALGOXBT', 'ANKRXBT', 'ATOMXBT', 'BCHXBT', 'COMPXBT', 'DOTXBT', 'FILXBT', 'GRTXBT',
                'LINKXBT', 'MANAXBT', 'MATICXBT', 'MINAXBT', 'MKRXBT', 'PAXGXBT', 'SANDXBT', 'SCXBT', 'SNXXBT', 'SOLXBT',
                'TBTCXBT', 'TRXXBT', 'UNIXBT', 'WBTCXBT', 'XETCXXBT', 'XETHXXBT', 'XLTCXXBT', 'XMLNXXBT', 'XXDGXXBT',
                'XXLMXXBT', 'XXMRXXBT', 'XXRPXXBT', 'XZECXXBT', 'ZRXXBT'
            ]
        elif ecosystem == "USD":
            pairs = [
                '1INCHUSD', 'AAVEUSD', 'ACAUSD', 'ACHUSD', 'ACTUSD', 'ACXUSD', 'ADAUSD', 'ADXUSD', 'AEROUSD', 'AEVOUSD',
                'AGLDUSD', 'AI16ZUSD', 'AIRUSD', 'AIXBTUSD', 'AKTUSD', 'ALCHUSD', 'ALCXUSD', 'ALGOUSD', 'ALICEUSD',
                'ALPHAUSD', 'ALTUSD', 'ANKRUSD', 'ANLOGUSD', 'ANONUSD', 'APENFTUSD', 'APEUSD', 'API3USD', 'APTUSD',
                'APUUSD', 'ARBUSD', 'ARCUSD', 'ARKMUSD', 'ARPAUSD', 'ARUSD', 'ASTRUSD', 'ATHUSD', 'ATLASUSD', 'ATOMUSD',
                'AUCTIONUSD', 'AUDIOUSD', 'AUDUSD', 'AVAAIUSD', 'AVAXUSD', 'AXSUSD', 'B3USD', 'BABYUSD', 'BADGERUSD',
                'BALUSD', 'BANANAS31USD', 'BANDUSD', 'BATUSD', 'BCHUSD', 'BEAMUSD', 'BERAUSD', 'BICOUSD', 'BIGTIMEUSD',
                'BIOUSD', 'BITUSD', 'BLURUSD', 'BLZUSD', 'BMTUSD', 'BNBUSD', 'BNCUSD', 'BNTUSD', 'BOBAUSD', 'BODENUSD',
                'BONDUSD', 'BONKUSD', 'BRICKUSD', 'BSXUSD', 'BTTUSD', 'C98USD', 'CAKEUSD', 'CELOUSD', 'CELRUSD', 'CFGUSD',
                'CHEEMSUSD', 'CHRUSD', 'CHZUSD', 'CLANKERUSD', 'CLOUDUSD', 'CLVUSD', 'COMPUSD', 'CORNUSD', 'COTIUSD',
                'COWUSD', 'CPOOLUSD', 'CQTUSD', 'CROUSD', 'CRVUSD', 'CSMUSD', 'CTSIUSD', 'CVCUSD', 'CVXUSD', 'CXTUSD',
                'CYBERUSD', 'DAIUSD', 'DASHUSD', 'DBRUSD', 'DENTUSD', 'DOGSUSD', 'DOLOUSD', 'DOTUSD', 'DRIFTUSD',
                'DRVUSD', 'DUCKUSD', 'DYDXUSD', 'DYMUSD', 'EDGEUSD', 'EGLDUSD', 'EIGENUSD', 'ELXUSD', 'ENAUSD', 'ENJUSD',
                'ENSUSD', 'EOSUSD', 'ETHFIUSD', 'ETHPYUSD', 'ETHWUSD', 'EULUSD', 'EUROPUSD', 'EURQUSD', 'EURRUSD',
                'EURTUSD', 'EWTUSD', 'FARMUSD', 'FARTCOINUSD', 'FETUSD', 'FHEUSD', 'FIDAUSD', 'FILUSD', 'FISUSD',
                'FLOKIUSD', 'FLOWUSD', 'FLRUSD', 'FLUXUSD', 'FORTHUSD', 'FWOGUSD', 'FXSUSD', 'GALAUSD', 'GALUSD',
                'GARIUSD', 'GFIUSD', 'GHIBLIUSD', 'GHSTUSD', 'GIGAUSD', 'GLMRUSD', 'GMTUSD', 'GMXUSD', 'GNOUSD',
                'GOATUSD', 'GRASSUSD', 'GRIFFAINUSD', 'GRTUSD', 'GSTUSD', 'GTCUSD', 'GUNUSD', 'GUSD', 'HDXUSD',
                'HFTUSD', 'HMSTRUSD', 'HNTUSD', 'HONEYUSD', 'HPOS10IUSD', 'ICPUSD', 'ICXUSD', 'IDEXUSD', 'IMXUSD',
                'INITUSD', 'INJUSD', 'INTRUSD', 'IPUSD', 'JAILSTOOLUSD', 'JASMYUSD', 'JSTUSD', 'JTOUSD', 'JUNOUSD',
                'JUPUSD', 'KAITOUSD', 'KARUSD', 'KASUSD', 'KAVAUSD', 'KEEPUSD', 'KERNELUSD', 'KEYUSD', 'KILTUSD',
                'KINTUSD', 'KINUSD', 'KMNOUSD', 'KNCUSD', 'KP3RUSD', 'KSMUSD', 'KUJIUSD', 'KUSD', 'L3USD', 'LAYERUSD',
                'LCXUSD', 'LDOUSD', 'LINKUSD', 'LITUSD', 'LMWRUSD', 'LOCKINUSD', 'LPTUSD', 'LQTYUSD', 'LRCUSD',
                'LSETHUSD', 'LSKUSD', 'LUNA2USD', 'LUNAUSD', 'MANAUSD', 'MASKUSD', 'MATICUSD', 'MCUSD', 'MELANIAUSD',
                'MEMEUSD', 'METISUSD', 'MEUSD', 'MEWUSD', 'MICHIUSD', 'MINAUSD', 'MIRUSD', 'MKRUSD', 'MNGOUSD',
                'MNTUSD', 'MOGUSD', 'MOODENGUSD', 'MOONUSD', 'MORPHOUSD', 'MOVEUSD', 'MOVRUSD', 'MSOLUSD', 'MUBARAKUSD',
                'MULTIUSD', 'MVUSD', 'MXCUSD', 'NANOUSD', 'NEARUSD', 'NEIROUSD', 'NILUSD', 'NMRUSD', 'NODLUSD',
                'NOSUSD', 'NOTUSD', 'NTRNUSD', 'NYMUSD', 'OCEANUSD', 'ODOSUSD', 'OGNUSD', 'OMGUSD', 'OMNIUSD',
                'OMUSD', 'ONDOUSD', 'OPUSD', 'ORCAUSD', 'ORDERUSD', 'OSMOUSD', 'OXTUSD', 'OXYUSD', 'PAXGUSD',
                'PDAUSD', 'PENDLEUSD', 'PENGUUSD', 'PEPEUSD', 'PERPUSD', 'PHAUSD', 'PLUMEUSD', 'PNUTUSD', 'POLISUSD',
                'POLSUSD', 'POLUSD', 'PONDUSD', 'PONKEUSD', 'POPCATUSD', 'PORTALUSD', 'POWRUSD', 'PRCLUSD', 'PRIMEUSD',
                'PROMPTUSD', 'PSTAKEUSD', 'PUFFERUSD', 'PYTHUSD', 'PYUSDUSD', 'QNTUSD', 'QTUMUSD', 'RADUSD', 'RAREUSD',
                'RARIUSD', 'RAYUSD', 'RBCUSD', 'REDUSD', 'RENDERUSD', 'RENUSD', 'REPV2USD', 'REQUSD', 'REZUSD',
                'RLCUSD', 'RLUSDUSD', 'ROOKUSD', 'RPLUSD', 'RSRUSD', 'RUNEUSD', 'SAFEUSD', 'SAGAUSD', 'SAMOUSD',
                'SANDUSD', 'SBRUSD', 'SCRTUSD', 'SCUSD', 'SDNUSD', 'SEIUSD', 'SGBUSD', 'SHIBUSD', 'SIGMAUSD',
                'SKYUSD', 'SNEKUSD', 'SNXUSD', 'SOLUSD', 'SONICUSD', 'SPELLUSD', 'SPICEUSD', 'SPXUSD', 'SRMUSD',
                'SSVUSD', 'STEPUSD', 'STGUSD', 'STORJUSD', 'STRDUSD', 'STRKUSD', 'STXUSD', 'SUIUSD', 'SUNDOGUSD',
                'SUNUSD', 'SUPERUSD', 'SUSD', 'SUSHIUSD', 'SWARMSUSD', 'SWELLUSD', 'SYNUSD', 'SYRUPUSD', 'TAOUSD',
                'TBTCUSD', 'TEERUSD', 'TERMUSD', 'TIAUSD', 'TITCOINUSD', 'TLMUSD', 'TNSRUSD', 'TOKENUSD', 'TOKEUSD',
                'TONUSD', 'TOSHIUSD', 'TRACUSD', 'TREMPUSD', 'TRUMPUSD', 'TRUUSD', 'TRXUSD', 'TURBOUSD', 'TUSD',
                'TUSDUSD', 'TVKUSD', 'UFDUSD', 'UMAUSD', 'UNFIUSD', 'UNIUSD', 'USDCUSD', 'USDDUSD', 'USDGUSD',
                'USDQUSD', 'USDRUSD', 'USDSUSD', 'USDTZUSD', 'USTUSD', 'USUALUSD', 'VANRYUSD', 'VELODROMEUSD',
                'VINEUSD', 'VIRTUALUSD', 'VVVUSD', 'WALUSD', 'WAXLUSD', 'WBTCUSD', 'WCTUSD', 'WELLUSD', 'WENUSD',
                'WIFUSD', 'WINUSD', 'WLDUSD', 'WOOUSD', 'WUSD', 'XBTPYUSD', 'XCNUSD', 'XDGUSD', 'XETCZUSD',
                'XETHZUSD', 'XLTCZUSD', 'XMLNZUSD', 'XREPZUSD', 'XRPRLUSD', 'XRTUSD', 'XTZUSD', 'XXBTZUSD',
                'XXLMZUSD', 'XXMRZUSD', 'XXRPZUSD', 'XZECZUSD', 'YFIUSD', 'YGGUSD', 'ZEREBROUSD', 'ZETAUSD',
                'ZEURZUSD', 'ZEUSUSD', 'ZEXUSD', 'ZGBPZUSD', 'ZKUSD', 'ZORAUSD', 'ZROUSD', 'ZRXUSD'
            ]
        else:
            pairs = []
        # Update the pair list widget and combo boxes fix
        self.pair_list_widget.clear()
        self.pair_combo.clear()
        for pair in pairs:
            item = QtWidgets.QListWidgetItem(pair)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.pair_list_widget.addItem(item)
            self.pair_combo.addItem(pair)

    def on_ecosystem_changed(self):
        self.update_pairs_for_ecosystem()
        self.save_settings()

    def toggle_trading_pause(self, checked):
        """Toggle trading pause state and update UI accordingly."""
        self.trading_paused = checked
        if checked:
            self.pause_trading_button.setText("Resume Trading")
            self.trading_tab.setStyleSheet("""
                QWidget {
                    background-color: rgb(120, 100, 80);
                }
            """)
            self.log_message("[INFO] Trading paused - No new orders will be placed")
        else:
            self.pause_trading_button.setText("Pause Trading")
            self.trading_tab.setStyleSheet("")
            self.log_message("[INFO] Trading resumed - Orders will be placed normally")

    def update_cycle_timer(self, value):
        try:
            minutes = float(value)
            if minutes <= 0:
                return
            self.refresh_interval = int(minutes * 60)
            self.refresh_seconds_left = self.refresh_interval
            if self.bot_thread:
                self.bot_thread.set_cycle_interval(self.refresh_interval)
            self.log_message(f"Cycle timer updated to {minutes} minutes ({self.refresh_interval} seconds)")
        except Exception as e:
            self.log_message(f"[ERROR] Invalid cycle timer value: {e}")

    def refresh_strategy_combo(self):
        """Refresh the strategy combo box with the latest .py files in the strategies folder."""
        strategies_dir = os.path.join(os.path.dirname(__file__), "strategies")
        if not os.path.exists(strategies_dir):
            self.log_message(f"[ERROR] Strategies folder not found: {strategies_dir}")
            return
        files = [f for f in os.listdir(strategies_dir) if f.endswith('.py') and not f.startswith('_') and f != '__init__.py']
        strategy_names = [os.path.splitext(f)[0].replace('_', '-').title().replace('-', '-') for f in files]
        current_strategy = self.strategy_combo.currentText()
        self.strategy_combo.blockSignals(True)
        self.strategy_combo.clear()
        self.strategy_combo.addItems(strategy_names)
        # Try to restore previous selection
        if current_strategy in strategy_names:
            idx = strategy_names.index(current_strategy)
            self.strategy_combo.setCurrentIndex(idx)
        self.strategy_combo.blockSignals(False)
        self.log_message(f"[INFO] Refreshed strategies: {strategy_names}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = KrakenProGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 