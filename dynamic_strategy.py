print("\n\n")
print("-------------------------------------------------------------------------")
print(" ⭐⭐ Loaded Dynamic Strategy Version 1.3.1 (Breakout-Focused) ⭐⭐")
print(" ⭐⭐ This strategy focuses on strong breakouts with volume confirmation. ⭐⭐")
print(" ⭐⭐ It is more selective and will wait for optimal conditions. ⭐⭐")
print(" ⭐⭐ Breakout Strategy: ⭐⭐")
print(" ⭐⭐ - Uses 10-period high/low range for breakout detection ⭐⭐")
print(" ⭐⭐ - Requires 1.5% breakout above recent high ⭐⭐")
print(" ⭐⭐ - Requires 2x volume confirmation ⭐⭐")
print(" ⭐⭐ - Requires RSI < 70 to avoid overbought ⭐⭐")
print(" ⭐⭐ - Requires MACD positive and increasing ⭐⭐")
print(" ⭐⭐ - Maximum 2 pairs per cycle ⭐⭐")
print("-------------------------------------------------------------------------")
print("\n\n")

#1.3.1 fixed the error when the price is 0

import pandas as pd
import ta

def select_top_pairs(market_data, params=None):
    try:
        print("\n[STRATEGY] Starting pair selection cycle...")
        scores = []
        analyzed_pairs = 0
        skipped_pairs = 0
        
        for pair, df in market_data.items():
            analyzed_pairs += 1
            if df is None or df.empty or len(df) < 20:
                print(f"[STRATEGY] Skipping {pair}: Insufficient data (len={len(df) if df is not None else 'None'})")
                skipped_pairs += 1
                continue

            # Filter out illiquid pairs
            if df['volume'].tail(5).sum() == 0:
                print(f"[STRATEGY] Skipping {pair}: Low liquidity (zero volume in last 5 candles)")
                skipped_pairs += 1
                continue

            # Ensure numeric data
            df = df.copy()
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            
            # Check for NaN values after conversion
            if df['close'].isnull().any() or df['volume'].isnull().any() or df['high'].isnull().any() or df['low'].isnull().any():
                print(f"[STRATEGY] Skipping {pair}: Invalid price data (NaN values detected)")
                skipped_pairs += 1
                continue

            # Calculate indicators with error handling
            try:
                # RSI calculation
                rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
                if pd.isna(rsi):
                    print(f"[STRATEGY] Skipping {pair}: Invalid RSI value")
                    skipped_pairs += 1
                    continue

                # MACD calculation
                macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
                macd_line = macd.macd().iloc[-1]
                macd_signal = macd.macd_signal().iloc[-1]
                macd_hist = macd.macd_diff().iloc[-1]
                macd_hist_prev = macd.macd_diff().iloc[-2]
                
                if any(pd.isna(x) for x in [macd_line, macd_signal, macd_hist, macd_hist_prev]):
                    print(f"[STRATEGY] Skipping {pair}: Invalid MACD values")
                    skipped_pairs += 1
                    continue

                # Volume analysis with validation
                volume = df['volume'].iloc[-1]
                avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
                
                if pd.isna(volume) or pd.isna(avg_volume) or avg_volume == 0:
                    print(f"[STRATEGY] Skipping {pair}: Invalid volume data")
                    skipped_pairs += 1
                    continue
                
                volume_increase = volume > avg_volume * 2.0

                # Breakout detection with validation
                recent_high = df['high'].rolling(window=10).max().iloc[-2]
                recent_low = df['low'].rolling(window=10).min().iloc[-2]
                
                if pd.isna(recent_high) or pd.isna(recent_low):
                    print(f"[STRATEGY] Skipping {pair}: Invalid high/low data")
                    skipped_pairs += 1
                    continue
                
                price_range = recent_high - recent_low
                if price_range == 0:
                    print(f"[STRATEGY] Skipping {pair}: Zero price range")
                    skipped_pairs += 1
                    continue
                
                breakout_threshold = price_range * 0.015
                current_price = df['close'].iloc[-1]
                
                if pd.isna(current_price):
                    print(f"[STRATEGY] Skipping {pair}: Invalid current price")
                    skipped_pairs += 1
                    continue
                
                is_breakout = current_price > recent_high + breakout_threshold

                # Momentum calculation with validation
                prev_price = df['close'].iloc[-2]
                if pd.isna(prev_price) or prev_price == 0:
                    print(f"[STRATEGY] Skipping {pair}: Invalid previous price")
                    skipped_pairs += 1
                    continue
                
                price_change = (current_price - prev_price) / prev_price * 100
                strong_momentum = price_change > 0.8

                # Log detailed analysis for each pair
                print(f"\n[STRATEGY] Analyzing {pair}:")
                print(f"  - Current Price: {current_price:.8f}")
                print(f"  - RSI: {rsi:.2f}")
                print(f"  - MACD Histogram: {macd_hist:.8f} (Previous: {macd_hist_prev:.8f})")
                print(f"  - Volume: {volume:.2f} (Avg: {avg_volume:.2f}, {volume/avg_volume:.2f}x)")
                print(f"  - Breakout: {is_breakout} (Threshold: {breakout_threshold:.8f})")
                print(f"  - Price Change: {price_change:.2f}%")

                # Scoring system - more selective
                score = 0
                conditions_met = []

                # RSI condition
                if 30 <= rsi <= 70:
                    score += 2
                    conditions_met.append("RSI in range")

                # MACD conditions
                if macd_hist > 0 and macd_hist > macd_hist_prev:
                    score += 2
                    conditions_met.append("MACD positive and increasing")

                # Volume condition
                if volume_increase:
                    score += 2
                    conditions_met.append("Volume > 2x average")

                # Breakout condition
                if is_breakout:
                    score += 3
                    conditions_met.append("Strong breakout")

                # Momentum condition
                if strong_momentum:
                    score += 2
                    conditions_met.append("Strong momentum")

                # Log conditions met
                if conditions_met:
                    print(f"  - Conditions met: {', '.join(conditions_met)}")
                    print(f"  - Final score: {score}")
                else:
                    print("  - No conditions met")

                # Only add to scores if meets minimum criteria
                if score >= 6:
                    scores.append((pair, score, price_change, volume))
                    print(f"  - ✅ Added to potential trades")

            except Exception as e:
                print(f"[STRATEGY] Error analyzing {pair}: {str(e)}")
                skipped_pairs += 1
                continue

        # Select top pairs
        selected = [(pair, score) for pair, score, change, vol in scores]
        selected = sorted(selected, key=lambda x: x[1], reverse=True)
        top_pairs = [pair for pair, score in selected[:2]]

        # Log final selection
        print(f"\n[STRATEGY] Analysis complete:")
        print(f"  - Analyzed pairs: {analyzed_pairs}")
        print(f"  - Skipped pairs: {skipped_pairs}")
        print(f"  - Pairs meeting criteria: {len(scores)}")
        if top_pairs:
            print(f"  - Selected pairs: {top_pairs}")
        else:
            print("  - No pairs selected - waiting for better conditions")

        return top_pairs

    except Exception as e:
        print(f"[STRATEGY] Error in select_top_pairs: {str(e)}")
        import traceback
        traceback.print_exc()
        return [] 