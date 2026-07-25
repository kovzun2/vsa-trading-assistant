import os
import sys
import time
import json
import sqlite3
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

# Add parent to path if placed in scratch or tools, but we place it in root
sys.path.insert(0, os.path.dirname(__file__))
import core

load_dotenv()

def run_replay(symbol=core.DEFAULT_FUT_SYMBOL, spot_symbol=core.DEFAULT_SPOT_SYMBOL, days=60):
    print(f"Downloading {days} days of 15m data for {symbol} and {spot_symbol}...")
    
    ticker_fut = core.yf.Ticker(symbol)
    df_fut = ticker_fut.history(period=f"{days}d", interval="15m")
    
    ticker_spot = core.yf.Ticker(spot_symbol)
    df_spot = ticker_spot.history(period=f"{days}d", interval="15m")
    
    if df_fut.empty or df_spot.empty:
        print("No data.")
        return
        
    df_fut.index = pd.to_datetime(df_fut.index, utc=True)
    df_spot.index = pd.to_datetime(df_spot.index, utc=True)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("No API key in .env")
        return
        
    books_context = core.load_books()
    system_blocks = core.get_system_prompt_blocks(spot_symbol, core.DEFAULT_DEPOSIT, core.DEFAULT_RISK_PERCENT, books_context)
    
    limit = 100 # limit to 100 requests to avoid massive costs
    count = 0
    
    print(f"Total rows in futures data: {len(df_fut)}")
    
    original_ticker = core.yf.Ticker
    original_datetime = core.datetime
    
    # We will pick indices starting from 100 to end
    for i in range(100, len(df_fut)):
        if count >= limit:
            print(f"Reached {limit} requests limit.")
            break
            
        current_time = df_fut.index[i]
        
        window_fut = df_fut.loc[:current_time].tail(60)
        window_spot = df_spot.loc[:current_time].tail(60)
        
        if len(window_fut) < 60:
            continue
            
        class MockTicker:
            def __init__(self, sym):
                self.sym = sym
            def history(self, *args, **kwargs):
                if self.sym == spot_symbol: return window_spot
                if self.sym == symbol: return window_fut
                return pd.DataFrame()
                
        core.yf.Ticker = MockTicker
        
        class MockDatetime:
            @classmethod
            def now(cls, tz=None):
                t = current_time + pd.Timedelta(minutes=16)
                if tz:
                    return t.tz_convert(tz) if t.tzinfo else t.replace(tzinfo=tz)
                return t.tz_localize(None)
                
        core.datetime = MockDatetime
        
        try:
            candles, basis, is_stale = core.get_market_data(spot_symbol, symbol, core.DEFAULT_TZ_OFFSET)
        finally:
            core.yf.Ticker = original_ticker
            core.datetime = original_datetime
            
        if not candles or is_stale:
            continue
            
        # Optional: Only trade in NY session to save API calls
        if candles[-1].get("session") != "NY":
            continue
            
        user_prompt = (
            "Данные последних свечей (интервал 15м) в формате JSON:\n"
            f"{json.dumps(candles, indent=2, ensure_ascii=False)}"
        )
        messages = [
            {"role": "system", "content": system_blocks},
            {"role": "user", "content": user_prompt},
        ]
        
        print(f"[{count+1}/{limit}] {current_time} (NY) - Requesting LLM...")
        ok, raw_reply, usage = core.call_llm(messages, api_key, os.getenv("MODEL", core.DEFAULT_MODEL))
        if ok:
            display, signal = core.format_response(raw_reply, core.DEFAULT_DEPOSIT, core.DEFAULT_RISK_PERCENT, basis, candles[-1]['close'])
            core.log_signal(os.getenv("MODEL", core.DEFAULT_MODEL), candles, usage, raw_reply, signal)
            print(f"Logged signal. Direction: {signal.get('direction') if signal else 'None'}")
        else:
            print(f"LLM Error: {raw_reply}")
            
        count += 1
        time.sleep(2) # minimal delay

if __name__ == "__main__":
    run_replay()
