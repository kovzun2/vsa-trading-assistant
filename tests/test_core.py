import sys
import os
import pytest
import pickle
from unittest.mock import patch
import pandas as pd
from datetime import datetime, timezone

# Add parent directory to sys.path to import core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import core

@pytest.fixture
def mock_yfinance(monkeypatch):
    fixture_path = os.path.join(os.path.dirname(__file__), "fixture_15m.pkl")
    with open(fixture_path, "rb") as f:
        data = pickle.load(f)

    class MockTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period, interval):
            if self.symbol == core.DEFAULT_SPOT_SYMBOL:
                return data["df_spot"]
            elif self.symbol == core.DEFAULT_FUT_SYMBOL:
                return data["df_fut"]
            return pd.DataFrame()

    monkeypatch.setattr(core.yf, "Ticker", MockTicker)
    return data

@patch('core.datetime')
def test_get_market_data(mock_datetime, mock_yfinance):
    # Simulate current time right after the last fixture candle + 10 mins
    fixture_data = mock_yfinance
    df_fut = fixture_data["df_fut"]
    last_candle_time = df_fut.index[-1].tz_convert('UTC').tz_localize(None)
    
    mock_now = last_candle_time + pd.Timedelta(minutes=10)
    mock_datetime.now.return_value = mock_now.replace(tzinfo=timezone.utc)
    
    candles, basis, is_stale = core.get_market_data()
    
    assert candles is not None
    assert len(candles) == 60
    assert basis != 0.0
    assert not is_stale
    
    # Check VSA features
    last_candle = candles[-1]
    assert "rel_volume" in last_candle
    assert "spread" in last_candle
    assert "close_pos" in last_candle
    assert "session" in last_candle
    assert "has_gap" in last_candle

@patch('core.datetime')
def test_get_market_data_stale(mock_datetime, mock_yfinance):
    fixture_data = mock_yfinance
    df_fut = fixture_data["df_fut"]
    last_candle_time = df_fut.index[-1].tz_convert('UTC').tz_localize(None)
    
    # 40 mins after the open of the last closed candle.
    # The candle closes at +15m, so it will be 25 mins old, which is > 20m threshold.
    mock_now = last_candle_time + pd.Timedelta(minutes=40)
    mock_datetime.now.return_value = mock_now.replace(tzinfo=timezone.utc)
    
    candles, basis, is_stale = core.get_market_data()
    
    assert is_stale == True

def test_calculate_position_size_valid():
    res = core.calculate_position_size(deposit=1000, risk_percent=1, entry_price=1.1000, stop_price=1.0950, last_close=1.0990, contract_size=100000)
    assert "error" not in res
    assert res["direction_inferred"] == "long"
    assert res["stop_distance"] == 0.0050
    assert res["risk_amount_usd"] == 10.0
    assert res["lots"] == 0.02

def test_calculate_position_size_invalid():
    # Negative deposit
    res = core.calculate_position_size(deposit=-1000, risk_percent=1, entry_price=1.1000, stop_price=1.0950)
    assert "error" in res
    
    # Stop too small
    res = core.calculate_position_size(deposit=1000, risk_percent=1, entry_price=1.1000, stop_price=1.10001)
    assert "error" in res
    assert "минимума" in res["error"]
    
    # Entry too far from market
    res = core.calculate_position_size(deposit=1000, risk_percent=1, entry_price=1.1500, stop_price=1.1400, last_close=1.1000)
    assert "error" in res
    assert "далеко от рынка" in res["error"]
    
    # Lot limit exceeded
    res = core.calculate_position_size(deposit=1000000, risk_percent=5, entry_price=1.1000, stop_price=1.0990)
    assert "error" in res
    assert "лимита" in res["error"]

def test_extract_signal():
    # Standard Markdown
    text1 = "Here is the signal:\n```json\n{\"direction\": \"long\", \"entry\": 1.1, \"stop\": 1.09}\n```\nGood luck!"
    assert core.extract_signal(text1) == {"direction": "long", "entry": 1.1, "stop": 1.09}
    
    # Fallback (No markdown tags)
    text2 = "I recommend:\n{\"direction\": \"short\", \"entry\": 1.1, \"stop\": 1.11}\nWatch out for volatility."
    assert core.extract_signal(text2) == {"direction": "short", "entry": 1.1, "stop": 1.11}
    
    # Invalid JSON
    text3 = "{\"direction\": \"long\", \"entry\": }"
    assert core.extract_signal(text3) is None
