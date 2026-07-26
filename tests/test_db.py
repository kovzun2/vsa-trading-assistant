import sys
import os
import sqlite3
import pytest

# Add parent directory to sys.path to import core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import core


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Изолирует БД тестов: подменяет путь signals.sqlite на временную папку."""
    db_path = str(tmp_path / "test_signals.sqlite")
    monkeypatch.setattr(core, "DB_PATH", db_path)
    core.init_db()
    return db_path


def _make_candles(times_prices):
    return [
        {"time": t, "open": p, "high": p + 0.0010, "low": p - 0.0010, "close": p, "volume": 100}
        for t, p in times_prices
    ]


def test_init_db_has_last_candle_time_column(temp_db):
    with sqlite3.connect(temp_db) as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(signals)").fetchall()]
    assert "last_candle_time" in cols
    assert "outcome" in cols


def test_log_signal_stores_last_candle_time(temp_db):
    candles = _make_candles([("2026-07-24 10:00:00", 1.1000)])
    core.log_signal("test-model", candles, {"prompt_tokens": 10, "completion_tokens": 5},
                    "reply", {"direction": "none"})

    assert core.get_last_analyzed_candle_time() == "2026-07-24 10:00:00"

    with sqlite3.connect(temp_db) as conn:
        row = conn.execute("SELECT model, prompt_tokens, completion_tokens, direction FROM signals").fetchone()
    assert row == ("test-model", 10, 5, "none")


def test_log_signal_without_candles_keeps_null_time(temp_db):
    core.log_signal("test-model", [], {}, "reply", None)
    assert core.get_last_analyzed_candle_time() is None

    # После сигнала со свечами — возвращается именно время свечи
    candles = _make_candles([("2026-07-24 11:00:00", 1.1000)])
    core.log_signal("test-model", candles, {}, "reply", None)
    assert core.get_last_analyzed_candle_time() == "2026-07-24 11:00:00"


def test_get_journal_stats_winrate(temp_db):
    for direction, outcome in [("long", "hit_tp"), ("short", "hit_tp"), ("long", "hit_sl"),
                               ("long", "pending"), ("none", "pending")]:
        with sqlite3.connect(temp_db) as conn:
            conn.execute("INSERT INTO signals (direction, outcome) VALUES (?, ?)", (direction, outcome))

    stats = core.get_journal_stats()
    assert stats["hit_tp"] == 2
    assert stats["hit_sl"] == 1
    assert stats["pending"] == 1  # direction 'none' не считается сделкой
    assert stats["total_closed"] == 3
    assert stats["winrate"] == 66.7


def test_estimate_cost():
    # Claude с кэшем: (1000 - 400) * 3.00/M + 400 * 0.30/M + 100 * 15.00/M
    cost = core.estimate_cost("anthropic/claude-3.5-sonnet", 1000, 100, cached_tokens=400)
    assert cost == pytest.approx(600 * 3.00 / 1e6 + 400 * 0.30 / 1e6 + 100 * 15.00 / 1e6)

    # Gemini: без учёта кэша
    cost = core.estimate_cost("google/gemini-1.5-pro", 1000, 100)
    assert cost == pytest.approx(1000 * 1.25 / 1e6 + 100 * 5.00 / 1e6)

    # Fallback (glm и прочие)
    cost = core.estimate_cost("z-ai/glm-5.2", 1000, 100)
    assert cost == pytest.approx(1000 * 0.50 / 1e6 + 100 * 1.50 / 1e6)


def _insert_pending(temp_db, direction, entry, stop, take, candle_time):
    with sqlite3.connect(temp_db) as conn:
        cur = conn.execute(
            "INSERT INTO signals (direction, entry, stop, take, outcome, last_candle_time) "
            "VALUES (?, ?, ?, ?, 'pending', ?)",
            (direction, entry, stop, take, candle_time),
        )
        return cur.lastrowid


def _outcome_of(temp_db, signal_id):
    with sqlite3.connect(temp_db) as conn:
        return conn.execute("SELECT outcome FROM signals WHERE id = ?", (signal_id,)).fetchone()[0]


def test_check_pending_outcomes_hit_tp(temp_db):
    sig_id = _insert_pending(temp_db, "long", 1.1000, 1.0950, 1.1050, "2026-07-24 10:00:00")
    candles = [
        # Касание входа, без TP/SL
        {"time": "2026-07-24 10:15:00", "high": 1.1020, "low": 1.0990},
        # Касание тейка
        {"time": "2026-07-24 10:30:00", "high": 1.1060, "low": 1.1000},
    ]
    assert core.check_pending_outcomes(candles) == 1
    assert _outcome_of(temp_db, sig_id) == "hit_tp"


def test_check_pending_outcomes_hit_sl_conservative(temp_db):
    sig_id = _insert_pending(temp_db, "long", 1.1000, 1.0950, 1.1050, "2026-07-24 10:00:00")
    # Одна свеча задевает и вход, и TP, и SL -> консервативно фиксируем SL
    candles = [{"time": "2026-07-24 10:15:00", "high": 1.1060, "low": 1.0940}]
    assert core.check_pending_outcomes(candles) == 1
    assert _outcome_of(temp_db, sig_id) == "hit_sl"


def test_check_pending_outcomes_short_hit_sl(temp_db):
    sig_id = _insert_pending(temp_db, "short", 1.1000, 1.1050, 1.0950, "2026-07-24 10:00:00")
    candles = [
        {"time": "2026-07-24 10:15:00", "high": 1.1010, "low": 1.0990},  # касание входа
        {"time": "2026-07-24 10:30:00", "high": 1.1060, "low": 1.1000},  # касание стопа
    ]
    assert core.check_pending_outcomes(candles) == 1
    assert _outcome_of(temp_db, sig_id) == "hit_sl"


def test_check_pending_outcomes_no_entry_touch_stays_pending(temp_db):
    sig_id = _insert_pending(temp_db, "long", 1.1000, 1.0950, 1.1050, "2026-07-24 10:00:00")
    # Цена ушла вверх без касания входа — сделка не открывалась
    candles = [{"time": "2026-07-24 10:15:00", "high": 1.1100, "low": 1.1010}]
    assert core.check_pending_outcomes(candles) == 0
    assert _outcome_of(temp_db, sig_id) == "pending"


def test_check_pending_outcomes_ignores_old_candles(temp_db):
    sig_id = _insert_pending(temp_db, "long", 1.1000, 1.0950, 1.1050, "2026-07-24 10:00:00")
    # Свечи ДО свечи анализа не должны учитываться
    candles = [{"time": "2026-07-24 09:45:00", "high": 1.1060, "low": 1.0940}]
    assert core.check_pending_outcomes(candles) == 0
    assert _outcome_of(temp_db, sig_id) == "pending"
