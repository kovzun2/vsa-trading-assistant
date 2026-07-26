import os
import re
import json
import time

import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import sqlite3
import hashlib
from dotenv import load_dotenv

load_dotenv()

# --- Настройки по умолчанию ---
DEFAULT_SPOT_SYMBOL = "EURUSD=X"
DEFAULT_FUT_SYMBOL = "6E=F"
DEFAULT_TZ_OFFSET = 3
DEFAULT_DEPOSIT = 1000.0
DEFAULT_RISK_PERCENT = 1.0
DEFAULT_MODEL = "z-ai/glm-5.2"
DEFAULT_CONTRACT_SIZE = 100_000  # 1 лот EURUSD = 100 000 EUR

# База знаний: структурированный референс по VSA/VPA (~35k токенов).
# Собран из книг Анны Коуллинг и Тома Вильямса (см. VSA_REFERENCE.md).
# Использование референса вместо полных книг (~500k токенов) экономит
# стоимость API и ускоряет ответы при сохранении покрытия методологии.
BOOK_FILES = [
    "VSA_REFERENCE.md",
]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_book_cache = None


def load_books():
    """Кэширует и возвращает содержимое книг-методичек."""
    global _book_cache
    if _book_cache is not None:
        return _book_cache
    books_content = ""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for file in BOOK_FILES:
        filepath = os.path.join(base_dir, file)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                books_content += f"\n--- НАЧАЛО {file} ---\n"
                books_content += f.read()
                books_content += f"\n--- КОНЕЦ {file} ---\n"
        else:
            raise FileNotFoundError(f"Критическая ошибка: файл базы знаний VSA не найден по пути {filepath}")
    _book_cache = books_content
    return _book_cache


def get_market_data(spot_symbol=DEFAULT_SPOT_SYMBOL, fut_symbol=DEFAULT_FUT_SYMBOL,
                    tz_offset=DEFAULT_TZ_OFFSET, n_candles=60):
    """Последние n свечей 15м: цены и объемы из фьючерса, расчет базиса по споту."""
    try:
        df_spot = yf.Ticker(spot_symbol).history(period="1mo", interval="15m")
        df_fut = yf.Ticker(fut_symbol).history(period="1mo", interval="15m")
        if df_spot.empty or df_fut.empty:
            return None, 0.0, False
            
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        
        df = pd.merge(
            df_spot[['Close']].rename(columns={'Close': 'Spot_Close'}),
            df_fut[['Open', 'High', 'Low', 'Close', 'Volume']],
            left_index=True, right_index=True, how='inner'
        )
        if df.empty:
            return None, 0.0, False
            
        df.index = df.index.tz_convert('UTC').tz_localize(None)
        
        # Отбрасываем незакрытую свечу
        df = df[df.index + pd.Timedelta(minutes=15) <= now_utc].copy()
        if df.empty:
            return None, 0.0, False
            
        is_stale = False
        last_closed_time = df.index[-1] + pd.Timedelta(minutes=15)
        if (now_utc - last_closed_time).total_seconds() > 20 * 60:
            is_stale = True

        df['basis'] = df['Close'] - df['Spot_Close']
        basis_median = df['basis'].tail(20).median()
        
        df['SMA20_Volume'] = df['Volume'].rolling(window=20).mean()
        df['rel_volume'] = np.where(df['SMA20_Volume'] > 0, df['Volume'] / df['SMA20_Volume'], 1.0)
        df['spread'] = df['High'] - df['Low']
        df['close_pos'] = np.where(df['spread'] > 0, (df['Close'] - df['Low']) / df['spread'], 0.5)
        
        df['time_diff'] = df.index.to_series().diff()
        df['has_gap'] = df['time_diff'] > pd.Timedelta(minutes=15)
        
        df = df.tail(n_candles).copy()
        df.index = df.index + pd.Timedelta(hours=tz_offset)
        
        def get_session(hour_utc):
            if 23 <= hour_utc or hour_utc < 8:
                return "Asia"
            elif 7 <= hour_utc < 16:
                return "London" if hour_utc < 12 else "Overlap"
            elif 16 <= hour_utc < 21:
                return "NY"
            else:
                return "Other"

        candles = []
        for index, row in df.iterrows():
            hour_utc = (index - pd.Timedelta(hours=tz_offset)).hour
            candles.append({
                "time": str(index),
                "open": round(row['Open'], 5),
                "high": round(row['High'], 5),
                "low": round(row['Low'], 5),
                "close": round(row['Close'], 5),
                "volume": int(row['Volume']),
                "rel_volume": round(float(row['rel_volume']) if not pd.isna(row['rel_volume']) else 1.0, 2),
                "spread": round(float(row['spread']), 5),
                "close_pos": round(float(row['close_pos']), 2),
                "session": get_session(hour_utc),
                "has_gap": bool(row['has_gap']) if not pd.isna(row['has_gap']) else False
            })
        return candles, round(basis_median, 5), is_stale
    except Exception as e:
        print(f"Ошибка при получении котировок: {e}")
        return None, 0.0, False


def calculate_position_size(deposit, risk_percent, entry_price, stop_price, last_close=None,
                            contract_size=DEFAULT_CONTRACT_SIZE):
    """Расчёт объёма позиции в лотах с валидацией."""
    if deposit <= 0:
        return {"error": f"Сигнал отклонён: некорректный депозит {deposit}"}
    if not (0 < risk_percent <= 100):
        return {"error": f"Сигнал отклонён: некорректный риск {risk_percent}%"}

    if entry_price is None or stop_price is None:
        return {"error": "Сигнал отклонён: отсутствуют уровни"}
    try:
        entry = float(entry_price)
        stop = float(stop_price)
    except (TypeError, ValueError):
        return {"error": "Сигнал отклонён: уровни не являются числами"}

    direction = "long" if stop < entry else "short"

    if last_close is not None:
        if abs(entry - last_close) > 0.0050:
            return {"error": f"Сигнал отклонён: цена входа {entry} далеко от рынка {last_close}"}

    stop_distance = abs(entry - stop)
    if stop_distance < 0.0005:
        return {"error": f"Сигнал отклонён: стоп {round(stop_distance, 5)} < минимума 0.0005"}

    risk_amount = deposit * (risk_percent / 100.0)
    loss_per_lot = stop_distance * contract_size
    if loss_per_lot == 0:
        return {"error": "Сигнал отклонён: loss_per_lot = 0"}
    lots = risk_amount / loss_per_lot
    
    if lots > 100:
        return {"error": f"Сигнал отклонён: объем {round(lots, 2)} > лимита 100 лотов"}

    return {
        "direction_inferred": direction,
        "risk_amount_usd": round(risk_amount, 2),
        "stop_distance": round(stop_distance, 5),
        "lots": round(lots, 4),
        "units": int(round(lots * contract_size)),
        "contract_size": contract_size,
    }


_SIGNAL_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def extract_signal(text):
    """Извлекает сигнал из ответа LLM с фолбэками."""
    matches = _SIGNAL_JSON_RE.findall(text)
    json_str = matches[-1] if matches else None
    
    if not json_str:
        fallback_re = re.compile(r"\{.*?\}", re.DOTALL)
        matches_fallback = fallback_re.findall(text)
        if matches_fallback:
            json_str = matches_fallback[-1]
            
    if not json_str:
        return None
        
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def round_to_tick(price, tick_size=0.00005):
    """Округляет уровень к шагу цены (тик-сайзу) фьючерса 6E."""
    if price is None:
        return None
    return round(round(float(price) / tick_size) * tick_size, 5)


def build_position_block(deposit, risk_percent, signal, basis, last_close, contract_size=DEFAULT_CONTRACT_SIZE):
    """Текстовый блок с расчётом позиции на основе сигнала LLM."""
    if not signal:
        return ""
    direction = str(signal.get("direction", "")).lower()
    if direction in ("", "none", "flat", "wait"):
        return "\n\n---\n**🤖 Сигнал:** Вне рынка (None)"

    entry = round_to_tick(signal.get("entry"))
    stop = round_to_tick(signal.get("stop"))
    take = round_to_tick(signal.get("take"))

    calc = calculate_position_size(deposit, risk_percent, entry, stop, last_close, contract_size)
    if "error" in calc:
         return f"\n\n---\n**📐 Расчёт позиции:**\n❌ {calc['error']}"
         
    spot_entry = round(entry - basis, 5) if entry else None
    spot_stop = round(stop - basis, 5) if stop else None
    spot_take = round(take - basis, 5) if take else None
    
    lines = [
        "\n\n---\n**📐 Расчёт позиции (Spot MT5):**",
        f"- Направление: **{calc['direction_inferred'].upper()}**",
        f"- Цена входа: **{spot_entry}** (Фьючерс: {entry})",
        f"- Стоп-лосс: **{spot_stop}** (Фьючерс: {stop})",
    ]
    if spot_take is not None:
        lines.append(f"- Тейк-профит: **{spot_take}** (Фьючерс: {take})")
    lines += [
        f"- Размер стопа: {calc['stop_distance']} (в ценах фьючерса)",
        f"- Риск на сделку: {calc['risk_amount_usd']} USD",
        f"- Объём: **{calc['lots']} лотов** ({calc['units']} ед.)",
    ]
    return "\n".join(lines)


def format_response(raw_text, deposit, risk_percent, basis, last_close, contract_size=DEFAULT_CONTRACT_SIZE):
    """Убирает JSON-блок из ответа LLM и добавляет блок расчёта позиции."""
    signal = extract_signal(raw_text)
    display_text = _SIGNAL_JSON_RE.sub("", raw_text).rstrip()
    block = build_position_block(deposit, risk_percent, signal, basis, last_close, contract_size)
    if block:
        display_text += block
    return display_text, signal


def get_system_prompt_blocks(spot_symbol, deposit, risk_percent, books_context=None):
    """Системный промпт, разбитый на блоки для Prompt Caching (Anthropic)."""
    if books_context is None:
        books_context = load_books()
    text1 = (
        f"В качестве базы знаний (контекста) тебе предоставлены книги по методологии VSA и VPA.\n"
        f"Изучи их внимательно:\n{books_context}"
    )
    text2 = (
        f"Ты профессиональный трейдер, который в совершенстве владеет знаниями об VSA и VPA.\n"
        f"Твой торговый инструмент: фьючерс EUR/USD (CME 6E). Цены фьючерсные, шаг цены (тик) 0.00005.\n"
        f"Настройки риск-менеджмента: депозит {deposit} USD, риск {risk_percent}%.\n\n"
        "Веди диалог с пользователем. Если он просит анализ, ты получишь JSON с последними свечами фьючерса. "
        "Опирайся на VSA, предлагай уровни входа, стопа и тейка.\n\n"
        "ВАЖНО: объём позиции НЕ считай сам — его рассчитает система на Python по твоим уровням.\n"
        "ВАЖНО: не придумывай цены! Уровни входа и стопа должны опираться на уровни рынка в предоставленном JSON.\n"
        "ВАЖНО (ЗАКРЫТИЕ ПЯТНИЦЫ): Если последняя свеча относится к концу пятничной сессии (пятница вечер), "
        "помни, что снижения объёма обусловлены фиксацией позиций перед выходными, а не истинным VSA-сигналом No Supply/No Demand. "
        "Не рекомендуй открывать новые сделки на самом закрытии недели (выдавай direction: \"none\").\n\n"
        "РЕЖИМ НАБЛЮДЕНИЯ И ГИПОТЕЗ:\n"
        "1. В каждом анализе оценивай свежую закрывшуюся свечу и указывай, сбылась ли предыдущая гипотеза или контекст изменился.\n"
        "2. В конце анализа ОБЯЗАТЕЛЬНО формулируй сценарное условие на СЛЕДУЮЩУЮ 15-минутную свечу (формат IF-THEN):\n"
        "   Например: '🔮 СЦЕНАРИЙ НА СЛЕДУЮЩУЮ СВЕЧУ: Если следующая 15м свеча закроется бычьей с ростом объема (rel_volume > 1.2) — это подтвердит тест поддержки и готовность к LONG. Если свеча закроется ниже 1.1390 — сценарий отменяется, ждём.'\n"
        "3. Выдавай торговый сигнал в JSON (long|short) ТОЛЬКО тогда, когда паттерн и условия сценария ПОЛНОСТЬЮ сформировались на закрывшейся свече. Во всех остальных случаях пиши direction: \"none\" и готовь сценарий на следующий бар.\n\n"
        "Если есть сигнал, в конце ответа добавь ровно один блок в формате JSON:\n"
        "```json\n"
        '{"direction": "long|short|none", "entry": <entry>, "stop": <stop>, "take": <take>}\n'
        "```\n"
        "Если сигнала нет — пиши direction: \"none\" и пустые поля."
    )
    return [
        {"type": "text", "text": text1, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": text2},
    ]


def call_llm(messages, api_key, model, retries=2, timeout=60):
    """Вызов OpenRouter с простым retry/backoff."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    last_err = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 400 and "temperature" in response.text.lower():
                payload.pop("temperature", None)
                response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
                
            response.raise_for_status()
            data = response.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content")
                if content is None:
                    content = ""
            else:
                content = ""
                
            usage = data.get("usage", {})
            return True, content, usage
            
        except requests.RequestException as e:
            last_err = str(e)
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 429:
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        time.sleep(int(retry_after))
                        continue
                last_err = f"{e} - {e.response.text}"
                
            if attempt < retries:
                time.sleep(2 ** attempt)
                
    return False, f"Ошибка API OpenRouter: {last_err}", {}


def create_vsa_chart(candles, signal=None, basis=0.0):
    """Создаёт интерактивный Plotly свечной график с VSA-объёмами и уровнями ордера."""
    if not candles:
        return None
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = pd.DataFrame(candles)
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("График 6E=F (CME 15m) & VSA Уровни", "VSA Объём и Относительный объем (rel_vol)"),
        row_width=[0.25, 0.75]
    )
    
    # 1. Свечи (Candlestick)
    fig.add_trace(
        go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="6E=F Futures",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )
    
    # Подсветка объёма: Ultra High (rel_vol >= 1.5) - пурпурный, Low Volume (<= 0.4) - желтый, Normal - голубой
    colors = []
    for rv in df['rel_volume']:
        if rv >= 1.5:
            colors.append('#ab47bc')
        elif rv <= 0.4:
            colors.append('#ffee58')
        else:
            colors.append('#42a5f5')
            
    fig.add_trace(
        go.Bar(
            x=df['time'],
            y=df['volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.85
        ),
        row=2, col=1
    )
    
    # 2. Уровни сигнала (Entry, Stop, Take)
    if signal and isinstance(signal, dict):
        direction = str(signal.get("direction", "")).lower()
        entry = signal.get("entry")
        stop = signal.get("stop")
        take = signal.get("take")
        
        if direction in ("long", "short") and entry and stop:
            fig.add_hline(
                y=entry, line_dash="dash", line_color="#29b6f6", line_width=2,
                annotation_text=f"ENTRY ({direction.upper()}): {entry}", annotation_position="top right",
                row=1, col=1
            )
            fig.add_hline(
                y=stop, line_dash="dash", line_color="#ef5350", line_width=2,
                annotation_text=f"STOP: {stop}", annotation_position="bottom right",
                row=1, col=1
            )
            if take:
                fig.add_hline(
                    y=take, line_dash="dash", line_color="#66bb6a", line_width=2,
                    annotation_text=f"TAKE: {take}", annotation_position="top right",
                    row=1, col=1
                )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        showlegend=False
    )
    return fig


DB_PATH = os.path.join(os.path.dirname(__file__), "signals.sqlite")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                candles_hash TEXT,
                model TEXT,
                prompt_tokens INTEGER,
                cached_tokens INTEGER,
                completion_tokens INTEGER,
                cost REAL,
                raw_reply TEXT,
                direction TEXT,
                entry REAL,
                stop REAL,
                take REAL,
                outcome TEXT DEFAULT 'pending'
            )
        ''')
        try:
            conn.execute("ALTER TABLE signals ADD COLUMN outcome TEXT DEFAULT 'pending'")
        except sqlite3.OperationalError:
            pass
        # Миграция: время последней свечи анализа — для дедупликации циклов бота
        try:
            conn.execute("ALTER TABLE signals ADD COLUMN last_candle_time TEXT")
        except sqlite3.OperationalError:
            pass


def get_all_signals():
    """Возвращает все сигналы из БД для торгового журнала."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, model, direction, entry, stop, take, cost, outcome FROM signals ORDER BY id DESC")
        return [dict(row) for row in cursor.fetchall()]


def get_last_analyzed_candle_time():
    """Возвращает время последней проанализированной свечи из БД (для дедупликации циклов бота)."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT last_candle_time FROM signals "
            "WHERE last_candle_time IS NOT NULL ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return row[0] if row else None


def update_signal_outcome(signal_id, outcome):
    """Обновляет статус исхода сделки (hit_tp, hit_sl, canceled, pending)."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE signals SET outcome = ? WHERE id = ?", (outcome, signal_id))


def get_journal_stats():
    """Рассчитывает статистику Winrate и исходов из БД."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT outcome, COUNT(*) FROM signals WHERE direction IN ('long', 'short') GROUP BY outcome")
        rows = dict(cursor.fetchall())
        
        tp = rows.get('hit_tp', 0)
        sl = rows.get('hit_sl', 0)
        canceled = rows.get('canceled', 0)
        pending = rows.get('pending', 0)
        total_closed = tp + sl
        winrate = round((tp / total_closed) * 100, 1) if total_closed > 0 else 0.0
        
        return {
            "hit_tp": tp,
            "hit_sl": sl,
            "canceled": canceled,
            "pending": pending,
            "total_closed": total_closed,
            "winrate": winrate
        }


def _evaluate_signal_outcome(sig, candles):
    """Определяет исход сигнала по свечам, вышедшим после его свечи анализа.

    Сделка считается открытой только после касания уровня входа.
    Если в одной свече задеты и TP, и SL — фиксируем SL (консервативно).
    """
    direction = sig["direction"]
    entry = float(sig["entry"])
    stop = float(sig["stop"])
    take = float(sig["take"]) if sig["take"] is not None else None

    started = False
    for c in candles:
        # Время свечей и last_candle_time в одном строковом формате — лексикографическое сравнение корректно
        if c["time"] <= sig["last_candle_time"]:
            continue
        high, low = c["high"], c["low"]
        if not started:
            if not (low <= entry <= high):
                continue
            started = True
        if direction == "long":
            hit_sl = low <= stop
            hit_tp = take is not None and high >= take
        else:
            hit_sl = high >= stop
            hit_tp = take is not None and low <= take
        if hit_sl:
            return "hit_sl"
        if hit_tp:
            return "hit_tp"
    return None


def check_pending_outcomes(candles):
    """Авто-трекинг: обновляет исходы pending-сигналов по касанию TP/SL новыми свечами."""
    if not candles:
        return 0
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, direction, entry, stop, take, last_candle_time FROM signals
            WHERE outcome = 'pending' AND direction IN ('long', 'short')
              AND entry IS NOT NULL AND stop IS NOT NULL AND last_candle_time IS NOT NULL
        ''')
        pending = [dict(row) for row in cursor.fetchall()]
        updated = 0
        for sig in pending:
            outcome = _evaluate_signal_outcome(sig, candles)
            if outcome:
                conn.execute("UPDATE signals SET outcome = ? WHERE id = ?", (outcome, sig["id"]))
                updated += 1
    return updated


def estimate_cost(model, prompt_tokens, completion_tokens, cached_tokens=0):
    """Рассчитывает ориентировочную стоимость вызова, если API вернул 0."""
    model_lower = str(model).lower()
    if "claude-3.5-sonnet" in model_lower or "claude-3-5-sonnet" in model_lower:
        p_cost = max(0, prompt_tokens - cached_tokens) * (3.00 / 1_000_000) + cached_tokens * (0.30 / 1_000_000)
        c_cost = completion_tokens * (15.00 / 1_000_000)
        return p_cost + c_cost
    elif "gemini-1.5-pro" in model_lower:
        return prompt_tokens * (1.25 / 1_000_000) + completion_tokens * (5.00 / 1_000_000)
    else:
        return prompt_tokens * (0.50 / 1_000_000) + completion_tokens * (1.50 / 1_000_000)


def format_usage_summary(usage, model=""):
    """Форматирует информацию об использованных токенах и стоимости для вывода."""
    if not usage:
        return ""
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    
    details = usage.get("prompt_tokens_details", {})
    cached_tokens = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    if not cached_tokens:
        cached_tokens = usage.get("native_tokens_cached", usage.get("cached_tokens", 0))
        
    cost = usage.get("total_cost", 0.0)
    if not cost or cost == 0.0:
        cost = estimate_cost(model, prompt_tokens, completion_tokens, cached_tokens)
    
    parts = [f"🔤 Токены: {total_tokens:,} (Промпт: {prompt_tokens:,}, Ответ: {completion_tokens:,})"]
    if cached_tokens > 0:
        parts.append(f"⚡ Скэшировано: {cached_tokens:,}")
    parts.append(f"💵 Стоимость: ${cost:.6f}")
    return " | ".join(parts)


def get_db_stats():
    """Возвращает суммарную статистику вызовов, токенов и расходов из БД с дорасчетом стоимости."""
    try:
        init_db()
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, model, prompt_tokens, completion_tokens, cached_tokens FROM signals WHERE cost IS NULL OR cost = 0.0")
            rows_to_update = cursor.fetchall()
            for r in rows_to_update:
                r_id, r_model, r_prompt, r_comp, r_cached = r
                est = estimate_cost(r_model or "z-ai/glm-5.2", r_prompt or 0, r_comp or 0, r_cached or 0)
                conn.execute("UPDATE signals SET cost = ? WHERE id = ?", (est, r_id))
            conn.commit()

            cursor.execute("SELECT COUNT(*), SUM(prompt_tokens), SUM(completion_tokens), SUM(cost) FROM signals")
            row = cursor.fetchone()
            prompt = row[1] or 0
            completion = row[2] or 0
            cost = row[3] or 0.0
            return {
                "total_calls": row[0] or 0,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "total_cost": cost
            }
    except Exception as e:
        print(f"Ошибка получения статистики из БД: {e}")
        return {"total_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost": 0.0}


def get_mt5_account_info():
    """Подключается к локальному терминалу MetaTrader 5 и забирает баланс счета."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return False, "Терминал MetaTrader 5 не запущен или недоступен.", None
        acc = mt5.account_info()
        if acc is None:
            mt5.shutdown()
            return False, "Не удалось получить информацию о счете MT5.", None
        info = {
            "login": acc.login,
            "server": acc.server,
            "name": acc.name,
            "balance": acc.balance,
            "equity": acc.equity,
            "currency": acc.currency,
            "leverage": acc.leverage
        }
        mt5.shutdown()
        return True, f"Успешно! Аккаунт #{acc.login} ({acc.server})", info
    except Exception as e:
        return False, f"Ошибка подключения MT5: {e}", None


def log_signal(model, candles, usage, raw_reply, signal):
    try:
        init_db()
        candles_str = json.dumps(candles, sort_keys=True)
        candles_hash = hashlib.md5(candles_str.encode('utf-8')).hexdigest()
        
        prompt_tokens = usage.get("prompt_tokens", 0) if usage else 0
        completion_tokens = usage.get("completion_tokens", 0) if usage else 0
        
        details = usage.get("prompt_tokens_details", {}) if usage else {}
        cached_tokens = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
        if not cached_tokens and usage:
            cached_tokens = usage.get("native_tokens_cached", usage.get("cached_tokens", 0))

        cost = usage.get("total_cost", 0.0) if usage else 0.0
        if not cost or cost == 0.0:
            cost = estimate_cost(model, prompt_tokens, completion_tokens, cached_tokens)
        
        direction = signal.get("direction", "") if signal else ""
        entry = signal.get("entry") if signal else None
        stop = signal.get("stop") if signal else None
        take = signal.get("take") if signal else None
        last_candle_time = candles[-1].get("time") if candles else None

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                INSERT INTO signals (candles_hash, model, prompt_tokens, completion_tokens, cost, raw_reply, direction, entry, stop, take, last_candle_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (candles_hash, model, prompt_tokens, completion_tokens, cost, raw_reply, direction, entry, stop, take, last_candle_time))
    except Exception as e:
        print(f"Ошибка записи в БД: {e}")
