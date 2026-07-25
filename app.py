import os
import json
import time
from datetime import datetime
import pandas as pd

import streamlit as st
from dotenv import load_dotenv

import importlib
import core
importlib.reload(core)

load_dotenv()

st.set_page_config(page_title="VSA Trading Assistant", layout="wide")

BOOKS_CONTEXT = core.load_books()


def build_api_messages(messages, spot_symbol, deposit, risk, books_context):
    """Формирует список сообщений для API, сжимая старые сообщения и вкладывая полный JSON только в последнее."""
    api_messages = [{
        "role": "system",
        "content": core.get_system_prompt_blocks(spot_symbol, deposit, risk, books_context),
    }]
    recent = messages[-12:]
    for i, m in enumerate(recent):
        if m.get("role") == "system":
            continue
        content = m["content"] if i == len(recent) - 1 else m.get("display", m["content"])
        api_messages.append({"role": m["role"], "content": content})
    return api_messages


def run_analysis_step(candles, basis, spot_symbol, deposit, risk, model, api_key, is_auto=False):
    """Выполняет цикл анализа рынка нейросетью с сохранением результатов."""
    st.session_state.candles = candles
    st.session_state.basis = basis
    st.session_state.last_close = candles[-1]['close']
    st.session_state.last_auto_candle = candles[-1]['time']

    prefix = "🔔 [АВТО-МОНИТОРИНГ 15М] Свеча " + candles[-1]['time'] if is_auto else "📉 Ручной анализ свечей"
    user_msg = (
        f"Проанализируй график. Вот данные 60 последних закрытых свечей (15м, история за 15 часов):\n"
        f"{json.dumps(candles, indent=2, ensure_ascii=False)}"
    )
    display_msg = f"{prefix}. Проверить гипотезу по новой свече {candles[-1]['time']} и сформулировать сценарные условия."

    st.session_state.messages.append({"role": "user", "content": user_msg, "display": display_msg})
    api_messages = build_api_messages(st.session_state.messages, spot_symbol, deposit, risk, BOOKS_CONTEXT)

    ok, raw_reply, usage = core.call_llm(api_messages, api_key, model)

    if ok:
        display_reply, signal = core.format_response(raw_reply, deposit, risk, basis, candles[-1]['close'])
        if signal:
            st.session_state.last_signal = signal
        
        core.log_signal(model, candles, usage, raw_reply, signal)
        
        st.session_state.messages.append(
            {"role": "assistant", "content": raw_reply, "display": display_reply, "usage": usage}
        )
        return True, display_reply
    else:
        return False, raw_reply


# --- Боковая панель (Настройки) ---
with st.sidebar:
    st.header("⚙️ Настройки")
    api_key = st.text_input("OpenRouter API Key", value=os.getenv("OPENROUTER_API_KEY", ""), type="password")
    model = st.selectbox("LLM Модель", [
        "z-ai/glm-5.2",
        "xiaomi/mimo-v2.5",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-1.5-pro",
    ])

    st.divider()
    st.subheader("Торговля & MT5")
    
    if st.button("🔗 Синхронизировать с MT5", use_container_width=True):
        ok, msg, mt5_info = core.get_mt5_account_info()
        if ok and mt5_info:
            st.session_state.mt5_info = mt5_info
            st.session_state.deposit_synced = mt5_info["balance"]
            st.success(f"Подключено: #{mt5_info['login']} ({mt5_info['server']})")
        else:
            st.error(msg)
            
    mt5_info = st.session_state.get("mt5_info")
    if mt5_info:
        st.info(f"🟢 **MT5:** #{mt5_info['login']} ({mt5_info['server']})\nБаланс: **${mt5_info['balance']:.2f} {mt5_info['currency']}** | Плечо: 1:{mt5_info['leverage']}")
        default_dep = float(mt5_info["balance"])
    else:
        default_dep = float(st.session_state.get("deposit_synced", core.DEFAULT_DEPOSIT))

    spot_symbol = st.text_input("Тикер цен (MT5/Спот)", value=core.DEFAULT_SPOT_SYMBOL)
    fut_symbol = st.text_input("Тикер объемов (Фьючерс)", value=core.DEFAULT_FUT_SYMBOL)
    tz_offset = st.number_input("Часовой пояс MT5 (от UTC)", min_value=-12, max_value=12, value=core.DEFAULT_TZ_OFFSET)
    deposit = st.number_input("Депозит (USD)", value=default_dep)
    risk = st.number_input("Риск на сделку (%)", value=core.DEFAULT_RISK_PERCENT, step=0.1)

    st.divider()
    st.subheader("📊 Расход токенов (БД)")
    stats = core.get_db_stats()
    st.markdown(f"- **Запросов всего:** {stats['total_calls']}")
    st.markdown(f"- **Всего токенов:** {stats['total_tokens']:,}")
    st.markdown(f"  - Промпт: {stats['prompt_tokens']:,}")
    st.markdown(f"  - Ответы: {stats['completion_tokens']:,}")
    st.markdown(f"- **Общие затраты:** `${stats['total_cost']:.6f}`")

# --- Инициализация истории чата ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Привет! Я твой торговый ассистент по VSA. Запусти 'Живой мониторинг' или нажми 'Разовый анализ'.",
            "display": "Привет! Я твой торговый ассистент по VSA. Запусти 'Живой мониторинг' или нажми 'Разовый анализ'.",
        }
    ]

# --- ОСНОВНОЙ ИНТЕРФЕЙС С ВКЛАДКАМИ ---
st.title("📈 VSA Trading Assistant")

tab1, tab2 = st.tabs(["📈 Ассистент & График", "📊 Журнал сделок & Winrate"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("📊 Режимы наблюдения")
        is_weekend = datetime.today().weekday() >= 5
        if is_weekend:
            st.warning("⚠️ Внимание: Рынок закрыт (выходные). Анализируются котировки закрытия пятницы. Сигналы перед выходными носят ознакомительный характер, реальный вход не рекомендуется из-за риска гепа при открытии в воскресенье.")
            
        # 1. Переключатель Живого мониторинга
        is_monitoring = st.toggle("▶️ Включить живой 15м мониторинг сессии", value=st.session_state.get("monitoring_active", False))
        st.session_state.monitoring_active = is_monitoring

        if is_monitoring:
            st.info("🟢 **Мониторинг активен:** Бот следит за закрытием каждой 15м свечи, проверяет прошлые гипотезы и готовит новые сценарии.")
            
            if not api_key:
                st.error("Введите API ключ в настройках (слева).")
            else:
                candles, basis, is_stale = core.get_market_data(spot_symbol, fut_symbol, tz_offset)
                if candles and not is_stale:
                    last_time = candles[-1]['time']
                    if last_time != st.session_state.get("last_auto_candle"):
                        with st.spinner(f"Обнаружена новая свеча ({last_time})! Проверка гипотезы..."):
                            ok, reply = run_analysis_step(candles, basis, spot_symbol, deposit, risk, model, api_key, is_auto=True)
                            if ok:
                                st.rerun()

        st.divider()

        # 2. Кнопка ручного анализа
        if st.button("🔄 Сделать разовый анализ", use_container_width=True):
            if not api_key:
                st.error("Введите API ключ в настройках (слева).")
            else:
                with st.spinner("Загрузка котировок с Yahoo Finance (цены + объемы)..."):
                    candles, basis, is_stale = core.get_market_data(spot_symbol, fut_symbol, tz_offset)
                if candles:
                    with st.spinner("Бот проводит VSA-анализ..."):
                        ok, reply = run_analysis_step(candles, basis, spot_symbol, deposit, risk, model, api_key, is_auto=False)
                        if ok:
                            st.rerun()
                        else:
                            st.error(f"Ошибка вызова LLM: {reply}")
                else:
                    st.error("Не удалось получить данные с Yahoo Finance.")

    with col1:
        # Автоматическая загрузка котировок для графика при открытии
        if "candles" not in st.session_state:
            with st.spinner("Загрузка графика..."):
                candles, basis, _ = core.get_market_data(spot_symbol, fut_symbol, tz_offset)
                if candles:
                    st.session_state.candles = candles
                    st.session_state.basis = basis

        if st.session_state.get("candles"):
            fig = core.create_vsa_chart(
                st.session_state.candles,
                signal=st.session_state.get("last_signal"),
                basis=st.session_state.get("basis", 0.0)
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("💬 Чат с ботом & Ведение сценария")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg.get("display", msg["content"]))
                if msg.get("usage"):
                    caption = core.format_usage_summary(msg["usage"])
                    if caption:
                        st.caption(caption)

        if prompt := st.chat_input("Напишите вопрос боту (например: 'Что думаешь по поводу последнего сценария?'):"):
            if not api_key:
                st.error("Введите API ключ в настройках (слева).")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt, "display": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                api_messages = build_api_messages(st.session_state.messages, spot_symbol, deposit, risk, BOOKS_CONTEXT)

                with st.chat_message("assistant"):
                    with st.spinner("Бот печатает..."):
                        ok, raw_reply, usage = core.call_llm(api_messages, api_key, model)
                    
                    if ok:
                        basis = st.session_state.get("basis", 0.0)
                        last_close = st.session_state.get("last_close", None)
                        display_reply, signal = core.format_response(raw_reply, deposit, risk, basis, last_close)
                        if signal:
                            st.session_state.last_signal = signal
                        
                        core.log_signal(model, [], usage, raw_reply, signal)
                        
                        st.markdown(display_reply)
                        caption = core.format_usage_summary(usage)
                        if caption:
                            st.caption(caption)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": raw_reply, "display": display_reply, "usage": usage}
                        )
                    else:
                        st.error(f"Ошибка вызова LLM: {raw_reply}")

with tab2:
    st.subheader("📊 Торговый журнал & Статистика сигналов")
    
    j_stats = core.get_journal_stats()
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Winrate", f"{j_stats['winrate']}%")
    m2.metric("Успешных (TP)", j_stats['hit_tp'])
    m3.metric("Убыточных (SL)", j_stats['hit_sl'])
    m4.metric("В процессе", j_stats['pending'])
    m5.metric("Всего закрыто", j_stats['total_closed'])
    
    st.divider()
    
    signals_list = core.get_all_signals()
    if signals_list:
        df_signals = pd.DataFrame(signals_list)
        
        st.subheader("📋 История сгенерированных сигналов")
        
        col_select, col_outcome, col_btn = st.columns([2, 2, 1])
        with col_select:
            selected_id = st.selectbox("Выберите ID сигнала для отметки результата:", df_signals['id'].tolist())
        with col_outcome:
            new_outcome = st.selectbox("Результат сделки:", ["hit_tp", "hit_sl", "canceled", "pending"])
        with col_btn:
            st.write("")
            st.write("")
            if st.button("Сохранить статус", use_container_width=True):
                core.update_signal_outcome(selected_id, new_outcome)
                st.success(f"Сигнал #{selected_id} обновлен -> {new_outcome}")
                st.rerun()

        st.dataframe(
            df_signals,
            column_config={
                "id": "ID",
                "timestamp": "Время (UTC)",
                "model": "Модель LLM",
                "direction": "Сигнал",
                "entry": "Вход (Fut)",
                "stop": "Стоп (Fut)",
                "take": "Тейк (Fut)",
                "cost": "Стоимость ($)",
                "outcome": "Исход"
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("В базе данных пока нет записанных сигналов.")
