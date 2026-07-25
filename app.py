import os
import json
from datetime import datetime

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
        # Для последнего сообщения используем полный content, для предыдущих - очищенную отображаемую версию
        content = m["content"] if i == len(recent) - 1 else m.get("display", m["content"])
        api_messages.append({"role": m["role"], "content": content})
    return api_messages

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
    st.subheader("Торговля")
    spot_symbol = st.text_input("Тикер цен (MT5/Спот)", value=core.DEFAULT_SPOT_SYMBOL)
    fut_symbol = st.text_input("Тикер объемов (Фьючерс)", value=core.DEFAULT_FUT_SYMBOL)
    tz_offset = st.number_input("Часовой пояс MT5 (от UTC)", min_value=-12, max_value=12, value=core.DEFAULT_TZ_OFFSET)
    deposit = st.number_input("Депозит (USD)", value=core.DEFAULT_DEPOSIT)
    risk = st.number_input("Риск на сделку (%)", value=core.DEFAULT_RISK_PERCENT, step=0.1)

# --- Инициализация истории чата ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Привет! Я твой торговый ассистент по VSA. Нажми 'Анализ рынка' справа или задай вопрос.",
            "display": "Привет! Я твой торговый ассистент по VSA. Нажми 'Анализ рынка' справа или задай вопрос.",
        }
    ]

# --- ОСНОВНОЙ ИНТЕРФЕЙС ---
st.title("📈 VSA Trading Assistant")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📊 Анализ графика")
    is_weekend = datetime.today().weekday() >= 5
    if is_weekend:
        st.info("Сегодня выходной. Бот проанализирует данные за момент закрытия рынка в пятницу.")
    if st.button("🔄 Сделать анализ последних данных", use_container_width=True):
        if not api_key:
            st.error("Введите API ключ в настройках (слева).")
        else:
            with st.spinner("Загрузка котировок с Yahoo Finance (цены + объемы)..."):
                candles, basis, is_stale = core.get_market_data(spot_symbol, fut_symbol, tz_offset)
            if candles:
                st.session_state.basis = basis
                st.session_state.last_close = candles[-1]['close']
                st.success("Данные успешно получены!")
                st.write(f"**Последняя закрытая свеча:** {candles[-1]['time']}")
                st.json(candles[-1])

                user_msg = (
                    f"Проанализируй график. Вот данные последних свечей (15м):\n"
                    f"{json.dumps(candles, indent=2, ensure_ascii=False)}"
                )
                display_msg = "📉 Пожалуйста, проанализируй последние доступные рыночные данные."
                st.session_state.messages.append({"role": "user", "content": user_msg, "display": display_msg})

                api_messages = build_api_messages(st.session_state.messages, spot_symbol, deposit, risk, BOOKS_CONTEXT)

                with st.spinner("Бот проводит анализ..."):
                    ok, raw_reply, usage = core.call_llm(api_messages, api_key, model)

                if ok:
                    basis = st.session_state.get("basis", 0.0)
                    last_close = st.session_state.get("last_close", None)
                    display_reply, signal = core.format_response(raw_reply, deposit, risk, basis, last_close)
                    
                    core.log_signal(model, candles, usage, raw_reply, signal)
                    
                    st.session_state.messages.append(
                        {"role": "assistant", "content": raw_reply, "display": display_reply}
                    )
                    st.rerun()
                else:
                    st.error(f"Ошибка вызова LLM: {raw_reply}")
            else:
                st.error("Не удалось получить данные с Yahoo Finance.")

with col1:
    st.subheader("💬 Чат с ботом")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("display", msg["content"]))

    if prompt := st.chat_input("Напишите вопрос боту (например: 'Где бы ты поставил стоп?'):"):
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
                    
                    core.log_signal(model, [], usage, raw_reply, signal)
                    
                    st.markdown(display_reply)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": raw_reply, "display": display_reply}
                    )
                else:
                    st.error(f"Ошибка вызова LLM: {raw_reply}")
