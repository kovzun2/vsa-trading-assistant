import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta

from dotenv import load_dotenv

import core

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("MODEL", core.DEFAULT_MODEL)

SPOT_SYMBOL = core.DEFAULT_SPOT_SYMBOL
FUT_SYMBOL = core.DEFAULT_FUT_SYMBOL
TIMEZONE_OFFSET = core.DEFAULT_TZ_OFFSET
DEPOSIT = core.DEFAULT_DEPOSIT
RISK_PERCENT = core.DEFAULT_RISK_PERCENT

logger.info("Инициализация бота, кэширование контекста книг...")
BOOKS_CONTEXT = core.load_books()

def analyze_market():
    logger.info("Начало анализа...")
    candles, basis, is_stale = core.get_market_data(SPOT_SYMBOL, FUT_SYMBOL, TIMEZONE_OFFSET)
    if not candles:
        logger.warning("Не удалось получить данные о свечах. Пропуск цикла.")
        return

    last_candle = candles[-1]
    
    if is_stale:
        logger.info(f"Данные устарели (последняя свеча {last_candle['time']}). Рынок закрыт, анализ пропущен.")
        return
        
    logger.info(f"Анализируется свеча: {last_candle['time']} | Close: {last_candle['close']} | Vol: {last_candle['volume']}")

    system_blocks = core.get_system_prompt_blocks(SPOT_SYMBOL, DEPOSIT, RISK_PERCENT, BOOKS_CONTEXT)
    user_prompt = (
        "Данные последних свечей (интервал 15м) в формате JSON:\n"
        f"{json.dumps(candles, indent=2, ensure_ascii=False)}"
    )

    messages = [
        {"role": "system", "content": system_blocks},
        {"role": "user", "content": user_prompt},
    ]

    logger.info(f"Отправка запроса в OpenRouter (модель: {MODEL})...")
    ok, raw_reply, usage = core.call_llm(messages, OPENROUTER_API_KEY, MODEL)
    
    if not ok:
        logger.error(f"Ошибка вызова LLM: {raw_reply}")
        return
        
    display_reply, signal = core.format_response(raw_reply, DEPOSIT, RISK_PERCENT, basis, last_candle['close'])
    core.log_signal(MODEL, candles, usage, raw_reply, signal)

    print("\n=== ОТВЕТ БОТА ===")
    print(display_reply)
    print("==================\n")

def run_loop():
    logger.info("Бот переходит в режим ожидания. Проверки в 00, 15, 30, 45 минут часа.")
    while True:
        now = datetime.now()
        # Find next 15-minute slot + 5 seconds
        minutes_to_next_slot = 15 - (now.minute % 15)
        next_run = now + timedelta(minutes=minutes_to_next_slot)
        next_run = next_run.replace(second=5, microsecond=0)
        
        sleep_seconds = (next_run - now).total_seconds()
        if sleep_seconds <= 0:
            sleep_seconds = 15 * 60  # fallback if already past 5s
            
        logger.info(f"Сон до следующего запуска: {sleep_seconds:.1f} сек ({next_run.strftime('%H:%M:%S')})")
        time.sleep(sleep_seconds)
        
        try:
            analyze_market()
        except Exception as e:
            logger.error(f"Ошибка в цикле анализа: {e}")

def main():
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        logger.error("КРИТИЧЕСКАЯ ОШИБКА: Не установлен OPENROUTER_API_KEY в файле .env")
        return
    logger.info(f"Запуск бота. Настройки: {FUT_SYMBOL}, Депозит: {DEPOSIT}$, Риск: {RISK_PERCENT}%")
    analyze_market()
    run_loop()

if __name__ == "__main__":
    main()
