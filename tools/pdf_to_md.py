import sys
import os

try:
    import pymupdf4llm
except ImportError:
    print("Ошибка: библиотека pymupdf4llm не установлена.")
    print("Пожалуйста, установите её с помощью команды: pip install pymupdf4llm")
    sys.exit(1)

def convert_pdf_to_md(pdf_path, output_path=None):
    """
    Конвертирует PDF в профессионально отформатированный Markdown.
    Сохраняет структуру, таблицы, жирный/курсивный текст и абзацы.
    """
    if not os.path.exists(pdf_path):
        print(f"Ошибка: Файл '{pdf_path}' не найден.")
        return
    
    if output_path is None:
        base_name = os.path.splitext(pdf_path)[0]
        output_path = f"{base_name}.md"

    print(f"Начинаю конвертацию '{pdf_path}' в Markdown...")
    print("Это может занять некоторое время в зависимости от размера файла...")
    
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_text)
            
        print(f"Успешно! Markdown сохранен в '{output_path}'")
    except Exception as e:
        print(f"Произошла ошибка при конвертации: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python pdf_to_md.py <путь_к_файлу.pdf> [путь_к_файлу.md]")
        sys.exit(1)
        
    input_pdf = sys.argv[1]
    output_md = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_pdf_to_md(input_pdf, output_md)
