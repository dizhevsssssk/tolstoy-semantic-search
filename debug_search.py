import os
import json

def debug_corpus():
    print("🔍 Поиск файла tolstoy_corpus...")
    
    possible_paths = [
        'data/tolstoy_corpus.json',  # ← главный путь
        'tolstoy_corpus.json',
        '../data/tolstoy_corpus.json',
        '../../data/tolstoy_corpus.json'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Файл найден: {path}")
            print(f"📏 Размер: {os.path.getsize(path)} байт")
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"📖 Первые 500 символов: {content[:500]}...")
                    
                    # Пробуем загрузить как JSON
                    data = json.loads(content)
                    print(f"📊 Тип данных: {type(data)}")
                    
                    if isinstance(data, list):
                        print(f"📚 Количество произведений: {len(data)}")
                        if len(data) > 0:
                            first_work = data[0]
                            print(f"📖 Первое произведение: {first_work.get('title', 'Нет названия')}")
                            print(f"📝 Длина текста: {len(first_work.get('text', ''))} символов")
                            print(f"📄 Начало текста: {first_work.get('text', '')[:200]}...")
                    
                return True
                
            except Exception as e:
                print(f"❌ Ошибка чтения: {e}")
                return False
        else:
            print(f"❌ Не найден: {path}")
    
    print("❌ Файл tolstoy_corpus не найден ни по одному пути")
    return False

if __name__ == "__main__":
    debug_corpus()