#!/usr/bin/env python3
"""
Тестирование приложения в production-режиме
"""

import os
import sys
import subprocess
import time
import requests

def test_app():
    """Запускает приложение в фоне и тестирует основные endpoints"""
    
    # Устанавливаем переменные окружения как в production
    os.environ['PORT'] = '8888'
    os.environ['FLASK_DEBUG'] = 'False'
    
    print("=== Тестирование Tolstoy Search в production-режиме ===")
    
    # Запускаем приложение в фоне
    print("1. Запуск приложения...")
    proc = subprocess.Popen(
        [sys.executable, 'app.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ
    )
    
    # Даем время на запуск
    time.sleep(5)
    
    try:
        # Проверяем, что процесс работает
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            print(f"Ошибка запуска приложения:\n{stderr.decode()}")
            return False
        
        print("2. Приложение запущено, тестируем endpoints...")
        
        # Тестируем главную страницу
        try:
            response = requests.get('http://localhost:8888/', timeout=10)
            if response.status_code == 200:
                print("   ✓ Главная страница работает")
            else:
                print(f"   ✗ Главная страница: статус {response.status_code}")
                return False
        except Exception as e:
            print(f"   ✗ Ошибка доступа к главной странице: {e}")
            return False
        
        # Тестируем поиск
        try:
            response = requests.get('http://localhost:8888/search?q=любовь&negative_q=война', timeout=10)
            if response.status_code == 200:
                print("   ✓ Поиск работает (с отрицательным запросом)")
            else:
                print(f"   ✗ Поиск: статус {response.status_code}")
                return False
        except Exception as e:
            print(f"   ✗ Ошибка поиска: {e}")
            return False
        
        # Тестируем API
        try:
            response = requests.get('http://localhost:8888/api/search?q=философия', timeout=10)
            if response.status_code == 200:
                print("   ✓ API работает")
            else:
                print(f"   ✗ API: статус {response.status_code}")
                return False
        except Exception as e:
            print(f"   ✗ Ошибка API: {e}")
            return False
        
        print("3. Все тесты пройдены успешно!")
        return True
        
    finally:
        # Останавливаем приложение
        print("4. Остановка приложения...")
        proc.terminate()
        proc.wait(timeout=5)
        print("=== Тестирование завершено ===")

if __name__ == '__main__':
    success = test_app()
    sys.exit(0 if success else 1)