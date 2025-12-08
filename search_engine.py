import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

class TolstoySearchEngine:
    def __init__(self, embeddings_path, metadata_path, info_path, model_name):
        print("=" * 50)
        print("Инициализация TolstoySearchEngine...")
        print("=" * 50)
        
        # Загружаем основные данные
        self.embeddings = np.load(embeddings_path)
        self.metadata = json.load(open(metadata_path, 'r', encoding='utf-8'))
        self.info = json.load(open(info_path, 'r', encoding='utf-8'))
        self.model = SentenceTransformer(model_name)
        
        # Загружаем тексты произведений
        self.works_texts = {}
        self._load_works_texts()
        
        # Загружаем эмбеддинги чанков
        self.chunk_embeddings = None
        self.chunk_data = []
        
        self._load_chunk_embeddings()
        
        if self.chunk_embeddings is not None:
            self._create_chunk_texts()  # Создаем реальные тексты для чанков
        
        print("=" * 50)
        print(f"✅ ПОИСКОВЫЙ ДВИЖОК ЗАГРУЖЕН")
        print(f"📚 Произведений: {len(self.metadata)}")
        print(f"📖 Текстов загружено: {len(self.works_texts)}")
        print(f"🔍 Отрывков: {len(self.chunk_data)}")
        print("=" * 50)

    def _load_works_texts(self):
        """Загружает тексты произведений из tolstoy_corpus"""
        print("🔍 Поиск текстов произведений...")
        
        possible_paths = [
            'data/tolstoy_corpus.json',  
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Найден файл: {path}")
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(f"📏 Размер файла: {len(content)} символов")
                        
                        # Пробуем загрузить как JSON
                        corpus_data = json.loads(content)
                        print(f"📊 Формат данных: {type(corpus_data)}")
                        
                        self._extract_texts_from_corpus(corpus_data)
                        print(f"✅ Загружено текстов произведений: {len(self.works_texts)}")
                        
                        if self.works_texts:
                            sample_title = list(self.works_texts.keys())[0]
                            sample_text = self.works_texts[sample_title]
                            print(f"📖 Пример: '{sample_title}' - {len(sample_text)} символов")
                            print(f"📄 Начало: {sample_text[:100]}...")
                        return
                        
                except Exception as e:
                    print(f"❌ Ошибка загрузки {path}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print("❌ Файл с текстами произведений не найден")
        print("⚠️ Будут использованы заглушки")

    def _extract_texts_from_corpus(self, corpus_data):
        """Извлекает тексты из корпуса"""
        if isinstance(corpus_data, list):
            print(f"📋 Загружается список из {len(corpus_data)} элементов")
            for i, work in enumerate(corpus_data):
                if isinstance(work, dict):
                    title = work.get('title', f'Произведение_{i}')
                    text = work.get('text', '')
                    
                    if text and len(text.strip()) > 0:
                        self.works_texts[title] = text.strip()
                    else:
                        print(f"⚠️ У произведения '{title}' нет текста")
                else:
                    print(f"⚠️ Неизвестный формат элемента {i}: {type(work)}")
        else:
            print(f"⚠️ Неизвестный формат корпуса: {type(corpus_data)}")

    def _load_chunk_embeddings(self):
        """Загружает эмбеддинги чанков"""
        print("🔍 Поиск эмбеддингов чанков...")
        
        try:
            possible_paths = [
                'data/tolstoy_chunk_embeddings.npy',  # ← главный путь
                'tolstoy_chunk_embeddings.npy',
                'data/tolstoy_chunk_embeddings_complete.npy',  # на всякий случай
                '../data/tolstoy_chunk_embeddings.npy'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.chunk_embeddings = np.load(path)
                    print(f"✅ Загружены векторы чанков: {self.chunk_embeddings.shape}")
                    self._create_chunk_structure()
                    return
            
            print("❌ Файл с эмбеддингами чанков не найден")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки эмбеддингов чанков: {e}")

    def _create_chunk_structure(self):
        """Создает структуру данных для чанков"""
        chunk_mapping = self.info.get('chunk_mapping', [])
        print(f"📋 Создание структуры для {len(chunk_mapping)} чанков...")
        
        for mapping in chunk_mapping:
            work_idx = mapping['work_idx']
            chunk_idx = mapping['chunk_idx']
            chunk_length = mapping['chunk_length']
            
            if work_idx < len(self.metadata):
                work = self.metadata[work_idx]
                
                self.chunk_data.append({
                    'work_title': work['title'],
                    'work_url': work['url'],
                    'work_id': work_idx,
                    'chunk_id': chunk_idx,
                    'original_length': chunk_length,
                    'word_count': chunk_length,
                    'total_chunks': work.get('num_chunks', 1),
                    'text': f"Отрывок {chunk_idx + 1} из '{work['title']}'"  # временная заглушка
                })

    def _create_chunk_texts(self):
        """Создает реальные тексты для чанков из полных текстов произведений"""
        print("✂️ Создание текстов отрывков из произведений...")
        
        texts_created = 0
        texts_missing = 0
        
        for i, chunk in enumerate(self.chunk_data):
            work_title = chunk['work_title']
            
            if work_title in self.works_texts:
                full_text = self.works_texts[work_title]
                chunk_text = self._extract_chunk_text(full_text, chunk)
                if chunk_text:
                    chunk['text'] = chunk_text
                    texts_created += 1
                    
                    # Выводим первый созданный отрывок для примера
                    if texts_created == 1:
                        print(f"📖 Пример отрывка: {chunk_text[:100]}...")
            else:
                texts_missing += 1
                # Оставляем заглушку, но более информативную
                chunk['text'] = f"Отрывок из '{work_title}'"
        
        print(f"✅ Создано текстов отрывков: {texts_created}")
        print(f"⚠️ Текстов не найдено: {texts_missing}")

    def _extract_chunk_text(self, full_text, chunk_info):
        """Извлекает конкретный отрывок из полного текста произведения"""
        if not full_text or len(full_text.strip()) == 0:
            return None
            
        words = full_text.split()
        total_words = len(words)
        
        if total_words == 0:
            return None
        
        # Вычисляем позицию чанка в тексте
        chunk_ratio = chunk_info['chunk_id'] / max(chunk_info['total_chunks'], 1)
        chunk_size = 250  # целевой размер отрывка в словах
        
        # Определяем позицию отрывка
        start_pos = int(total_words * chunk_ratio)
        start = max(0, start_pos - chunk_size // 2)
        end = min(total_words, start + chunk_size)
        
        # Извлекаем отрывок
        passage_words = words[start:end]
        
        if not passage_words:
            return None
        
        # Собираем в текст
        passage = ' '.join(passage_words)
        
        # Добавляем многоточия если это не начало/конец
        if start > 0:
            passage = '...' + passage
        if end < total_words:
            passage = passage + '...'
        
        # Убедимся, что текст заканчивается на законченной мысли
        passage = self._clean_passage_end(passage)
        
        return passage

    def _clean_passage_end(self, text):
        """Очищает конец отрывка, чтобы он заканчивался на законченной мысли"""
        # Ищем последнюю точку, восклицательный или вопросительный знак
        for end_char in ['.', '!', '?', ';']:
            last_pos = text.rfind(end_char)
            if last_pos != -1 and last_pos > len(text) * 0.7:  # Чтобы не обрезать слишком много
                return text[:last_pos + 1]
        
        return text

    def search_passages(self, query, top_k=15, min_similarity=0.3):
        """Поиск по отрывкам с реальными текстами"""
        print(f"🔍 Поиск отрывков для: '{query}'")
        
        if self.chunk_embeddings is None or len(self.chunk_data) == 0:
            print("⚠️ Эмбеддинги чанков не загружены, используем поиск по произведениям")
            return self._fallback_search_works(query, top_k)
        
        # Кодируем запрос
        query_vector = self.model.encode([query])
        query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
        
        # Вычисляем схожесть с чанками
        similarities = cosine_similarity(query_vector, self.chunk_embeddings)[0]
        
        # Получаем топ-K результатов
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            if idx >= len(self.chunk_data):
                continue
                
            similarity = similarities[idx]
            
            if similarity < min_similarity:
                continue
                
            chunk_info = self.chunk_data[idx]
            
            results.append({
                'rank': i + 1,
                'work_title': chunk_info['work_title'],
                'work_url': chunk_info['work_url'],
                'work_id': chunk_info['work_id'],
                'passage': chunk_info['text'],
                'similarity': float(similarity),
                'similarity_percent': round(float(similarity) * 100, 1),
                'word_count': chunk_info['word_count'],
                'passage_length': chunk_info['word_count']
            })
        
        print(f"✅ Найдено отрывков: {len(results)}")
        
        # Показываем отладочную информацию о первом результате
        if results:
            first_result = results[0]
            print(f"📖 Первый результат: '{first_result['work_title']}'")
            print(f"📄 Текст: {first_result['passage'][:100]}...")
        
        return results

    # Остальные методы без изменений...
    def _fallback_search_works(self, query, top_k=15):
        works_results = self.search_works(query, top_k)
        
        passages_results = []
        for work in works_results:
            work_text = self.works_texts.get(work['title'], '')
            if work_text:
                preview = work_text[:300] + "..." if len(work_text) > 300 else work_text
                passage_text = preview
            else:
                passage_text = f"📖 {work['title']}"
            
            passages_results.append({
                'rank': work['rank'],
                'work_title': work['title'],
                'work_url': work['url'],
                'work_id': self._get_work_id_by_title(work['title']),
                'passage': passage_text,
                'similarity': work['similarity'],
                'similarity_percent': work['similarity_percent'],
                'word_count': work['original_length'],
                'passage_length': work['original_length']
            })
        
        return passages_results

    def _get_work_id_by_title(self, title):
        for work in self.metadata:
            if work['title'] == title:
                return work['id']
        return 0

    def search_works(self, query, top_k=20):
        query_vector = self.model.encode([query])
        query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
        
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            work = self.metadata[idx]
            results.append({
                'rank': i + 1,
                'title': work['title'],
                'url': work['url'],
                'similarity': float(similarities[idx]),
                'similarity_percent': round(float(similarities[idx]) * 100, 1),
                'original_length': work['original_length']
            })
        
        return results

    def find_similar_works(self, work_id, top_k=5):
        work_vector = self.embeddings[work_id:work_id+1]
        similarities = cosine_similarity(work_vector, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[-top_k-1:][::-1]
        
        similar = []
        for idx in top_indices:
            if idx != work_id and len(similar) < top_k:
                work = self.metadata[idx]
                similar.append({
                    'title': work['title'],
                    'url': work['url'],
                    'similarity': float(similarities[idx]),
                    'similarity_percent': round(float(similarities[idx]) * 100, 1)
                })
        
        return similar
    
    def get_work_by_id(self, work_id):
        return self.metadata[work_id]
    
    def get_all_works(self):
        return self.metadata
