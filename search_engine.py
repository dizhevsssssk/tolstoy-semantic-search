import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

class TolstoySearchEngine:
    def __init__(self, embeddings_path, metadata_path, info_path, model_name, verbose=False):
        self.verbose = verbose
        
        # Загружаем основные данные
        self.embeddings = np.load(embeddings_path)
        self.metadata = json.load(open(metadata_path, 'r', encoding='utf-8'))
        self.info = json.load(open(info_path, 'r', encoding='utf-8'))
        self.model = SentenceTransformer(model_name)
        
        # Загружаем тексты произведений с пунктуацией
        self.works_texts = {}
        self._load_works_texts_with_punctuation()
        
        # Загружаем эмбеддинги чанков
        self.chunk_embeddings = None
        self.chunk_data = []
        self._load_chunk_embeddings()
        
        # Создаем тексты для чанков
        if self.chunk_embeddings is not None and self.works_texts:
            self._create_chunk_texts_from_originals()
        
        if self.verbose:
            print(f"Поисковый движок загружен: {len(self.metadata)} произведений, {len(self.chunk_data)} отрывков")

    def _load_works_texts_with_punctuation(self):
        """Загружает тексты произведений с пунктуацией"""
        path = 'data/tolstoy_corpus_with_punctuation.json'
        if not os.path.exists(path):
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
                if isinstance(corpus_data, list):
                    for work in corpus_data:
                        if isinstance(work, dict):
                            title = work.get('title', '')
                            text = work.get('text', '')
                            if title and text and len(text.strip()) > 0:
                                self.works_texts[title] = text.strip()
        except Exception as e:
            if self.verbose:
                print(f"Ошибка загрузки текстов: {e}")

    def _load_chunk_embeddings(self):
        """Загружает эмбеддинги чанков"""
        path = 'data/tolstoy_chunk_embeddings.npy'
        if not os.path.exists(path):
            return
        
        try:
            self.chunk_embeddings = np.load(path)
            self._create_chunk_structure()
        except Exception as e:
            if self.verbose:
                print(f"Ошибка загрузки эмбеддингов чанков: {e}")

    def _create_chunk_structure(self):
        """Создает структуру данных для чанков"""
        chunk_mapping = self.info.get('chunk_mapping', [])
        for mapping in chunk_mapping:
            work_idx = mapping['work_idx']
            if work_idx < len(self.metadata):
                work = self.metadata[work_idx]
                self.chunk_data.append({
                    'work_title': work['title'],
                    'work_url': work['url'],
                    'work_id': work_idx,
                    'chunk_id': mapping['chunk_idx'],
                    'original_length': mapping['chunk_length'],
                    'word_count': mapping['chunk_length'],
                    'total_chunks': work.get('num_chunks', 1),
                    'text': f"Отрывок {mapping['chunk_idx'] + 1} из '{work['title']}'"
                })

    def _clean_text(self, text, work_title):
        """Очищает текст от заголовков и метаданных"""
        if not text:
            return text
        
        lines = text.split('\n')
        cleaned_lines = []
        in_content = False
        
        for line in lines:
            line = line.strip()
            if not line and not in_content:
                continue
            
            # Пропускаем заголовки
            if (len(line) < 50 and line.isupper()) or line == work_title or line == work_title.upper():
                continue
            
            if not in_content:
                in_content = True
            cleaned_lines.append(line)
        
        cleaned_text = ' '.join(cleaned_lines)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def _extract_chunk(self, full_text, chunk_info):
        """Извлекает отрывок из текста"""
        if not full_text:
            return None
        
        work_title = chunk_info['work_title']
        clean_text = self._clean_text(full_text, work_title)
        
        if len(clean_text) < 500:
            return clean_text[:500] + "..." if len(clean_text) > 500 else clean_text
        
        total_chars = len(clean_text)
        chunk_ratio = chunk_info['chunk_id'] / max(chunk_info['total_chunks'], 1)
        chunk_size = 1000
        
        start_pos = int(total_chars * chunk_ratio)
        start = max(0, start_pos - 100)
        end = min(total_chars, start + chunk_size)
        
        # Ищем начало предложения
        for i in range(start, max(0, start - 200), -1):
            if i > 0 and clean_text[i-1] in '.!?':
                start = i
                break
        
        # Ищем конец предложения
        for i in range(end, min(total_chars, end + 200)):
            if i < total_chars and clean_text[i] in '.!?':
                end = i + 1
                break
        
        passage = clean_text[start:end]
        if start > 0:
            passage = '...' + passage
        if end < total_chars:
            passage = passage + '...'
        
        return passage

    def _create_chunk_texts_from_originals(self):
        """Создает тексты для чанков из оригинальных текстов"""
        for chunk in self.chunk_data:
            work_title = chunk['work_title']
            if work_title in self.works_texts:
                full_text = self.works_texts[work_title]
                chunk_text = self._extract_chunk(full_text, chunk)
                if chunk_text:
                    chunk['text'] = chunk_text

    def _split_into_sentences(self, text):
        """Разбивает текст на предложения"""
        if not text:
            return []
        # Простая разбивка по точкам, восклицательным и вопросительным знакам
        # Учитываем многоточия и сокращения
        import re
        # Регулярное выражение для разбивки на предложения
        sentence_endings = r'(?<=[.!?])\s+(?=[А-ЯA-Z])'
        sentences = re.split(sentence_endings, text)
        # Фильтруем пустые
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _find_most_relevant_sentence(self, passage, query):
        """Находит наиболее релевантное предложение в отрывке для заданного запроса"""
        sentences = self._split_into_sentences(passage)
        if not sentences:
            return passage, 0, 0.0  # Возвращаем весь отрывок, индекс 0, схожесть 0
        
        # Кодируем запрос один раз
        query_vector = self.model.encode([query])
        query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
        
        # Кодируем все предложения
        sentence_vectors = self.model.encode(sentences)
        sentence_vectors = sentence_vectors / np.linalg.norm(sentence_vectors, axis=1, keepdims=True)
        
        # Вычисляем схожесть
        similarities = cosine_similarity(query_vector, sentence_vectors)[0]
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        # Если схожесть очень низкая (меньше 0.1), не выделяем ничего
        if best_similarity < 0.1:
            return passage, -1, best_similarity
        
        return sentences[best_idx], best_idx, best_similarity

    def _highlight_most_relevant_sentence(self, passage, query):
        """Выделяет наиболее релевантное предложение в отрывке HTML-тегами"""
        sentences = self._split_into_sentences(passage)
        if not sentences:
            return passage, False
        
        best_sentence, best_idx, similarity = self._find_most_relevant_sentence(passage, query)
        if best_idx == -1:
            return passage, False
        
        # Заменяем предложение на версию с тегом
        sentences[best_idx] = f'<span class="highlight-sentence">{sentences[best_idx]}</span>'
        highlighted_passage = ' '.join(sentences)
        
        # Проверяем, что тег действительно добавлен
        if '<span class="highlight-sentence">' in highlighted_passage:
            return highlighted_passage, True
        else:
            return passage, False

    def search_passages(self, query, top_k=15, min_similarity=0.3):
        """Поиск по отрывкам"""
        if self.chunk_embeddings is None or len(self.chunk_data) == 0:
            return self._fallback_search_works(query, top_k)
        
        # Кодируем запрос
        query_vector = self.model.encode([query])
        query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
        
        # Вычисляем схожесть
        similarities = cosine_similarity(query_vector, self.chunk_embeddings)[0]
        
        # Получаем топ-K результатов
        top_indices = np.argsort(similarities)[-top_k*2:][::-1]
        
        results = []
        seen_texts = set()
        
        for idx in top_indices:
            if idx >= len(self.chunk_data):
                continue
                
            similarity = similarities[idx]
            if similarity < min_similarity:
                continue
                
            chunk_info = self.chunk_data[idx]
            passage_text = chunk_info['text']
            
            # Проверяем уникальность
            text_fingerprint = passage_text[:100].lower()
            if text_fingerprint in seen_texts:
                continue
            
            seen_texts.add(text_fingerprint)
            
            # Выделяем наиболее релевантное предложение
            highlighted_passage, has_highlight = self._highlight_most_relevant_sentence(passage_text, query)
            
            results.append({
                'rank': len(results) + 1,
                'work_title': chunk_info['work_title'],
                'work_url': chunk_info['work_url'],
                'work_id': chunk_info['work_id'],
                'passage': highlighted_passage,
                'similarity': float(similarity),
                'similarity_percent': round(float(similarity) * 100, 1),
                'word_count': chunk_info['word_count'],
                'passage_length': chunk_info['word_count'],
                'has_highlight': has_highlight
            })
            
            if len(results) >= top_k:
                break
        
        return results

    def _fallback_search_works(self, query, top_k=15):
        """Фолбэк поиск по произведениям"""
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
        """Находит ID произведения по названию"""
        for i, work in enumerate(self.metadata):
            if work['title'] == title:
                return i
        return 0

    def search_works(self, query, top_k=20):
        """Поиск по произведениям"""
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
        """Поиск похожих произведений"""
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
        """Получение произведения по ID"""
        return self.metadata[work_id]
    
    def get_all_works(self):
        """Получение всех произведений"""
        return self.metadata
