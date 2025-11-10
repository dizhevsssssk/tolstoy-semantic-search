import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TolstoySearchEngine:
    def __init__(self, embeddings_path, metadata_path, info_path, model_name):
        self.embeddings = np.load(embeddings_path)
        self.metadata = json.load(open(metadata_path, 'r', encoding='utf-8'))
        self.info = json.load(open(info_path, 'r', encoding='utf-8'))
        self.model = SentenceTransformer(model_name)
        
        # Нормализуем векторы для косинусной схожести
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        print(f"Поисковый движок загружен: {len(self.metadata)} произведений")
    
    def search(self, query, top_k=10):
        query_vector = self.model.encode([query])
        query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
        
        # Вычисляем косинусную схожесть
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        
        # Получаем топ-K результатов
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            work = self.metadata[idx]
            results.append({
                'rank': i + 1,
                'title': work['title'],
                'url': work['url'],
                'similarity': float(similarities[idx]),
                'similarity_percent': float(similarities[idx]) * 100,
                'original_length': work['original_length'],
                'num_chunks': work['num_chunks']
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
                    'similarity_percent': float(similarities[idx]) * 100
                })
        
        return similar
    
    def get_work_by_id(self, work_id):
        return self.metadata[work_id]
    
    def get_all_works(self):
        return self.metadata