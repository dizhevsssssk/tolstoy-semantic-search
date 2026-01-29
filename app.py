from flask import Flask, render_template, request, jsonify, session
import json
import numpy as np
from sklearn.decomposition import PCA
import plotly
import plotly.express as px
import plotly.utils
import os
from search_engine import TolstoySearchEngine

# Проверка наличия файлов данных
def check_data_files():
    required_files = [
        'data/tolstoy_embeddings_complete.npy',
        'data/tolstoy_metadata_complete.json', 
        'data/tolstoy_embeddings_info_complete.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Критическая ошибка: отсутствуют файлы данных:")
        for file in missing_files:
            print(f"   - {file}")
        print("Решение: Запустите vectorize_corpus.py для создания файлов")
        return False
    
    return True

# Инициализация поискового движка
try:
    if not check_data_files():
        print("Приложение не может быть запущено без файлов данных")
        exit(1)
    
    search_engine = TolstoySearchEngine(
        'data/tolstoy_embeddings_complete.npy',
        'data/tolstoy_metadata_complete.json', 
        'data/tolstoy_embeddings_info_complete.json',
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        verbose=False
    )
    
except Exception as e:
    print(f"Ошибка инициализации поискового движка: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

app = Flask(__name__)
app.secret_key = 'tolstoy_search_secret_key'


@app.route('/')
def index():
    if 'search_history' not in session:
        session['search_history'] = []
    
    popular_searches = [
        "философские размышления о жизни",
        "любовь и отношения", 
        "война и мир",
        "нравственные искания",
        "крестьянская жизнь",
        "религия и вера",
        "смысл жизни",
        "общество и мораль"
    ]
    
    return render_template('index.html', 
                         popular_searches=popular_searches,
                         history=session['search_history'][-5:])

@app.route('/search')
def search():
    query = request.args.get('q', '')
    negative_query = request.args.get('negative_q', '')
    search_type = request.args.get('type', 'passages')
    
    if not query:
        return render_template('search.html', results=[], query='', negative_query='', search_type=search_type)
    
    # Сохраняем в историю (только основной запрос)
    if 'search_history' not in session:
        session['search_history'] = []
    
    history = session['search_history'].copy()
    
    if query not in history:
        history.append(query)
        if len(history) > 20:
            history = history[-20:]
        session['search_history'] = history
        session.modified = True
    
    try:
        if search_type == 'passages':
            # Поиск по отрывкам с отрицательным запросом
            results = search_engine.search_passages(query, top_k=15, negative_query=negative_query)
        else:
            # Поиск по произведениям с отрицательным запросом
            results = search_engine.search_works(query, top_k=20, negative_query=negative_query)
        
        return render_template('search.html', results=results, query=query, negative_query=negative_query, search_type=search_type)
    
    except Exception as e:
        print(f"Ошибка при поиске: {e}")
        return render_template('search.html', results=[], query=query, negative_query=negative_query, search_type=search_type, error=str(e))
    
@app.route('/work/<int:work_id>')
def work_detail(work_id):
    work = search_engine.get_work_by_id(work_id)
    similar_works = search_engine.find_similar_works(work_id)
    
    return render_template('work.html', work=work, similar_works=similar_works)

@app.route('/visualization')
def visualization():
    try:
        embeddings = search_engine.embeddings
        metadata = search_engine.metadata
        
        pca = PCA(n_components=2, random_state=42)
        
        if len(embeddings) > 100:
            sample_indices = np.random.choice(len(embeddings), 100, replace=False)
            embeddings_sample = embeddings[sample_indices]
            titles_sample = [metadata[i]['title'] for i in sample_indices]
            lengths_sample = [metadata[i]['original_length'] for i in sample_indices]
        else:
            embeddings_sample = embeddings
            titles_sample = [work['title'] for work in metadata]
            lengths_sample = [work['original_length'] for work in metadata]
        
        embeddings_2d = pca.fit_transform(embeddings_sample)
        
        fig = px.scatter(
            x=embeddings_2d[:, 0], 
            y=embeddings_2d[:, 1],
            hover_name=titles_sample,
            size=lengths_sample,
            size_max=20,
            title="Семантическое пространство произведений Толстого",
            color=lengths_sample,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title="Главная компонента 1",
            yaxis_title="Главная компонента 2",
            showlegend=False,
            height=600
        )
        
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        print(f"Ошибка визуализации: {e}")
        fig = px.scatter(title="Ошибка загрузки данных")
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('visualization.html', graph_json=graph_json)

@app.route('/history')
def history():
    history = session.get('search_history', [])
    
    from flask import make_response
    response = make_response(render_template('history.html', history=history))
    # Отключаем кэширование для страницы истории
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/stats')
def stats():
    total_works = len(search_engine.metadata)
    total_tokens = sum(work['original_length'] for work in search_engine.metadata)
    avg_tokens = total_tokens // total_works if total_works > 0 else 0
    
    stats_data = {
        'total_works': total_works,
        'total_tokens': f"{total_tokens:,}",
        'avg_tokens_per_work': f"{avg_tokens:,}",
        'search_engine': "Активен",
        'data_status': "Полные данные"
    }
    
    return render_template('stats.html', stats=stats_data)

@app.route('/api/search')
def api_search():
    query = request.args.get('q', '')
    negative_query = request.args.get('negative_q', '')
    # Используем поиск по отрывкам как основной API
    results = search_engine.search_passages(query, top_k=10, negative_query=negative_query)
    return jsonify(results)

@app.route('/api/works')
def api_works():
    works = search_engine.get_all_works()
    return jsonify(works)

if __name__ == '__main__':
    print(f"Запуск Tolstoy Search. Загружено произведений: {len(search_engine.metadata)}")
    app.run(debug=True, host='0.0.0.0', port=5000)
