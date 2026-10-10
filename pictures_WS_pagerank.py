import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')

# ========================
# 1. ЗАГРУЗКА СЕТЕЙ
# ========================

def load_network(filepath):
    """Загружает сеть из GML, TXT/EDGES или MTX файла."""
    if filepath.endswith('.gml'):
        return load_cleaned_gml(filepath)
    elif filepath.endswith('.txt') or filepath.endswith('.edges'):
        G = nx.read_edgelist(filepath, nodetype=int)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
    elif filepath.endswith('.mtx'):
        from scipy.io import mmread
        matrix = mmread(filepath)
        if matrix.shape[0] == matrix.shape[1]:
            G = nx.from_scipy_sparse_array(matrix)
        else:
            G = nx.read_edgelist(filepath, nodetype=int)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
    else:
        raise ValueError(f"Неизвестный формат: {filepath}")

def load_cleaned_gml(path):
    """Читает GML файл, удаляет все нестандартные атрибуты."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = []
    allowed_keywords = {"graph", "node", "edge", "id", "label", "source", "target", "[", "]"}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        first_word = stripped.split()[0]
        if first_word in allowed_keywords:
            cleaned_lines.append(line)
        else:
            continue

    cleaned_text = "".join(cleaned_lines)
    try:
        return nx.parse_gml(cleaned_text, label='id')
    except Exception as e:
        return parse_gml_manual(cleaned_text)

def parse_gml_manual(content):
    """Ручной парсинг GML"""
    node_pattern = r'node\s*\[\s*id\s+(\d+)'
    nodes = re.findall(node_pattern, content)
    edge_pattern = r'edge\s*\[\s*id\s+\d+\s*source\s+(\d+)\s*target\s+(\d+)'
    edges = re.findall(edge_pattern, content)
    
    if not nodes:
        raise ValueError("Не найдены узлы в GML файле")
    
    G = nx.Graph()
    for node_id in nodes:
        G.add_node(int(node_id))
    
    edges_set = set()
    for source, target in edges:
        edge = tuple(sorted([int(source), int(target)]))
        edges_set.add(edge)
    
    for source, target in edges_set:
        G.add_edge(source, target)
    
    return G

def validate_network(G, min_nodes=10):
    """Проверяет сеть на пригодность для анализа."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    if n < min_nodes:
        return False, f"Слишком мало вершин ({n})"
    
    if m == 0:
        return False, "Нет рёбер"
    
    return True, "OK"

# ========================
# 2. ВЫЧИСЛЕНИЕ РЁБЕРНЫХ ГРАДИЕНТОВ
# ========================

def compute_edge_gradients(G, centrality_type):
    """Вычисляет рёберные градиенты для заданной центральности."""
    if G.number_of_nodes() < 10:
        return None
    
    try:
        if centrality_type == 'pagerank':
            centrality_dict = nx.pagerank(G, max_iter=1000, tol=1e-8)
        elif centrality_type == 'betweenness':
            if G.number_of_nodes() > 5000:
                centrality_dict = nx.betweenness_centrality(G, k=int(np.sqrt(G.number_of_nodes())))
            else:
                centrality_dict = nx.betweenness_centrality(G)
        elif centrality_type == 'closeness':
            if G.number_of_nodes() < 5000:
                centrality_dict = nx.closeness_centrality(G)
        elif centrality_type == 'eigenvector':
            centrality_dict = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
        elif centrality_type == 'kcore':
            centrality_dict = nx.core_number(G)
        elif centrality_type == 'degree':
            centrality_dict = dict(G.degree())
        else:
            raise ValueError(f"Неизвестный тип центральности: {centrality_type}")
    except Exception as e:
        print(f"    Ошибка при вычислении {centrality_type}: {e}")
        return None

    G_dir = G.to_directed()
    edges_to_remove = [(u, v) for u, v in G_dir.edges() if u > v]
    G_dir.remove_edges_from(edges_to_remove)
    
    try:
        B = nx.incidence_matrix(G_dir, oriented=True).toarray()
        f = np.array([centrality_dict[node] for node in G_dir.nodes()])
        gradients = B.T.dot(f)
        abs_gradients = np.abs(gradients)
        
        mean_val = np.mean(abs_gradients)
        if mean_val > 0:
            return abs_gradients / mean_val
        return abs_gradients
    except Exception as e:
        print(f"    Ошибка при вычислении градиентов: {e}")
        return None

# ========================
# 3. ПОСТРОЕНИЕ МАТРИЦЫ ГРАФИКОВ ДЛЯ WS МОДЕЛИ
# ========================

def plot_ws_matrix(n_nodes=500, k_values=None, p_values=None, centrality='pagerank', 
                   output_dir='.', bins=50, figsize=(20, 20)):
    """
    Строит матрицу графиков распределений градиентов для модели WS.
    
    Параметры:
    - n_nodes: число вершин (фиксировано)
    - k_values: список значений k (число соседей) - по оси Y
    - p_values: список значений p (вероятность переподключения) - по оси X
    - centrality: тип центральности
    - output_dir: директория для сохранения
    - bins: число бинов гистограммы
    - figsize: размер фигуры
    """
    
    if k_values is None:
        k_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    if p_values is None:
        p_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    n_rows = len(k_values)
    n_cols = len(p_values)
    
    print(f"\n{'='*60}")
    print(f"ПОСТРОЕНИЕ МАТРИЦЫ ГРАФИКОВ ДЛЯ WS МОДЕЛИ")
    print(f"{'='*60}")
    print(f"n = {n_nodes}, центральность: {centrality}")
    print(f"k = {k_values}")
    print(f"p = {p_values}")
    print(f"Всего графиков: {n_rows * n_cols}")
    
    # Создаем фигуру с субплотами
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Данные для хранения skewness
    skewness_matrix = np.zeros((n_rows, n_cols))
    skewness_matrix.fill(np.nan)
    
    # Обходим все комбинации k и p
    for i, k in enumerate(k_values):
        for j, p in enumerate(p_values):
            ax = axes[i, j]
            
            try:
                # Генерируем WS граф
                G = nx.watts_strogatz_graph(n=n_nodes, k=k, p=p)
                
                # Проверяем, что граф валидный
                valid, msg = validate_network(G)
                if not valid:
                    print(f"  k={k}, p={p}: {msg}")
                    ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # Вычисляем градиенты
                gradients = compute_edge_gradients(G, centrality)
                
                if gradients is None:
                    ax.text(0.5, 0.5, 'Ошибка', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # Фильтруем нулевые значения
                grad = gradients[gradients > 1e-12]
                
                if len(grad) < 10:
                    ax.text(0.5, 0.5, 'Мало данных', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # Вычисляем skewness
                skewness_val = skew(grad)
                skewness_matrix[i, j] = skewness_val
                
                # Строим гистограмму в линейном масштабе
                ax.hist(grad, bins=bins, density=True, alpha=0.7, 
                       color='steelblue', edgecolor='black', linewidth=0.3)
                
                # Добавляем текст с параметрами и skewness
                ax.text(0.05, 0.95, f'k={k}, p={p}\nξ={skewness_val:.3f}', 
                       transform=ax.transAxes, fontsize=7, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
                
                # Настройка осей
                ax.set_xlim([0, np.max(grad)* 1.05 if len(grad) > 0 else 5])
                ax.set_xticks([])  # Убираем метки для экономии места
                ax.set_yticks([])
                
            except Exception as e:
                print(f"  k={k}, p={p}: Ошибка - {e}")
                ax.text(0.5, 0.5, 'Ошибка', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
    
    # Добавляем общие подписи
    fig.text(0.5, 0.02, 'Вероятность переподключения p', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.02, 0.5, 'Число соседей k', va='center', rotation='vertical', fontsize=16, fontweight='bold')
    
    # Заголовок
    fig.suptitle(f'Распределение градиентов для WS модели (n={n_nodes}, центральность: {centrality})', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    
    # Сохраняем
    save_path = os.path.join(output_dir, f'ws_matrix_{centrality}_n{n_nodes}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nМатрица сохранена: {save_path}")
    plt.close()
    
    # Сохраняем матрицу skewness
    df_skewness = pd.DataFrame(skewness_matrix, 
                               index=[f'k={k}' for k in k_values],
                               columns=[f'p={p}' for p in p_values])
    csv_path = os.path.join(output_dir, f'ws_skewness_matrix_{centrality}_n{n_nodes}.csv')
    df_skewness.to_csv(csv_path)
    print(f"Матрица skewness сохранена: {csv_path}")
    
    return df_skewness


# ========================
# 4. ПОСТРОЕНИЕ ТЕПЛОВОЙ КАРТЫ ДЛЯ SKEWNESS
# ========================

def plot_skewness_heatmap(skewness_df, output_dir='.', centrality='pagerank'):
    """
    Строит тепловую карту для матрицы skewness.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Извлекаем значения k и p из индексов и колонок
    k_values = [int(idx.split('=')[1]) for idx in skewness_df.index]
    p_values = [float(col.split('=')[1]) for col in skewness_df.columns]
    
    # Строим тепловую карту
    im = ax.imshow(skewness_df.values, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=3)
    
    # Настройка осей
    ax.set_xticks(np.arange(len(p_values)))
    ax.set_yticks(np.arange(len(k_values)))
    ax.set_xticklabels(p_values)
    ax.set_yticklabels(k_values)
    
    # Поворачиваем метки X
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Добавляем значения в ячейки
    for i in range(len(k_values)):
        for j in range(len(p_values)):
            val = skewness_df.values[i, j]
            if not np.isnan(val):
                color = 'white' if val > 1.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
    
    # Подписи
    ax.set_xlabel('Вероятность переподключения p', fontsize=14, fontweight='bold')
    ax.set_ylabel('Число соседей k', fontsize=14, fontweight='bold')
    ax.set_title(f'Skewness градиентов для WS модели (центральность: {centrality})', 
                fontsize=16, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Skewness (ξ)', fontsize=12)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'ws_skewness_heatmap_{centrality}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Тепловая карта сохранена: {save_path}")
    plt.close()


# ========================
# 5. ОСНОВНАЯ ФУНКЦИЯ
# ========================

def run_ws_analysis(n_nodes=500, 
                    k_values=None, 
                    p_values=None, 
                    centrality='pagerank',
                    output_dir='C:/Proj_edge_distr/results_ws'):
    """
    Запускает полный анализ WS модели.
    """
    # Создаем директорию для результатов
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Результаты будут сохранены в: {output_dir}")
    
    print("\n" + "="*60)
    print("АНАЛИЗ WS МОДЕЛИ")
    print("="*60)
    
    # Строим матрицу графиков
    skewness_df = plot_ws_matrix(
        n_nodes=n_nodes,
        k_values=k_values,
        p_values=p_values,
        centrality=centrality,
        output_dir=output_dir
    )
    
    # Строим тепловую карту
    plot_skewness_heatmap(skewness_df, output_dir, centrality)
    
    print(f"\n💚 Все результаты сохранены в папку: {output_dir}")
    
    return skewness_df


# ========================
# 6. ЗАПУСК
# ========================

if __name__ == "__main__":
    # Параметры для анализа
    N_NODES = 500  # фиксированное число вершин
    
    # Значения k (число соседей) - по оси Y (строки)
    K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    # Значения p (вероятность переподключения) - по оси X (колонки)
    P_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Запускаем анализ
    skewness_df = run_ws_analysis(
        n_nodes=N_NODES,
        k_values=K_VALUES,
        p_values=P_VALUES,
        centrality='pagerank',
        output_dir='C:/Proj_edge_distr/results_ws'
    )
    
    # Выводим результаты
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ (матрица skewness):")
    print("="*60)
    print(skewness_df.round(3))