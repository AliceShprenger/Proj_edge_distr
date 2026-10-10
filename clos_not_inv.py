"""
ДОКАЗАТЕЛЬСТВО: closeness НЕ ЯВЛЯЕТСЯ ИНВАРИАНТОМ
Метод: двухвыборочный KS-тест
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. ЗАГРУЗКА СЕТЕЙ
# ============================================================

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
    except Exception:
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

# ============================================================
# 2. ВЫЧИСЛЕНИЕ ГРАДИЕНТОВ ДЛЯ closeness
# ============================================================

def compute_edge_gradients_closeness(G):
    """Вычисляет рёберные градиенты для closeness-центральности."""
    if G.number_of_nodes() < 10:
        return None
    
    try:
        centrality_dict = nx.closeness_centrality(G)
    except Exception as e:
        print(f"    ❌ Ошибка: {e}")
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
    except Exception:
        return None

# ============================================================
# 3. СБОР РЕАЛЬНЫХ СЕТЕЙ
# ============================================================

def collect_real_networks(folder_path, max_networks=None, max_nodes=5000):
    """Собирает градиенты для всех реальных сетей."""
    print("\n" + "="*60)
    print("СБОР РЕАЛЬНЫХ СЕТЕЙ (closeness)")
    print("="*60)
    
    extensions = ['.gml', '.txt', '.edges', '.mtx']
    all_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                all_files.append(os.path.join(root, f))
    
    print(f"Найдено файлов: {len(all_files)}")
    
    gradients_dict = {}
    
    for idx, filepath in enumerate(all_files):
        if max_networks and idx >= max_networks:
            break
            
        filename = os.path.basename(filepath)
        print(f"  {idx+1}. {filename}")
        
        try:
            G = load_network(filepath)
        except Exception as e:
            print(f"    Ошибка загрузки: {e}")
            continue
        
        if G is None:
            continue
        
        valid, msg = validate_network(G)
        if not valid:
            print(f"    Пропуск: {msg}")
            continue
        
        # ✅ ПРОВЕРКА: слишком много вершин (ДЛЯ РЕАЛЬНЫХ СЕТЕЙ)
        if G.number_of_nodes() > max_nodes:
            print(f"    ❌ Слишком много вершин ({G.number_of_nodes()}) → пропускаем")
            continue
        
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc)
            print(f"    Взята гигантская компонента: {len(largest_cc)} вершин")
        
        gradients = compute_edge_gradients_closeness(G)
        
        if gradients is None:
            print(f"    Не удалось вычислить градиенты")
            continue
        
        gradients = gradients[gradients > 1e-12]
        if len(gradients) < 50:
            print(f"    Недостаточно данных ({len(gradients)} ребер)")
            continue
        
        key = filename
        gradients_dict[key] = gradients
        print(f"    ✅ {len(gradients)} ребер, {G.number_of_nodes()} вершин")
    
    print(f"\nСобрано реальных сетей: {len(gradients_dict)}")
    return gradients_dict

# ============================================================
# 4. ГЕНЕРАЦИЯ СЛУЧАЙНЫХ СЕТЕЙ
# ============================================================

def generate_random_networks(n_networks=44, max_nodes=5000):
    """Генерирует случайные сети для closeness."""
    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ СЛУЧАЙНЫХ СЕТЕЙ (closeness)")
    print("="*60)
    
    gradients_dict = {}
    n_nodes = 500  # уже маленькое число, но оставим проверку на всякий случай
    
    # Параметры моделей (равномерно разбросанные)
    models = []
    
    # ER
    for p in [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 0.5]:
        models.append(('ER', {'n': n_nodes, 'p': p}))
    
    # BA
    for m in [1, 2, 3, 4, 5, 7, 10, 15]:
        models.append(('BA', {'n': n_nodes, 'm': m}))
    
    # WS
    for k in [1, 2, 3, 5, 7, 10]:
        for p in [0.0, 0.05, 0.1, 0.3, 0.5, 0.8]:
            models.append(('WS', {'n': n_nodes, 'k': k, 'p': p}))
    
    import random
    random.seed(42)
    random.shuffle(models)
    models = models[:n_networks]
    
    print(f"Будет сгенерировано {len(models)} сетей (n_nodes={n_nodes})")
    
    for i, (model_name, params) in enumerate(models):
        try:
            print(f"  {i+1}. {model_name} {params}")
            
            # ✅ ПРОВЕРКА ДО СОЗДАНИЯ: слишком много вершин
            if params.get('n', 0) > max_nodes:
                print(f"    ❌ Слишком много вершин ({params['n']}) → пропускаем")
                continue
            
            if model_name == 'ER':
                G = nx.erdos_renyi_graph(**params)
            elif model_name == 'BA':
                G = nx.barabasi_albert_graph(**params)
            elif model_name == 'WS':
                G = nx.watts_strogatz_graph(**params)
            else:
                continue
            
            if G.number_of_edges() < 10:
                print(f"    ❌ Слишком мало ребер")
                continue
            
            # ✅ ПРОВЕРКА ПОСЛЕ СОЗДАНИЯ (на всякий случай)
            if G.number_of_nodes() > max_nodes:
                print(f"    ❌ Слишком много вершин ({G.number_of_nodes()}) → пропускаем")
                continue
            
            gradients = compute_edge_gradients_closeness(G)
            if gradients is None:
                continue
            
            gradients = gradients[gradients > 1e-12]
            if len(gradients) < 50:
                continue
            
            key = f"{model_name}_{i+1}"
            gradients_dict[key] = gradients
            print(f"    ✅ {len(gradients)} ребер, {G.number_of_nodes()} вершин")
                
        except Exception as e:
            print(f"    ❌ Ошибка: {e}")
    
    print(f"\nСгенерировано случайных сетей: {len(gradients_dict)}")
    return gradients_dict

# ============================================================
# 5. KS-TEST
# ============================================================

def run_ks_test(real_gradients, random_gradients):
    """
    Двухвыборочный KS-тест для всех пар сетей.
    """
    print("\n" + "="*70)
    print("ДВУХВЫБОРОЧНЫЙ KS-TEST")
    print("="*70)
    
    results = []
    
    for real_name, real_grad in real_gradients.items():
        for rand_name, rand_grad in random_gradients.items():
            
            real_grad = real_grad[real_grad > 1e-12]
            rand_grad = rand_grad[rand_grad > 1e-12]
            
            if len(real_grad) < 50 or len(rand_grad) < 50:
                continue
            
            ks_stat, ks_p = ks_2samp(real_grad, rand_grad)
            
            results.append({
                'real_network': real_name,
                'random_network': rand_name,
                'ks_pvalue': ks_p,
                'distributions_different': ks_p < 0.05
            })
    
    df_results = pd.DataFrame(results)
    
    total_pairs = len(df_results)
    different_pairs = df_results['distributions_different'].sum()
    same_pairs = total_pairs - different_pairs
    
    print(f"\nВсего пар (реальная vs случайная): {total_pairs}")
    print(f"Пары, где распределения РАЗЛИЧАЮТСЯ: {different_pairs} ({different_pairs/total_pairs*100:.1f}%)")
    print(f"Пары, где распределения НЕ РАЗЛИЧАЮТСЯ: {same_pairs} ({same_pairs/total_pairs*100:.1f}%)")
    
    print("\n" + "="*70)
    print("ВЫВОД:")
    print("="*70)
    
    if same_pairs / total_pairs > 0.5:
        print("✅ В БОЛЬШИНСТВЕ случаев (>{:.1f}%) распределения НЕ РАЗЛИЧАЮТСЯ".format(same_pairs/total_pairs*100))
        print("   → closeness НЕ ЯВЛЯЕТСЯ ИНВАРИАНТОМ")
    else:
        print("⚠️ Требуется дополнительный анализ")
    
    return df_results

# ============================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_ks_results(df_ks, save_path=None):
    """Визуализирует результаты KS-теста."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Гистограмма p-values
    axes[0].hist(df_ks['ks_pvalue'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
    axes[0].set_xlabel('KS p-value')
    axes[0].set_ylabel('Frequency')
    
    same_pairs = (df_ks['ks_pvalue'] > 0.05).sum()
    total_pairs = len(df_ks)
    axes[0].set_title(f'Closeness: KS-test\n{same_pairs/total_pairs*100:.1f}% пар НЕ различаются')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Scatter: statistic vs p-value
    axes[1].scatter(range(len(df_ks)), df_ks['ks_pvalue'], 
                   alpha=0.5, color='steelblue', s=10)
    axes[1].axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
    axes[1].set_xlabel('Пара (реальная vs случайная)')
    axes[1].set_ylabel('KS p-value')
    axes[1].set_title('KS p-value для каждой пары')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

# ============================================================
# 7. ЗАПУСК
# ============================================================

def main():
    """Основная функция."""
    NETWORKS_FOLDER = "C:/Proj_edge_distr/cite_A/"
    OUTPUT_DIR = "C:/Proj_edge_distr/closeness_ks_test/"
    MAX_NODES = 5000  # ← ЛИМИТ ПО ЧИСЛУ ВЕРШИН
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print("="*70)
    print("ДОКАЗАТЕЛЬСТВО: closeness НЕ ЯВЛЯЕТСЯ ИНВАРИАНТОМ")
    print("="*70)
    print(f"Максимальное число вершин: {MAX_NODES}")
    
    # 1. Собираем реальные сети (с проверкой на max_nodes)
    real_gradients = collect_real_networks(NETWORKS_FOLDER, max_networks=None, max_nodes=MAX_NODES)
    
    # 2. Генерируем случайные сети (с проверкой на max_nodes)
    random_gradients = generate_random_networks(n_networks=44, max_nodes=MAX_NODES)
    
    if len(real_gradients) == 0 or len(random_gradients) == 0:
        print("\n❌ Недостаточно данных для анализа")
        return
    
    print(f"\nРеальные сети: {len(real_gradients)}")
    print(f"Случайные сети: {len(random_gradients)}")
    
    # 3. KS-тест
    df_ks = run_ks_test(real_gradients, random_gradients)
    
    # 4. Визуализация
    plot_ks_results(df_ks, save_path=os.path.join(OUTPUT_DIR, 'ks_test_results.png'))
    
    # 5. Сохраняем результаты
    df_ks.to_csv(os.path.join(OUTPUT_DIR, 'ks_test_results.csv'), index=False)
    
    print(f"\n✅ Результаты сохранены в: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()