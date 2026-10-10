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

# --- Опциональные зависимости для .mat ---
try:
    from scipy.io import loadmat
    SCIPY_MAT_AVAILABLE = True
except ImportError:
    SCIPY_MAT_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    import mat73
    MAT73_AVAILABLE = True
except ImportError:
    MAT73_AVAILABLE = False


# ========================
# 1. ЗАГРУЗКА СЕТЕЙ
# ========================

def load_network(filepath):
    """Загружает сеть из GML, TXT/EDGES, MTX или MAT файла."""
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
    elif filepath.endswith('.mat'):
        return load_mat_network(filepath)
    else:
        raise ValueError(f"Неизвестный формат: {filepath}")


def load_mat_network(filepath):
    """
    Загружает НЕориентированный граф из .mat.
    Поддерживает v4/v6/v7 (scipy) и v7.3 (mat73/h5py).
    Ищет матрицу смежности или список рёбер.
    Возвращает крупнейшую компоненту связности.
    """
    data = None

    # --- v4/v6/v7 через scipy ---
    if SCIPY_MAT_AVAILABLE:
        try:
            data = loadmat(filepath, squeeze_me=True, struct_as_record=False)
        except NotImplementedError:
            data = None  # v7.3 — нужен HDF5
        except Exception as e:
            print(f"    ⚠️ scipy.io.loadmat: {e}")
            data = None

    # --- v7.3 через mat73 ---
    if data is None and MAT73_AVAILABLE:
        try:
            data = mat73.loadmat(filepath)
            print("    ℹ️ Использован mat73 (v7.3 / HDF5)")
        except Exception as e:
            print(f"    ⚠️ mat73: {e}")

    # --- v7.3 через h5py ---
    if data is None and H5PY_AVAILABLE:
        try:
            data = {}
            with h5py.File(filepath, 'r') as f:
                for key in f.keys():
                    data[key] = f[key][:]
            print("    ℹ️ Использован h5py (v7.3 / HDF5)")
        except Exception as e:
            print(f"    ⚠️ h5py: {e}")

    if data is None:
        raise ValueError(f"Не удалось прочитать .mat файл: {filepath}")

    # Убираем служебные ключи
    data = {k: v for k, v in data.items() if not k.startswith('__')}
    if not data:
        raise ValueError(f"В .mat файле нет переменных: {filepath}")

    print(f"    Переменные в .mat: {list(data.keys())}")

    # --- Ищем матрицу смежности / список рёбер ---
    G = None
    for key, val in data.items():
        try:
            from scipy.sparse import issparse
            if issparse(val):
                arr = val.toarray()
            else:
                arr = np.asarray(val)
        except Exception:
            continue

        if arr.size == 0:
            continue

        # Квадратная 2D матрица — кандидат на adjacency
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 1:
            G = nx.from_numpy_array(arr)
            print(f"    ✓ Граф построен из '{key}' (матрица смежности {arr.shape})")
            break

        # N x 2 — список рёбер
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 1:
            try:
                edges = [(int(u), int(v)) for u, v in arr]
                G = nx.Graph()
                G.add_edges_from(edges)
                print(f"    ✓ Граф построен из '{key}' (список рёбер, {len(edges)} шт.)")
                break
            except (ValueError, TypeError):
                continue

        # Плоский список рёбер
        if arr.ndim == 1 and arr.size >= 4 and arr.size % 2 == 0:
            try:
                edges = [(int(arr[i]), int(arr[i + 1])) for i in range(0, arr.size, 2)]
                G = nx.Graph()
                G.add_edges_from(edges)
                print(f"    ✓ Граф построен из '{key}' (плоский список рёбер)")
                break
            except (ValueError, TypeError):
                continue

    if G is None:
        raise ValueError(f"Не найдена матрица смежности или список рёбер в {filepath}")

    # Приводим к неориентированному виду
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    # Если несвязный — берём крупнейшую компоненту
    if G.number_of_nodes() > 0 and not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        print(f"    ℹ️ Несвязный граф: {G.number_of_nodes()} -> {len(largest_cc)} вершин (крупнейшая КС)")
        G = G.subgraph(largest_cc).copy()

    return G


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
        print(f"    👿 Ошибка при вычислении {centrality_type}: {e}")
        return None

    G_dir = G.to_directed()
    edges_to_remove = [(u, v) for u, v in G_dir.edges() if u > v]
    G_dir.remove_edges_from(edges_to_remove)

    try:
        # Прямое вычисление градиентов без плотной матрицы инцидентности:
        # B.T @ f эквивалентно f[v] - f[u] для каждого ориентированного ребра.
        # Это O(m) по памяти, а не O(n*m).
        nodes = list(G_dir.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        f = np.fromiter((centrality_dict[n] for n in nodes),
                        dtype=np.float64, count=len(nodes))

        edges = list(G_dir.edges())
        u_idx = np.fromiter((idx[u] for u, v in edges),
                            dtype=np.int64, count=len(edges))
        v_idx = np.fromiter((idx[v] for u, v in edges),
                            dtype=np.int64, count=len(edges))

        gradients = f[v_idx] - f[u_idx]
        abs_gradients = np.abs(gradients)

        mean_val = np.mean(abs_gradients)
        if mean_val > 0:
            return abs_gradients / mean_val
        return abs_gradients
    except Exception as e:
        print(f"    👿 Ошибка при вычислении градиентов: {e}")
        return None


# ========================
# 3. ПОСТРОЕНИЕ ГИСТОГРАММЫ (ОБЫЧНЫЙ МАСШТАБ)
# ========================

def plot_histogram(gradients, network_name, centrality_type, save_path=None):
    """
    Строит гистограмму распределения градиентов в обычном (линейном) масштабе.
    """
    grad = gradients[gradients > 1e-12]
    if len(grad) < 50:
        print(f"  {network_name} ({centrality_type}): недостаточно данных")
        return None

    skewness_val = skew(grad)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(grad, bins=500, density=True, alpha=0.7,
            color='steelblue', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Градиент', fontsize=12)
    ax.set_ylabel('Плотность', fontsize=12)
    ax.set_title(f'{network_name} ({centrality_type})\nSkewness = {skewness_val:.3f}', fontsize=14)
    ax.grid(True, alpha=0.3)

    info_text = (f'$n_{{edges}}$ = {len(grad):,}\n'
                 f'$\\xi$ = {skewness_val:.3f}')
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return skewness_val


# ========================
# 4. ГЕНЕРАЦИЯ СЛУЧАЙНЫХ СЕТЕЙ
# ========================

def generate_random_networks(n_networks=10, centrality='betweenness', output_dir='.'):
    """Генерирует случайные графы разных моделей и строит гистограммы."""
    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ И АНАЛИЗ СЛУЧАЙНЫХ СЕТЕЙ")
    print("=" * 60)

    results = []
    n_nodes = 500

    models = [
        ('ER', {'n': n_nodes, 'p': 0.005}),
        ('ER', {'n': n_nodes, 'p': 0.005}),
        ('ER', {'n': n_nodes, 'p': 0.01}),
        ('ER', {'n': n_nodes, 'p': 0.01}),
        ('ER', {'n': n_nodes, 'p': 0.05}),
        ('ER', {'n': n_nodes, 'p': 0.05}),
        ('ER', {'n': n_nodes, 'p': 0.1}),
        ('ER', {'n': n_nodes, 'p': 0.1}),
        ('ER', {'n': n_nodes, 'p': 0.3}),
        ('ER', {'n': n_nodes, 'p': 0.3}),
        ('ER', {'n': n_nodes, 'p': 0.5}),
        ('ER', {'n': n_nodes, 'p': 0.5}),
        ('ER', {'n': n_nodes, 'p': 0.7}),
        ('ER', {'n': n_nodes, 'p': 0.7}),
        ('ER', {'n': n_nodes, 'p': 0.9}),
        ('ER', {'n': n_nodes, 'p': 0.9}),
        ('ER', {'n': n_nodes, 'p': 1}),
        ('ER', {'n': n_nodes, 'p': 1}),

        ('BA', {'n': n_nodes, 'm': 2}),
        ('BA', {'n': n_nodes, 'm': 2}),
        ('BA', {'n': n_nodes, 'm': 3}),
        ('BA', {'n': n_nodes, 'm': 3}),
        ('BA', {'n': n_nodes, 'm': 5}),
        ('BA', {'n': n_nodes, 'm': 5}),
        ('BA', {'n': n_nodes, 'm': 7}),
        ('BA', {'n': n_nodes, 'm': 7}),
        ('BA', {'n': n_nodes, 'm': 9}),
        ('BA', {'n': n_nodes, 'm': 9}),
        ('BA', {'n': n_nodes, 'm': 11}),
        ('BA', {'n': n_nodes, 'm': 11}),
        ('BA', {'n': n_nodes, 'm': 13}),
        ('BA', {'n': n_nodes, 'm': 13}),
        ('BA', {'n': n_nodes, 'm': 15}),
        ('BA', {'n': n_nodes, 'm': 15}),
        ('BA', {'n': n_nodes, 'm': 17}),
        ('BA', {'n': n_nodes, 'm': 17}),
        ('BA', {'n': n_nodes, 'm': 19}),
        ('BA', {'n': n_nodes, 'm': 19}),
        ('BA', {'n': n_nodes, 'm': 21}),
        ('BA', {'n': n_nodes, 'm': 21}),

        ('WS', {'n': n_nodes, 'k': 2, 'p': 0.05}),
        ('WS', {'n': n_nodes, 'k': 3, 'p': 0.05}),
        ('WS', {'n': n_nodes, 'k': 4, 'p': 0.05}),
        ('WS', {'n': n_nodes, 'k': 5, 'p': 0.05}),
        ('WS', {'n': n_nodes, 'k': 7, 'p': 0.05}),
        ('WS', {'n': n_nodes, 'k': 10, 'p': 0.05}),
        ('WS', {'n': n_nodes, 'k': 15, 'p': 0.05}),
        ('WS', {'n': n_nodes, 'k': 20, 'p': 0.05}),
        ('WS', {'n': n_nodes, 'k': 25, 'p': 0.05}),
        ('WS', {'n': n_nodes, 'k': 2, 'p': 0.1}),
        ('WS', {'n': n_nodes, 'k': 3, 'p': 0.1}),
        ('WS', {'n': n_nodes, 'k': 4, 'p': 0.1}),
        ('WS', {'n': n_nodes, 'k': 5, 'p': 0.1}),
        ('WS', {'n': n_nodes, 'k': 7, 'p': 0.1}),
        ('WS', {'n': n_nodes, 'k': 10, 'p': 0.1}),
        ('WS', {'n': n_nodes, 'k': 15, 'p': 0.1}),
        ('WS', {'n': n_nodes, 'k': 20, 'p': 0.1}),
        ('WS', {'n': n_nodes, 'k': 25, 'p': 0.1}),
        ('WS', {'n': n_nodes, 'k': 2, 'p': 0.2}),
        ('WS', {'n': n_nodes, 'k': 3, 'p': 0.2}),
        ('WS', {'n': n_nodes, 'k': 4, 'p': 0.2}),
        ('WS', {'n': n_nodes, 'k': 5, 'p': 0.2}),
        ('WS', {'n': n_nodes, 'k': 7, 'p': 0.2}),
        ('WS', {'n': n_nodes, 'k': 10, 'p': 0.2}),
        ('WS', {'n': n_nodes, 'k': 15, 'p': 0.2}),
        ('WS', {'n': n_nodes, 'k': 20, 'p': 0.2}),
        ('WS', {'n': n_nodes, 'k': 25, 'p': 0.2}),
        ('WS', {'n': n_nodes, 'k': 2, 'p': 0.5}),
        ('WS', {'n': n_nodes, 'k': 3, 'p': 0.5}),
        ('WS', {'n': n_nodes, 'k': 4, 'p': 0.5}),
        ('WS', {'n': n_nodes, 'k': 5, 'p': 0.5}),
        ('WS', {'n': n_nodes, 'k': 7, 'p': 0.5}),
        ('WS', {'n': n_nodes, 'k': 10, 'p': 0.5}),
        ('WS', {'n': n_nodes, 'k': 15, 'p': 0.5}),
        ('WS', {'n': n_nodes, 'k': 20, 'p': 0.5}),
        ('WS', {'n': n_nodes, 'k': 25, 'p': 0.5}),
        ('WS', {'n': n_nodes, 'k': 2, 'p': 0.9}),
        ('WS', {'n': n_nodes, 'k': 3, 'p': 0.9}),
        ('WS', {'n': n_nodes, 'k': 4, 'p': 0.9}),
        ('WS', {'n': n_nodes, 'k': 5, 'p': 0.9}),
        ('WS', {'n': n_nodes, 'k': 7, 'p': 0.9}),
        ('WS', {'n': n_nodes, 'k': 10, 'p': 0.9}),
        ('WS', {'n': n_nodes, 'k': 15, 'p': 0.9}),
        ('WS', {'n': n_nodes, 'k': 20, 'p': 0.9}),
        ('WS', {'n': n_nodes, 'k': 25, 'p': 0.9}),
    ]

    models = models[:n_networks]

    for i, (model_name, params) in enumerate(models):
        try:
            print(f"\n  Генерация {model_name} {i + 1}/{len(models)}...")

            if model_name == 'ER':
                G = nx.erdos_renyi_graph(**params)
            elif model_name == 'BA':
                G = nx.barabasi_albert_graph(**params)
            elif model_name == 'WS':
                G = nx.watts_strogatz_graph(**params)
            else:
                continue

            if G.number_of_edges() < 10:
                print(f"    👿 Слишком мало ребер")
                continue

            print(f"    Вершин: {G.number_of_nodes()}, Рёбер: {G.number_of_edges()}")

            gradients = compute_edge_gradients(G, centrality)
            if gradients is None:
                print(f"    👿 Не удалось вычислить градиенты")
                continue

            gradients = gradients[gradients > 1e-12]
            if len(gradients) < 10:
                print(f"    👿 Недостаточно данных после фильтрации")
                continue

            name = f"{model_name}_{i + 1}"
            save_path = os.path.join(output_dir, f'random_{centrality}_{name}.png')
            skewness_val = plot_histogram(gradients, name, centrality, save_path=save_path)

            if skewness_val is not None:
                results.append({
                    'filename': name,
                    'centrality': centrality,
                    'type': 'Random',
                    'model': model_name,
                    'n_nodes': G.number_of_nodes(),
                    'n_edges': G.number_of_edges(),
                    'skewness': skewness_val
                })

        except Exception as e:
            print(f"    👿 Ошибка: {e}")

    df = pd.DataFrame(results)
    return df


# ========================
# 5. АНАЛИЗ РЕАЛЬНЫХ СЕТЕЙ
# ========================

def analyze_real_networks(folder_path, centrality='betweenness', output_dir='.',
                          max_networks=None, min_nodes=10):
    """Анализирует реальные сети и строит гистограммы."""
    print("\n" + "=" * 60)
    print("АНАЛИЗ РЕАЛЬНЫХ СЕТЕЙ")
    print("=" * 60)

    extensions = ['.gml', '.txt', '.edges', '.mtx', '.mat']
    all_files = []

    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                all_files.append(os.path.join(root, f))

    print(f"Найдено файлов: {len(all_files)}")

    results = []

    for idx, filepath in enumerate(all_files):
        if max_networks and idx >= max_networks:
            break

        filename = os.path.basename(filepath)
        print(f"\n{'=' * 60}")
        print(f"Обработка: {filename}")
        print('=' * 60)

        try:
            G = load_network(filepath)
        except Exception as e:
            print(f"  Ошибка загрузки: {e}")
            continue

        if G is None:
            continue

        # Если несвязный — берём крупнейшую КС
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc)
            print(f"  Взята гигантская компонента: {len(largest_cc)} вершин")

        valid, msg = validate_network(G, min_nodes=min_nodes)
        if not valid:
            print(f"  Пропуск: {msg}")
            continue

        gradients = compute_edge_gradients(G, centrality)

        if gradients is None or len(gradients) < 10:
            print(f"  Недостаточно данных")
            continue

        gradients = gradients[gradients > 1e-12]
        if len(gradients) < 10:
            print(f"  Недостаточно данных после фильтрации")
            continue

        print(f"  Рёбер: {len(gradients)}")

        save_path = os.path.join(output_dir,
                                 f'real_{centrality}_{filename.replace(".", "_")}.png')
        skewness_val = plot_histogram(gradients, filename, centrality, save_path=save_path)

        if skewness_val is not None:
            results.append({
                'filename': filename,
                'centrality': centrality,
                'type': 'Real',
                'model': 'Real',
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'skewness': skewness_val
            })

    df = pd.DataFrame(results)
    return df


# ========================
# 7. ОСНОВНАЯ ФУНКЦИЯ
# ========================

def run_full_analysis(folder_path, centrality='betweenness', n_random=10,
                      max_networks=None, min_nodes=10):
    """
    Запускает полный анализ: реальные сети + случайные.
    Результаты сохраняются в C:/Proj_edge_distr/results_{centrality}/
    """
    base_dir = "C:/Proj_edge_distr/"
    output_dir = os.path.join(base_dir, f"results_{centrality}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Результаты будут сохранены в: {output_dir}")

    # Реальные сети
    real_df = analyze_real_networks(folder_path, centrality, output_dir,
                                    max_networks, min_nodes=min_nodes)

    # Случайные сети
    random_df = generate_random_networks(n_random, centrality, output_dir)

    # Сохраняем результаты
    df_combined = pd.concat([real_df, random_df], ignore_index=True)
    df_combined.to_csv(os.path.join(output_dir, 'full_analysis_results.csv'), index=False)
    print(f"\n💚 Все результаты сохранены в папку: {output_dir}")

    return df_combined, real_df, random_df


# ========================
# 8. ЗАПУСК
# ========================

if __name__ == "__main__":
    NETWORKS_FOLDER = "C:/Proj_edge_distr/selected/"

    df, real, random = run_full_analysis(
        folder_path=NETWORKS_FOLDER,
        centrality='pagerank',
        n_random=0,
        max_networks=None,
        min_nodes=0          # <-- вот здесь
    )