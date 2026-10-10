import os
import re
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import linregress
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')


# ========================
# 0. ОПЦИОНАЛЬНЫЕ ЗАВИСИМОСТИ ДЛЯ .mat
# ========================

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
    if filepath.endswith('.gml'):
        return load_cleaned_gml(filepath)
    elif filepath.endswith(('.txt', '.edges', '.edge')):
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


def load_cleaned_gml(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = []
    allowed_keywords = {"graph", "node", "edge", "id", "label",
                        "source", "target", "[", "]"}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        first_word = stripped.split()[0]
        if first_word in allowed_keywords:
            cleaned_lines.append(line)

    cleaned_text = "".join(cleaned_lines)
    try:
        return nx.parse_gml(cleaned_text, label='id')
    except Exception:
        return parse_gml_manual(cleaned_text)


def parse_gml_manual(content):
    node_pattern = r'node\s*\[\s*id\s+(\d+)'
    nodes = re.findall(node_pattern, content)
    edge_pattern = r'edge\s*\[\s*id\s+\d+\s*source\s+(\d+)\s*target\s+(\d+)'
    edges = re.findall(edge_pattern, content)

    G = nx.Graph()
    for node_id in nodes:
        G.add_node(int(node_id))

    edges_set = set()
    for source, target in edges:
        edges_set.add(tuple(sorted([int(source), int(target)])))

    for source, target in edges_set:
        G.add_edge(source, target)

    return G


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
            data = None  # это v7.3 — нужен HDF5
        except Exception:
            data = None

    # --- v7.3 через mat73 ---
    if data is None and MAT73_AVAILABLE:
        try:
            data = mat73.loadmat(filepath)
        except Exception:
            pass

    # --- v7.3 через h5py ---
    if data is None and H5PY_AVAILABLE:
        try:
            data = {}
            with h5py.File(filepath, 'r') as f:
                for key in f.keys():
                    data[key] = f[key][:]
        except Exception:
            data = None

    if data is None:
        raise ValueError(f"Не удалось прочитать .mat файл: {filepath}")

    # Убираем служебные ключи
    data = {k: v for k, v in data.items() if not k.startswith('__')}
    if not data:
        raise ValueError(f"В .mat файле нет переменных: {filepath}")

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
            break

        # N x 2 — список рёбер
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 1:
            try:
                edges = [(int(u), int(v)) for u, v in arr]
                G = nx.Graph()
                G.add_edges_from(edges)
                break
            except (ValueError, TypeError):
                continue

        # Плоский список рёбер
        if arr.ndim == 1 and arr.size >= 4 and arr.size % 2 == 0:
            try:
                edges = [(int(arr[i]), int(arr[i + 1])) for i in range(0, arr.size, 2)]
                G = nx.Graph()
                G.add_edges_from(edges)
                break
            except (ValueError, TypeError):
                continue

    if G is None:
        raise ValueError(f"Не найдена матрица смежности или список рёбер в {filepath}")

    # --- Приводим к неориентированному виду ---
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    # --- Если несвязный — берём крупнейшую компоненту ---
    if G.number_of_nodes() > 0 and not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    return G


# ========================
# 2. ПРОВЕРКА ШУМА (без графиков)
# ========================

def compute_noise_metrics(G):
    """
    Возвращает (is_noisy, noise_score, metrics).
    Полностью повторяет логику check_degree_noise, но без plot.
    """
    degrees = np.array([d for n, d in G.degree()])

    if len(degrees) < 10:
        return True, 1.0, {'error': 'too_few_nodes', 'n_nodes': len(degrees)}

    unique_deg, counts = np.unique(degrees, return_counts=True)
    p_k = counts / len(degrees)

    min_deg = max(unique_deg.min(), 1)
    max_deg = unique_deg.max()

    if min_deg == max_deg:
        return True, 1.0, {'error': 'only_one_degree', 'n_nodes': len(degrees)}

    log_bins = np.logspace(np.log10(min_deg), np.log10(max_deg),
                           num=min(30, len(unique_deg)))
    log_bins = np.unique(np.round(log_bins).astype(int))
    log_bins = log_bins[log_bins >= 1]

    if len(log_bins) < 3:
        return True, 1.0, {'error': 'too_few_bins', 'n_nodes': len(degrees)}

    binned_ks, binned_ps = [], []
    for i in range(len(log_bins) - 1):
        mask = (unique_deg >= log_bins[i]) & (unique_deg < log_bins[i + 1])
        if np.sum(mask) > 0:
            binned_ks.append(np.mean(unique_deg[mask]))
            binned_ps.append(np.mean(p_k[mask]))

    if len(binned_ks) < 3:
        return True, 1.0, {'error': 'too_few_bins_after_binning',
                           'n_nodes': len(degrees)}

    log_ks = np.log10(binned_ks)
    log_ps = np.log10(binned_ps)

    tail_start = max(1, len(log_ks) // 3)
    tail_ks = log_ks[tail_start:]
    tail_ps = log_ps[tail_start:]

    if len(tail_ks) < 3:
        return True, 1.0, {'error': 'too_short_tail', 'n_nodes': len(degrees)}

    slope, intercept, r_value, p_value, std_err = linregress(tail_ks, tail_ps)
    r_squared = r_value ** 2

    fitted = slope * tail_ks + intercept
    residuals = tail_ps - fitted
    rmse = np.sqrt(np.mean(residuals ** 2))
    signal_range = np.ptp(tail_ps)
    noise_ratio = rmse / (signal_range + 1e-10)

    if len(tail_ks) > 2:
        derivatives = np.diff(tail_ps) / np.diff(tail_ks)
        deriv_std = np.std(derivatives)
        deriv_mean = np.abs(np.mean(derivatives))
        smoothness = deriv_std / (deriv_mean + 1e-10)
    else:
        smoothness = 1.0

    noise_score = (1 - r_squared) * 0.5 + noise_ratio * 0.3 + min(smoothness, 1.0) * 0.2
    noise_score = min(noise_score, 1.0)

    is_noisy = (r_squared < 0.7) or (noise_score > 0.5)

    metrics = {
        'r_squared': r_squared,
        'slope': -slope,
        'noise_ratio': noise_ratio,
        'smoothness': smoothness,
        'noise_score': noise_score,
        'is_noisy': is_noisy,
        'tail_size': len(tail_ks),
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'max_degree': int(np.max(degrees)),
        'mean_degree': float(np.mean(degrees)),
        'degree_assortativity': nx.degree_assortativity_coefficient(G)
                                if G.number_of_edges() > 0 else 0.0,
    }
    return is_noisy, noise_score, metrics


# ========================
# 3. ОБРАБОТКА ОДНОЙ СЕТИ (для воркера)
# ========================

def process_one(filepath):
    """
    Возвращает dict с результатом для одной сети.
    Никогда не бросает исключение наружу — ошибки упаковываются в dict.
    """
    filename = os.path.basename(filepath)
    result = {
        'filename': filename,
        'filepath': filepath,
        'status': None,      # 'ok' | 'error'
        'reason': None,
    }

    try:
        G = load_network(filepath)
    except Exception as e:
        result['status'] = 'error'
        result['reason'] = f'load: {e}'
        return result

    if G is None:
        result['status'] = 'error'
        result['reason'] = 'G is None'
        return result

    if G.number_of_nodes() < 10 or G.number_of_edges() == 0:
        result['status'] = 'error'
        result['reason'] = f'too_small (V={G.number_of_nodes()}, E={G.number_of_edges()})'
        return result

    # Гигантская компонента (для .mat она уже взята внутри загрузчика)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    try:
        is_noisy, noise_score, metrics = compute_noise_metrics(G)
    except Exception as e:
        result['status'] = 'error'
        result['reason'] = f'metrics: {e}'
        return result

    if 'error' in metrics:
        result['status'] = 'error'
        result['reason'] = metrics['error']
        return result

    result['status'] = 'ok'
    result.update(metrics)
    return result


# ========================
# 4. ПАРТИЙНАЯ ОБРАБОТКА
# ========================

def filter_networks(folder_path, output_file='filtered_networks.txt',
                    min_nodes=600, max_nodes=None, workers=None, verbose=True):
    """
    Проходит по всем сетям в folder_path, отбирает те, у которых:
      - min_nodes < |V| < max_nodes   (если max_nodes задан)
      - не шумные (is_noisy == False)
    Сохраняет список в output_file (по одному пути на строку).
    Рядом кладёт подробный CSV и CSV с ошибками.
    """
    extensions = ('.gml', '.txt', '.edges', '.mtx', '.edge', '.mat')
    all_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(extensions):
                all_files.append(os.path.join(root, f))

    if verbose:
        print(f"Найдено файлов: {len(all_files)}")

    results = []

    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

    if verbose:
        print(f"Воркеров: {workers}")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, fp): fp for fp in all_files}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                results.append(fut.result())
            except Exception as e:
                fp = futures[fut]
                results.append({
                    'filename': os.path.basename(fp),
                    'filepath': fp,
                    'status': 'error',
                    'reason': f'worker: {e}',
                })
            if verbose and done % 50 == 0:
                print(f"  обработано {done}/{len(all_files)}")

    df = pd.DataFrame(results)

    # ---- Отбор ----
    ok = df[df['status'] == 'ok'].copy()
    errors = df[df['status'] == 'error'].copy()

    mask = (ok['n_nodes'] > min_nodes) & (~ok['is_noisy'])
    if max_nodes is not None:
        mask &= (ok['n_nodes'] < max_nodes)
    selected = ok[mask].copy()

    # ---- Сохранение списка путей ----
    with open(output_file, 'w', encoding='utf-8') as f:
        for p in selected['filepath'].tolist():
            f.write(p + '\n')

    # ---- Подробный CSV рядом ----
    csv_path = os.path.splitext(output_file)[0] + '_detailed.csv'
    selected.to_csv(csv_path, index=False, encoding='utf-8-sig', sep=';')

    # ---- Лог ошибок ----
    err_path = os.path.splitext(output_file)[0] + '_errors.csv'
    errors.to_csv(err_path, index=False, encoding='utf-8-sig', sep=';')

    if verbose:
        if max_nodes is not None:
            size_ok = ok[(ok['n_nodes'] > min_nodes) & (ok['n_nodes'] < max_nodes)]
            range_str = f"{min_nodes} < |V| < {max_nodes}"
        else:
            size_ok = ok[ok['n_nodes'] > min_nodes]
            range_str = f"|V| > {min_nodes}"

        print("\n" + "=" * 60)
        print("ИТОГ")
        print("=" * 60)
        print(f"  Всего файлов:        {len(df)}")
        print(f"  Успешно обработано:  {len(ok)}")
        print(f"  Ошибок:              {len(errors)}")
        print(f"  Размер {range_str}:  {len(size_ok)}")
        print(f"  Не шумных:           {len(ok[~ok['is_noisy']])}")
        print(f"  ✅ ПОДХОДЯТ ({range_str} И не шумные): {len(selected)}")
        print(f"\n  Список:  {output_file}")
        print(f"  Детали:  {csv_path}")
        print(f"  Ошибки:  {err_path}")

    return selected, ok, errors


# ========================
# 5. ЗАПУСК
# ========================

if __name__ == "__main__":
    selected, ok, errors = filter_networks(
        folder_path="C:/Proj_edge_distr/merged_matrices/",
        output_file="C:/Proj_edge_distr/filtered_networks.txt",
        min_nodes=600,
        max_nodes=50000,       # <-- верхняя граница; None = без ограничения
        workers=None,          # None = авто (CPU-1)
        verbose=True,
    )