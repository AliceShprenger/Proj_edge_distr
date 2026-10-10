import os
import re
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import skew, weibull_min, mannwhitneyu, pearsonr
from scipy import stats
from matplotlib.ticker import LogLocator
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
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
    # Загружаем сеть
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
            data = None  # это v7.3 — нужен HDF5
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
            # sparse -> dense
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

    # --- Приводим к неориентированному виду ---
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    # --- Если несвязный — берём крупнейшую компоненту ---
    if G.number_of_nodes() > 0 and not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        print(f"    ℹ️ Несвязный граф: {G.number_of_nodes()} -> {len(largest_cc)} вершин (крупнейшая КС)")
        G = G.subgraph(largest_cc).copy()

    return G


def load_cleaned_gml(path):
    # Читает GML файл, удаляет все нестандартные атрибуты.
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
    return nx.parse_gml(cleaned_text, label='id')


def validate_network(G, min_nodes=10):
    # Проверяем сеть на пригодность для анализа.
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n < min_nodes:
        return False, f"Слишком мало вершин ({n} < {min_nodes})"

    if m == 0:
        return False, "Нет рёбер"

    return True, "OK"


# ========================
# 2. ВЫЧИСЛЕНИЕ РЁБЕРНЫХ ГРАДИЕНТОВ
# ========================

def compute_edge_gradients(G, centrality_type):
    # Вычисляем рёберные градиенты для заданной центральности.
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

    G_dir = G.to_directed()  # Делаем орграф, теперь рёбра раздваиваются и направлены в разные стороны
    edges_to_remove = [(u, v) for u, v in G_dir.edges() if u > v]
    G_dir.remove_edges_from(edges_to_remove)  # удаляем у каждой пары рёбер одно ребро
    # нам не важны направления рёбер, так как смотрим модуль градиента.

    try:
        B = nx.incidence_matrix(G_dir, oriented=True).toarray()
        f = np.array([centrality_dict[node] for node in G_dir.nodes()])
        gradients = B.T.dot(f)  # транспонированная матрица инцидентности - это матрица оператора кограничного отображения
        abs_gradients = np.abs(gradients)
        mean_val = np.mean(abs_gradients)
        if mean_val > 0:
            return abs_gradients / mean_val  # Нормируем
        return abs_gradients
    except Exception as e:
        print(f"    👿 Ошибка при вычислении градиентов: {e}")
        return None


# ========================
# 3. ПОСТРОЕНИЕ ГРАФИКОВ
# ========================

def plot_qq_pp_histogram(gradients, network_name, centrality_type, save_path=None):
    # Строит три графика: QQ-plot, P-P plot, гистограмма с линией Weibull.
    grad = gradients[gradients > 1e-12]
    if len(grad) < 50:
        print(f"  {network_name} ({centrality_type}): недостаточно данных")
        return None, None, None

    try:
        params = weibull_min.fit(grad, floc=0)
        c, loc, scale = params
        skewness_val = skew(grad)
    except Exception as e:
        print(f"  {network_name} ({centrality_type}): ошибка - {e}")
        return None, None, None

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f'{network_name} ({centrality_type})\nWeibull c={c:.3f}, scale={scale:.3f}, skewness={skewness_val:.3f}',
                 fontsize=14, fontweight='bold')

    n = len(grad)
    sorted_data = np.sort(grad)
    p = np.arange(1, n + 1) / (n + 1)

    # QQ-plot
    theoretical_qq = weibull_min.ppf(p, c=c, scale=scale)  # теоретические квантили
    axes[0].scatter(theoretical_qq, sorted_data, alpha=0.5, s=10, color='blue')
    axes[0].plot([min(theoretical_qq), max(theoretical_qq)],
                 [min(theoretical_qq), max(theoretical_qq)],
                 'r--', linewidth=2, label='y=x (идеал)')
    axes[0].set_xlabel('Теоретические квантили (Weibull)')
    axes[0].set_ylabel('Эмпирические квантили')
    axes[0].set_title('QQ-plot')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    residuals = sorted_data - theoretical_qq
    rmse = np.sqrt(np.mean(residuals ** 2))
    axes[0].text(0.05, 0.95, f'RMSE={rmse:.4f}', transform=axes[0].transAxes,
                 fontsize=10, verticalalignment='top')

    # P-P plot
    empirical_cdf = np.arange(1, n + 1) / n
    theoretical_pp = weibull_min.cdf(sorted_data, c=c, scale=scale)
    axes[1].scatter(theoretical_pp, empirical_cdf, alpha=0.5, s=10, color='green')
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x (идеал)')
    axes[1].set_xlabel('Теоретическая CDF (Weibull)')
    axes[1].set_ylabel('Эмпирическая CDF')
    axes[1].set_title('P-P plot')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    residuals_pp = empirical_cdf - theoretical_pp
    rmse_pp = np.sqrt(np.mean(residuals_pp ** 2))
    axes[1].text(0.05, 0.95, f'RMSE={rmse_pp:.4f}', transform=axes[1].transAxes,
                 fontsize=10, verticalalignment='top')

    # Гистограмма
    min_val = max(grad.min(), 1e-6)
    max_val = grad.max()
    margin = 0.5
    min_val_extended = min_val / (10 ** margin)
    max_val_extended = max_val * (10 ** margin)

    n_bins = 60
    bins = np.logspace(np.log10(min_val_extended), np.log10(max_val_extended), n_bins)
    axes[2].hist(grad, bins=bins, density=True, alpha=0.5,
                 label='Данные', color='gray', edgecolor='black', linewidth=0.5)

    x_vals = np.logspace(np.log10(min_val_extended), np.log10(max_val_extended), 2000)
    weibull_pdf = weibull_min.pdf(x_vals, c=c, scale=scale)
    weibull_pdf_norm = weibull_pdf / np.trapz(weibull_pdf, x_vals)
    axes[2].loglog(x_vals, weibull_pdf_norm, 'r-', linewidth=2.5,
                   label=f'Weibull (c={c:.3f})')

    axes[2].set_xlim(min_val_extended, max_val_extended)
    y_min, y_max = axes[2].get_ylim()
    axes[2].set_ylim(y_min / 10, y_max * 10)

    axes[2].xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    axes[2].xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
    axes[2].yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    axes[2].yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=10))
    axes[2].grid(True, which='major', linestyle='-', alpha=0.3, linewidth=0.8)
    axes[2].grid(True, which='minor', linestyle=':', alpha=0.15, linewidth=0.5)

    axes[2].set_xlabel('Градиент (log scale)', fontsize=12)
    axes[2].set_ylabel('Плотность (log scale)', fontsize=12)
    axes[2].set_title('Градиент-плотность с кривой Вейбулла', fontsize=12)
    axes[2].legend(loc='best', fontsize=10)

    info_text = (f'$n_{{edges}}$={len(grad):,}\n'
                 f'$c$={c:.3f}\n'
                 f'$\\xi$={skewness_val:.3f}')
    axes[2].text(0.05, 0.05, info_text, transform=axes[2].transAxes,
                 fontsize=10, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))

    plt.tight_layout()
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return c, scale, skewness_val


# ========================
# 4. ГЕНЕРАЦИЯ СЛУЧАЙНЫХ СЕТЕЙ
# ========================

def generate_random_networks(n_networks=10, centrality='betweenness', output_dir='.'):
    """
    Генерирует случайные графы разных моделей с равномерно разбросанными параметрами.
    """
    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ И АНАЛИЗ СЛУЧАЙНЫХ СЕТЕЙ (равномерные параметры)")
    print("=" * 60)

    results = []
    n_nodes = 500  # фиксируем число вершин для всех моделей

    # Определяем диапазоны параметров для каждой модели
    param_ranges = {
        'ER': {
            'p': (0.001, 0.5)  # вероятность появления ребра
        },
        'BA': {
            'm': (1, 30)  # число рёбер для новой вершины
        },
        'WS': {
            'k': (2, 15),     # число соседей в кольце
            'p': (0.01, 0.9)  # вероятность переброски
        }
    }

    # Генерируем случайные параметры для каждой модели
    models = []

    # Равномерно распределяем количество сетей по моделям
    n_per_model = n_networks // 3
    remainder = n_networks % 3

    # ER
    for i in range(n_per_model + (1 if remainder > 0 else 0)):
        p = random.uniform(*param_ranges['ER']['p'])
        models.append(('ER', {'n': n_nodes, 'p': p}))

    # BA
    for i in range(n_per_model + (1 if remainder > 1 else 0)):
        m = random.randint(*param_ranges['BA']['m'])
        models.append(('BA', {'n': n_nodes, 'm': m}))

    # WS
    for i in range(n_per_model):
        k = random.randint(*param_ranges['WS']['k'])
        p = random.uniform(*param_ranges['WS']['p'])
        models.append(('WS', {'n': n_nodes, 'k': k, 'p': p}))

    # Перемешиваем, чтобы модели шли в случайном порядке
    random.shuffle(models)

    for i, (model_name, params) in enumerate(models):
        try:
            print(f"\n  Генерация {model_name} {i + 1}/{len(models)}...")
            print(f"    Параметры: {params}")

            if model_name == 'ER':
                G = nx.erdos_renyi_graph(**params)
            elif model_name == 'BA':
                G = nx.barabasi_albert_graph(**params)
            elif model_name == 'WS':
                G = nx.watts_strogatz_graph(**params)
            else:
                continue

            if G.number_of_edges() < 10:
                print(f"    👿 Слишком мало ребер ({G.number_of_edges()})")
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
            c, scale, skewness_val = plot_qq_pp_histogram(
                gradients, name, centrality, save_path=save_path
            )

            if c is not None:
                results.append({
                    'filename': name,
                    'centrality': centrality,
                    'type': 'Random',
                    'model': model_name,
                    'params': str(params),
                    'n_nodes': G.number_of_nodes(),
                    'n_edges': G.number_of_edges(),
                    'weibull_c': c,
                    'weibull_scale': scale,
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
                          max_networks=None, min_nodes=3000):
    # Анализируем реальные сети.
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
    print(f"Минимальное число вершин для анализа: {min_nodes}")

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

        # Применяем порог по числу вершин ПОСЛЕ взятия крупнейшей КС
        # (для .mat крупнейшая КС уже взята внутри load_mat_network)
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

        gradients = gradients[gradients > 1e-12]  # убираем ошибки округления
        if len(gradients) < 10:
            print(f"  Недостаточно данных после фильтрации")
            continue

        print(f"  Вершин: {G.number_of_nodes()}, Рёбер: {len(gradients)}")

        save_path = os.path.join(output_dir, f'real_{centrality}_{filename.replace(".", "_")}.png')
        c, scale, skewness_val = plot_qq_pp_histogram(
            gradients, filename, centrality, save_path=save_path
        )

        if c is not None:
            results.append({
                'filename': filename,
                'centrality': centrality,
                'type': 'Real',
                'model': 'Real',
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'weibull_c': c,
                'weibull_scale': scale,
                'skewness': skewness_val
            })

    df = pd.DataFrame(results)
    return df


# ========================
# 6. ROC-АНАЛИЗ
# ========================

def plot_roc_curve(df, feature='weibull_c', label_col='type', output_dir='.', save_name=None, invert=False):
    # Строит ROC-кривую для заданного признака.
    # Если invert=True, признак инвертируется (подходит для Weibull c).

    if len(df) == 0:
        print("😐 Нет данных для ROC-анализа")
        return None

    if feature not in df.columns:
        print(f"😐 Признак {feature} не найден в данных")
        return None

    # Кодируем метки
    le = LabelEncoder()
    y_true = le.fit_transform(df[label_col])

    # Проверяем, что есть оба класса
    if len(np.unique(y_true)) < 2:
        print(f"😐 В данных только один класс: {le.classes_}")
        return None

    X = df[feature].values

    # Инвертируем признак, если нужно (для c нужно)
    if invert:
        X = -X
        feature_display = f"{feature} (инвертированный)"
    else:
        feature_display = feature

    # ROC-кривая
    fpr, tpr, thresholds = roc_curve(y_true, X)
    roc_auc = auc(fpr, tpr)

    # Оптимальный порог (Youden's J)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # График
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random (AUC = 0.5)')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100,
                label=f'Optimal threshold = {optimal_threshold:.3f}', zorder=5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(f'ROC-кривая для признака "{feature_display}"\nРеальные vs Случайные сети', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_name is None:
        save_name = f'roc_{feature}.png'
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Вывод статистики
    print(f"\n{'=' * 60}")
    print(f"ROC-АНАЛИЗ ДЛЯ ПРИЗНАКА: {feature_display}")
    print('=' * 60)
    print(f"AUC = {roc_auc:.4f}")

    # Для инвертированного признака показываем исходный порог
    if invert:
        original_threshold = -optimal_threshold
        print(f"Оптимальный порог (инвертированный) = {optimal_threshold:.4f}")
        print(f"Оптимальный порог (исходный c) = {original_threshold:.4f}")
    else:
        print(f"Оптимальный порог = {optimal_threshold:.4f}")

    print(f"TPR (чувствительность) = {tpr[optimal_idx]:.4f}")
    print(f"FPR (1-специфичность) = {fpr[optimal_idx]:.4f}")

    # Интерпретация
    if roc_auc >= 0.9:
        print("💚 Отлично! AUC > 0.9 — выдающееся разделение!")
    elif roc_auc >= 0.8:
        print("💚 Хорошо! AUC > 0.8 — хорошее разделение!")
    elif roc_auc >= 0.7:
        print("😐 Приемлемо! AUC > 0.7 — удовлетворительное разделение.")
    else:
        print("👿 Плохо! AUC < 0.7 — разделение слабое.")

    return {'auc': roc_auc, 'optimal_threshold': optimal_threshold, 'fpr': fpr, 'tpr': tpr}


def plot_roc_comparison(df, features=['weibull_c', 'skewness'], label_col='type', output_dir='.'):
    """
    Сравнивает ROC-кривые для разных признаков.
    """
    if len(df) == 0:
        print("😐 Нет данных для ROC-анализа")
        return

    le = LabelEncoder()
    y_true = le.fit_transform(df[label_col])

    if len(np.unique(y_true)) < 2:
        print(f"😐 В данных только один класс: {le.classes_}")
        return

    plt.figure(figsize=(10, 8))
    colors = ['darkorange', 'blue', 'green', 'red', 'purple']

    auc_values = {}

    for i, feature in enumerate(features):
        if feature not in df.columns:
            print(f"😐 Признак {feature} не найден в данных")
            continue

        # Для weibull_c используем инвертированный признак
        if feature == 'weibull_c':
            X = -df[feature].values
            display_name = 'weibull_c (инвертированный)'
        else:
            X = df[feature].values
            display_name = feature

        fpr, tpr, _ = roc_curve(y_true, X)
        roc_auc = auc(fpr, tpr)
        auc_values[display_name] = roc_auc

        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f'{display_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Сравнение ROC-кривых\nРеальные vs Случайные сети', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, 'roc_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Вывод сравнения
    print(f"\n{'=' * 60}")
    print("СРАВНЕНИЕ ПРИЗНАКОВ ПО AUC")
    print('=' * 60)
    for feature, auc_val in sorted(auc_values.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: AUC = {auc_val:.4f}")

    # Комбинация признаков (используем инвертированный c)
    try:
        X_comb = np.column_stack([-df['weibull_c'].values, df['skewness'].values])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_comb, y_true)
        y_score = clf.predict_proba(X_comb)[:, 1]
        auc_comb = roc_auc_score(y_true, y_score)
        print(f"\n  Комбинация (инвертированный c + skewness): AUC = {auc_comb:.4f}")

        if auc_comb > max(auc_values.values()):
            print("  💚 Комбинация признаков улучшает разделение!")
        else:
            print("  🥶 Комбинация признаков не улучшает разделение.")
    except Exception as e:
        print(f"  👿 Ошибка при комбинации признаков: {e}")


# ========================
# 7. СРАВНИТЕЛЬНЫЕ ГРАФИКИ С КОЭФФИЦИЕНТОМ ПИРСОНА
# ========================

def plot_comparison(real_df, random_df, centrality, output_dir='.'):
    # Строит сравнительные графики с коэффициентом Пирсона.
    if len(real_df) == 0 or len(random_df) == 0:
        print("Недостаточно данных для сравнения")
        return None

    # Подготовка данных для корреляции
    combined_c = list(real_df['weibull_c'].dropna()) + list(random_df['weibull_c'].dropna())
    combined_skew = list(real_df['skewness'].dropna()) + list(random_df['skewness'].dropna())
    labels = [1] * len(real_df) + [0] * len(random_df)

    # Корреляции Пирсона
    r_c, p_c = pearsonr(combined_c, labels)
    r_skew, p_skew = pearsonr(combined_skew, labels)

    # Корреляция между c и skewness (для всех данных)
    combined_all = pd.concat([real_df, random_df], ignore_index=True)
    r_c_skew, p_c_skew = pearsonr(combined_all['weibull_c'], combined_all['skewness'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Сравнение реальных и случайных сетей\nЦентральность: {centrality}',
                 fontsize=14, fontweight='bold')

    # 1. Scatter: weibull_c (Real vs Random)
    axes[0].scatter([0] * len(real_df), real_df['weibull_c'],
                    color='blue', alpha=0.7, label='Real', s=50)
    axes[0].scatter([1] * len(random_df), random_df['weibull_c'],
                    color='red', alpha=0.7, label='Random', s=50)
    axes[0].axhline(y=1, color='green', linestyle='--', linewidth=2, label='c=1')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Real', 'Random'])
    axes[0].set_ylabel('Weibull c')
    axes[0].set_title('Параметр формы Weibull')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.05, 0.95, f'Пирсон: r = {r_c:.3f} (p = {p_c:.4f})',
                 transform=axes[0].transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # 2. Scatter: skewness (Real vs Random)
    axes[1].scatter([0] * len(real_df), real_df['skewness'],
                    color='blue', alpha=0.7, label='Real', s=50)
    axes[1].scatter([1] * len(random_df), random_df['skewness'],
                    color='red', alpha=0.7, label='Random', s=50)
    axes[1].axhline(y=2, color='green', linestyle='--', linewidth=2, label='skewness=2')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Real', 'Random'])
    axes[1].set_ylabel('Skewness')
    axes[1].set_title('Коэффициент асимметрии')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.05, 0.95, f'Пирсон: r = {r_skew:.3f} (p = {p_skew:.4f})',
                 transform=axes[1].transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # 3. Scatter: c vs skewness
    axes[2].scatter(real_df['weibull_c'], real_df['skewness'],
                    color='blue', alpha=0.7, label='Real', s=50)
    axes[2].scatter(random_df['weibull_c'], random_df['skewness'],
                    color='red', alpha=0.7, label='Random', s=50)
    axes[2].axhline(y=2, color='green', linestyle='--', linewidth=2, label='skewness=2')
    axes[2].axvline(x=1, color='orange', linestyle='--', linewidth=2, label='c=1')
    axes[2].set_xlabel('Weibull c')
    axes[2].set_ylabel('Skewness')
    axes[2].set_title('c vs skewness')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].text(0.05, 0.95, f'Пирсон: r = {r_c_skew:.3f} (p = {p_c_skew:.4f})',
                 transform=axes[2].transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Вывод корреляций в консоль
    print("\n" + "=" * 70)
    print(f"КОРРЕЛЯЦИИ ПИРСОНА ДЛЯ {centrality.upper()}")
    print("=" * 70)
    print(f"  weibull_c vs тип сети:  r = {r_c:.4f}  (p = {p_c:.4f})")
    print(f"  skewness vs тип сети:   r = {r_skew:.4f}  (p = {p_skew:.4f})")
    print(f"  weibull_c vs skewness:  r = {r_c_skew:.4f}  (p = {p_c_skew:.4f})")

    # Интерпретация
    print("\nИнтерпретация:")
    if abs(r_skew) > 0.7:
        print(f"  💚 skewness: СИЛЬНАЯ связь с типом сети")
    elif abs(r_skew) > 0.5:
        print(f"  💚 skewness: УМЕРЕННАЯ связь с типом сети")
    elif abs(r_skew) > 0.3:
        print(f"  😐 skewness: СЛАБАЯ связь с типом сети")
    else:
        print(f"  👿 skewness: НЕТ связи с типом сети")

    if abs(r_c) > 0.7:
        print(f"  💚 weibull_c: СИЛЬНАЯ связь с типом сети")
    elif abs(r_c) > 0.5:
        print(f"  💚 weibull_c: УМЕРЕННАЯ связь с типом сети")
    elif abs(r_c) > 0.3:
        print(f"  😐 weibull_c: СЛАБАЯ связь с типом сети")
    else:
        print(f"  👿 weibull_c: НЕТ связи с типом сети")

    return {
        'r_c': r_c, 'p_c': p_c,
        'r_skew': r_skew, 'p_skew': p_skew,
        'r_c_skew': r_c_skew, 'p_c_skew': p_c_skew
    }


# ========================
# 8. СВОДКА
# ========================

def print_comparison_summary(real_df, random_df):
    # Выводит сравнительную сводку.
    print("\n" + "=" * 70)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("РЕАЛЬНЫЕ СЕТИ:")
    print("-" * 70)
    print(f"  Количество: {len(real_df)}")
    print(f"  Weibull c: mean={real_df['weibull_c'].mean():.3f} ± {real_df['weibull_c'].std():.3f}")
    print(f"  Доля c < 1: {(real_df['weibull_c'] < 1).mean() * 100:.1f}%")
    print(f"  Skewness: mean={real_df['skewness'].mean():.3f} ± {real_df['skewness'].std():.3f}")
    print(f"  Доля skewness > 2: {(real_df['skewness'] > 2).mean() * 100:.1f}%")

    print("\n" + "-" * 70)
    print("СЛУЧАЙНЫЕ СЕТИ:")
    print("-" * 70)
    print(f"  Количество: {len(random_df)}")
    print(f"  Weibull c: mean={random_df['weibull_c'].mean():.3f} ± {random_df['weibull_c'].std():.3f}")
    print(f"  Доля c < 1: {(random_df['weibull_c'] < 1).mean() * 100:.1f}%")
    print(f"  Skewness: mean={random_df['skewness'].mean():.3f} ± {random_df['skewness'].std():.3f}")
    print(f"  Доля skewness > 2: {(random_df['skewness'] > 2).mean() * 100:.1f}%")

    print("\n" + "-" * 70)
    print("СТАТИСТИЧЕСКИЕ ТЕСТЫ (Mann-Whitney):")
    print("-" * 70)

    stat, p = mannwhitneyu(real_df['weibull_c'], random_df['weibull_c'])
    print(f"  Weibull c: p={p:.4f} {'💚 Различаются' if p < 0.05 else '👿 Не различаются'}")

    stat, p = mannwhitneyu(real_df['skewness'], random_df['skewness'])
    print(f"  Skewness: p={p:.4f} {'💚 Различаются' if p < 0.05 else '👿 Не различаются'}")

    print("\n" + "=" * 70)


# ========================
# 9. ОСНОВНАЯ ФУНКЦИЯ
# ========================

def run_full_analysis(folder_path, centrality='betweenness', n_random=10,
                      max_networks=None, min_nodes=3000):
    """Запускает полный анализ: реальные сети + случайные + ROC + Пирсон."""
    base_dir = "C:/Proj_edge_distr/"  # Сохраняем всё в эту папку
    output_dir = os.path.join(base_dir, f"results_{centrality}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Результаты будут сохранены в: {output_dir}")

    # Вычисления:
    # Реальные сети
    real_df = analyze_real_networks(folder_path, centrality, output_dir,
                                    max_networks, min_nodes=min_nodes)

    # Случайные сети
    random_df = generate_random_networks(n_random, centrality, output_dir)

    # Коэф корреляции Пирсона
    pearson_results = None
    if len(real_df) > 0 and len(random_df) > 0:
        pearson_results = plot_comparison(real_df, random_df, centrality, output_dir)
        print_comparison_summary(real_df, random_df)

    # ROC-анализ
    df_combined = None
    if len(real_df) > 0 and len(random_df) > 0:
        df_combined = pd.concat([real_df, random_df], ignore_index=True)

        print("\n" + "=" * 70)
        print("ROC-АНАЛИЗ ДЛЯ КЛАССИФИКАЦИИ")
        print("=" * 70)
        print(f"Всего сетей: {len(df_combined)}")
        print(f"  Реальных: {len(real_df)}")
        print(f"  Случайных: {len(random_df)}")

        # ROC для Weibull c (с инвертированием!)
        plot_roc_curve(df_combined, feature='weibull_c', output_dir=output_dir,
                       save_name='roc_weibull_c.png', invert=True)

        # ROC для Skewness (без инвертирования)
        plot_roc_curve(df_combined, feature='skewness', output_dir=output_dir,
                       save_name='roc_skewness.png', invert=False)

        # Сравнение ROC-кривых
        plot_roc_comparison(df_combined, features=['weibull_c', 'skewness'], output_dir=output_dir)

        # Сохраняем объединенные результаты
        df_combined.to_csv(os.path.join(output_dir, 'full_analysis_results.csv'), index=False)

    print(f"\n💚 Все результаты сохранены в папку: {output_dir}")

    return df_combined, real_df, random_df, pearson_results


# ========================
# 10. ЗАПУСК
# ========================

if __name__ == "__main__":
    NETWORKS_FOLDER = "C:/Proj_edge_distr/merged_matrices/"

    df, real, random, pearson = run_full_analysis(
        folder_path=NETWORKS_FOLDER,
        centrality='betweenness',
        n_random=48,
        max_networks=None,
        min_nodes=3000          # <-- порог по числу вершин (для крупных сетей)
    )

    # Вывод корреляций в конце
    # r - Коэффициент корреляции Пирсона
    # p - p-value (Вероятность ошибиться, если сказать, что связь есть), Маленькое p (< 0.05) - связь не случайна
    if pearson is not None:
        print("\n" + "=" * 70)
        print("ИТОГОВЫЕ КОРРЕЛЯЦИИ ПИРСОНА")
        print("=" * 70)
        print(f"  weibull_c vs тип сети:  r = {pearson['r_c']:.4f}  (p = {pearson['p_c']:.4f})")
        print(f"  skewness vs тип сети:   r = {pearson['r_skew']:.4f}  (p = {pearson['p_skew']:.4f})")
        print(f"  weibull_c vs skewness:  r = {pearson['r_c_skew']:.4f}  (p = {pearson['p_c_skew']:.4f})")