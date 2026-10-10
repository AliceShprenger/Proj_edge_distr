import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import skew, weibull_min, lognorm, genpareto
from scipy import stats
from matplotlib.ticker import LogLocator
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
        print(f"    ❌ Ошибка при вычислении {centrality_type}: {e}")
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
        print(f"    ❌ Ошибка при вычислении градиентов: {e}")
        return None

# ========================
# 3. ПОСТРОЕНИЕ ГРАФИКОВ (3 ГРАФИКА ДЛЯ КАЖДОЙ СЕТИ)
# ========================

def plot_qq_pp_histogram(gradients, network_name, centrality_type, save_path=None):
    """
    Строит три графика для PageRank.
    ИСПОЛЬЗУЕТ АВТОМАТИЧЕСКИЙ ПЕРЕБОР ПОРОГА (от 90% до 99.5%).
    Находит точку, где RMSE гибридной модели минимален.
    Для маленьких хвостов автоматически выбирает Exponential.
    """
    grad = gradients[gradients > 1e-12]
    if len(grad) < 50:
        print(f"  {network_name} ({centrality_type}): недостаточно данных")
        return None, None, None
    
    n = len(grad)
    sorted_grad = np.sort(grad)
    p = np.arange(1, n+1) / (n+1)
    skewness_val = skew(grad)
    
    # ================== АВТОМАТИЧЕСКИЙ ПОИСК ЛУЧШЕГО ПОРОГА ==================
    best_rmse = float('inf')
    best_cutoff_idx = 0
    best_tail_min = 0
    best_params = {}
    best_tail_name = "Unknown"
    
    # Перебираем пороги от 90% до 99.5% с шагом 0.5%
    percentiles = np.arange(90.0, 100.0, 0.5)
    
    for perc in percentiles:
        cutoff_idx = int(n * (perc / 100.0))
        if cutoff_idx >= n - 5:  # Хвост должен быть минимум 5 точек
            continue
        if cutoff_idx < 10:      # Тело должно быть минимум 10 точек
            continue
            
        body_data = sorted_grad[:cutoff_idx]
        tail_data = sorted_grad[cutoff_idx:]
        
        # 1. Подгоняем тело (Логнормальное)
        try:
            s_b, loc_b, scale_b = lognorm.fit(body_data)
        except:
            continue
            
        # 2. Подгоняем хвост
        tail_min = tail_data.min()
        tail_shifted = tail_data - tail_min
        
        try:
            if len(tail_data) < 50:
                # Если хвост маленький -> Экспонента
                from scipy.stats import expon
                c_t, loc_t, scale_t = expon.fit(tail_shifted)
                t_name = "Exponential"
            else:
                # Если хвост большой -> Парето
                c_t, loc_t, scale_t = genpareto.fit(tail_shifted)
                t_name = "Pareto"
        except:
            continue
            
        # 3. Строим теоретические квантили
        theo_qq = np.zeros(n)
        p_body = p[:cutoff_idx]
        theo_qq[:cutoff_idx] = lognorm.ppf(p_body, s_b, loc=loc_b, scale=scale_b)
        
        p_tail = p[cutoff_idx:]
        if len(p_tail) > 0:
            if t_name == "Exponential":
                theo_qq[cutoff_idx:] = expon.ppf(p_tail, loc=loc_t, scale=scale_t) + tail_min
            else:
                theo_qq[cutoff_idx:] = genpareto.ppf(p_tail, c_t, loc=loc_t, scale=scale_t) + tail_min
        
        # 4. Считаем RMSE
        residuals = sorted_grad - theo_qq
        current_rmse = np.sqrt(np.mean(residuals**2))
        
        # 5. Запоминаем лучший вариант
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_cutoff_idx = cutoff_idx
            best_tail_min = tail_min
            best_tail_name = t_name
            best_params = {
                's': s_b, 'loc': loc_b, 'scale': scale_b,
                'c_tail': c_t, 'loc_tail': loc_t, 'scale_tail': scale_t
            }
    # =================================================================
    
    # Если ничего не найдено (аварийный случай), берем 95%
    if best_cutoff_idx == 0:
        cutoff_idx = int(n * 0.95)
        body_data = sorted_grad[:cutoff_idx]
        tail_data = sorted_grad[cutoff_idx:]
        s_b, loc_b, scale_b = lognorm.fit(body_data)
        tail_min = tail_data.min()
        tail_shifted = tail_data - tail_min
        if len(tail_data) < 50:
            from scipy.stats import expon
            c_t, loc_t, scale_t = expon.fit(tail_shifted)
            best_tail_name = "Exponential"
        else:
            c_t, loc_t, scale_t = genpareto.fit(tail_shifted)
            best_tail_name = "Pareto"
        best_params = {'s': s_b, 'loc': loc_b, 'scale': scale_b, 'c_tail': c_t, 'loc_tail': loc_t, 'scale_tail': scale_t}
        best_tail_min = tail_min
        best_rmse = 999.9
        
    # ================== СТРОИМ ГРАФИКИ ==================
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    percent = (best_cutoff_idx / n) * 100
    fig.suptitle(f'{network_name} ({centrality_type})\nГИБРИД (Оптим. порог {percent:.1f}%): Log-Normal + {best_tail_name}\n'
                 f'LogNorm s={best_params["s"]:.3f}, {best_tail_name} c={best_params["c_tail"]:.3f}',
                 fontsize=14, fontweight='bold')
    
    # Пересчитываем теоретические квантили для лучшего порога
    cutoff = best_cutoff_idx
    p_body = p[:cutoff]
    theoretical_qq = np.zeros(n)
    theoretical_qq[:cutoff] = lognorm.ppf(p_body, best_params['s'], 
                                          loc=best_params['loc'], scale=best_params['scale'])
    p_tail = p[cutoff:]
    if len(p_tail) > 0:
        if best_tail_name == "Exponential":
            theoretical_qq[cutoff:] = expon.ppf(p_tail, loc=best_params['loc_tail'], scale=best_params['scale_tail']) + best_tail_min
        else:
            theoretical_qq[cutoff:] = genpareto.ppf(p_tail, best_params['c_tail'], 
                                                    loc=best_params['loc_tail'], scale=best_params['scale_tail']) + best_tail_min
    
    # --- 1. QQ-PLOT ---
    axes[0].scatter(theoretical_qq, sorted_grad, alpha=0.5, s=10, color='blue')
    axes[0].plot([0, max(theoretical_qq)], [0, max(theoretical_qq)], 'r--', linewidth=2, label='y=x (идеал)')
    axes[0].set_xlabel('Теоретические квантили (Гибрид)')
    axes[0].set_ylabel('Эмпирические квантили')
    axes[0].set_title('QQ-plot (проверка ХВОСТОВ)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.05, 0.95, f'RMSE={best_rmse:.4f}', transform=axes[0].transAxes, fontsize=10, verticalalignment='top')
    
    # --- 2. P-P PLOT ---
    empirical_cdf = p
    theoretical_cdf = np.zeros(n)
    theoretical_cdf[:cutoff] = lognorm.cdf(sorted_grad[:cutoff], best_params['s'], 
                                           loc=best_params['loc'], scale=best_params['scale'])
    if best_tail_name == "Exponential":
        theoretical_cdf[cutoff:] = expon.cdf(sorted_grad[cutoff:] - best_tail_min, 
                                             loc=best_params['loc_tail'], scale=best_params['scale_tail'])
    else:
        theoretical_cdf[cutoff:] = genpareto.cdf(sorted_grad[cutoff:] - best_tail_min, 
                                                 best_params['c_tail'], loc=best_params['loc_tail'], scale=best_params['scale_tail'])
    
    axes[1].scatter(theoretical_cdf, empirical_cdf, alpha=0.5, s=10, color='green')
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x (идеал)')
    axes[1].set_xlabel('Теоретическая CDF (Гибрид)')
    axes[1].set_ylabel('Эмпирическая CDF')
    axes[1].set_title('P-P plot (проверка ЦЕНТРА)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    residuals_pp = empirical_cdf - theoretical_cdf
    rmse_pp = np.sqrt(np.mean(residuals_pp**2))
    axes[1].text(0.05, 0.95, f'RMSE={rmse_pp:.4f}', transform=axes[1].transAxes, fontsize=10, verticalalignment='top')
    
    # --- 3. ГИСТОГРАММА С ГИБРИДНОЙ КРИВОЙ ---
    min_val = max(grad.min(), 1e-6)
    max_val = grad.max()
    margin = 0.5
    min_val_extended = min_val / (10**margin)
    max_val_extended = max_val * (10**margin)
    
    n_bins = 60
    bins = np.logspace(np.log10(min_val_extended), np.log10(max_val_extended), n_bins)
    
    axes[2].hist(grad, bins=bins, density=True, alpha=0.5, 
                 label='Данные', color='gray', edgecolor='black', linewidth=0.5)
    
    # Строим гибридную кривую плотности
    x_vals = np.logspace(np.log10(min_val_extended), np.log10(max_val_extended), 2000)
    y_vals = np.zeros_like(x_vals)
    
    threshold = sorted_grad[cutoff]
    mask_body = x_vals <= threshold
    mask_tail = x_vals > threshold
    
    if np.any(mask_body):
        y_vals[mask_body] = lognorm.pdf(x_vals[mask_body], best_params['s'], 
                                        loc=best_params['loc'], scale=best_params['scale'])
    if np.any(mask_tail):
        if best_tail_name == "Exponential":
            y_vals[mask_tail] = expon.pdf(x_vals[mask_tail] - best_tail_min, 
                                          loc=best_params['loc_tail'], scale=best_params['scale_tail'])
        else:
            y_vals[mask_tail] = genpareto.pdf(x_vals[mask_tail] - best_tail_min, 
                                              best_params['c_tail'], loc=best_params['loc_tail'], scale=best_params['scale_tail'])
    
    if np.trapz(y_vals, x_vals) > 0:
        y_vals = y_vals / np.trapz(y_vals, x_vals)
    
    axes[2].loglog(x_vals, y_vals, 'r-', linewidth=2.5, label='Гибридная модель')
    axes[2].set_xlim(min_val_extended, max_val_extended)
    axes[2].set_xlabel('Градиент (log scale)', fontsize=12)
    axes[2].set_ylabel('Плотность (log scale)', fontsize=12)
    axes[2].set_title('Градиент-плотность (Оптимальный порог)')
    axes[2].legend(loc='best')
    axes[2].grid(True, which='major', alpha=0.3)
    
    info_text = (f'$n_{{edges}}$={len(grad):,}\n'
                 f'Порог: {percent:.1f}%\n'
                 f'Тело: $s$={best_params["s"]:.3f}\n'
                 f'Хвост: {best_tail_name} $c$={best_params["c_tail"]:.3f}')
    axes[2].text(0.05, 0.05, info_text, transform=axes[2].transAxes, 
                 fontsize=10, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_params['s'], best_params['c_tail'], skewness_val

# ========================
# 4. ГЕНЕРАЦИЯ СЛУЧАЙНЫХ СЕТЕЙ
# ========================

def generate_random_networks(n_networks=10, centrality='betweenness'):
    """
    Генерирует случайные графы разных моделей и строит для них графики.
    """
    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ И АНАЛИЗ СЛУЧАЙНЫХ СЕТЕЙ")
    print("="*60)
    
    results = []
    n_nodes = 500
    
    models = [
        ('ER', {'n': n_nodes, 'p': 0.005}),
        ('ER', {'n': n_nodes, 'p': 0.01}),
        ('ER', {'n': n_nodes, 'p': 0.02}),
        ('ER', {'n': n_nodes, 'p': 0.05}),
        ('ER', {'n': n_nodes, 'p': 0.1}),
        ('ER', {'n': n_nodes, 'p': 0.5}),
        ('BA', {'n': n_nodes, 'm': 2}),
        ('BA', {'n': n_nodes, 'm': 3}),
        ('BA', {'n': n_nodes, 'm': 4}),
        ('BA', {'n': n_nodes, 'm': 7}),
        ('BA', {'n': n_nodes, 'm': 10}),
        ('BA', {'n': n_nodes, 'm': 15}),
        ('WS', {'n': n_nodes, 'k': 10, 'p': 0.05}),
        ('WS', {'n': n_nodes, 'k': 10, 'p': 0.1}),
        ('WS', {'n': n_nodes, 'k': 10, 'p': 0.5}),
        ('WS', {'n': n_nodes, 'k': 10, 'p': 0.9}),
        ('WS', {'n': n_nodes, 'k': 5, 'p': 0.05}),
        ('WS', {'n': n_nodes, 'k': 5, 'p': 0.1}),
        ('WS', {'n': n_nodes, 'k': 5, 'p': 0.5}),
        ('WS', {'n': n_nodes, 'k': 5, 'p': 0.9}),
    ]
    
    models = models[:n_networks]
    
    for i, (model_name, params) in enumerate(models):
        try:
            print(f"\n  Генерация {model_name} {i+1}/{len(models)}...")
            
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
            
            print(f"    Вершин: {G.number_of_nodes()}, Рёбер: {G.number_of_edges()}")
            
            gradients = compute_edge_gradients(G, centrality)
            if gradients is None:
                print(f"    ❌ Не удалось вычислить градиенты")
                continue
            
            gradients = gradients[gradients > 1e-12]
            if len(gradients) < 10:
                print(f"    ❌ Недостаточно данных после фильтрации")
                continue
            
            # Строим графики
            name = f"{model_name}_{i+1}"
            c, scale, skewness_val = plot_qq_pp_histogram(
                gradients, name, centrality,
                save_path=f'random_{centrality}_{name}.png'
            )
            
            if c is not None:
                results.append({
                    'filename': name,
                    'centrality': centrality,
                    'type': 'Random',
                    'model': model_name,
                    'n_nodes': G.number_of_nodes(),
                    'n_edges': G.number_of_edges(),
                    'weibull_c': c,
                    'weibull_scale': scale,
                    'skewness': skewness_val
                })
                
        except Exception as e:
            print(f"    ❌ Ошибка: {e}")
    
    df = pd.DataFrame(results)
    return df

# ========================
# 5. АНАЛИЗ РЕАЛЬНЫХ СЕТЕЙ
# ========================

def analyze_real_networks(folder_path, centrality='betweenness', max_networks=None):
    """
    Анализирует реальные сети и строит для каждой 3 графика.
    """
    print("\n" + "="*60)
    print("АНАЛИЗ РЕАЛЬНЫХ СЕТЕЙ")
    print("="*60)
    
    extensions = ['.gml', '.txt', '.edges', '.mtx']
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
        print(f"\n{'='*60}")
        print(f"Обработка: {filename}")
        print('='*60)
        
        try:
            G = load_network(filepath)
        except Exception as e:
            print(f"  Ошибка загрузки: {e}")
            continue
        
        if G is None:
            continue
        
        valid, msg = validate_network(G)
        if not valid:
            print(f"  Пропуск: {msg}")
            continue
        
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc)
            print(f"  Взята гигантская компонента: {len(largest_cc)} вершин")
        
        gradients = compute_edge_gradients(G, centrality)
        
        if gradients is None or len(gradients) < 10:
            print(f"  Недостаточно данных")
            continue
        
        gradients = gradients[gradients > 1e-12]
        if len(gradients) < 10:
            print(f"  Недостаточно данных после фильтрации")
            continue
        
        print(f"  Рёбер: {len(gradients)}")
        
        # Строим 3 графика
        c, scale, skewness_val = plot_qq_pp_histogram(
            gradients, filename, centrality,
            save_path=f'real_{centrality}_{filename.replace(".", "_")}.png'
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
# 6. СРАВНИТЕЛЬНЫЕ ГРАФИКИ
# ========================

def plot_comparison(real_df, random_df, centrality, save_path=None):
    """
    Строит сравнительные графики для реальных и случайных сетей.
    """
    if len(real_df) == 0 or len(random_df) == 0:
        print("Недостаточно данных для сравнения")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Сравнение реальных и случайных сетей\nЦентральность: {centrality}', 
                 fontsize=14, fontweight='bold')
    
    # 1. Weibull c
    axes[0].scatter([0]*len(real_df), real_df['weibull_c'], 
                   color='blue', alpha=0.7, label='Real', s=50)
    axes[0].scatter([1]*len(random_df), random_df['weibull_c'], 
                   color='red', alpha=0.7, label='Random', s=50)
    axes[0].axhline(y=1, color='green', linestyle='--', linewidth=2, label='c=1')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Real', 'Random'])
    axes[0].set_ylabel('Weibull c')
    axes[0].set_title('Параметр формы Weibull')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Skewness
    axes[1].scatter([0]*len(real_df), real_df['skewness'], 
                   color='blue', alpha=0.7, label='Real', s=50)
    axes[1].scatter([1]*len(random_df), random_df['skewness'], 
                   color='red', alpha=0.7, label='Random', s=50)
    axes[1].axhline(y=2, color='green', linestyle='--', linewidth=2, label='skewness=2')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Real', 'Random'])
    axes[1].set_ylabel('Skewness')
    axes[1].set_title('Коэффициент асимметрии')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. c vs skewness
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
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_comparison_summary(real_df, random_df):
    """Выводит сравнительную сводку."""
    print("\n" + "="*70)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("="*70)
    
    print("\n" + "-"*70)
    print("РЕАЛЬНЫЕ СЕТИ:")
    print("-"*70)
    print(f"  Количество: {len(real_df)}")
    print(f"  Weibull c: mean={real_df['weibull_c'].mean():.3f} ± {real_df['weibull_c'].std():.3f}")
    print(f"  Доля c < 1: {(real_df['weibull_c'] < 1).mean()*100:.1f}%")
    print(f"  Skewness: mean={real_df['skewness'].mean():.3f} ± {real_df['skewness'].std():.3f}")
    print(f"  Доля skewness > 2: {(real_df['skewness'] > 2).mean()*100:.1f}%")
    
    print("\n" + "-"*70)
    print("СЛУЧАЙНЫЕ СЕТИ:")
    print("-"*70)
    print(f"  Количество: {len(random_df)}")
    print(f"  Weibull c: mean={random_df['weibull_c'].mean():.3f} ± {random_df['weibull_c'].std():.3f}")
    print(f"  Доля c < 1: {(random_df['weibull_c'] < 1).mean()*100:.1f}%")
    print(f"  Skewness: mean={random_df['skewness'].mean():.3f} ± {random_df['skewness'].std():.3f}")
    print(f"  Доля skewness > 2: {(random_df['skewness'] > 2).mean()*100:.1f}%")
    
    from scipy.stats import mannwhitneyu
    
    print("\n" + "-"*70)
    print("СТАТИСТИЧЕСКИЕ ТЕСТЫ:")
    print("-"*70)
    
    stat, p = mannwhitneyu(real_df['weibull_c'], random_df['weibull_c'])
    print(f"  Weibull c (Mann-Whitney): p={p:.4f} {'✅ Различаются' if p < 0.05 else '❌ Не различаются'}")
    
    stat, p = mannwhitneyu(real_df['skewness'], random_df['skewness'])
    print(f"  Skewness (Mann-Whitney): p={p:.4f} {'✅ Различаются' if p < 0.05 else '❌ Не различаются'}")
    
    print("\n" + "="*70)

# ========================
# 7. ОСНОВНАЯ ФУНКЦИЯ
# ========================

def run_full_analysis(folder_path, centrality='betweenness', n_random=10, max_networks=None):
    """
    Запускает полный анализ: реальные сети + случайные.
    """
    # Реальные сети
    real_df = analyze_real_networks(folder_path, centrality, max_networks)
    
    # Случайные сети
    random_df = generate_random_networks(n_random, centrality)
    
    # Сравнение
    if len(real_df) > 0 and len(random_df) > 0:
        plot_comparison(real_df, random_df, centrality, 'comparison.png')
        print_comparison_summary(real_df, random_df)
    
    # Сохраняем результаты
    df_combined = pd.concat([real_df, random_df], ignore_index=True)
    df_combined.to_csv('full_analysis_results.csv', index=False)
    print(f"\nРезультаты сохранены в full_analysis_results.csv")
    
    return df_combined, real_df, random_df

# ========================
# 8. ЗАПУСК
# ========================

if __name__ == "__main__":
    NETWORKS_FOLDER = "C:/Proj_edge_distr/cite_A/"
    
    df, real, random = run_full_analysis(
        folder_path=NETWORKS_FOLDER,
        centrality='pagerank',
        n_random=20,
        max_networks=None
    )