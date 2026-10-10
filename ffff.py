import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import (skew, kurtosis, lognorm, weibull_min, gamma, kstest, 
                         norm, mannwhitneyu, ks_2samp, pareto, expon, burr, 
                         invweibull, genpareto, fisk)
from scipy import stats
import powerlaw
from collections import defaultdict
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
    
    if centrality_type == 'pagerank':
        centrality_dict = nx.pagerank(G)
    elif centrality_type == 'betweenness':
        if G.number_of_nodes() > 5000:
            centrality_dict = nx.betweenness_centrality(G, k=int(np.sqrt(G.number_of_nodes())))
        else:
            centrality_dict = nx.betweenness_centrality(G)
    elif centrality_type == 'closeness':
        centrality_dict = nx.closeness_centrality(G)
    elif centrality_type == 'eigenvector':
        try:
            centrality_dict = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            centrality_dict = {node: 0 for node in G.nodes()}
    elif centrality_type == 'kcore':
        centrality_dict = nx.core_number(G)
    elif centrality_type == 'degree':
        centrality_dict = dict(G.degree())
    else:
        raise ValueError("Неизвестный тип центральности")

    G_dir = G.to_directed()
    edges_to_remove = [(u, v) for u, v in G_dir.edges() if u > v]
    G_dir.remove_edges_from(edges_to_remove)
    
    B = nx.incidence_matrix(G_dir, oriented=True).toarray()
    f = np.array([centrality_dict[node] for node in G_dir.nodes()])
    gradients = B.T.dot(f)
    abs_gradients = np.abs(gradients)
    
    mean_val = np.mean(abs_gradients)
    if mean_val > 0:
        return abs_gradients / mean_val
    return abs_gradients

# ========================
# 3. ПРОВЕРКА ВСЕХ РАСПРЕДЕЛЕНИЙ
# ========================

def test_all_distributions(gradients):
    """Проверяет все heavy-tailed распределения и возвращает лучшее."""
    grad = gradients[gradients > 1e-12]
    if len(grad) < 10:
        return None, None
    
    if len(grad) > 10000:
        grad = np.random.choice(grad, 10000, replace=False)
    
    distributions = {
        'Weibull': {
            'dist': weibull_min,
            'params': ['c', 'scale'],
            'description': 'Распределение Вейбулла'
        },
        'Lognormal': {
            'dist': lognorm,
            'params': ['s', 'scale'],
            'description': 'Логнормальное распределение'
        },
        'Gamma': {
            'dist': gamma,
            'params': ['a', 'scale'],
            'description': 'Гамма-распределение'
        },
        'Pareto': {
            'dist': pareto,
            'params': ['b', 'scale'],
            'description': 'Распределение Парето'
        },
        'Exponential': {
            'dist': expon,
            'params': ['scale'],
            'description': 'Экспоненциальное распределение'
        },
        'Burr': {
            'dist': burr,
            'params': ['c', 'd', 'scale'],
            'description': 'Распределение Барра'
        },
        'Frechet': {
            'dist': invweibull,
            'params': ['c', 'scale'],
            'description': 'Распределение Фреше'
        },
        'GPD': {
            'dist': genpareto,
            'params': ['c', 'scale'],
            'description': 'Обобщенное распределение Парето'
        },
        'Fisk': {
            'dist': fisk,
            'params': ['c', 'scale'],
            'description': 'Распределение Фиска'
        }
    }
    
    results = {}
    
    for name, dist_info in distributions.items():
        try:
            params = dist_info['dist'].fit(grad, floc=0)
            ks_stat, ks_p = kstest(grad, dist_info['dist'].name, args=params)
            log_likelihood = np.sum(dist_info['dist'].logpdf(grad, *params))
            k = len(params)
            aic = -2 * log_likelihood + 2 * k
            
            results[name] = {
                'params': params,
                'params_names': dist_info['params'],
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_p,
                'aic': aic,
                'fitted': True,
                'description': dist_info['description']
            }
        except Exception as e:
            results[name] = {
                'fitted': False,
                'error': str(e),
                'description': dist_info['description']
            }
    
    best_ks = max([(name, r['ks_pvalue']) for name, r in results.items() 
                   if r.get('fitted', False)],
                  key=lambda x: x[1], default=(None, 0))
    
    best_aic = min([(name, r['aic']) for name, r in results.items() 
                    if r.get('fitted', False)],
                   key=lambda x: x[1], default=(None, float('inf')))
    
    return results, {'best_ks': best_ks, 'best_aic': best_aic}

# ========================
# 4. АНАЛИЗ СЕТИ
# ========================

def analyze_network_all_distributions(gradients, network_name=None):
    """Анализирует сеть со всеми распределениями."""
    grad = gradients[gradients > 1e-12]
    if len(grad) < 10:
        return None
    
    results = {
        'network': network_name,
        'n_edges': len(grad),
        'skewness': skew(grad),
        'kurtosis': kurtosis(grad),
        'min': np.min(grad),
        'max': np.max(grad),
        'mean': np.mean(grad),
        'std': np.std(grad),
        'median': np.median(grad)
    }
    
    dist_results, best = test_all_distributions(grad)
    results['distributions'] = dist_results
    results['best_ks'] = best['best_ks']
    results['best_aic'] = best['best_aic']
    
    if best['best_ks'][0] is not None:
        best_name = best['best_ks'][0]
        best_params = dist_results[best_name]['params']
        best_params_names = dist_results[best_name]['params_names']
        
        results['best_distribution_ks'] = best_name
        results['best_ks_pvalue'] = best['best_ks'][1]
        
        for i, param_name in enumerate(best_params_names):
            results[f'{best_name}_{param_name}'] = best_params[i]
    
    if best['best_aic'][0] is not None:
        results['best_distribution_aic'] = best['best_aic'][0]
        results['best_aic_value'] = best['best_aic'][1]
    
    return results

# ========================
# 5. АНАЛИЗ ВСЕХ СЕТЕЙ
# ========================

def analyze_all_networks_all_distributions(folder_path, centrality_types=None):
    """Анализирует все сетевые файлы с проверкой всех распределений."""
    if centrality_types is None:
        centrality_types = ['betweenness']
    
    extensions = ['.gml', '.txt', '.edges', '.mtx']
    all_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                all_files.append(os.path.join(root, f))
    
    print(f"Найдено файлов: {len(all_files)}")
    
    all_results = []
    all_gradients = {}
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        print(f"\nОбработка: {filename}")
        
        try:
            G = load_network(filepath)
        except Exception as e:
            print(f"  Ошибка загрузки {filename}: {e}")
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
        
        for cent in centrality_types:
            print(f"  Центральность: {cent}")
            gradients = compute_edge_gradients(G, cent)
            
            if gradients is None or len(gradients) < 10:
                print(f"    Недостаточно данных")
                continue
            
            gradients = gradients[gradients > 1e-12]
            if len(gradients) < 10:
                print(f"    Недостаточно данных после фильтрации")
                continue
            
            print(f"    Рёбер: {len(gradients)}, Skewness: {skew(gradients):.3f}")
            
            dist_results = analyze_network_all_distributions(gradients, f"{filename}_{cent}")
            
            if dist_results:
                dist_results.update({
                    'filename': filename,
                    'centrality': cent,
                    'n_nodes': G.number_of_nodes(),
                    'n_edges': G.number_of_edges(),
                    'density': nx.density(G),
                    'clustering': nx.average_clustering(G)
                })
                
                all_results.append(dist_results)
                
                key = f"{filename}_{cent}"
                all_gradients[key] = gradients
    
    df = pd.DataFrame(all_results)
    return df, all_gradients

# ========================
# 6. ГЕНЕРАЦИЯ СЛУЧАЙНЫХ СЕТЕЙ
# ========================

def generate_random_networks_all_distributions(n_nodes_list=[100, 500], n_samples=3):
    """Генерирует случайные графы разных моделей."""
    results = []
    gradients_dict = {}
    centrality = 'betweenness'
    
    for N in n_nodes_list:
        # ER
        for p in [0.001, 0.005, 0.01, 0.05]:
            for i in range(n_samples):
                try:
                    G = nx.erdos_renyi_graph(N, p)
                    if G.number_of_edges() > 10:
                        grad = compute_edge_gradients(G, centrality)
                        if grad is not None and len(grad) > 10:
                            grad = grad[grad > 1e-12]
                            if len(grad) > 10:
                                dist_results = analyze_network_all_distributions(grad, f"ER_p={p}_N={N}_{i}")
                                if dist_results:
                                    dist_results.update({
                                        'filename': f"ER_p={p}_N={N}_{i}",
                                        'centrality': centrality,
                                        'type': 'Random',
                                        'model': 'ER',
                                        'n_nodes': G.number_of_nodes(),
                                        'n_edges': G.number_of_edges(),
                                        'density': nx.density(G),
                                        'clustering': nx.average_clustering(G)
                                    })
                                    results.append(dist_results)
                                    key = f"ER_p={p}_N={N}_{i}_{centrality}"
                                    gradients_dict[key] = grad
                except Exception as e:
                    pass
        
        # BA
        for m in [1, 2, 3, 5]:
            for i in range(n_samples):
                try:
                    if m < N:
                        G = nx.barabasi_albert_graph(N, m)
                        if G.number_of_edges() > 10:
                            grad = compute_edge_gradients(G, centrality)
                            if grad is not None and len(grad) > 10:
                                grad = grad[grad > 1e-12]
                                if len(grad) > 10:
                                    dist_results = analyze_network_all_distributions(grad, f"BA_m={m}_N={N}_{i}")
                                    if dist_results:
                                        dist_results.update({
                                            'filename': f"BA_m={m}_N={N}_{i}",
                                            'centrality': centrality,
                                            'type': 'Random',
                                            'model': 'BA',
                                            'n_nodes': G.number_of_nodes(),
                                            'n_edges': G.number_of_edges(),
                                            'density': nx.density(G),
                                            'clustering': nx.average_clustering(G)
                                        })
                                        results.append(dist_results)
                                        key = f"BA_m={m}_N={N}_{i}_{centrality}"
                                        gradients_dict[key] = grad
                except:
                    pass
        
        # WS
        for p in [0.0, 0.05, 0.1, 0.2, 0.5, 0.8]:
            for i in range(n_samples):
                try:
                    G = nx.watts_strogatz_graph(N, min(10, N-1), p)
                    if G.number_of_edges() > 10:
                        grad = compute_edge_gradients(G, centrality)
                        if grad is not None and len(grad) > 10:
                            grad = grad[grad > 1e-12]
                            if len(grad) > 10:
                                dist_results = analyze_network_all_distributions(grad, f"WS_p={p}_N={N}_{i}")
                                if dist_results:
                                    dist_results.update({
                                        'filename': f"WS_p={p}_N={N}_{i}",
                                        'centrality': centrality,
                                        'type': 'Random',
                                        'model': 'WS',
                                        'n_nodes': G.number_of_nodes(),
                                        'n_edges': G.number_of_edges(),
                                        'density': nx.density(G),
                                        'clustering': nx.average_clustering(G)
                                    })
                                    results.append(dist_results)
                                    key = f"WS_p={p}_N={N}_{i}_{centrality}"
                                    gradients_dict[key] = grad
                except:
                    pass
    
    df = pd.DataFrame(results)
    return df, gradients_dict

# ========================
# 7. ФУНКЦИИ ВИЗУАЛИЗАЦИИ (ИСПРАВЛЕННЫЕ)
# ========================

def plot_all_distributions_comparison(df, save_path=None):
    """Визуализирует сравнение всех распределений."""
    if len(df) == 0:
        print("Нет данных для визуализации")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Лучшее распределение по KS
    if 'best_distribution_ks' in df.columns:
        best_ks = df['best_distribution_ks'].value_counts()
        axes[0,0].bar(best_ks.index, best_ks.values, color='skyblue', edgecolor='black')
        axes[0,0].set_xlabel('Распределение')
        axes[0,0].set_ylabel('Количество сетей')
        axes[0,0].set_title('Лучшее распределение по KS тесту')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
    
    # 2. Лучшее распределение по AIC
    if 'best_distribution_aic' in df.columns:
        best_aic = df['best_distribution_aic'].value_counts()
        axes[0,1].bar(best_aic.index, best_aic.values, color='lightgreen', edgecolor='black')
        axes[0,1].set_xlabel('Распределение')
        axes[0,1].set_ylabel('Количество сетей')
        axes[0,1].set_title('Лучшее распределение по AIC')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
    
    # 3. Сравнение KS p-value для разных распределений
    distributions = ['Weibull', 'Lognormal', 'Gamma', 'Pareto', 'Burr', 'Frechet', 'GPD', 'Fisk']
    available_dists = []
    p_values = {}
    
    for dist in distributions:
        p_values[dist] = []
        for _, row in df.iterrows():
            if 'distributions' in row and dist in row['distributions']:
                if row['distributions'][dist].get('fitted', False):
                    p_values[dist].append(row['distributions'][dist]['ks_pvalue'])
                    if dist not in available_dists:
                        available_dists.append(dist)
    
    if len(available_dists) > 0:
        data_to_plot = [p_values[d] for d in available_dists if len(p_values[d]) > 0]
        if data_to_plot:
            axes[1,0].boxplot(data_to_plot, labels=[d for d in available_dists if len(p_values[d]) > 0], 
                             patch_artist=True)
            axes[1,0].axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
            axes[1,0].set_ylabel('KS p-value')
            axes[1,0].set_title('Сравнение KS p-value для разных распределений')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
    
    # 4. Распределение лучших по типам сетей
    if 'type' in df.columns and 'best_distribution_ks' in df.columns:
        real = df[df['type'] != 'Random']
        random = df[df['type'] == 'Random']
        
        real_best = real['best_distribution_ks'].value_counts()
        random_best = random['best_distribution_ks'].value_counts()
        
        all_dists = list(set(real_best.index) | set(random_best.index))
        all_dists.sort()
        
        plot_data = pd.DataFrame({
            'Real': [real_best.get(d, 0) for d in all_dists],
            'Random': [random_best.get(d, 0) for d in all_dists]
        }, index=all_dists)
        
        plot_data.plot(kind='bar', ax=axes[1,1], color=['blue', 'red'], alpha=0.7)
        axes[1,1].set_xlabel('Распределение')
        axes[1,1].set_ylabel('Количество сетей')
        axes[1,1].set_title('Лучшее распределение: Real vs Random')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_distribution_fit_comparison(grad, dist_results, title, save_path=None):
    """
    Сравнивает все подогнанные распределения для одной сети.
    Исправленная версия с логарифмическим масштабом.
    """
    grad = grad[grad > 1e-12]
    if len(grad) < 50:
        return
    
    # Получаем все подогнанные распределения
    fitted_dists = [(name, info) for name, info in dist_results.items() 
                    if info.get('fitted', False)]
    
    if len(fitted_dists) == 0:
        print("Нет подогнанных распределений")
        return
    
    # Сортируем по KS p-value
    fitted_dists.sort(key=lambda x: x[1]['ks_pvalue'], reverse=True)
    
    # Берем топ-5 для визуализации
    top_dists = fitted_dists[:5]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- 1. Гистограмма с подгонками (лог-лог масштаб) ---
    # Используем логарифмические бины
    min_val = max(grad.min(), 1e-6)
    max_val = grad.max()
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
    
    # Гистограмма в логарифмическом масштабе
    counts, bin_edges, _ = axes[0].hist(grad, bins=bins, density=True, 
                                         alpha=0.5, label='Data', 
                                         color='gray', edgecolor='black')
    
    # Центры бинов для построения кривых
    x_vals = np.logspace(np.log10(min_val), np.log10(max_val), 1000)
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    # Словарь для поиска распределений
    dist_map = {
        'Lognormal': lognorm,
        'Weibull': weibull_min,
        'Gamma': gamma,
        'Pareto': pareto,
        'Exponential': expon,
        'Burr': burr,
        'Frechet': invweibull,
        'GPD': genpareto,
        'Fisk': fisk
    }
    
    # Рисуем подогнанные распределения
    for i, (name, info) in enumerate(top_dists):
        try:
            dist = dist_map.get(name)
            if dist is None:
                continue
            
            # Вычисляем PDF
            pdf = dist.pdf(x_vals, *info['params'])
            
            # Нормализуем для сравнения с гистограммой
            pdf_norm = pdf / np.trapz(pdf, x_vals)
            
            axes[0].loglog(x_vals, pdf_norm, 
                          color=colors[i % len(colors)], 
                          linewidth=2.5,
                          label=f"{name} (KS p={info['ks_pvalue']:.4f})")
        except Exception as e:
            pass
    
    axes[0].set_xlabel('Градиент (log scale)')
    axes[0].set_ylabel('Плотность (log scale)')
    axes[0].set_title(f'Сравнение распределений (лог-лог): {title[:30]}...')
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(True, alpha=0.3, which='both')
    
    # --- 2. KS p-values для всех распределений ---
    names = []
    p_values = []
    colors_bar = []
    
    for name, info in fitted_dists:
        names.append(name)
        p_values.append(info['ks_pvalue'])
        colors_bar.append('green' if info['ks_pvalue'] > 0.05 else 
                         'orange' if info['ks_pvalue'] > 0.01 else 'red')
    
    # Сортируем по p-value для лучшей читаемости
    sorted_indices = np.argsort(p_values)[::-1]
    names_sorted = [names[i] for i in sorted_indices]
    p_values_sorted = [p_values[i] for i in sorted_indices]
    colors_bar_sorted = [colors_bar[i] for i in sorted_indices]
    
    bars = axes[1].barh(names_sorted, p_values_sorted, color=colors_bar_sorted)
    axes[1].axvline(x=0.05, color='green', linestyle='--', linewidth=2, label='p=0.05 (хорошо)')
    axes[1].axvline(x=0.01, color='red', linestyle='--', linewidth=2, label='p=0.01 (плохо)')
    axes[1].set_xlabel('KS p-value')
    axes[1].set_title('Сравнение KS p-values')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # Добавляем значения на график
    for i, (name, p) in enumerate(zip(names_sorted, p_values_sorted)):
        axes[1].text(p + 0.01, i, f'{p:.4f}', va='center', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_all_networks_distributions(df, gradients_dict, n_networks=6, save_path=None):
    """
    Строит графики распределений для нескольких сетей на одной фигуре.
    """
    # Фильтруем реальные сети
    if 'type' in df.columns:
        real_df = df[df['type'] != 'Random']
    else:
        real_df = df
    
    # Берем n_networks сетей с наибольшей скошенностью
    real_df = real_df.sort_values('skewness', ascending=False).head(n_networks)
    
    # Вычисляем количество строк и столбцов
    n_cols = min(3, n_networks)
    n_rows = (n_networks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_networks == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Словарь для поиска распределений
    dist_map = {
        'Lognormal': lognorm,
        'Weibull': weibull_min,
        'Gamma': gamma,
        'Pareto': pareto,
        'Exponential': expon,
        'Burr': burr,
        'Frechet': invweibull,
        'GPD': genpareto,
        'Fisk': fisk
    }
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    
    for idx, (_, row) in enumerate(real_df.iterrows()):
        if idx >= n_networks:
            break
        
        filename = row['filename']
        cent = row['centrality']
        key = f"{filename}_{cent}"
        
        if key not in gradients_dict:
            axes[idx].text(0.5, 0.5, f'No data for {filename[:15]}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_visible(True)
            continue
        
        grad = gradients_dict[key]
        grad = grad[grad > 1e-12]
        
        if len(grad) < 50:
            axes[idx].text(0.5, 0.5, f'Insufficient data\n{filename[:15]}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_visible(True)
            continue
        
        # Подгоняем лучшие распределения
        dist_results = row['distributions']
        best_ks = row.get('best_distribution_ks', 'Unknown')
        best_p = row.get('best_ks_pvalue', 0)
        
        # Лог-лог гистограмма
        min_val = max(grad.min(), 1e-6)
        max_val = grad.max()
        bins = np.logspace(np.log10(min_val), np.log10(max_val), 30)
        
        axes[idx].hist(grad, bins=bins, density=True, alpha=0.4, 
                      color='gray', edgecolor='black')
        
        # Подогнанные распределения (топ-3)
        fitted_dists = [(name, info) for name, info in dist_results.items() 
                       if info.get('fitted', False)]
        fitted_dists.sort(key=lambda x: x[1]['ks_pvalue'], reverse=True)
        
        x_vals = np.logspace(np.log10(min_val), np.log10(max_val), 200)
        
        for i, (name, info) in enumerate(fitted_dists[:3]):
            try:
                dist = dist_map.get(name)
                if dist is None:
                    continue
                pdf = dist.pdf(x_vals, *info['params'])
                pdf_norm = pdf / np.trapz(pdf, x_vals)
                axes[idx].loglog(x_vals, pdf_norm, 
                               color=colors[i % len(colors)], 
                               linewidth=2,
                               label=f"{name} (p={info['ks_pvalue']:.3f})")
            except:
                pass
        
        # Определяем цвет заголовка
        color = 'green' if best_p > 0.05 else 'orange' if best_p > 0.01 else 'red'
        axes[idx].set_xlabel('Градиент')
        axes[idx].set_ylabel('Плотность')
        axes[idx].set_title(f'{filename[:20]}\nBest: {best_ks} (p={best_p:.4f})\nskew={row["skewness"]:.2f}', 
                           color=color, fontsize=9)
        axes[idx].legend(loc='best', fontsize=7)
        axes[idx].grid(True, alpha=0.3, which='both')
    
    # Скрываем пустые подграфики
    for idx in range(len(real_df), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ========================
# 8. ВЫВОД СТАТИСТИКИ
# ========================

def print_all_distributions_summary(df):
    """Выводит сводку по всем распределениям."""
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ПРОВЕРКИ ВСЕХ РАСПРЕДЕЛЕНИЙ")
    print("="*80)
    
    if len(df) == 0:
        print("Нет данных!")
        return
    
    print(f"\nВсего сетей: {len(df)}")
    print(f"Средняя скошенность: {df['skewness'].mean():.3f} ± {df['skewness'].std():.3f}")
    
    if 'best_distribution_ks' in df.columns:
        print("\n" + "-"*80)
        print("ЛУЧШИЕ РАСПРЕДЕЛЕНИЯ ПО KS ТЕСТУ:")
        best_ks = df['best_distribution_ks'].value_counts()
        for dist, count in best_ks.items():
            print(f"  {dist}: {count} ({count/len(df)*100:.1f}%)")
    
    if 'best_distribution_aic' in df.columns:
        print("\n" + "-"*80)
        print("ЛУЧШИЕ РАСПРЕДЕЛЕНИЯ ПО AIC:")
        best_aic = df['best_distribution_aic'].value_counts()
        for dist, count in best_aic.items():
            print(f"  {dist}: {count} ({count/len(df)*100:.1f}%)")
    
    if 'type' in df.columns:
        real = df[df['type'] != 'Random']
        random = df[df['type'] == 'Random']
        
        print("\n" + "-"*80)
        print("РЕАЛЬНЫЕ СЕТИ:")
        print(f"  Количество: {len(real)}")
        print(f"  Средняя скошенность: {real['skewness'].mean():.3f} ± {real['skewness'].std():.3f}")
        print(f"  Доля с skewness > 2: {(real['skewness'] > 2).mean()*100:.1f}%")
        if 'best_distribution_ks' in real.columns:
            best_real = real['best_distribution_ks'].value_counts()
            print(f"  Лучшее распределение (KS): {best_real.index[0] if len(best_real) > 0 else 'N/A'}")
        
        print("\nСЛУЧАЙНЫЕ СЕТИ:")
        print(f"  Количество: {len(random)}")
        print(f"  Средняя скошенность: {random['skewness'].mean():.3f} ± {random['skewness'].std():.3f}")
        print(f"  Доля с skewness > 2: {(random['skewness'] > 2).mean()*100:.1f}%")
        if 'best_distribution_ks' in random.columns:
            best_random = random['best_distribution_ks'].value_counts()
            print(f"  Лучшее распределение (KS): {best_random.index[0] if len(best_random) > 0 else 'N/A'}")
    
    print("\n" + "="*80)

def print_distribution_test_results(df, network_name=None):
    """Выводит детальные результаты для конкретной сети."""
    if network_name is None:
        network_name = df.iloc[0]['filename']
    
    row = df[df['filename'] == network_name]
    if len(row) == 0:
        print(f"Сеть {network_name} не найдена")
        return
    
    row = row.iloc[0]
    
    print("\n" + "="*80)
    print(f"ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ: {network_name}")
    print("="*80)
    
    print(f"\nОсновная статистика:")
    print(f"  Количество ребер: {row['n_edges']}")
    print(f"  Skewness: {row['skewness']:.3f}")
    print(f"  Kurtosis: {row['kurtosis']:.3f}")
    print(f"  Mean: {row['mean']:.3f}")
    print(f"  Std: {row['std']:.3f}")
    
    print(f"\nЛучшее распределение (KS): {row.get('best_distribution_ks', 'N/A')}")
    print(f"  KS p-value: {row.get('best_ks_pvalue', 'N/A'):.4f}")
    print(f"Лучшее распределение (AIC): {row.get('best_distribution_aic', 'N/A')}")
    print(f"  AIC: {row.get('best_aic_value', 'N/A'):.2f}")
    
    print("\nРезультаты для всех распределений:")
    for name, info in row['distributions'].items():
        if info.get('fitted', False):
            print(f"\n  {name}:")
            print(f"    KS p-value: {info['ks_pvalue']:.4f}")
            print(f"    AIC: {info['aic']:.2f}")
            params_str = ', '.join([f"{p}={v:.3f}" for p, v in zip(info['params_names'], info['params'])])
            print(f"    Parameters: {params_str}")
        else:
            print(f"\n  {name}: не подогнан")

# ========================
# 9. ОСНОВНАЯ ФУНКЦИЯ
# ========================

def run_full_analysis_all_distributions(real_networks_folder, centrality_types=['betweenness'], n_random_samples=3):
    """Запускает полный анализ с проверкой всех распределений."""
    print("="*80)
    print("ШАГ 1: Анализ реальных сетей (все распределения)")
    print("="*80)
    
    df_real, gradients_real = analyze_all_networks_all_distributions(
        real_networks_folder, 
        centrality_types=centrality_types
    )
    
    if len(df_real) == 0:
        print("Не найдено реальных сетей для анализа!")
        return None, None
    
    print(f"\nПроанализировано {len(df_real)} сетей")
    df_real.to_csv('results_real_all_distributions.csv', index=False)
    
    print("\n" + "="*80)
    print("ШАГ 2: Генерация случайных сетей")
    print("="*80)
    
    df_random, gradients_random = generate_random_networks_all_distributions(
        n_nodes_list=[100, 500], 
        n_samples=n_random_samples
    )
    df_random.to_csv('results_random_all_distributions.csv', index=False)
    
    print(f"\nСгенерировано {len(df_random)} случайных сетей")
    
    df_combined = pd.concat([df_real, df_random], ignore_index=True)
    all_gradients = {**gradients_real, **gradients_random}
    
    print("\n" + "="*80)
    print("ШАГ 3: Визуализация")
    print("="*80)
    
    # 1. Сравнение всех распределений
    plot_all_distributions_comparison(df_combined, 'all_distributions_comparison.png')
    
    # 2. Детальные графики для нескольких сетей
    sample_networks = list(gradients_real.keys())[:3]
    for key in sample_networks:
        if key in gradients_real:
            filename = key.replace('_betweenness', '')
            dist_row = df_real[df_real['filename'] == filename]
            if len(dist_row) > 0:
                grad = gradients_real[key]
                dist_results = dist_row.iloc[0]['distributions']
                plot_distribution_fit_comparison(
                    grad, dist_results, key, 
                    f'fit_all_dists_{key.replace("/", "_")}.png'
                )
    
    # 3. Графики для всех сетей на одной фигуре
    plot_all_networks_distributions(
        df_combined, 
        all_gradients, 
        n_networks=6,
        save_path='all_networks_distributions.png'
    )
    
    print_all_distributions_summary(df_combined)
    
    if len(df_real) > 0:
        print_distribution_test_results(df_real, df_real.iloc[0]['filename'])
    
    print("\n" + "="*80)
    print("АНАЛИЗ ЗАВЕРШЁН!")
    print("="*80)
    
    return df_combined, all_gradients

# ========================
# 10. ЗАПУСК
# ========================

if __name__ == "__main__":
    NETWORKS_FOLDER = "C:/Proj_edge_distr/cite_A/"
    
    df, gradients = run_full_analysis_all_distributions(
        real_networks_folder=NETWORKS_FOLDER,
        centrality_types=['betweenness'],
        n_random_samples=3
    )