import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import skew, kurtosis, lognorm, weibull_min, gamma, kstest, norm
from scipy import stats
import powerlaw
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ========================
# 1. ЗАГРУЗКА СЕТЕЙ (ВАШИ РАБОЧИЕ ФУНКЦИИ)
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
    """Ручной парсинг GML (ваша существующая функция)"""
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


# ========================
# 2. ПРОВЕРКА КАЧЕСТВА СЕТИ
# ========================

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
# 3. ВЫЧИСЛЕНИЕ РЁБЕРНЫХ ГРАДИЕНТОВ (ВАША ФУНКЦИЯ)
# ========================

def compute_edge_gradients(G, centrality_type):
    """Вычисляет рёберные градиенты для заданной центральности."""
    if G.number_of_nodes() < 10:
        return None
    
    # Вычисляем центральность
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
    else:
        raise ValueError("Неизвестный тип центральности")

    # Ориентируем рёбра
    G_dir = G.to_directed()
    edges_to_remove = [(u, v) for u, v in G_dir.edges() if u > v]
    G_dir.remove_edges_from(edges_to_remove)
    
    # Матрица инцидентности и градиенты
    B = nx.incidence_matrix(G_dir, oriented=True).toarray()
    f = np.array([centrality_dict[node] for node in G_dir.nodes()])
    gradients = B.T.dot(f)
    abs_gradients = np.abs(gradients)
    
    # Нормализация
    mean_val = np.mean(abs_gradients)
    if mean_val > 0:
        return abs_gradients / mean_val
    return abs_gradients


# ========================
# 4. АНАЛИЗ РАСПРЕДЕЛЕНИЙ
# ========================

def analyze_distribution(gradients, network_name=None):
    """
    Анализирует распределение градиентов.
    Проверяет: логнормальное, heavy-tailed, степенной закон.
    """
    # Фильтруем данные
    grad = gradients[gradients > 1e-12]
    if len(grad) < 10:
        return None
    
    # Ограничиваем для скорости
    if len(grad) > 10000:
        grad = np.random.choice(grad, 10000, replace=False)
    
    results = {
        'network': network_name,
        'n_edges': len(grad),
        'skewness': skew(grad),
        'kurtosis': kurtosis(grad),
        'min': np.min(grad),
        'max': np.max(grad),
        'mean': np.mean(grad),
        'std': np.std(grad),
        'median': np.median(grad),
        'distributions': {}
    }
    
    # --- 1. Логнормальное распределение ---
    try:
        log_grad = np.log(grad)
        mu, sigma = norm.fit(log_grad)
        
        # KS тест
        ks_stat, ks_p = kstest(log_grad, 'norm', args=(mu, sigma))
        
        results['distributions']['lognormal'] = {
            'mu': mu,
            'sigma': sigma,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_p,
            'fitted': True
        }
    except Exception as e:
        results['distributions']['lognormal'] = {'fitted': False, 'error': str(e)}
    
    # --- 2. Проверка на heavy-tailed ---
    try:
        sorted_grad = np.sort(grad)[::-1]
        k = min(len(sorted_grad) // 10, 500)
        
        if k > 10:
            log_ratios = np.log(sorted_grad[:k] / sorted_grad[k])
            hill_alpha = 1 / np.mean(log_ratios)
            
            results['distributions']['heavy_tail'] = {
                'hill_alpha': hill_alpha,
                'tail_fraction': k / len(grad),
                'is_heavy_tailed': hill_alpha < 4
            }
        else:
            results['distributions']['heavy_tail'] = {'fitted': False}
    except:
        results['distributions']['heavy_tail'] = {'fitted': False}
    
    # --- 3. Степенной закон (powerlaw) ---
    try:
        fit = powerlaw.Fit(grad, verbose=False, discrete=False)
        R, p = fit.distribution_compare('power_law', 'lognormal')
        
        results['distributions']['powerlaw'] = {
            'alpha': fit.power_law.alpha,
            'xmin': fit.power_law.xmin,
            'vs_lognormal_R': R,
            'vs_lognormal_p': p,
            'is_powerlaw': p > 0.05
        }
    except:
        results['distributions']['powerlaw'] = {'fitted': False}
    
    # --- 4. Weibull ---
    try:
        params = weibull_min.fit(grad, floc=0)
        ks_stat, ks_p = kstest(grad, 'weibull_min', args=params)
        
        results['distributions']['weibull'] = {
            'c': params[0],
            'scale': params[2],
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_p,
            'fitted': True
        }
    except:
        results['distributions']['weibull'] = {'fitted': False}
    
    # --- 5. Gamma ---
    try:
        params = gamma.fit(grad, floc=0)
        ks_stat, ks_p = kstest(grad, 'gamma', args=params)
        
        results['distributions']['gamma'] = {
            'a': params[0],
            'scale': params[2],
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_p,
            'fitted': True
        }
    except:
        results['distributions']['gamma'] = {'fitted': False}
    
    # --- Определяем лучшее распределение ---
    best_dist = None
    best_p = -1
    
    for dist_name, dist_info in results['distributions'].items():
        if dist_info.get('fitted', False) and 'ks_pvalue' in dist_info:
            if dist_info['ks_pvalue'] > best_p:
                best_p = dist_info['ks_pvalue']
                best_dist = dist_name
    
    results['best_distribution'] = best_dist if best_dist else 'unknown'
    results['best_ks_pvalue'] = best_p if best_p > 0 else 0
    
    return results


# ========================
# 5. АНАЛИЗ ВСЕХ СЕТЕЙ
# ========================

def analyze_all_networks(folder_path, centrality_types=None):
    """Анализирует все сетевые файлы в папке."""
    if centrality_types is None:
        centrality_types = ['betweenness']
    
    # Ищем все файлы с сетями
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
        
        # Берём гигантскую компоненту
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
            
            # Убираем нулевые значения
            gradients = gradients[gradients > 1e-12]
            if len(gradients) < 10:
                print(f"    Недостаточно данных после фильтрации")
                continue
            
            print(f"    Рёбер: {len(gradients)}, Skewness: {skew(gradients):.3f}")
            
            # Анализируем распределение
            dist_results = analyze_distribution(gradients, f"{filename}_{cent}")
            
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

def generate_random_networks(n_nodes_list=[100, 500], n_samples=3):
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
                                dist_results = analyze_distribution(grad, f"ER_p={p}_N={N}_{i}")
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
                                    dist_results = analyze_distribution(grad, f"BA_m={m}_N={N}_{i}")
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
                                dist_results = analyze_distribution(grad, f"WS_p={p}_N={N}_{i}")
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
# 7. ВИЗУАЛИЗАЦИЯ
# ========================

def plot_distribution_analysis(df, save_path=None):
    """Визуализирует результаты анализа распределений."""
    
    if len(df) == 0:
        print("Нет данных для визуализации")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Скошенность vs KS p-value для логнормального
    df_lognorm = df[df['distributions'].apply(lambda x: x.get('lognormal', {}).get('fitted', False))]
    if len(df_lognorm) > 0:
        lognorm_p = df_lognorm['distributions'].apply(lambda x: x['lognormal']['ks_pvalue'])
        axes[0,0].scatter(df_lognorm['skewness'], lognorm_p, alpha=0.7, s=30)
        axes[0,0].axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
        axes[0,0].set_xlabel('Skewness')
        axes[0,0].set_ylabel('KS p-value (LogNormal)')
        axes[0,0].set_title('Соответствие логнормальному распределению')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
    
    # 2. Hill alpha vs skewness
    df_ht = df[df['distributions'].apply(lambda x: x.get('heavy_tail', {}).get('fitted', False))]
    if len(df_ht) > 0:
        hill_alpha = df_ht['distributions'].apply(lambda x: x['heavy_tail']['hill_alpha'])
        axes[0,1].scatter(df_ht['skewness'], hill_alpha, alpha=0.7, s=30)
        axes[0,1].axhline(y=4, color='red', linestyle='--', label='Heavy-tailed threshold')
        axes[0,1].set_xlabel('Skewness')
        axes[0,1].set_ylabel('Hill alpha')
        axes[0,1].set_title('Heavy-tailed оценка')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # 3. Лучшее распределение
    best_dist = df['best_distribution'].value_counts()
    if len(best_dist) > 0:
        axes[1,0].bar(best_dist.index, best_dist.values, color='skyblue', edgecolor='black')
        axes[1,0].set_xlabel('Распределение')
        axes[1,0].set_ylabel('Количество сетей')
        axes[1,0].set_title('Лучшее распределение')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Скошенность для реальных vs случайных
    if 'type' in df.columns:
        real = df[df['type'] != 'Random']
        random = df[df['type'] == 'Random']
        
        axes[1,1].hist(real['skewness'], bins=15, alpha=0.5, label='Real', color='blue', edgecolor='black')
        axes[1,1].hist(random['skewness'], bins=15, alpha=0.5, label='Random', color='red', edgecolor='black')
        axes[1,1].axvline(x=2, color='green', linestyle='--', linewidth=2, label='Threshold')
        axes[1,1].set_xlabel('Skewness')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Скошенность: Real vs Random')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_fit_comparison(grad, dist_results, title, save_path=None):
    """Сравнивает подогнанные распределения для одной сети."""
    
    grad = grad[grad > 1e-12]
    if len(grad) < 50:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Гистограмма с подгонками
    axes[0].hist(grad, bins=50, density=True, alpha=0.5, label='Data', color='gray', edgecolor='black')
    
    x_vals = np.linspace(min(grad), max(grad), 1000)
    
    # Логнормальное
    if dist_results['distributions'].get('lognormal', {}).get('fitted', False):
        ln = dist_results['distributions']['lognormal']
        ln_pdf = (1/(x_vals * ln['sigma'] * np.sqrt(2*np.pi))) * \
                 np.exp(-(np.log(x_vals) - ln['mu'])**2 / (2*ln['sigma']**2))
        axes[0].plot(x_vals, ln_pdf, 'g-', linewidth=2, label=f"LogNormal (KS p={ln['ks_pvalue']:.3f})")
    
    # Weibull
    if dist_results['distributions'].get('weibull', {}).get('fitted', False):
        wb = dist_results['distributions']['weibull']
        wb_pdf = weibull_min.pdf(x_vals, c=wb['c'], scale=wb['scale'])
        axes[0].plot(x_vals, wb_pdf, 'r--', linewidth=2, label=f"Weibull (KS p={wb['ks_pvalue']:.3f})")
    
    # Gamma
    if dist_results['distributions'].get('gamma', {}).get('fitted', False):
        gm = dist_results['distributions']['gamma']
        gm_pdf = gamma.pdf(x_vals, a=gm['a'], scale=gm['scale'])
        axes[0].plot(x_vals, gm_pdf, 'b-.', linewidth=2, label=f"Gamma (KS p={gm['ks_pvalue']:.3f})")
    
    axes[0].set_xlabel('Градиент')
    axes[0].set_ylabel('Плотность')
    axes[0].set_title(f'Сравнение распределений: {title[:30]}...')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. QQ-plot против логнормального
    log_grad = np.log(grad)
    if dist_results['distributions'].get('lognormal', {}).get('fitted', False):
        ln = dist_results['distributions']['lognormal']
        stats.probplot(log_grad, dist="norm", plot=axes[1])
        axes[1].set_title(f'QQ-plot (LogNormal): μ={ln["mu"]:.2f}, σ={ln["sigma"]:.2f}')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_distribution_summary(df):
    """Выводит сводку по распределениям."""
    
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА РАСПРЕДЕЛЕНИЙ")
    print("="*70)
    
    if len(df) == 0:
        print("Нет данных!")
        return
    
    print(f"\nВсего сетей: {len(df)}")
    print(f"Средняя скошенность: {df['skewness'].mean():.3f} ± {df['skewness'].std():.3f}")
    print(f"Медианная скошенность: {df['skewness'].median():.3f}")
    
    # Лучшие распределения
    best_counts = df['best_distribution'].value_counts()
    print("\nЛучшие распределения:")
    for dist, count in best_counts.items():
        print(f"  {dist}: {count} ({count/len(df)*100:.1f}%)")
    
    # Проверка логнормального
    lognorm_ok = df[df['distributions'].apply(
        lambda x: x.get('lognormal', {}).get('fitted', False) and 
                  x['lognormal']['ks_pvalue'] > 0.05
    )]
    print(f"\nСети, соответствующие логнормальному (KS p > 0.05): {len(lognorm_ok)}/{len(df)}")
    
    # Heavy-tailed
    heavy_tailed = df[df['distributions'].apply(
        lambda x: x.get('heavy_tail', {}).get('fitted', False) and 
                  x['heavy_tail'].get('is_heavy_tailed', False)
    )]
    print(f"Сети с heavy-tailed распределением: {len(heavy_tailed)}/{len(df)}")
    
    # Разделение реальных/случайных
    if 'type' in df.columns:
        real = df[df['type'] != 'Random']
        random = df[df['type'] == 'Random']
        
        if len(real) > 0:
            print("\n" + "-"*70)
            print("РЕАЛЬНЫЕ СЕТИ:")
            print(f"  Количество: {len(real)}")
            print(f"  Средняя скошенность: {real['skewness'].mean():.3f} ± {real['skewness'].std():.3f}")
            print(f"  Доля с skewness > 2: {(real['skewness'] > 2).mean()*100:.1f}%")
            print(f"  Лучшее распределение: {real['best_distribution'].value_counts().index[0] if len(real['best_distribution'].value_counts()) > 0 else 'N/A'}")
        
        if len(random) > 0:
            print("\nСЛУЧАЙНЫЕ СЕТИ:")
            print(f"  Количество: {len(random)}")
            print(f"  Средняя скошенность: {random['skewness'].mean():.3f} ± {random['skewness'].std():.3f}")
            print(f"  Доля с skewness > 2: {(random['skewness'] > 2).mean()*100:.1f}%")
            print(f"  Лучшее распределение: {random['best_distribution'].value_counts().index[0] if len(random['best_distribution'].value_counts()) > 0 else 'N/A'}")
    
    print("\n" + "="*70)


# ========================
# 8. ОСНОВНАЯ ФУНКЦИЯ
# ========================

def run_full_analysis(real_networks_folder, centrality_types=['betweenness'], n_random_samples=3):
    """Запускает полный анализ с проверкой распределений."""
    
    print("="*70)
    print("ШАГ 1: Анализ реальных сетей")
    print("="*70)
    
    df_real, gradients_real = analyze_all_networks(
        real_networks_folder, 
        centrality_types=centrality_types
    )
    
    if len(df_real) == 0:
        print("Не найдено реальных сетей для анализа!")
        return None, None
    
    print(f"\nПроанализировано {len(df_real)} сетей")
    df_real.to_csv('results_real_distributions.csv', index=False)
    
    print("\n" + "="*70)
    print("ШАГ 2: Генерация случайных сетей")
    print("="*70)
    
    df_random, gradients_random = generate_random_networks(
        n_nodes_list=[100, 500], 
        n_samples=n_random_samples
    )
    df_random.to_csv('results_random_distributions.csv', index=False)
    
    print(f"\nСгенерировано {len(df_random)} случайных сетей")
    
    # Объединяем результаты
    df_combined = pd.concat([df_real, df_random], ignore_index=True)
    all_gradients = {**gradients_real, **gradients_random}
    
    print("\n" + "="*70)
    print("ШАГ 3: Визуализация")
    print("="*70)
    
    # Общий анализ распределений
    plot_distribution_analysis(df_combined, 'distribution_analysis.png')
    
    # Показываем несколько примеров реальных сетей
    sample_networks = list(gradients_real.keys())[:3]
    for key in sample_networks:
        if key in gradients_real:
            # Находим соответствующие результаты
            filename = key.replace('_betweenness', '')
            dist_row = df_real[df_real['filename'] == filename]
            if len(dist_row) > 0:
                grad = gradients_real[key]
                plot_fit_comparison(grad, dist_row.iloc[0], key, f'fit_{key.replace("/", "_")}.png')
    
    # Выводим сводку
    print_distribution_summary(df_combined)
    
    print("\n" + "="*70)
    print("АНАЛИЗ ЗАВЕРШЁН!")
    print("="*70)
    
    return df_combined, all_gradients


# ========================
# 9. ЗАПУСК
# ========================

if __name__ == "__main__":
    # Путь к папке с реальными сетями
    NETWORKS_FOLDER = "C:/Proj_edge_distr/cite_A/"
    
    # Запускаем анализ для betweenness
    df, gradients = run_full_analysis(
        real_networks_folder=NETWORKS_FOLDER,
        centrality_types=['betweenness'],
        n_random_samples=3
    )