import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import skew, gamma
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
            if G.number_of_nodes() > 5000:
                centrality_dict = nx.closeness_centrality(G, u=None, distance=None, wf_improved=True)
            else:
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
# 3. ПОСТРОЕНИЕ ГРАФИКОВ (3 ГРАФИКА ДЛЯ КАЖДОЙ СЕТИ) - Gamma версия
# ========================

def plot_qq_pp_histogram(gradients, network_name, centrality_type, save_path=None):
    """
    Строит три графика для центральности.
    Использует ГАММА-распределение (Gamma distribution).
    """
    grad = gradients[gradients > 1e-12]
    if len(grad) < 50:
        print(f"  {network_name} ({centrality_type}): недостаточно данных")
        return None, None, None
    
    # Подгоняем Гамма (Gamma)
    try:
        # Gamma.fit возвращает (shape, loc, scale)
        a, loc, scale = gamma.fit(grad)
        skewness_val = skew(grad)
    except Exception as e:
        print(f"  {network_name} ({centrality_type}): ошибка подгонки Gamma - {e}")
        return None, None, None
    
    # Создаем фигуру с тремя графиками
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f'{network_name} ({centrality_type})\nGamma a={a:.3f}, scale={scale:.3f}, skewness={skewness_val:.3f}', 
                 fontsize=14, fontweight='bold')
    
    n = len(grad)
    sorted_data = np.sort(grad)
    p = np.arange(1, n+1) / (n+1)
    
    # --- 1. QQ-PLOT ---
    theoretical_qq = gamma.ppf(p, a, loc=loc, scale=scale)
    
    axes[0].scatter(theoretical_qq, sorted_data, alpha=0.5, s=10, color='blue')
    axes[0].plot([min(theoretical_qq), max(theoretical_qq)], 
                [min(theoretical_qq), max(theoretical_qq)], 
                'r--', linewidth=2, label='y=x (идеал)')
    axes[0].set_xlabel('Теоретические квантили (Gamma)')
    axes[0].set_ylabel('Эмпирические квантили')
    axes[0].set_title('QQ-plot (проверка ХВОСТОВ)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    residuals = sorted_data - theoretical_qq
    rmse = np.sqrt(np.mean(residuals**2))
    axes[0].text(0.05, 0.95, f'RMSE={rmse:.4f}', transform=axes[0].transAxes, 
                fontsize=10, verticalalignment='top')
    
    # --- 2. P-P PLOT ---
    empirical_cdf = np.arange(1, n+1) / n
    theoretical_pp = gamma.cdf(sorted_data, a, loc=loc, scale=scale)
    
    axes[1].scatter(theoretical_pp, empirical_cdf, alpha=0.5, s=10, color='green')
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x (идеал)')
    axes[1].set_xlabel('Теоретическая CDF (Gamma)')
    axes[1].set_ylabel('Эмпирическая CDF')
    axes[1].set_title('P-P plot (проверка ЦЕНТРА)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    residuals_pp = empirical_cdf - theoretical_pp
    rmse_pp = np.sqrt(np.mean(residuals_pp**2))
    axes[1].text(0.05, 0.95, f'RMSE={rmse_pp:.4f}', transform=axes[1].transAxes, 
                fontsize=10, verticalalignment='top')
    
    # --- 3. ГИСТОГРАММА С КРИВОЙ GAMMA ---
    min_val = max(grad.min(), 1e-6)
    max_val = grad.max()
    margin = 0.5
    min_val_extended = min_val / (10**margin)
    max_val_extended = max_val * (10**margin)
    
    n_bins = 60
    bins = np.logspace(np.log10(min_val_extended), np.log10(max_val_extended), n_bins)
    
    axes[2].hist(grad, bins=bins, density=True, alpha=0.5, 
                label='Данные', color='gray', edgecolor='black', linewidth=0.5)
    
    x_vals = np.logspace(np.log10(min_val_extended), np.log10(max_val_extended), 2000)
    gamma_pdf = gamma.pdf(x_vals, a, loc=loc, scale=scale)
    gamma_pdf_norm = gamma_pdf / np.trapz(gamma_pdf, x_vals)
    
    axes[2].loglog(x_vals, gamma_pdf_norm, 'r-', linewidth=2.5, 
                  label=f'Gamma (a={a:.3f})')
    
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
    axes[2].set_title('Градиент-плотность с кривой Gamma', fontsize=12)
    axes[2].legend(loc='best', fontsize=10)
    
    info_text = (f'$n_{{edges}}$={len(grad):,}\n'
                 f'$a$={a:.3f}\n'
                 f'$scale$={scale:.3f}\n'
                 f'$\\xi$={skewness_val:.3f}')
    axes[2].text(0.05, 0.05, info_text, transform=axes[2].transAxes, 
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Закрываем фигуру, чтобы не всплывала
    
    return a, scale, skewness_val

# ========================
# 4. ГЕНЕРАЦИЯ СЛУЧАЙНЫХ СЕТЕЙ
# ========================

def generate_random_networks(n_networks=10, centrality='closeness', output_dir='.'):
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
            save_path = os.path.join(output_dir, f'random_{centrality}_{name}.png')
            a, scale, skewness_val = plot_qq_pp_histogram(
                gradients, name, centrality,
                save_path=save_path
            )
            
            if a is not None:
                results.append({
                    'filename': name,
                    'centrality': centrality,
                    'type': 'Random',
                    'model': model_name,
                    'n_nodes': G.number_of_nodes(),
                    'n_edges': G.number_of_edges(),
                    'gamma_a': a,
                    'gamma_scale': scale,
                    'skewness': skewness_val
                })
                
        except Exception as e:
            print(f"    ❌ Ошибка: {e}")
    
    df = pd.DataFrame(results)
    return df

# ========================
# 5. АНАЛИЗ РЕАЛЬНЫХ СЕТЕЙ
# ========================

def analyze_real_networks(folder_path, centrality='closeness', output_dir='.', max_networks=None):
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
        save_path = os.path.join(output_dir, f'real_{centrality}_{filename.replace(".", "_")}.png')
        a, scale, skewness_val = plot_qq_pp_histogram(
            gradients, filename, centrality,
            save_path=save_path
        )
        
        if a is not None:
            results.append({
                'filename': filename,
                'centrality': centrality,
                'type': 'Real',
                'model': 'Real',
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'gamma_a': a,
                'gamma_scale': scale,
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
    
    # 1. Gamma a (параметр формы)
    axes[0].scatter([0]*len(real_df), real_df['gamma_a'], 
                   color='blue', alpha=0.7, label='Real', s=50)
    axes[0].scatter([1]*len(random_df), random_df['gamma_a'], 
                   color='red', alpha=0.7, label='Random', s=50)
    axes[0].axhline(y=1, color='green', linestyle='--', linewidth=2, label='a=1')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Real', 'Random'])
    axes[0].set_ylabel('Gamma a')
    axes[0].set_title('Параметр формы Gamma')
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
    
    # 3. a vs skewness
    axes[2].scatter(real_df['gamma_a'], real_df['skewness'], 
                   color='blue', alpha=0.7, label='Real', s=50)
    axes[2].scatter(random_df['gamma_a'], random_df['skewness'], 
                   color='red', alpha=0.7, label='Random', s=50)
    axes[2].axhline(y=2, color='green', linestyle='--', linewidth=2, label='skewness=2')
    axes[2].axvline(x=1, color='orange', linestyle='--', linewidth=2, label='a=1')
    axes[2].set_xlabel('Gamma a')
    axes[2].set_ylabel('Skewness')
    axes[2].set_title('a vs skewness')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_comparison_summary(real_df, random_df):
    """Выводит сравнительную сводку."""
    print("\n" + "="*70)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("="*70)
    
    print("\n" + "-"*70)
    print("РЕАЛЬНЫЕ СЕТИ:")
    print("-"*70)
    print(f"  Количество: {len(real_df)}")
    print(f"  Gamma a: mean={real_df['gamma_a'].mean():.3f} ± {real_df['gamma_a'].std():.3f}")
    print(f"  Доля a < 1: {(real_df['gamma_a'] < 1).mean()*100:.1f}%")
    print(f"  Skewness: mean={real_df['skewness'].mean():.3f} ± {real_df['skewness'].std():.3f}")
    print(f"  Доля skewness > 2: {(real_df['skewness'] > 2).mean()*100:.1f}%")
    
    print("\n" + "-"*70)
    print("СЛУЧАЙНЫЕ СЕТИ:")
    print("-"*70)
    print(f"  Количество: {len(random_df)}")
    print(f"  Gamma a: mean={random_df['gamma_a'].mean():.3f} ± {random_df['gamma_a'].std():.3f}")
    print(f"  Доля a < 1: {(random_df['gamma_a'] < 1).mean()*100:.1f}%")
    print(f"  Skewness: mean={random_df['skewness'].mean():.3f} ± {random_df['skewness'].std():.3f}")
    print(f"  Доля skewness > 2: {(random_df['skewness'] > 2).mean()*100:.1f}%")
    
    from scipy.stats import mannwhitneyu
    
    print("\n" + "-"*70)
    print("СТАТИСТИЧЕСКИЕ ТЕСТЫ:")
    print("-"*70)
    
    stat, p = mannwhitneyu(real_df['gamma_a'], random_df['gamma_a'])
    print(f"  Gamma a (Mann-Whitney): p={p:.4f} {'✅ Различаются' if p < 0.05 else '❌ Не различаются'}")
    
    stat, p = mannwhitneyu(real_df['skewness'], random_df['skewness'])
    print(f"  Skewness (Mann-Whitney): p={p:.4f} {'✅ Различаются' if p < 0.05 else '❌ Не различаются'}")
    
    print("\n" + "="*70)

# ========================
# 7. ОСНОВНАЯ ФУНКЦИЯ
# ========================

def run_full_analysis(folder_path, centrality='pagerank', n_random=10, max_networks=None):
    """
    Запускает полный анализ: реальные сети + случайные.
    Результаты сохраняются в C:/Proj_edge_distr/results_{centrality}/
    """
    import os
    
    # Создаем папку для результатов в нужном месте
    base_dir = "C:/Proj_edge_distr/"
    output_dir = os.path.join(base_dir, f"results_{centrality}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Результаты будут сохранены в: {output_dir}")
    
    # Реальные сети
    real_df = analyze_real_networks(folder_path, centrality, output_dir, max_networks)
    
    # Случайные сети
    random_df = generate_random_networks(n_random, centrality, output_dir)
    
    # Сравнение
    if len(real_df) > 0 and len(random_df) > 0:
        plot_comparison(real_df, random_df, centrality, save_path=os.path.join(output_dir, 'comparison.png'))
        print_comparison_summary(real_df, random_df)
    
    # Сохраняем результаты
    df_combined = pd.concat([real_df, random_df], ignore_index=True)
    df_combined.to_csv(os.path.join(output_dir, 'full_analysis_results.csv'), index=False)
    print(f"\n✅ Все результаты сохранены в папку: {output_dir}")
    
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