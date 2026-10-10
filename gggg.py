import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import skew
import powerlaw
from scipy import stats

# ========================
# 1. ЗАГРУЗКА СЕТЕЙ (универсальная)
# ========================

def load_network(filepath):
    """Загружает сеть из GML, TXT/EDGES или MTX файла."""
    if filepath.endswith('.gml'):
        # Ваш существующий парсер GML
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

# (Ваша существующая функция load_cleaned_gml остаётся без изменений)
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
# 2. ВЫЧИСЛЕНИЕ ГРАДИЕНТОВ
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
# 3. ПОЛНЫЙ АНАЛИЗ ОДНОЙ СЕТИ
# ========================

def analyze_single_network(filepath, centrality_type):
    """Полный анализ одной сети: статистики + распределение."""
    try:
        G = load_network(filepath)
    except Exception as e:
        print(f"  Ошибка загрузки {filepath}: {e}")
        return None, None
    
    if G.number_of_nodes() < 10:
        return None, None
    
    gradients = compute_edge_gradients(G, centrality_type)
    if gradients is None or len(gradients) == 0:
        return None, None
    # Убираем нулевые значения
    gradients = gradients[gradients > 1e-12]
    if len(gradients) < 10:
        return None, None
    
    # --- СТАТИСТИКИ ---
    stats = {
        'filename': os.path.basename(filepath),
        'centrality': centrality_type,
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'mean': np.mean(gradients),
        'std': np.std(gradients),
        'skewness': skew(gradients),
        'size': len(gradients)
    }
    
    # --- СТЕПЕННОЙ ЗАКОН (powerlaw) ---
    try:
        # Ограничиваем количество данных для ускорения
        if len(gradients) > 5000:
            # Берем выборку
            sample = np.random.choice(gradients, 5000, replace=False)
        else:
            sample = gradients
        
        # Фильтруем нулевые и отрицательные значения
        sample = sample[sample > 0]
        
        if len(sample) > 10:  # Минимум данных для подгонки
            # Устанавливаем жесткие ограничения
            fit = powerlaw.Fit(
                sample, 
                verbose=False, 
                discrete=False,
                xmin=None,           # автоопределение
                xmax=None,           # автоопределение
                estimate_discrete=False  # не оценивать дискретность
            )
            
            # Ограничиваем поиск xmin
            # Находим только если данных достаточно
            if len(sample) > 100:
                alpha = fit.power_law.alpha
                xmin = fit.power_law.xmin
                R, p = fit.distribution_compare('power_law', 'lognormal')
                stats['alpha'] = alpha
                stats['xmin'] = xmin
                stats['p_powerlaw'] = p
                stats['tail_fraction'] = np.mean(sample >= xmin)
            else:
                # Слишком мало данных
                stats['alpha'] = np.nan
                stats['xmin'] = np.nan
                stats['p_powerlaw'] = np.nan
                stats['tail_fraction'] = np.nan
        else:
            stats['alpha'] = np.nan
            stats['xmin'] = np.nan
            stats['p_powerlaw'] = np.nan
            stats['tail_fraction'] = np.nan
            
    except Exception as e:
        print(f"Ошибка powerlaw: {e}")
        stats['alpha'] = np.nan
        stats['xmin'] = np.nan
        stats['p_powerlaw'] = np.nan
        stats['tail_fraction'] = np.nan
    return stats, gradients


# ========================
# 4. АНАЛИЗ ВСЕХ СЕТЕЙ В ПАПКЕ
# ========================

def analyze_all_networks(folder_path, centrality_types=None):
    """Анализирует все сетевые файлы в папке для всех центральностей."""
    if centrality_types is None:
        centrality_types = ['pagerank', 'betweenness', 'closeness', 'eigenvector', 'kcore']
    
    # Ищем все файлы с сетями
    extensions = ['.gml', '.txt', '.edges', '.mtx']
    files = [f for f in os.listdir(folder_path) if any(f.endswith(ext) for ext in extensions)]
    
    all_stats = []
    all_gradients = {}  # для сохранения распределений
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        print(f"\nОбработка: {filename}")
        
        for cent in centrality_types:
            print(f"  Центральность: {cent}")
            stats, gradients = analyze_single_network(filepath, cent)
        
            if stats is not None:
                all_stats.append(stats)
                key = f"{filename}_{cent}"
                all_gradients[key] = gradients
            print(1)
    
    df = pd.DataFrame(all_stats)
    return df, all_gradients


# ========================
# 5. ГЕНЕРАЦИЯ СЛУЧАЙНЫХ СЕТЕЙ
# ========================

def generate_random_networks(n_nodes=1000, n_samples=5):
    """Генерирует случайные графы разных моделей."""
    results = []
    gradients_dict = {}
    
    # ER
    for p in [0.001, 0.005, 0.01, 0.05, 0.1]:
        for i in range(n_samples):
            G = nx.erdos_renyi_graph(n_nodes, p)
            cent = 'betweenness'
            grad = compute_edge_gradients(G, cent)
            if grad is not None and len(grad) > 10:
                grad = grad[grad > 1e-12]
                results.append({
                    'filename': f'ER_p={p}_{i}',
                    'centrality': cent,
                    'n_nodes': G.number_of_nodes(),
                    'n_edges': G.number_of_edges(),
                    'density': nx.density(G),
                    'skewness': skew(grad),
                    'type': 'Random'
                })
                gradients_dict[f'ER_p={p}_{i}_{cent}'] = grad
    
    # BA
    for m in [1, 2, 3, 5, 10]:
        for i in range(n_samples):
            try:
                G = nx.barabasi_albert_graph(n_nodes, m)
                cent = 'betweenness'
                grad = compute_edge_gradients(G, cent)
                if grad is not None and len(grad) > 10:
                    grad = grad[grad > 1e-12]
                    results.append({
                        'filename': f'BA_m={m}_{i}',
                        'centrality': cent,
                        'n_nodes': G.number_of_nodes(),
                        'n_edges': G.number_of_edges(),
                        'density': nx.density(G),
                        'skewness': skew(grad),
                        'type': 'Random'
                    })
                    gradients_dict[f'BA_m={m}_{i}_{cent}'] = grad
            except:
                pass
    
    # WS
    for p in [0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]:
        for i in range(n_samples):
            try:
                G = nx.watts_strogatz_graph(n_nodes, 10, p)
                cent = 'betweenness'
                grad = compute_edge_gradients(G, cent)
                if grad is not None and len(grad) > 10:
                    grad = grad[grad > 1e-12]
                    results.append({
                        'filename': f'WS_p={p}_{i}',
                        'centrality': cent,
                        'n_nodes': G.number_of_nodes(),
                        'n_edges': G.number_of_edges(),
                        'density': nx.density(G),
                        'skewness': skew(grad),
                        'type': 'Random'
                    })
                    gradients_dict[f'WS_p={p}_{i}_{cent}'] = grad
            except:
                pass
    
    df = pd.DataFrame(results)
    return df, gradients_dict


# ========================
# 6. ПОСТРОЕНИЕ ГРАФИКОВ
# ========================

def plot_skewness_comparison(df_real, df_random, centrality):
    """График 1: Сравнение скошенности."""
    df_real_cent = df_real[df_real['centrality'] == centrality].copy()
    df_random_cent = df_random[df_random['centrality'] == centrality].copy()
    
    df_real_cent['type'] = 'Real'
    df_random_cent['type'] = 'Random'
    df_combined = pd.concat([df_real_cent, df_random_cent])
    
    plt.figure(figsize=(10, 6))

    sns.boxplot(
    data=df_combined, 
    x='type', 
    y='skewness',
    hue='type',           # <-- добавляем
    palette={'Real': 'blue', 'Random': 'orange'}  # <-- задаём цвета
    )

    sns.boxplot(data=df_combined, x='type', y='skewness')
    plt.yscale('log')
    plt.title(f'Скошенность распределения градиентов: {centrality}')
    plt.ylabel('Skewness (log scale)')
    plt.axhline(y=2, color='red', linestyle='--', label='Порог разделения (skew=2)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'skewness_boxplot_{centrality}.png', dpi=300)
    plt.show()


def plot_skewness_vs_size(df_real, df_random, centrality):
    """График 2: Скошенность vs размер сети."""
    df_real_cent = df_real[df_real['centrality'] == centrality].copy()
    df_random_cent = df_random[df_random['centrality'] == centrality].copy()
    
    df_real_cent['type'] = 'Real'
    df_random_cent['type'] = 'Random'
    df_combined = pd.concat([df_real_cent, df_random_cent])
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_combined, x='n_edges', y='skewness', hue='type', alpha=0.7, s=50)
    plt.xscale('log')
    plt.axhline(y=2, color='red', linestyle='--', label='Порог разделения')
    plt.legend()
    plt.title(f'Скошенность vs количество рёбер: {centrality}')
    plt.xlabel('log(|E|)')
    plt.ylabel('Skewness')
    plt.tight_layout()
    plt.savefig(f'skewness_vs_size_{centrality}.png', dpi=300)
    plt.show()


def plot_skewness_histogram(df_real, df_random, centrality):
    """График 3: Гистограмма скошенности."""
    df_real_cent = df_real[df_real['centrality'] == centrality]
    df_random_cent = df_random[df_random['centrality'] == centrality]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_real_cent, x='skewness', bins=20, alpha=0.5, color='blue', label='Real', kde=True)
    sns.histplot(data=df_random_cent, x='skewness', bins=20, alpha=0.5, color='red', label='Random', kde=True)
    plt.axvline(x=2, color='green', linestyle='--', label='Порог разделения')
    plt.legend()
    plt.title(f'Распределение скошенности: {centrality}')
    plt.xlabel('Skewness')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'skewness_histogram_{centrality}.png', dpi=300)
    plt.show()


def plot_loglog(gradients, title, subplot_idx=None):
    """График 4: Log-log распределение (ваша существующая функция)."""
    bins = min(25, len(gradients) // 4)
    counts, bin_edges = np.histogram(gradients, bins=bins, density=True)
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = counts > 0
    x = x[mask]
    counts = counts[mask]
    sorted_idx = np.argsort(x)
    x = x[sorted_idx]
    counts = counts[sorted_idx]
    
    if subplot_idx is not None:
        plt.subplot(1, 3, subplot_idx)
    plt.plot(x, counts, 'o-', markersize=3, linewidth=1, color='salmon')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("log(|df|)")
    plt.ylabel("log(Плотность)")
    plt.title(title)
    plt.grid(True, which="major", ls="--")


def plot_qq(gradients, title):
    """График 5: QQ-plot для проверки степенного закона."""
    plt.figure(figsize=(6, 6))
    stats.probplot(gradients, dist="norm", plot=plt)
    plt.title(f'QQ-plot: {title}')
    plt.tight_layout()
    plt.savefig(f'qqplot_{title.replace("/", "_")}.png', dpi=300)
    plt.show()


# ========================
# 7. ОСНОВНАЯ ФУНКЦИЯ (запуск всего)
# ========================

def run_full_analysis(real_networks_folder, centrality='betweenness', n_random=5):
    """
    Запускает полный анализ:
    1. Анализирует все реальные сети в папке
    2. Генерирует случайные сети
    3. Строит все графики
    """
    print("="*60)
    print("ШАГ 1: Анализ реальных сетей")
    print("="*60)
    
    df_real, gradients_real = analyze_all_networks(
        real_networks_folder, 
        centrality_types=[centrality]
    )
    
    if len(df_real) == 0:
        print("Не найдено реальных сетей для анализа!")
        return
    
    print(f"\nПроанализировано {len(df_real)} сетей")
    print(df_real[['filename', 'n_nodes', 'n_edges', 'skewness', 'p_powerlaw']].to_string())
    
    # Сохраняем результаты
    df_real.to_csv('results_real_networks.csv', index=False)
    
    print("\n" + "="*60)
    print("ШАГ 2: Генерация случайных сетей")
    print("="*60)
    
    df_random, gradients_random = generate_random_networks(n_nodes=500, n_samples=n_random)
    df_random.to_csv('results_random_networks.csv', index=False)
    
    print(f"\nСгенерировано {len(df_random)} случайных сетей")
    
    print("\n" + "="*60)
    print("ШАГ 3: Построение графиков")
    print("="*60)
    
    # График 1: Boxplot скошенности
    plot_skewness_comparison(df_real, df_random, centrality)
    
    # График 2: Scatter plot
    plot_skewness_vs_size(df_real, df_random, centrality)
    
    # График 3: Гистограмма скошенности
    plot_skewness_histogram(df_real, df_random, centrality)
    
    # График 4: Log-log для нескольких реальных сетей
    plt.figure(figsize=(15, 5))
    for i, (key, grad) in enumerate(list(gradients_real.items())[:3]):
        if grad is not None and len(grad) > 10:
            plot_loglog(grad, key[:20], i+1)
    plt.tight_layout()
    plt.savefig('loglog_real_networks.png', dpi=300)
    plt.show()
    
    # График 5: QQ-plot для сетей с высокой скошенностью
    high_skew = df_real[df_real['skewness'] > 2].head(3)
    for _, row in high_skew.iterrows():
        key = f"{row['filename']}_{row['centrality']}"
        if key in gradients_real:
            plot_qq(gradients_real[key], row['filename'])
    
    print("\n" + "="*60)
    print("АНАЛИЗ ЗАВЕРШЁН!")
    print("="*60)
    
    # Вывод статистики
    print("\nСтатистика по реальным сетям:")
    print(f"  Средняя скошенность: {df_real['skewness'].mean():.2f} ± {df_real['skewness'].std():.2f}")
    print(f"  Медианная скошенность: {df_real['skewness'].median():.2f}")
    print(f"  Доля сетей с skewness > 2: {100 * (df_real['skewness'] > 2).mean():.1f}%")
    
    print("\nСтатистика по случайным сетям:")
    print(f"  Средняя скошенность: {df_random['skewness'].mean():.2f} ± {df_random['skewness'].std():.2f}")
    print(f"  Медианная скошенность: {df_random['skewness'].median():.2f}")
    print(f"  Доля сетей с skewness > 2: {100 * (df_random['skewness'] > 2).mean():.1f}%")
    
    return df_real, df_random


# ========================
# 8. ЗАПУСК
# ========================

if __name__ == "__main__":
    # Укажите путь к папке с вашими реальными сетями
    NETWORKS_FOLDER = "C:/Proj_edge_distr/cite_A/"
    
    # Запускаем анализ для центральности betweenness
    df_real, df_random = run_full_analysis(
        real_networks_folder=NETWORKS_FOLDER,
        centrality='betweenness',
        n_random=5
    )