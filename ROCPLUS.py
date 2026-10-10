import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import skew, kurtosis, mannwhitneyu
from scipy import stats
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
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
# 3. ВЫЧИСЛЕНИЕ СТАТИСТИЧЕСКИХ ХАРАКТЕРИСТИК
# ========================

def compute_all_stats(gradients):
    """
    Вычисляет все статистические характеристики распределения градиентов.
    """
    grad = gradients[gradients > 1e-12]
    if len(grad) < 10:
        return None
    
    mean = np.mean(grad)
    median = np.median(grad)
    std = np.std(grad)
    max_val = np.max(grad)
    min_val = np.min(grad)
    
    # Квантили
    q25 = np.percentile(grad, 25)
    q50 = np.percentile(grad, 50)
    q75 = np.percentile(grad, 75)
    q90 = np.percentile(grad, 90)
    q95 = np.percentile(grad, 95)
    q99 = np.percentile(grad, 99)
    
    # --- ОСНОВНЫЕ ПРИЗНАКИ (уже были) ---
    stats = {
        'skewness': skew(grad),
        'kurtosis': kurtosis(grad),
        'variance': np.var(grad),
        'cv': std / mean if mean > 0 else 0,
        'median_mean_ratio': median / mean if mean > 0 else 0,
        'q90_q50_ratio': q90 / q50 if q50 > 0 else 0,
        'max_mean_ratio': max_val / mean if mean > 0 else 0,
        'range': max_val - min_val,
        'mad': np.mean(np.abs(grad - median)),
        'gini': compute_gini(grad),
        'entropy': compute_entropy(grad)
    }
    
    # --- НОВЫЕ ПРИЗНАКИ ДЛЯ CLOSENESS ---
    
    # 1. Логарифмические преобразования
    stats['log_variance'] = np.log(stats['variance'] + 1e-10)
    stats['log_range'] = np.log(stats['range'] + 1e-10)
    stats['log_max_mean'] = np.log(stats['max_mean_ratio'] + 1e-10)
    
    # 2. Другие отношения квантилей
    stats['q75_q25_ratio'] = q75 / q25 if q25 > 0 else 0
    stats['q95_q50_ratio'] = q95 / q50 if q50 > 0 else 0
    stats['q99_q50_ratio'] = q99 / q50 if q50 > 0 else 0
    
    # 3. Характеристики на основе порядка статистик
    stats['mean_median_diff'] = mean - median
    stats['mean_median_rel'] = (mean - median) / median if median > 0 else 0
    stats['pearson_skewness'] = 3 * (mean - median) / std if std > 0 else 0
    
    # 4. Характеристики разброса
    iqr = q75 - q25
    stats['iqr'] = iqr
    stats['iqr_median_ratio'] = iqr / median if median > 0 else 0
    stats['std_iqr_ratio'] = std / iqr if iqr > 0 else 0
    
    # 5. Доли
    stats['frac_above_mean'] = np.mean(grad > mean)
    stats['frac_above_2median'] = np.mean(grad > 2 * median)
    stats['frac_above_q90'] = np.mean(grad > q90)
    
    return stats

def compute_gini(grad):
    """Вычисляет коэффициент Джини."""
    sorted_grad = np.sort(grad)
    n = len(grad)
    return (2 * np.sum((np.arange(1, n+1) * sorted_grad)) / (n * np.sum(sorted_grad))) - (n + 1) / n

def compute_entropy(grad):
    """Вычисляет энтропию распределения."""
    hist, _ = np.histogram(grad, bins=20, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist)) / np.log(len(hist)) if len(hist) > 0 else 0
# ========================
# 4. ПОСТРОЕНИЕ ГИСТОГРАММЫ
# ========================

def plot_histogram(gradients, network_name, centrality_type, save_path=None):
    """Строит гистограмму распределения градиентов."""
    grad = gradients[gradients > 1e-12]
    if len(grad) < 50:
        print(f"  {network_name} ({centrality_type}): недостаточно данных")
        return None
    
    stats_dict = compute_all_stats(grad)
    if stats_dict is None:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(grad, bins=50, density=True, alpha=0.7, 
            color='steelblue', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Градиент', fontsize=12)
    ax.set_ylabel('Плотность', fontsize=12)
    ax.set_title(f'{network_name} ({centrality_type})\nSkewness = {stats_dict["skewness"]:.3f}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    info_text = (f'$n_{{edges}}$ = {len(grad):,}\n'
                 f'$\\xi$ = {stats_dict["skewness"]:.3f}\n'
                 f'Kurtosis = {stats_dict["kurtosis"]:.3f}\n'
                 f'CV = {stats_dict["cv"]:.3f}')
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    
    plt.tight_layout()
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats_dict

# ========================
# 5. ГЕНЕРАЦИЯ СЛУЧАЙНЫХ СЕТЕЙ
# ========================

def generate_random_networks(n_networks=34, centrality='betweenness', output_dir='.'):
    """Генерирует случайные графы разных моделей."""
    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ И АНАЛИЗ СЛУЧАЙНЫХ СЕТЕЙ")
    print("="*60)
    
    results = []
    n_nodes = 500
    
    models = [
        ('ER', {'n': n_nodes, 'p': 0.002}),
        ('ER', {'n': n_nodes, 'p': 0.005}),
        ('ER', {'n': n_nodes, 'p': 0.01}),
        ('ER', {'n': n_nodes, 'p': 0.02}),
        ('ER', {'n': n_nodes, 'p': 0.04}),
        ('ER', {'n': n_nodes, 'p': 0.05}),
        ('ER', {'n': n_nodes, 'p': 0.08}),
        ('ER', {'n': n_nodes, 'p': 0.1}),
        ('ER', {'n': n_nodes, 'p': 0.2}),
        ('ER', {'n': n_nodes, 'p': 0.3}),
        ('ER', {'n': n_nodes, 'p': 0.5}),
        ('ER', {'n': n_nodes, 'p': 0.7}),
        ('BA', {'n': n_nodes, 'm': 2}),
        ('BA', {'n': n_nodes, 'm': 3}),
        ('BA', {'n': n_nodes, 'm': 4}),
        ('BA', {'n': n_nodes, 'm': 5}),
        ('BA', {'n': n_nodes, 'm': 6}),
        ('BA', {'n': n_nodes, 'm': 7}),
        ('BA', {'n': n_nodes, 'm': 8}),
        ('BA', {'n': n_nodes, 'm': 10}),
        ('BA', {'n': n_nodes, 'm': 12}),
        ('BA', {'n': n_nodes, 'm': 15}),
        ('WS', {'n': n_nodes, 'k': 5, 'p': 0.05}),
        ('WS', {'n': n_nodes, 'k': 5, 'p': 0.1}),
        ('WS', {'n': n_nodes, 'k': 5, 'p': 0.5}),
        ('WS', {'n': n_nodes, 'k': 5, 'p': 0.9}),
        ('WS', {'n': n_nodes, 'k': 7, 'p': 0.05}),
        ('WS', {'n': n_nodes, 'k': 7, 'p': 0.1}),
        ('WS', {'n': n_nodes, 'k': 7, 'p': 0.5}),
        ('WS', {'n': n_nodes, 'k': 7, 'p': 0.9}),
        ('WS', {'n': n_nodes, 'k': 10, 'p': 0.05}),
        ('WS', {'n': n_nodes, 'k': 10, 'p': 0.1}),
        ('WS', {'n': n_nodes, 'k': 10, 'p': 0.5}),
        ('WS', {'n': n_nodes, 'k': 10, 'p': 0.9}),
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
            
            name = f"{model_name}_{i+1}"
            save_path = os.path.join(output_dir, f'random_{centrality}_{name}.png')
            stats_dict = plot_histogram(gradients, name, centrality, save_path=save_path)
            
            if stats_dict is not None:
                results.append({
                    'filename': name,
                    'centrality': centrality,
                    'type': 'Random',
                    'model': model_name,
                    'n_nodes': G.number_of_nodes(),
                    'n_edges': G.number_of_edges(),
                    **stats_dict
                })
                
        except Exception as e:
            print(f"    ❌ Ошибка: {e}")
    
    df = pd.DataFrame(results)
    return df

# ========================
# 6. АНАЛИЗ РЕАЛЬНЫХ СЕТЕЙ
# ========================

def analyze_real_networks(folder_path, centrality='betweenness', output_dir='.', max_networks=None):
    """Анализирует реальные сети."""
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
        
        save_path = os.path.join(output_dir, f'real_{centrality}_{filename.replace(".", "_")}.png')
        stats_dict = plot_histogram(gradients, filename, centrality, save_path=save_path)
        
        if stats_dict is not None:
            results.append({
                'filename': filename,
                'centrality': centrality,
                'type': 'Real',
                'model': 'Real',
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                **stats_dict
            })
    
    df = pd.DataFrame(results)
    return df

# ========================
# 7. ROC-АНАЛИЗ ДЛЯ ВСЕХ ПРИЗНАКОВ
# ========================

def plot_roc_curve(df, feature, label_col='type', output_dir='.', save_name=None):
    """Строит ROC-кривую для заданного признака."""
    if len(df) == 0:
        print(f"⚠️ Нет данных для ROC-анализа признака {feature}")
        return None
    
    if feature not in df.columns:
        print(f"⚠️ Признак {feature} не найден в данных")
        return None
    
    le = LabelEncoder()
    y_true = le.fit_transform(df[label_col])
    
    if len(np.unique(y_true)) < 2:
        print(f"⚠️ В данных только один класс")
        return None
    
    X = df[feature].values
    
    fpr, tpr, thresholds = roc_curve(y_true, X)
    roc_auc = auc(fpr, tpr)
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
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
    plt.title(f'ROC-кривая для признака "{feature}"\nРеальные vs Случайные сети', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # if save_name is None:
    #     save_name = f'roc_{feature}.png'
    # save_path = os.path.join(output_dir, save_name)
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"ROC-АНАЛИЗ ДЛЯ ПРИЗНАКА: {feature}")
    print('='*60)
    print(f"AUC = {roc_auc:.4f}")
    print(f"Оптимальный порог = {optimal_threshold:.4f}")
    print(f"TPR (чувствительность) = {tpr[optimal_idx]:.4f}")
    print(f"FPR (1-специфичность) = {fpr[optimal_idx]:.4f}")
    
    if roc_auc >= 0.9:
        print("✅ Отлично! AUC > 0.9 — выдающееся разделение!")
    elif roc_auc >= 0.8:
        print("✅ Хорошо! AUC > 0.8 — хорошее разделение!")
    elif roc_auc >= 0.7:
        print("⚠️ Приемлемо! AUC > 0.7 — удовлетворительное разделение.")
    else:
        print("❌ Плохо! AUC < 0.7 — разделение слабое.")
    
    return {'auc': roc_auc, 'optimal_threshold': optimal_threshold}

def plot_roc_comparison(df, features, label_col='type', output_dir='.'):
    """Сравнивает ROC-кривые для разных признаков."""
    if len(df) == 0:
        print("⚠️ Нет данных для ROC-анализа")
        return
    
    le = LabelEncoder()
    y_true = le.fit_transform(df[label_col])
    
    if len(np.unique(y_true)) < 2:
        print(f"⚠️ В данных только один класс")
        return
    
    plt.figure(figsize=(10, 8))
    colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    auc_values = {}
    
    for i, feature in enumerate(features):
        if feature not in df.columns:
            print(f"⚠️ Признак {feature} не найден в данных")
            continue
        
        X = df[feature].values
        fpr, tpr, _ = roc_curve(y_true, X)
        roc_auc = auc(fpr, tpr)
        auc_values[feature] = roc_auc
        
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, 
                 label=f'{feature} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Сравнение ROC-кривых для разных признаков', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # save_path = os.path.join(output_dir, 'roc_comparison.png')
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print("СРАВНЕНИЕ ПРИЗНАКОВ ПО AUC")
    print('='*60)
    for feature, auc_val in sorted(auc_values.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: AUC = {auc_val:.4f}")
    
    return auc_values

# ========================
# 8. СРАВНИТЕЛЬНЫЕ ГРАФИКИ
# ========================

def plot_comparison(real_df, random_df, feature, centrality, output_dir='.'):
    """Строит сравнительные графики для заданного признака."""
    if len(real_df) == 0 or len(random_df) == 0:
        print("Недостаточно данных для сравнения")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Сравнение реальных и случайных сетей\nЦентральность: {centrality}, Признак: {feature}', 
                 fontsize=14, fontweight='bold')
    
    axes[0].scatter([0]*len(real_df), real_df[feature], 
                   color='blue', alpha=0.7, label='Real', s=50)
    axes[0].scatter([1]*len(random_df), random_df[feature], 
                   color='red', alpha=0.7, label='Random', s=50)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Real', 'Random'])
    axes[0].set_ylabel(feature)
    axes[0].set_title(f'Распределение {feature}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(real_df[feature], bins=10, alpha=0.5, label='Real', color='blue', edgecolor='black')
    axes[1].hist(random_df[feature], bins=10, alpha=0.5, label='Random', color='red', edgecolor='black')
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Гистограмма {feature}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # save_path = os.path.join(output_dir, f'comparison_{feature}.png')
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_summary(real_df, random_df, feature):
    """Выводит сводку по признаку."""
    print("\n" + "-"*70)
    print(f"ПРИЗНАК: {feature}")
    print("-"*70)
    print("РЕАЛЬНЫЕ СЕТИ:")
    print(f"  mean = {real_df[feature].mean():.4f} ± {real_df[feature].std():.4f}")
    print(f"  median = {real_df[feature].median():.4f}")
    
    print("\nСЛУЧАЙНЫЕ СЕТИ:")
    print(f"  mean = {random_df[feature].mean():.4f} ± {random_df[feature].std():.4f}")
    print(f"  median = {random_df[feature].median():.4f}")
    
    stat, p = mannwhitneyu(real_df[feature], random_df[feature])
    print(f"\nMann-Whitney U test: p = {p:.4f} {'✅ Различаются' if p < 0.05 else '❌ Не различаются'}")

# ========================
# 9. ОСНОВНАЯ ФУНКЦИЯ
# ========================

def run_full_analysis(folder_path, centrality='betweenness', n_random=34, max_networks=None):
    """Запускает полный анализ: реальные сети + случайные + ROC для всех признаков."""
    base_dir = "C:/Proj_edge_distr/"
    output_dir = os.path.join(base_dir, f"results_{centrality}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Результаты будут сохранены в: {output_dir}")
    
    # Реальные сети
    real_df = analyze_real_networks(folder_path, centrality, output_dir, max_networks)
    
    # Случайные сети
    random_df = generate_random_networks(n_random, centrality, output_dir)
    
    if len(real_df) == 0 or len(random_df) == 0:
        print("❌ Недостаточно данных для анализа")
        return None, real_df, random_df
    
    # Объединяем
    df_combined = pd.concat([real_df, random_df], ignore_index=True)
    
    # Список признаков для ROC-анализа
    features = ['skewness', 'kurtosis', 'variance', 'cv', 'median_mean_ratio', 
                'q90_q50_ratio', 'max_mean_ratio', 'range', 'mad', 'gini', 'entropy']
    
    print("\n" + "="*70)
    print(f"АНАЛИЗ ДЛЯ ЦЕНТРАЛЬНОСТИ: {centrality}")
    print("="*70)
    print(f"Всего сетей: {len(df_combined)}")
    print(f"  Реальных: {len(real_df)}")
    print(f"  Случайных: {len(random_df)}")
    
    # Для каждого признака строим сравнение и ROC
    results_roc = {}
    
    for feature in features:
        if feature not in df_combined.columns:
            print(f"⚠️ Признак {feature} не найден, пропускаем")
            continue
        
        # Сравнительные графики
        plot_comparison(real_df, random_df, feature, centrality, output_dir)
        print_summary(real_df, random_df, feature)
        
        # ROC-анализ
        roc_result = plot_roc_curve(df_combined, feature, output_dir=output_dir, 
                                    save_name=f'roc_{feature}.png')
        if roc_result:
            results_roc[feature] = roc_result['auc']
    
    # Сравнение всех ROC-кривых
    if results_roc:
        plot_roc_comparison(df_combined, list(results_roc.keys()), output_dir=output_dir)
    
    # Сохраняем результаты
    # df_combined.to_csv(os.path.join(output_dir, 'full_analysis_results.csv'), index=False)
    
    # Вывод лучшего признака
    if results_roc:
        best_feature = max(results_roc, key=results_roc.get)
        print(f"\n{'='*70}")
        print(f"ЛУЧШИЙ ПРИЗНАК: {best_feature} (AUC = {results_roc[best_feature]:.4f})")
        print("="*70)
    
    print(f"\n✅ Все результаты сохранены в папку: {output_dir}")
    
    return df_combined, real_df, random_df

# ========================
# 10. ЗАПУСК
# ========================

if __name__ == "__main__":
    NETWORKS_FOLDER = "C:/Proj_edge_distr/cite_A/"
    
    # Можно менять центральность: 'betweenness', 'pagerank', 'closeness', 'eigenvector', 'kcore', 'degree'
    df, real, random = run_full_analysis(
        folder_path=NETWORKS_FOLDER,
        centrality='closeness',
        n_random=34,
        max_networks=None
    )