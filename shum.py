import os
import re
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')


# ========================
# 1. ЗАГРУЗКА СЕТЕЙ
# ========================

def load_network(filepath):
    """Загружает сеть из GML, TXT/EDGES, MTX или EDGE файла."""
    if filepath.endswith('.gml'):
        return load_cleaned_gml(filepath)
    elif filepath.endswith('.txt') or filepath.endswith('.edges') or filepath.endswith('.edge'):
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

    cleaned_text = "".join(cleaned_lines)
    try:
        return nx.parse_gml(cleaned_text, label='id')
    except:
        return parse_gml_manual(cleaned_text)

def parse_gml_manual(content):
    """Ручной парсинг GML"""
    node_pattern = r'node\s*\[\s*id\s+(\d+)'
    nodes = re.findall(node_pattern, content)
    edge_pattern = r'edge\s*\[\s*id\s+\d+\s*source\s+(\d+)\s*target\s+(\d+)'
    edges = re.findall(edge_pattern, content)
    
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
# 2. ОСНОВНАЯ ФУНКЦИЯ: ПРОВЕРКА ЗАШУМЛЕННОСТИ
# ========================

def check_degree_noise(G, network_name, output_dir=None, plot=True):
    """
    Проверяет распределение степеней вершин на зашумленность.
    
    Параметры:
    - G: граф NetworkX
    - network_name: имя сети для заголовков
    - output_dir: папка для сохранения графиков (если None - не сохраняет)
    - plot: строить ли графики
    
    Возвращает:
    - is_noisy: True если сеть слишком зашумлена
    - noise_score: число от 0 (идеально) до 1 (очень шумно)
    - metrics: словарь с метриками
    """
    degrees = np.array([d for n, d in G.degree()])
    
    # Проверка на минимальный размер
    if len(degrees) < 10:
        return True, 1.0, {'error': 'too_few_nodes', 'n_nodes': len(degrees)}
    
    # Уникальные степени и их частоты
    unique_deg, counts = np.unique(degrees, return_counts=True)
    p_k = counts / len(degrees)
    
    # ===== ЛОГАРИФМИЧЕСКОЕ БИННИНГОВАНИЕ =====
    min_deg = max(unique_deg.min(), 1)
    max_deg = unique_deg.max()
    
    if min_deg == max_deg:
        return True, 1.0, {'error': 'only_one_degree', 'n_nodes': len(degrees)}
    
    log_bins = np.logspace(np.log10(min_deg), np.log10(max_deg), num=min(30, len(unique_deg)))
    log_bins = np.unique(np.round(log_bins).astype(int))
    log_bins = log_bins[log_bins >= 1]
    
    if len(log_bins) < 3:
        return True, 1.0, {'error': 'too_few_bins', 'n_nodes': len(degrees)}
    
    # Усредняем точки в бинах
    binned_ks = []
    binned_ps = []
    for i in range(len(log_bins)-1):
        mask = (unique_deg >= log_bins[i]) & (unique_deg < log_bins[i+1])
        if np.sum(mask) > 0:
            binned_ks.append(np.mean(unique_deg[mask]))
            binned_ps.append(np.mean(p_k[mask]))
    
    if len(binned_ks) < 3:
        return True, 1.0, {'error': 'too_few_bins_after_binning', 'n_nodes': len(degrees)}
    
    # Логарифмы для регрессии
    log_ks = np.log10(binned_ks)
    log_ps = np.log10(binned_ps)
    
    # ===== АНАЛИЗ ХВОСТА (берем последние 2/3 точек) =====
    tail_start = max(1, len(log_ks) // 3)
    tail_ks = log_ks[tail_start:]
    tail_ps = log_ps[tail_start:]
    
    if len(tail_ks) < 3:
        return True, 1.0, {'error': 'too_short_tail', 'n_nodes': len(degrees)}
    
    # Линейная регрессия в лог-логе
    slope, intercept, r_value, p_value, std_err = linregress(tail_ks, tail_ps)
    r_squared = r_value ** 2
    
    # ===== ВЫЧИСЛЕНИЕ МЕР ШУМА =====
    fitted = slope * tail_ks + intercept
    residuals = tail_ps - fitted
    rmse = np.sqrt(np.mean(residuals**2))
    signal_range = np.ptp(tail_ps)  # размах сигнала
    noise_ratio = rmse / (signal_range + 1e-10)
    
    # Гладкость хвоста (стабильность производной)
    if len(tail_ks) > 2:
        derivatives = np.diff(tail_ps) / np.diff(tail_ks)
        deriv_std = np.std(derivatives)
        deriv_mean = np.abs(np.mean(derivatives))
        smoothness = deriv_std / (deriv_mean + 1e-10)
    else:
        smoothness = 1.0
    
    # ===== ИТОГОВАЯ ОЦЕНКА =====
    noise_score = (1 - r_squared) * 0.5 + noise_ratio * 0.3 + min(smoothness, 1.0) * 0.2
    noise_score = min(noise_score, 1.0)
    
    # Критерий: слишком зашумлено если R² < 0.7 ИЛИ noise_score > 0.5
    is_noisy = (r_squared < 0.7) or (noise_score > 0.5)
    
    metrics = {
        'r_squared': r_squared,
        'slope': -slope,  # показатель степени (положительный)
        'noise_ratio': noise_ratio,
        'smoothness': smoothness,
        'noise_score': noise_score,
        'is_noisy': is_noisy,
        'tail_size': len(tail_ks),
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'max_degree': np.max(degrees),
        'mean_degree': np.mean(degrees),
        'degree_assortativity': nx.degree_assortativity_coefficient(G) if G.number_of_edges() > 0 else 0
    }
    
    # ===== ПОСТРОЕНИЕ ГРАФИКОВ =====
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        status = '⚠️ ШУМНАЯ' if is_noisy else '✅ ЧИСТАЯ'
        fig.suptitle(f'Проверка зашумленности: {network_name}\n' + 
                    f'R²={r_squared:.3f}, Noise Score={noise_score:.3f}, {status}', 
                    fontsize=14, fontweight='bold')
        
        # График 1: PDF в лог-лог с биннингом
        axes[0].loglog(unique_deg, p_k, 'b.', alpha=0.4, label='Исходные точки', markersize=6)
        axes[0].loglog(binned_ks, binned_ps, 'ro-', alpha=0.8, 
                      label='Лог-биннинг', markersize=5, linewidth=1.5)
        
        # Линия регрессии для хвоста
        x_fit = np.logspace(np.log10(binned_ks[tail_start]), np.log10(binned_ks[-1]), 50)
        y_fit = 10**(slope * np.log10(x_fit) + intercept)
        axes[0].loglog(x_fit, y_fit, 'g--', linewidth=2, 
                      label=f'Хвост: γ={-slope:.2f}, R²={r_squared:.3f}')
        
        axes[0].set_xlabel('Степень (k)', fontsize=12)
        axes[0].set_ylabel('P(k)', fontsize=12)
        axes[0].set_title('Распределение степеней (лог-лог)', fontsize=12)
        axes[0].legend(fontsize=9, loc='lower left')
        axes[0].grid(True, alpha=0.3)
        
        # График 2: Только хвост с регрессией
        axes[1].scatter(tail_ks, tail_ps, c='red', alpha=0.7, s=50, label='Хвост')
        axes[1].plot(tail_ks, fitted, 'g-', linewidth=2, label=f'Регрессия, R²={r_squared:.3f}')
        
        # Доверительный интервал (1.96 * std_err)
        ci = 1.96 * std_err
        axes[1].fill_between(tail_ks, fitted - ci, fitted + ci, alpha=0.2, color='green')
        
        axes[1].set_xlabel('log10(Степень)', fontsize=12)
        axes[1].set_ylabel('log10(P(k))', fontsize=12)
        axes[1].set_title('Анализ хвоста', fontsize=12)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        # График 3: Остатки
        axes[2].scatter(tail_ks, residuals, alpha=0.7, s=40, color='purple')
        axes[2].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[2].set_xlabel('log10(Степень)', fontsize=12)
        axes[2].set_ylabel('Остатки', fontsize=12)
        axes[2].set_title('Остатки регрессии', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        # Информационное поле
        info_text = (f'R² = {r_squared:.4f}\n'
                    f'γ = {-slope:.3f}\n'
                    f'RMSE = {rmse:.4f}\n'
                    f'Noise Score = {noise_score:.3f}\n'
                    f'|V| = {G.number_of_nodes():,}\n'
                    f'|E| = {G.number_of_edges():,}\n'
                    f'k_max = {np.max(degrees)}')
        axes[2].text(0.05, 0.95, info_text, transform=axes[2].transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            safe_name = re.sub(r'[^\w\-_]', '_', network_name)[:50]
            save_path = os.path.join(output_dir, f'noise_check_{safe_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    return is_noisy, noise_score, metrics


# ========================
# 3. ПАРТИЙНАЯ ПРОВЕРКА ВСЕХ СЕТЕЙ В ПАПКЕ
# ========================

def check_all_networks(folder_path, output_dir='noise_reports', max_networks=None, verbose=True):
    """
    Проверяет все сети в папке на зашумленность.
    
    Возвращает:
    - df: DataFrame со всеми метриками
    - clean_networks: список имен чистых сетей
    - noisy_networks: список имен шумных сетей
    """
    import pandas as pd
    
    # Поиск файлов - ДОБАВЛЕН .edge
    extensions = ['.gml', '.txt', '.edges', '.mtx', '.edge']
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                all_files.append(os.path.join(root, f))
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"ПРОВЕРКА ЗАШУМЛЕННОСТИ СЕТЕЙ")
        print(f"{'='*70}")
        print(f"Найдено файлов: {len(all_files)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    clean_networks = []
    noisy_networks = []
    errors = []
    
    for idx, filepath in enumerate(all_files):
        if max_networks and idx >= max_networks:
            break
        
        filename = os.path.basename(filepath)
        
        if verbose:
            print(f"\n  [{idx+1}/{min(len(all_files), max_networks or len(all_files))}] {filename}")
        
        try:
            G = load_network(filepath)
        except Exception as e:
            if verbose:
                print(f"    ❌ Ошибка загрузки: {e}")
            errors.append({'filename': filename, 'error': str(e)})
            continue
        
        if G is None:
            errors.append({'filename': filename, 'error': 'G is None'})
            continue
        
        # Проверка на пустую сеть
        if G.number_of_nodes() < 10 or G.number_of_edges() == 0:
            if verbose:
                print(f"    ⚠️ Слишком маленькая сеть (V={G.number_of_nodes()}, E={G.number_of_edges()})")
            errors.append({'filename': filename, 'error': 'too_small'})
            continue
        
        # Берем гигантскую компоненту
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc)
            if verbose:
                print(f"    Взята гигантская компонента: {len(largest_cc)} вершин")
        
        # Вывод информации о размере сети
        if verbose:
            print(f"    |V| = {G.number_of_nodes():,}, |E| = {G.number_of_edges():,}")
        
        # Проверка зашумленности
        is_noisy, noise_score, metrics = check_degree_noise(
            G, filename, output_dir, plot=True
        )
        
        # Проверяем, не вернулась ли ошибка
        if 'error' in metrics:
            if verbose:
                print(f"    ⚠️ Ошибка: {metrics['error']}")
            errors.append({'filename': filename, 'error': metrics['error']})
            continue
        
        metrics['filename'] = filename
        metrics['is_noisy'] = is_noisy
        metrics['noise_score'] = noise_score
        results.append(metrics)
        
        if is_noisy:
            noisy_networks.append(filename)
            if verbose:
                print(f"    ⚠️ ШУМНАЯ (score={noise_score:.3f}, R²={metrics['r_squared']:.3f})")
        else:
            clean_networks.append(filename)
            if verbose:
                print(f"    ✅ ЧИСТАЯ (score={noise_score:.3f}, R²={metrics['r_squared']:.3f})")
    
    # Создаем DataFrame
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        df.to_csv(os.path.join(output_dir, 'noise_report.csv'), index=False)
        
        if verbose:
            print(f"\n{'='*70}")
            print("ИТОГОВАЯ СТАТИСТИКА")
            print(f"{'='*70}")
            print(f"  Всего проверено: {len(df)}")
            print(f"  Чистых сетей: {len(clean_networks)} ({len(clean_networks)/len(df)*100:.1f}%)")
            print(f"  Шумных сетей: {len(noisy_networks)} ({len(noisy_networks)/len(df)*100:.1f}%)")
            if errors:
                print(f"  Ошибок: {len(errors)}")
            print(f"  Средний R²: {df['r_squared'].mean():.3f} ± {df['r_squared'].std():.3f}")
            print(f"  Средний Noise Score: {df['noise_score'].mean():.3f} ± {df['noise_score'].std():.3f}")
            print(f"  Средний γ (показатель степени): {df['slope'].mean():.3f} ± {df['slope'].std():.3f}")
            
            if len(noisy_networks) > 0:
                print(f"\n  Шумные сети:")
                for n in noisy_networks[:10]:
                    print(f"    - {n}")
                if len(noisy_networks) > 10:
                    print(f"    ... и еще {len(noisy_networks)-10}")
            
            if len(clean_networks) > 0:
                print(f"\n  Чистые сети (топ-10):")
                for n in clean_networks[:10]:
                    print(f"    - {n}")
                if len(clean_networks) > 10:
                    print(f"    ... и еще {len(clean_networks)-10}")
            
            print(f"\n  📊 Отчет сохранен: {os.path.join(output_dir, 'noise_report.csv')}")
    
    return df, clean_networks, noisy_networks


# ========================
# 4. БЫСТРАЯ ПРОВЕРКА ОДНОЙ СЕТИ
# ========================

def quick_check(filepath, output_dir='.'):
    """
    Быстрая проверка одной сети.
    """
    G = load_network(filepath)
    filename = os.path.basename(filepath)
    
    if G is None:
        print("❌ Не удалось загрузить сеть")
        return None
    
    if G.number_of_nodes() < 10:
        print("❌ Сеть слишком маленькая")
        return None
    
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc)
        print(f"Взята гигантская компонента: {len(largest_cc)} вершин")
    
    print(f"\n📊 Размер сети:")
    print(f"  |V| = {G.number_of_nodes():,}")
    print(f"  |E| = {G.number_of_edges():,}")
    
    is_noisy, noise_score, metrics = check_degree_noise(
        G, filename, output_dir, plot=True
    )
    
    if 'error' in metrics:
        print(f"❌ Ошибка: {metrics['error']}")
        return None
    
    print(f"\n{'='*50}")
    print(f"РЕЗУЛЬТАТ ПРОВЕРКИ: {filename}")
    print(f"{'='*50}")
    print(f"  Вершин: {metrics['n_nodes']:,}")
    print(f"  Рёбер: {metrics['n_edges']:,}")
    print(f"  Max степень: {metrics['max_degree']}")
    print(f"  R² хвоста: {metrics['r_squared']:.4f}")
    print(f"  Показатель степени γ: {metrics['slope']:.3f}")
    print(f"  Noise Score: {metrics['noise_score']:.4f}")
    print(f"  Статус: {'⚠️ ШУМНАЯ' if is_noisy else '✅ ЧИСТАЯ'}")
    
    return is_noisy, noise_score, metrics


# ========================
# 5. ЗАПУСК
# ========================

if __name__ == "__main__":
    # Проверка всех сетей в папке
    df, clean, noisy = check_all_networks(
        folder_path="C:/Proj_edge_distr/cite_A/",
        output_dir="C:/Proj_edge_distr/noise_reports/",
        max_networks=None
    )