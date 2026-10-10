import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')

# ========================
# 1. ВЫЧИСЛЕНИЕ РЁБЕРНЫХ ГРАДИЕНТОВ
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
        print(f"    Ошибка при вычислении {centrality_type}: {e}")
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
        print(f"    Ошибка при вычислении градиентов: {e}")
        return None


# ========================
# 2. ПОСТРОЕНИЕ МАТРИЦЫ ГРАФИКОВ ДЛЯ ER МОДЕЛИ (110 графиков)
# ========================

def plot_er_matrix(n_nodes=500, p_values=None, centrality='pagerank', 
                   output_dir='.', bins=500, figsize=(30, 40)):
    """
    Строит матрицу графиков распределений градиентов для модели ER.
    
    Расположение (слева направо p УМЕНЬШАЮТСЯ):
        p_high  p_med  p_low
        p_high  p_med  p_low
        p_high  p_med  p_low
    """
    
    if p_values is None:
        # Генерируем 110 значений p в логарифмическом масштабе от 0.001 до 0.95
        p_values = np.logspace(-3, 0, 110)  # от 0.001 до ~0.95
    
    # Сортируем p_values по ВОЗРАСТАНИЮ (от малого к большому)
    p_values = sorted(p_values)
    
    # Определяем размеры матрицы (10 колонок, 11 строк = 110 графиков)
    n_cols = 10
    n_rows = int(np.ceil(len(p_values) / n_cols))
    
    print(f"Всего графиков: {len(p_values)}")
    print(f"Строк: {n_rows}, Колонок: {n_cols}")
    
    # Создаем фигуру с субплотами
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Если только одна строка, превращаем axes в 2D массив
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Данные для хранения skewness
    skewness_dict = {}
    
    # Для каждого p_value создаем график
    for idx, p in enumerate(p_values):
        # Вычисляем позицию в матрице
        # Строка: idx // n_cols (снизу вверх)
        row = n_rows - 1 - (idx // n_cols)
        # Колонка: зеркально отражаем, чтобы p УМЕНЬШАЛИСЬ слева направо
        col = n_cols - 1 - (idx % n_cols)
        
        ax = axes[row, col]
        
        # Показываем прогресс
        if idx % 10 == 0:
            print(f"  Обработка {idx+1}/{len(p_values)} (p={p:.5f})...")
        
        try:
            # Генерируем ER граф
            G = nx.erdos_renyi_graph(n=n_nodes, p=p)
            
            # Проверяем, что граф валидный
            if G.number_of_nodes() < 10 or G.number_of_edges() == 0:
                ax.text(0.5, 0.5, f'p={p:.4f}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=6)
                ax.set_xticks([])
                ax.set_yticks([])
                skewness_dict[p] = np.nan
                continue
            
            # Вычисляем градиенты
            gradients = compute_edge_gradients(G, centrality)
            
            if gradients is None:
                ax.text(0.5, 0.5, f'p={p:.4f}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=6)
                ax.set_xticks([])
                ax.set_yticks([])
                skewness_dict[p] = np.nan
                continue
            
            # Фильтруем нулевые значения
            grad = gradients[gradients > 1e-12]
            
            if len(grad) < 10:
                ax.text(0.5, 0.5, f'p={p:.4f}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=6)
                ax.set_xticks([])
                ax.set_yticks([])
                skewness_dict[p] = np.nan
                continue
            
            # Вычисляем skewness
            skewness_val = skew(grad)
            skewness_dict[p] = skewness_val
            
            # Строим гистограмму
            ax.hist(grad, bins=bins, density=True, alpha=0.7, 
                   color='steelblue', edgecolor='black', linewidth=0.2)
            
            # Добавляем информацию на график (только p и skewness)
            ax.text(0.05, 0.95, f'p={p:.4f}\nξ={skewness_val:.2f}', 
                   transform=ax.transAxes, fontsize=5, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray', pad=0.5))
            
            # Настройка осей
            if len(grad) > 0:
                x_max = np.percentile(grad, 95)
                if x_max < 0.1:
                    x_max = np.max(grad) * 1.1
                ax.set_xlim([0, x_max * 1.05])
            else:
                ax.set_xlim([0, 5])
            
            # Убираем все метки для экономии места
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Убираем рамки для экономии места
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
        except Exception as e:
            print(f"  p={p:.5f}: Ошибка - {e}")
            ax.text(0.5, 0.5, f'p={p:.4f}\nОшибка', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
            skewness_dict[p] = np.nan
    
    # Скрываем пустые подграфики
    for idx in range(len(p_values), n_rows * n_cols):
        row = n_rows - 1 - (idx // n_cols)
        col = n_cols - 1 - (idx % n_cols)
        axes[row, col].axis('off')
    
    # Добавляем общие подписи (увеличиваем размер)
    fig.text(0.5, 0.01, 'Градиент', ha='center', fontsize=18, fontweight='bold')
    fig.text(0.01, 0.5, 'Плотность', va='center', rotation='vertical', fontsize=18, fontweight='bold')
    
    # Заголовок
    fig.suptitle(f'Распределение градиентов для ER модели (110 значений p)\nn={n_nodes}, центральность: {centrality}', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    
    # Сохраняем
    save_path = os.path.join(output_dir, f'er_matrix_{centrality}_n{n_nodes}_110.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nМатрица сохранена: {save_path}")
    plt.close()
    
    # Сохраняем skewness
    df_skewness = pd.DataFrame([skewness_dict]).T
    df_skewness.columns = ['skewness']
    df_skewness.index.name = 'p'
    csv_path = os.path.join(output_dir, f'er_skewness_{centrality}_n{n_nodes}_110.csv')
    df_skewness.to_csv(csv_path)
    print(f"Данные skewness сохранены: {csv_path}")
    
    return df_skewness


# ========================
# 3. ПОСТРОЕНИЕ ГРАФИКА ЗАВИСИМОСТИ SKEWNESS ОТ p
# ========================

def plot_skewness_vs_p(skewness_df, output_dir='.', centrality='pagerank'):
    """
    Строит график зависимости skewness от p.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Извлекаем данные
    p_values = skewness_df.index.values.astype(float)
    skew_values = skewness_df['skewness'].values
    
    # Сортируем по p
    sorted_idx = np.argsort(p_values)
    p_sorted = p_values[sorted_idx]
    skew_sorted = skew_values[sorted_idx]
    
    # Строим график
    ax.plot(p_sorted, skew_sorted, 'o-', color='steelblue', linewidth=2, markersize=4, alpha=0.7)
    
    # Настройка
    ax.set_xlabel('Вероятность p', fontsize=14, fontweight='bold')
    ax.set_ylabel('Skewness (ξ)', fontsize=14, fontweight='bold')
    ax.set_title(f'Зависимость skewness от p для ER модели\n(центральность: {centrality})', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Логарифмическая шкала для p
    ax.set_xscale('log')
    
    # Добавляем горизонтальную линию на y=0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Добавляем вертикальные линии для основных значений p
    for p_val in [0.001, 0.01, 0.1, 0.5]:
        ax.axvline(x=p_val, color='red', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'er_skewness_vs_p_{centrality}_110.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"График зависимости сохранен: {save_path}")
    plt.close()


# ========================
# 4. ПОСТРОЕНИЕ ТЕПЛОВОЙ КАРТЫ SKEWNESS (ОПЦИОНАЛЬНО)
# ========================

def plot_skewness_heatmap(skewness_df, output_dir='.', centrality='pagerank'):
    """
    Строит тепловую карту для skewness (сгруппированную по диапазонам p).
    """
    # Группируем по диапазонам p
    p_values = skewness_df.index.values.astype(float)
    skew_values = skewness_df['skewness'].values
    
    # Создаем категории для p
    p_categories = pd.cut(p_values, bins=20)
    
    # Группируем и вычисляем среднее
    df_grouped = pd.DataFrame({'p': p_values, 'skewness': skew_values})
    df_grouped['p_cat'] = p_categories
    grouped_mean = df_grouped.groupby('p_cat')['skewness'].mean()
    
    # Строим тепловую карту
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Преобразуем в матрицу (1 строка, 20 колонок)
    heatmap_data = grouped_mean.values.reshape(1, -1)
    
    im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=3)
    
    # Настройка осей
    ax.set_xticks(np.arange(len(grouped_mean)))
    ax.set_xticklabels([f'{cat.left:.3f}' for cat in grouped_mean.index], rotation=45, ha='right', fontsize=8)
    ax.set_yticks([])
    
    # Добавляем значения
    for i, val in enumerate(grouped_mean.values):
        color = 'white' if val > 1.5 else 'black'
        ax.text(i, 0, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
    
    ax.set_xlabel('Диапазоны p', fontsize=12, fontweight='bold')
    ax.set_title(f'Средняя skewness по диапазонам p\n(центральность: {centrality})', 
                fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Skewness (ξ)', fontsize=12)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'er_skewness_heatmap_{centrality}_110.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Тепловая карта сохранена: {save_path}")
    plt.close()


# ========================
# 5. ОСНОВНАЯ ФУНКЦИЯ
# ========================

def run_er_analysis(n_nodes=500, 
                    p_values=None, 
                    centrality='pagerank',
                    output_dir='C:/Proj_edge_distr/results_er'):
    """
    Запускает полный анализ ER модели (110 графиков).
    """
    # Создаем директорию для результатов
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Результаты будут сохранены в: {output_dir}")
    
    print("\n" + "="*60)
    print("АНАЛИЗ ER МОДЕЛИ (110 графиков)")
    print("="*60)
    
    # Строим матрицу графиков
    skewness_df = plot_er_matrix(
        n_nodes=n_nodes,
        p_values=p_values,
        centrality=centrality,
        output_dir=output_dir
    )
    
    # Строим график зависимости skewness от p
    plot_skewness_vs_p(skewness_df, output_dir, centrality)
    
    # Строим тепловую карту
    plot_skewness_heatmap(skewness_df, output_dir, centrality)
    
    print(f"\n💚 Все результаты сохранены в папку: {output_dir}")
    
    return skewness_df


# ========================
# 6. ЗАПУСК
# ========================

if __name__ == "__main__":
    # Параметры для анализа
    N_NODES = 500  # фиксированное число вершин
    
    # Генерируем 110 значений p в логарифмическом масштабе
    # от 0.001 до 0.95
    P_VALUES = np.logspace(-3, -0.02, 110)  # от 0.001 до ~0.95
    
    # Запускаем анализ
    skewness_df = run_er_analysis(
        n_nodes=N_NODES,
        p_values=P_VALUES,
        centrality='pagerank',
        output_dir='C:/Proj_edge_distr/results_er'
    )
    
    # Выводим первые и последние результаты
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ (skewness):")
    print("="*60)
    print(f"Всего значений: {len(skewness_df)}")
    print("\nПервые 10 значений:")
    print(skewness_df.head(10).round(3))
    print("\nПоследние 10 значений:")
    print(skewness_df.tail(10).round(3))