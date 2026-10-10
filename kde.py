import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import skew
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms.community import (
    greedy_modularity_communities,
    label_propagation_communities,
)
import warnings
warnings.filterwarnings('ignore')

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("⚠ python-louvain не установлен — Louvain будет недоступен")


# ========================
# 0. СОХРАНЕНИЕ ТАБЛИЦ
# ========================

def _save_table(df, path_no_ext):
    csv_path = path_no_ext + '.csv'
    df.to_csv(csv_path, index=False, sep=';', encoding='utf-8-sig')
    try:
        xlsx_path = path_no_ext + '.xlsx'
        df.to_excel(xlsx_path, index=False)
    except Exception:
        pass


# ========================
# 1. ЗАГРУЗКА СЕТЕЙ
# ========================

def validate_network(G, min_nodes=10):
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

def _compute_centrality(G, centrality_type):
    if centrality_type == 'pagerank':
        return nx.pagerank(G, max_iter=1000, tol=1e-8)
    elif centrality_type == 'betweenness':
        if G.number_of_nodes() > 5000:
            return nx.betweenness_centrality(G, k=int(np.sqrt(G.number_of_nodes())))
        return nx.betweenness_centrality(G)
    elif centrality_type == 'closeness':
        return nx.closeness_centrality(G)
    elif centrality_type == 'eigenvector':
        return nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
    elif centrality_type == 'kcore':
        return nx.core_number(G)
    elif centrality_type == 'degree':
        return dict(G.degree())
    else:
        raise ValueError(f"Неизвестный тип центральности: {centrality_type}")


def compute_edge_gradients_with_edges(G, centrality_type):
    if G.number_of_nodes() < 10:
        return None, None
    try:
        centrality_dict = _compute_centrality(G, centrality_type)
    except Exception as e:
        print(f"    👿 Ошибка при вычислении {centrality_type}: {e}")
        return None, None
    G_dir = G.to_directed()
    edges_to_remove = [(u, v) for u, v in G_dir.edges() if u > v]
    G_dir.remove_edges_from(edges_to_remove)
    edge_list = list(G_dir.edges())
    try:
        B = nx.incidence_matrix(G_dir, oriented=True).toarray()
        f = np.array([centrality_dict[node] for node in G_dir.nodes()])
        gradients = B.T.dot(f)
        abs_gradients = np.abs(gradients)
        mean_val = np.mean(abs_gradients)
        if mean_val > 0:
            abs_gradients = abs_gradients / mean_val
        return abs_gradients, edge_list
    except Exception as e:
        print(f"    👿 Ошибка при вычислении градиентов: {e}")
        return None, None


# ========================
# 3. РАЗДЕЛЕНИЕ РЁБЕР НА ГРУППЫ
# ========================

def _auto_bins(g):
    n = len(g)
    if n < 2:
        return 10
    iqr = np.subtract(*np.percentile(g, [75, 25]))
    if iqr <= 0:
        iqr = np.std(g)
    if iqr <= 0:
        return 10
    h = 2 * iqr / (n ** (1 / 3))
    if h <= 0:
        return 10
    return int(np.clip((g.max() - g.min()) / h, 10, 1000))


def assign_edge_groups_peaks(gradients, bins=None,
                             smooth_window=3,
                             prominence_frac=0.1,
                             distance=None,
                             height_frac=0.0,
                             verbose=False):
    g = np.asarray(gradients).ravel()
    g = g[np.isfinite(g)]

    if bins is None:
        bins = _auto_bins(g)

    counts, bin_edges = np.histogram(g, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if smooth_window and smooth_window > 1:
        counts_s = uniform_filter1d(counts.astype(float), size=smooth_window)
    else:
        counts_s = counts.astype(float)

    if distance is None:
        distance = max(1, bins // 50)

    counts_padded = np.concatenate([[0.0], counts_s, [0.0]])

    kwargs = dict(
        prominence=prominence_frac * counts_padded.max(),
        distance=distance,
    )
    if height_frac and height_frac > 0:
        kwargs['height'] = height_frac * counts_padded.max()

    peaks_padded, props = find_peaks(counts_padded, **kwargs)
    peaks = peaks_padded - 1
    peaks = peaks[(peaks >= 0) & (peaks < len(counts_s))]

    info = {
        'counts': counts,
        'counts_smooth': counts_s,
        'bin_centers': bin_centers,
        'peaks': peaks,
        'bins': bins,
        'distance': distance,
    }

    if len(peaks) == 0:
        info['peak_values'] = np.array([])
        info['peak_heights'] = np.array([])
        if verbose:
            print(f"    Peaks: bins={bins}, smooth={smooth_window}, "
                  f"пиков=0 → одна группа")
        return np.zeros_like(g, dtype=int), np.array([]), info

    peak_values = bin_centers[peaks]
    peak_heights = counts_s[peaks]
    info['peak_values'] = peak_values
    info['peak_heights'] = peak_heights

    labels = np.argmin(np.abs(g[:, None] - peak_values[None, :]), axis=1)

    if len(peak_values) > 1:
        boundaries = (peak_values[:-1] + peak_values[1:]) / 2
    else:
        boundaries = np.array([])

    if verbose:
        print(f"    Peaks: bins={bins}, smooth={smooth_window}, "
              f"пиков={len(peaks)}, групп={len(peak_values)}, "
              f"значения пиков={np.round(peak_values, 3)}, "
              f"высоты пиков={np.round(peak_heights, 1)}")

    return labels, boundaries, info


def assign_edge_groups(gradients, method='peaks', n_groups=None,
                       bins=None, smooth_window=3,
                       prominence_frac=0.1, distance=None, height_frac=0.0,
                       bw_method=0.01, prominence=0.01,
                       verbose=False):
    g = np.asarray(gradients).ravel()
    g = g[np.isfinite(g)]

    if method == 'peaks':
        labels, boundaries, info = assign_edge_groups_peaks(
            g, bins=bins, smooth_window=smooth_window,
            prominence_frac=prominence_frac, distance=distance,
            height_frac=height_frac, verbose=verbose
        )
        return labels, boundaries

    elif method == 'kde':
        kde = gaussian_kde(g, bw_method=bw_method)
        x = np.linspace(g.min(), g.max(), 5000)
        y = kde(x)
        peaks, _ = find_peaks(y, prominence=prominence * y.max())
        peak_values = x[peaks]
        if len(peak_values) == 0:
            return np.zeros_like(g, dtype=int), np.array([])
        labels = np.argmin(np.abs(g[:, None] - peak_values[None, :]), axis=1)
        boundaries = (peak_values[:-1] + peak_values[1:]) / 2 if len(peak_values) > 1 else np.array([])
        if verbose:
            print(f"    KDE: пиков={len(peaks)}, значения={np.round(peak_values, 3)}")
        return labels, boundaries

    elif method == 'gmm':
        if n_groups is None:
            X = g.reshape(-1, 1)
            bics = []
            for k in range(1, 15):
                try:
                    gm = GaussianMixture(n_components=k, covariance_type='full',
                                         random_state=0).fit(X)
                    bics.append(gm.bic(X))
                except Exception:
                    bics.append(np.inf)
            n_groups = int(np.argmin(bics)) + 1
            if verbose:
                print(f"    GMM: BIC выбрал {n_groups} компонент")
        X = g.reshape(-1, 1)
        gm = GaussianMixture(n_components=n_groups, covariance_type='full',
                             random_state=0).fit(X)
        labels = gm.predict(X)
        means = gm.means_.ravel()
        order = np.argsort(means)
        remap = {old: new for new, old in enumerate(order)}
        labels = np.array([remap[l] for l in labels])
        sorted_means = means[order]
        boundaries = (sorted_means[:-1] + sorted_means[1:]) / 2
        return labels, boundaries

    elif method == 'jenks':
        try:
            import jenkspy
        except ImportError:
            raise ImportError("Установите jenkspy: pip install jenkspy")
        if n_groups is None:
            n_groups = 4
        breaks = jenkspy.jenks_breaks(g, n_classes=n_groups)
        boundaries = np.array(breaks[1:-1])
        labels = np.digitize(g, boundaries)
        return labels, boundaries

    else:
        raise ValueError(f"Неизвестный метод: {method}")


def build_node_group_table(G, edge_list, edge_labels, n_groups=None,
                           add_totals=True):
    if n_groups is None:
        n_groups = int(edge_labels.max()) + 1

    counts = {node: np.zeros(n_groups, dtype=int) for node in G.nodes()}
    for (u, v), lab in zip(edge_list, edge_labels):
        counts[u][lab] += 1
        counts[v][lab] += 1

    rows = []
    for node in G.nodes():
        row = {'node': node}
        for k in range(n_groups):
            row[f'group_{k}'] = int(counts[node][k])
        row['total'] = int(counts[node].sum())
        rows.append(row)

    df = pd.DataFrame(rows)

    if add_totals:
        group_cols = [f'group_{k}' for k in range(n_groups)]
        totals = {'node': 'TOTAL'}
        for col in group_cols:
            totals[col] = int(df[col].sum())
        totals['total'] = int(df['total'].sum())
        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    return df


# ========================
# 3b. АГРЕГАЦИЯ ГРАДИЕНТОВ НА ВЕРШИНЫ
# ========================

def compute_node_dominant_group(edge_list, edge_labels, n_groups,
                                 rule='majority', group_rank=None):
    node_counts = {}
    for (u, v), lab in zip(edge_list, edge_labels):
        if u not in node_counts:
            node_counts[u] = np.zeros(n_groups, dtype=int)
        if v not in node_counts:
            node_counts[v] = np.zeros(n_groups, dtype=int)
        node_counts[u][lab] += 1
        node_counts[v][lab] += 1

    total_per_group = np.zeros(n_groups, dtype=int)
    for lab in edge_labels:
        total_per_group[lab] += 1

    node_group = {}
    for node, cnt in node_counts.items():
        if cnt.sum() == 0:
            node_group[node] = -1
            continue

        if rule == 'majority':
            node_group[node] = int(np.argmax(cnt))
        elif rule == 'highest_rank':
            if group_rank is None:
                raise ValueError("rule='highest_rank' требует group_rank")
            present = np.where(cnt > 0)[0]
            best = min(present, key=lambda g: group_rank[g])
            node_group[node] = int(best)
        elif rule == 'lowest_rank':
            if group_rank is None:
                raise ValueError("rule='lowest_rank' требует group_rank")
            present = np.where(cnt > 0)[0]
            best = max(present, key=lambda g: group_rank[g])
            node_group[node] = int(best)
        elif rule == 'most_rare':
            present = np.where(cnt > 0)[0]
            best = min(present, key=lambda g: total_per_group[g])
            node_group[node] = int(best)
        else:
            raise ValueError(f"Неизвестное правило: {rule}")

    return node_group


def compute_node_group_profile(edge_list, edge_labels, gradients,
                                n_groups, metric='mean_gradient'):
    node_grads = {}
    node_labels = {}
    for (u, v), g, lab in zip(edge_list, gradients, edge_labels):
        node_grads.setdefault(u, []).append(g)
        node_grads.setdefault(v, []).append(g)
        node_labels.setdefault(u, []).append(lab)
        node_labels.setdefault(v, []).append(lab)

    node_score = {}
    for node in node_grads:
        grads = np.asarray(node_grads[node])
        labs = np.asarray(node_labels[node])

        if metric == 'mean_gradient':
            node_score[node] = float(grads.mean())
        elif metric == 'max_gradient':
            node_score[node] = float(grads.max())
        elif metric == 'entropy':
            p = np.bincount(labs, minlength=n_groups).astype(float)
            p = p / p.sum() if p.sum() > 0 else p
            p = p[p > 0]
            node_score[node] = float(-(p * np.log(p)).sum())
        else:
            raise ValueError(f"Неизвестная метрика: {metric}")

    return node_score


def compute_node_mean_group(edge_list, edge_labels):
    """
    Для каждой вершины: средний номер группы по её рёбрам.

    score(v) = Σ (метка группы ребра) / deg(v)

    Чем меньше score — тем ближе вершина к группе 0 (красная).
    """
    node_sum = {}
    node_cnt = {}
    for (u, v), lab in zip(edge_list, edge_labels):
        node_sum[u] = node_sum.get(u, 0) + int(lab)
        node_sum[v] = node_sum.get(v, 0) + int(lab)
        node_cnt[u] = node_cnt.get(u, 0) + 1
        node_cnt[v] = node_cnt.get(v, 0) + 1

    node_score = {}
    for node in node_sum:
        if node_cnt[node] > 0:
            node_score[node] = node_sum[node] / node_cnt[node]
        else:
            node_score[node] = np.nan
    return node_score


# ========================
# 3c. ВЫДЕЛЕНИЕ СООБЩЕСТВ
# ========================

def detect_communities_louvain(G, seed=0):
    if not HAS_LOUVAIN:
        return None
    Gu = G.to_undirected() if G.is_directed() else G
    return community_louvain.best_partition(Gu, random_state=seed)


def detect_communities_greedy(G):
    communities = greedy_modularity_communities(G)
    partition = {}
    for cid, nodes in enumerate(communities):
        for n in nodes:
            partition[n] = cid
    return partition


def detect_communities_label_prop(G, seed=0):
    communities = label_propagation_communities(G)
    partition = {}
    for cid, nodes in enumerate(communities):
        for n in nodes:
            partition[n] = cid
    return partition


def compare_partitions(labels_a, labels_b):
    nmi = normalized_mutual_info_score(labels_a, labels_b)
    ari = adjusted_rand_score(labels_a, labels_b)
    return nmi, ari


def internal_edge_ratio_per_group(edge_list, edge_labels, partition,
                                   n_groups):
    rows = []
    for g in range(n_groups):
        edges_g = [e for e, lab in zip(edge_list, edge_labels) if lab == g]
        n_int = 0
        n_ext = 0
        for (u, v) in edges_g:
            cu = partition.get(u, -1)
            cv = partition.get(v, -1)
            if cu == cv and cu != -1:
                n_int += 1
            else:
                n_ext += 1
        total = n_int + n_ext
        rows.append({
            'group': g,
            'n_edges': total,
            'n_internal': n_int,
            'n_external': n_ext,
            'frac_internal': (n_int / total) if total > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ========================
# 4. ВИЗУАЛИЗАЦИЯ
# ========================

RANK_COLORS = [
    '#e41a1c', '#ff7f00', '#ffd700', '#4daf4a', '#00bfff',
    '#377eb8', '#984ea3', '#a65628', '#f781bf', '#999999',
]


def _get_rank_colors(n_groups):
    colors = []
    for i in range(n_groups):
        if i < len(RANK_COLORS):
            colors.append(RANK_COLORS[i])
        else:
            gray = max(0.3, 0.7 - 0.03 * (i - len(RANK_COLORS)))
            colors.append((gray, gray, gray))
    return colors


def _lighten_color(color, amount=0.6):
    import matplotlib.colors as mcolors
    c = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    return tuple(c + (white - c) * amount)


def _compute_layout(G, layout, seed):
    if layout == 'spring':
        return nx.spring_layout(G, seed=seed, k=1.0 / np.sqrt(G.number_of_nodes()))
    elif layout == 'kamada_kawai':
        return nx.kamada_kawai_layout(G)
    elif layout == 'circular':
        return nx.circular_layout(G)
    else:
        raise ValueError(f"Неизвестный layout: {layout}")


def _draw_edges_by_groups(G, pos, ax, edge_list, edge_labels, n_groups,
                          group_colors,
                          edge_width=0.5, edge_alpha=0.5):
    edges_by_group = {g: [] for g in range(n_groups)}
    for (u, v), lab in zip(edge_list, edge_labels):
        edges_by_group[lab].append((u, v))

    for g in range(n_groups - 1, -1, -1):
        if not edges_by_group[g]:
            continue
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edges_by_group[g],
            edge_color=[group_colors[g]],
            width=edge_width,
            alpha=edge_alpha,
        )


def _draw_single_group_edges(G, pos, ax, edge_list, edge_labels, target_group,
                              group_color, edge_width=1.0, edge_alpha=0.9,
                              background_alpha=0.05):
    if background_alpha > 0:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edge_list,
            edge_color='lightgray',
            width=0.3, alpha=background_alpha,
        )
    edges_target = [e for e, lab in zip(edge_list, edge_labels)
                    if lab == target_group]
    if edges_target:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edges_target,
            edge_color=[group_color],
            width=edge_width, alpha=edge_alpha,
        )


def plot_histogram_with_groups(gradients, labels, boundaries,
                               network_name, centrality_type,
                               bins=300, info=None, save_path=None):
    grad = gradients[gradients > 1e-12]
    if len(grad) < 50:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(grad, bins=bins, density=True, alpha=0.5,
            color='steelblue', edgecolor='black', linewidth=0.4)

    if info is not None and 'bin_centers' in info and 'peaks' in info:
        bc = info['bin_centers']
        pk = info['peaks']
        counts = info['counts']
        if len(counts) > 0:
            total = counts.sum()
            binw = bc[1] - bc[0] if len(bc) > 1 else 1.0
            dens = counts / (total * binw) if total > 0 else counts
            ax.plot(bc, dens, color='black', linewidth=1, alpha=0.7, label='сглаж. гист.')
            ax.plot(bc[pk], dens[pk], 'rv', markersize=10,
                    label=f'пики ({len(pk)})')

    for b in boundaries:
        ax.axvline(b, color='red', linestyle='--', linewidth=1.0, alpha=0.7)

    n_groups_actual = int(labels.max()) + 1
    ax.set_xlabel('Градиент', fontsize=12)
    ax.set_ylabel('Плотность', fontsize=12)
    ax.set_title(f'{network_name} ({centrality_type})\n'
                 f'Групп: {n_groups_actual}', fontsize=14)
    ax.grid(True, alpha=0.3)
    if info is not None and 'peaks' in info and len(info['peaks']) > 0:
        ax.legend(loc='upper right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_network_with_edge_groups(
    G, edge_list, edge_labels, gradients,
    network_name, centrality_type,
    peak_values=None, peak_heights=None,
    layout='spring', node_size=15, edge_width=0.8,
    edge_alpha=0.6, figsize=(14, 14),
    seed=0, save_path=None
):
    n_groups = int(edge_labels.max()) + 1
    pos = _compute_layout(G, layout, seed)
    rank_colors = _get_rank_colors(n_groups)
    group_colors = {g: rank_colors[g] for g in range(n_groups)}

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                           node_color='lightgray', edgecolors='gray',
                           linewidths=0.3)

    edges_by_group = {k: [] for k in range(n_groups)}
    for (u, v), lab in zip(edge_list, edge_labels):
        edges_by_group[lab].append((u, v))

    for g in range(n_groups - 1, -1, -1):
        if not edges_by_group[g]:
            continue
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges_by_group[g],
                               edge_color=[group_colors[g]],
                               width=edge_width, alpha=edge_alpha)

    legend_handles = []
    for g in range(n_groups):
        grads_g = gradients[edge_labels == g]
        n_edges_g = len(grads_g)
        if n_edges_g > 0:
            lo, hi = grads_g.min(), grads_g.max()
            if peak_values is not None and g < len(peak_values):
                label = (f"G{g}: пик≈{peak_values[g]:.2f} "
                         f"[{lo:.2f}–{hi:.2f}] n={n_edges_g}")
            else:
                label = f"G{g}  n={n_edges_g}"
        else:
            label = f"G{g}  (пусто)"
        handle = plt.Line2D([0], [0], color=group_colors[g],
                            linewidth=2.5, label=label)
        legend_handles.append(handle)

    ax.legend(handles=legend_handles, loc='upper right', fontsize=8,
              framealpha=0.9, ncol=1, title="Группы рёбер")

    ax.set_title(f"{network_name} | n_узлов={G.number_of_nodes()}, "
                 f"n_рёбер={len(edge_list)}, групп={n_groups}",
                 fontsize=11, pad=6)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=0.94, bottom=0)
    plt.tight_layout(pad=0.2)

    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_network_with_communities(
    G, partition,
    edge_list, edge_labels, n_groups,
    network_name, centrality_type,
    method_name='Louvain',
    layout='spring', seed=0, figsize=(14, 14),
    node_size=45, edge_width=0.6, edge_alpha=0.55,
    edge_lighten=0.6, save_path=None
):
    pos = _compute_layout(G, layout, seed)

    rank_colors = _get_rank_colors(n_groups)
    group_colors = {g: rank_colors[g] for g in range(n_groups)}
    edge_colors = {g: _lighten_color(group_colors[g], amount=edge_lighten)
                   for g in range(n_groups)}

    nodes = list(G.nodes())
    comm_ids = [partition.get(n, -1) for n in nodes]
    unique_comms = sorted(set(comm_ids))
    n_comms = len(unique_comms)
    cmap = plt.get_cmap('tab20', max(n_comms, 2))
    comm_colors = {c: cmap(i) for i, c in enumerate(unique_comms)}

    fig, ax = plt.subplots(figsize=figsize)

    if edge_list is not None and edge_labels is not None:
        _draw_edges_by_groups(G, pos, ax, edge_list, edge_labels, n_groups,
                              edge_colors,
                              edge_width=edge_width, edge_alpha=edge_alpha)
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray',
                               width=edge_width, alpha=edge_alpha)

    for c in unique_comms:
        nodes_c = [n for n, cc in zip(nodes, comm_ids) if cc == c]
        if not nodes_c:
            continue
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=nodes_c,
            node_size=node_size,
            node_color=[comm_colors[c]],
            edgecolors='black', linewidths=0.3,
        )

    ax.set_title(f"{network_name} | {method_name} — {n_comms} сообществ\n"
                 f"(рёбра — светлые, по группам градиента)",
                 fontsize=11, pad=6)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=0.94, bottom=0)
    plt.tight_layout(pad=0.2)

    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_network_single_group_communities(
    G, partition,
    edge_list, edge_labels, n_groups, target_group,
    network_name, centrality_type,
    method_name='Louvain',
    layout='spring', seed=0, figsize=(14, 14),
    node_size=45, edge_width=1.2, edge_alpha=0.9,
    background_alpha=0.05,
    save_path=None
):
    pos = _compute_layout(G, layout, seed)

    rank_colors = _get_rank_colors(n_groups)
    target_color = rank_colors[target_group]

    edges_target = [(u, v) for (u, v), lab in zip(edge_list, edge_labels)
                    if lab == target_group]

    target_nodes = set()
    for u, v in edges_target:
        target_nodes.add(u)
        target_nodes.add(v)

    comm_ids = [partition.get(n, -1) for n in target_nodes]
    unique_comms = sorted(set(comm_ids))
    n_comms = len(unique_comms)
    cmap = plt.get_cmap('tab20', max(n_comms, 2))
    comm_colors = {c: cmap(i) for i, c in enumerate(unique_comms)}

    fig, ax = plt.subplots(figsize=figsize)

    if background_alpha > 0:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edge_list,
            edge_color='lightgray',
            width=0.3, alpha=background_alpha,
        )

    if edges_target:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edges_target,
            edge_color=[target_color],
            width=edge_width, alpha=edge_alpha,
        )

    for c in unique_comms:
        nodes_c = [n for n in target_nodes if partition.get(n, -1) == c]
        if not nodes_c:
            continue
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=nodes_c,
            node_size=node_size,
            node_color=[comm_colors[c]],
            edgecolors='black', linewidths=0.3,
        )

    ax.set_title(f"{network_name} | {method_name} — {n_comms} сообществ\n"
                 f"Рёбра ТОЛЬКО группы G{target_group} "
                 f"(n={len(edges_target)}), "
                 f"вершин затронуто: {len(target_nodes)}",
                 fontsize=11, pad=6)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=0.94, bottom=0)
    plt.tight_layout(pad=0.2)

    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_network_with_mean_group_nodes(
    G, node_score,
    edge_list, edge_labels, n_groups,
    network_name, centrality_type,
    layout='spring', seed=0, figsize=(14, 14),
    node_size_min=20, node_size_max=150,
    edge_width=0.6, edge_alpha=0.55,
    edge_lighten=0.6,
    highlight_top_frac=0.05,
    save_path=None
):
    """
    Вершины раскрашены МОНОХРОМНО КРАСНЫМ по среднему номеру группы:
      - score ≈ 0 (все рёбра из группы 0)  →  НАСЫЩЕННЫЙ КРАСНЫЙ
      - score ≈ n_groups-1 (старшие группы)  →  БЛЕДНЫЙ / почти белый
    Рёбра — по своим группам, светлыми оттенками.

    Реализация: используем cmap='Reds' (0=белый, 1=тёмно-красный),
    но передаём инвертированные значения, чтобы получить
    обратную зависимость.
    """
    pos = _compute_layout(G, layout, seed)

    # --- цвета групп для рёбер ---
    rank_colors = _get_rank_colors(n_groups)
    group_colors = {g: rank_colors[g] for g in range(n_groups)}
    edge_colors = {g: _lighten_color(group_colors[g], amount=edge_lighten)
                   for g in range(n_groups)}

    # --- scores вершин (средний номер группы) ---
    nodes = list(G.nodes())
    scores = np.array([node_score.get(n, np.nan) for n in nodes])

    # --- нормировка ---
    smin = 0.0
    smax = max(1.0, float(n_groups - 1))
    scores_clipped = np.clip(scores, smin, smax)
    scores_norm = (scores_clipped - smin) / (smax - smin)  # [0, 1]

    # --- ИНВЕРСИЯ: 0 → 1 (красный), max → 0 (белый) ---
    # Именно это даёт "чем меньше score, тем краснее".
    color_values = 1.0 - scores_norm

    # --- изолированные вершины (NaN) — нейтральный серый ---
    mask_valid = ~np.isnan(scores)
    color_values[~mask_valid] = 0.0  # белый/бледный

    # --- размеры по score (чем выше score, тем крупнее) ---
    # Если хотите размер тоже "чем меньше, тем крупнее" — инвертируйте так:
    # size_values = 1.0 - scores_norm
    size_values = scores_norm
    sizes = node_size_min + size_values * (node_size_max - node_size_min)

    fig, ax = plt.subplots(figsize=figsize)

    # --- рёбра: светлые, по группам ---
    if edge_list is not None and edge_labels is not None:
        _draw_edges_by_groups(G, pos, ax, edge_list, edge_labels, n_groups,
                              edge_colors,
                              edge_width=edge_width, edge_alpha=edge_alpha)
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray',
                               width=edge_width, alpha=edge_alpha)

    # --- вершины: монохромный красный ---
    # Передаём color_values (инвертированные), vmin=0, vmax=1.
    # Теперь: color_values=1 → тёмно-красный, color_values=0 → белый.
    # А так как color_values = 1 - scores_norm, то:
    #   score = 0      → color_values = 1 → ТЁМНО-КРАСНЫЙ
    #   score = max    → color_values = 0 → БЕЛЫЙ
    sc = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=nodes, node_size=sizes,
        node_color=color_values, cmap='Reds',
        vmin=0.0, vmax=1.0,
        edgecolors='black', linewidths=0.3,
    )

    # --- подсветка топ-5% самых насыщенных (наименьший score) ---
    if highlight_top_frac > 0:
        valid_idx = np.where(mask_valid)[0]
        if len(valid_idx) > 0:
            k = max(1, int(len(valid_idx) * highlight_top_frac))
            sorted_valid = valid_idx[np.argsort(scores[valid_idx])]
            top_nodes = [nodes[i] for i in sorted_valid[:k]]
            top_sizes = [sizes[i] * 2.5 for i in sorted_valid[:k]]
            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                nodelist=top_nodes, node_size=top_sizes,
                node_color='crimson',
                edgecolors='black', linewidths=0.8,
                label=f"Топ-{k} (минимальный score)",
            )
            ax.legend(loc='upper right', fontsize=9)

    # --- colorbar ---
    # ВАЖНО: colorbar показывает color_values (инвертированные),
    # поэтому подпишем шкалу в терминах score.
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.01)
    cbar.set_label('Средний номер группы (score)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    # Переворачиваем метки шкалы, чтобы было видно реальный score:
    ticks = np.linspace(0, 1, 6)
    # color_values=0 ↔ score=max; color_values=1 ↔ score=0
    tick_labels = [f"{(1 - t) * smax:.1f}" for t in ticks]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)

    ax.set_title(
        f"{network_name} | Вершины по среднему номеру группы\n"
        f"Чем меньше score, тем насыщеннее КРАСНЫЙ "
        f"(0 = тёмно-красный, {n_groups-1:.0f} = бледный)",
        fontsize=11, pad=6
    )
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=0.92, bottom=0)
    plt.tight_layout(pad=0.2)

    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches='tight', pad_inches=0.05)
    plt.close()


# ========================
# 5. АНАЛИЗ ER
# ========================

def analyze_er_edge_groups(output_dir, centrality='pagerank',
                           n_nodes=500, p_values=None,
                           method='peaks',
                           bins=None, smooth_window=3,
                           prominence_frac=0.1, distance=None, height_frac=0.0,
                           bw_method=0.01, prominence=0.01,
                           seed=None,
                           draw_network=True,
                           draw_mean_group=True,
                           draw_communities=True,
                           draw_per_group=False,
                           network_layout='spring',
                           edge_lighten=0.6,
                           edge_alpha=0.5,
                           edge_width=0.5,
                           per_group_methods=('louvain', 'greedy', 'labelprop')):
    """
    Параметры:
    ----------
    draw_network : bool
        Рисовать ли сеть с рёбрами по группам.
    draw_mean_group : bool
        Рисовать ли сеть, где вершины раскрашены по среднему номеру
        группы инцидентных рёбер (0=красный ... K=синий).
        Полностью отключает вычисление compute_node_mean_group,
        если False.
    draw_communities : bool
        Если False — ПОЛНОСТЬЮ отключает:
          - запуск Louvain / Greedy / Label Propagation,
          - расчёт NMI / ARI,
          - расчёт frac_internal,
          - сохранение *_internal_ratio_*.csv,
          - сохранение *_network_communities_*.png,
          - записи в comparison.
    draw_per_group : bool
        Рисовать ли per-group картинки. Зависит от draw_communities=True.
    per_group_methods : tuple
        Какие алгоритмы использовать для per-group картинок.
    """
    if p_values is None:
        p_values = [0.005, 0.01, 0.05]

    os.makedirs(output_dir, exist_ok=True)
    all_tables = {}
    summary_rows = []
    comparison_rows = []

    for p in p_values:
        print(f"\n  ER: n={n_nodes}, p={p}")
        G = nx.erdos_renyi_graph(n=n_nodes, p=p, seed=seed)
        if G.number_of_edges() < 10:
            print(f"    👿 Слишком мало рёбер")
            continue

        gradients, edge_list = compute_edge_gradients_with_edges(G, centrality)
        if gradients is None:
            print(f"    👿 Не удалось вычислить градиенты")
            continue

        mask = gradients > 1e-12
        gradients_f = gradients[mask]
        edge_list_f = [e for e, m in zip(edge_list, mask) if m]

        if len(gradients_f) < 10:
            print(f"    👿 Недостаточно данных после фильтрации")
            continue

        # ==========================================
        # БАЗОВЫЕ ВЫЧИСЛЕНИЯ (всегда)
        # ==========================================
        labels, boundaries = assign_edge_groups(
            gradients_f, method=method,
            bins=bins, smooth_window=smooth_window,
            prominence_frac=prominence_frac, distance=distance,
            height_frac=height_frac,
            bw_method=bw_method, prominence=prominence,
            verbose=True
        )
        n_groups = int(labels.max()) + 1

        if method == 'peaks':
            _, _, info = assign_edge_groups_peaks(
                gradients_f, bins=bins, smooth_window=smooth_window,
                prominence_frac=prominence_frac, distance=distance,
                height_frac=height_frac, verbose=False
            )
        else:
            info = None

        peak_values = info['peak_values'] if (info is not None) else None
        peak_heights = info['peak_heights'] if (info is not None) else None

        df = build_node_group_table(G, edge_list_f, labels, n_groups=n_groups)
        name = f"ER_n{n_nodes}_p{p}"

        _save_table(df, os.path.join(output_dir, f"{name}_node_groups"))
        all_tables[name] = df

        # (1) гистограмма — базовая, всегда
        plot_histogram_with_groups(
            gradients_f, labels, boundaries,
            name, centrality, bins=(info['bins'] if info else 300),
            info=info,
            save_path=os.path.join(output_dir, f"{name}_hist_groups.png")
        )

        # (2) сеть с рёбрами по группам — только если draw_network
        if draw_network:
            plot_network_with_edge_groups(
                G, edge_list_f, labels, gradients_f,
                name, centrality,
                peak_values=peak_values, peak_heights=peak_heights,
                layout=network_layout,
                seed=(seed if seed is not None else 0),
                save_path=os.path.join(output_dir, f"{name}_network_edges.png")
            )

        # (2b) сеть: вершины по СРЕДНЕМУ НОМЕРУ ГРУППЫ — только если draw_mean_group
        if draw_mean_group:
            node_score_mean = compute_node_mean_group(edge_list_f, labels)
            plot_network_with_mean_group_nodes(
                G, node_score_mean,
                edge_list_f, labels, n_groups,
                name, centrality,
                layout=network_layout,
                seed=(seed if seed is not None else 0),
                edge_width=edge_width,
                edge_alpha=edge_alpha,
                edge_lighten=edge_lighten,
                save_path=os.path.join(
                    output_dir,
                    f"{name}_network_mean_group.png"
                )
            )

        # ==========================================
        # ВЫДЕЛЕНИЕ СООБЩЕСТВ — только если draw_communities=True
        # ==========================================
        partitions = []
        partitions_dict = {}

        if draw_communities:
            print(f"    --- Community detection ---")

            partition_louvain = detect_communities_louvain(G, seed=(seed or 0))
            partition_greedy = detect_communities_greedy(G)
            partition_labelprop = detect_communities_label_prop(G)

            partitions = [
                ('louvain', partition_louvain, 'Louvain'),
                ('greedy', partition_greedy, 'Greedy Modularity'),
                ('labelprop', partition_labelprop, 'Label Propagation'),
            ]
            partitions_dict = {k: v for k, v, _ in partitions if v is not None}

            nodes_order = list(G.nodes())
            node_group_majority = compute_node_dominant_group(
                edge_list_f, labels, n_groups, rule='majority'
            )
            labels_yours = [node_group_majority.get(n, -1) for n in nodes_order]

            for key, partition, method_name in partitions:
                if partition is None:
                    print(f"    {method_name}: недоступен")
                    continue

                labels_comm = [partition.get(n, -1) for n in nodes_order]
                n_comms = len(set(labels_comm))
                nmi, ari = compare_partitions(labels_yours, labels_comm)
                print(f"    {method_name}: {n_comms} сообществ, "
                      f"NMI={nmi:.4f}, ARI={ari:.4f}")

                df_int = internal_edge_ratio_per_group(
                    edge_list_f, labels, partition, n_groups
                )
                df_int['network'] = name
                df_int['method'] = key
                _save_table(df_int,
                            os.path.join(output_dir,
                                         f"{name}_internal_ratio_{key}"))

                plot_network_with_communities(
                    G, partition,
                    edge_list_f, labels, n_groups,
                    name, centrality,
                    method_name=method_name,
                    layout=network_layout,
                    seed=(seed if seed is not None else 0),
                    edge_width=edge_width,
                    edge_alpha=edge_alpha,
                    edge_lighten=edge_lighten,
                    save_path=os.path.join(
                        output_dir,
                        f"{name}_network_communities_{key}.png"
                    )
                )

                comparison_rows.append({
                    'network': name,
                    'p': p,
                    'method': key,
                    'n_communities': n_comms,
                    'NMI_vs_majority': round(nmi, 4),
                    'ARI_vs_majority': round(ari, 4),
                })
        else:
            print(f"    (community detection пропущено — draw_communities=False)")

        # ==========================================
        # PER-GROUP — только если draw_per_group=True и есть partitions
        # ==========================================
        if draw_per_group:
            if not partitions_dict:
                print(f"    (per-group пропущено — нет данных сообществ)")
            else:
                print(f"    --- Per-group visualization ---")
                for g in range(n_groups):
                    for mkey in per_group_methods:
                        if mkey not in partitions_dict:
                            continue
                        partition = partitions_dict[mkey]
                        method_name = dict(
                            (k, n) for k, _, n in partitions
                        ).get(mkey, mkey)

                        fname = f"{name}_group{g}_communities_{mkey}.png"
                        plot_network_single_group_communities(
                            G, partition,
                            edge_list_f, labels, n_groups, g,
                            name, centrality,
                            method_name=method_name,
                            layout=network_layout,
                            seed=(seed if seed is not None else 0),
                            save_path=os.path.join(output_dir, fname)
                        )
                print(f"    Сгенерировано "
                      f"{n_groups * len(per_group_methods)} "
                      f"per-group картинок")

        summary_rows.append({
            'network': name,
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'n_groups': n_groups,
            'n_peaks': (len(info['peaks']) if info else None),
            'peak_values': (list(np.round(info['peak_values'], 4)) if info else None),
            'peak_heights': (list(np.round(info['peak_heights'], 1)) if info else None),
        })

        print(f"    Групп рёбер: {n_groups}")

    summary = pd.DataFrame(summary_rows)
    _save_table(summary, os.path.join(output_dir, "er_edge_groups_summary"))

    comparison = pd.DataFrame(comparison_rows)
    _save_table(comparison, os.path.join(output_dir, "community_comparison"))

    return all_tables, summary, comparison


# ========================
# 9. ЗАПУСК
# ========================

if __name__ == "__main__":
    base_dir = "C:/Proj_edge_distr/"
    er_output_dir = os.path.join(base_dir, "results_pagerank_ER_groups")

    all_tables, summary, comparison = analyze_er_edge_groups(
        output_dir=er_output_dir,
        centrality='pagerank',
        n_nodes=500,
        p_values=[0.4],
        method='peaks',
        bins=200,
        smooth_window=2, #2 #7
        prominence_frac=0.1,
        distance=2ц, #2 #None
        height_frac=0.0,
        seed=42,
        draw_network=True,           # (2) сеть с рёбрами по группам
        draw_mean_group=True,        # (2b) сеть: вершины по среднему номеру группы
        draw_communities=False,      # (3-5) сообщества (Louvain/Greedy/LPA)
        draw_per_group=False,        # per-group картинки
        per_group_methods=('louvain', 'greedy', 'labelprop'),
        network_layout='spring',
        edge_lighten=0,            # осветление рёбер (0 = исходный, 1 = белый)
        edge_alpha=0.5,
        edge_width=0.5,
    )

    print("\n\n===== СВОДКА ПО ER =====")
    print(summary)

    print("\n\n===== СРАВНЕНИЕ С СООБЩЕСТВАМИ =====")
    print(comparison)

    if all_tables:
        first_name = list(all_tables.keys())[0]
        print(f"\nТаблица для {first_name} (первые 10 строк):")
        print(all_tables[first_name].head(10))