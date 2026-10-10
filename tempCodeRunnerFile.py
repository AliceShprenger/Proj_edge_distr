import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats

G = nx.gnp_random_graph(3000, 5/3000, seed=42)
b = nx.betweenness_centrality(G, normalized=True)
b_vals = np.array([b[v] for v in G.nodes()])
b_mean = b_vals.mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) Гистограмма b_v / mean(b)
ax = axes[0]
ax.hist(b_vals / b_mean, bins=80, density=True, alpha=0.6,
        color='steelblue', edgecolor='black', linewidth=0.3,
        label='Эмпирика')
x = np.linspace(0, 5, 200)
ax.plot(x, stats.expon.pdf(x, scale=1), 'r-', lw=2,
        label='Exponential(1)')
ax.set_xlabel('b_v / mean(b)')
ax.set_ylabel('Плотность')
ax.set_title('(a) Распределение b_v')
ax.legend()
ax.set_xlim(0, 5)

# (b) CCDF в полу-лог
ax = axes[1]
sorted_b = np.sort(b_vals / b_mean)
ccdf = 1 - np.arange(1, len(sorted_b) + 1) / len(sorted_b)
ax.semilogy(sorted_b, ccdf, 'o', ms=3, alpha=0.5, color='steelblue',
            label='Эмпирика')
ax.semilogy(x, np.exp(-x), 'r-', lw=2, label='exp(-x)')
ax.set_xlabel('b_v / mean(b)')
ax.set_ylabel('P(b_v > x)')
ax.set_title('(b) CCDF: полу-лог (прямая = экспонента)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Линейная подгонка log(CCDF) vs x
mask = ccdf > 1e-3
slope, intercept, r, p, se = stats.linregress(
    sorted_b[mask], np.log(ccdf[mask]))
print(f"slope = {slope:.4f}, R² = {r**2:.4f}")
print(f"Если slope ≈ -1 и R² ≈ 1, то b_v экспоненциально с параметром 1.")