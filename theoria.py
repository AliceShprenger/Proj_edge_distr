import numpy as np
import networkx as nx
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ============================================================
# 1. Основные функции
# ============================================================

def generate_er(n, c, seed=42):
    p = c / n
    G = nx.gnp_random_graph(n, p, seed=seed)
    if not nx.is_connected(G):
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(comps[0]).copy()
    return G

def compute_gradient(G):
    b = nx.betweenness_centrality(G, normalized=True)
    b_vals = np.array([b[v] for v in G.nodes()])
    b_mean = b_vals.mean()
    X = np.array([abs(b[u] - b[v]) / b_mean for u, v in G.edges()])
    return X, b_vals, b_mean

def fit_lambda(data):
    """
    Оценка λ через линейную подгонку log(CCDF) vs x.
    Возвращает λ = -slope и R².
    """
    sorted_d = np.sort(data)
    ccdf = 1 - np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    mask = (ccdf > 1e-3) & (ccdf < 0.5)
    slope, intercept, r, p, se = stats.linregress(
        sorted_d[mask], np.log(ccdf[mask])
    )
    return -slope, r**2

# ============================================================
# 2. Эксперимент: много значений c
# ============================================================

n = 3000
c_values = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]

print("=" * 80)
print(f"ЭКСПЕРИМЕНТ: зависимость λ(c) при n = {n}")
print("=" * 80)
print(f"{'c':>5} | {'λ(X_e)':>9} | {'R²(X_e)':>8} | {'λ(b_v)':>9} | {'R²(b_v)':>8} | {'λ/√c':>8} | {'λ/log c':>8}")
print("-" * 80)

lambdas_X = []
lambdas_b = []
r2_X_list = []
r2_b_list = []

for c in c_values:
    G = generate_er(n=n, c=c, seed=42)
    X, b_vals, b_mean = compute_gradient(G)
    
    lam_X, r2_X = fit_lambda(X)
    lam_b, r2_b = fit_lambda(b_vals / b_mean)
    
    lambdas_X.append(lam_X)
    lambdas_b.append(lam_b)
    r2_X_list.append(r2_X)
    r2_b_list.append(r2_b)
    
    print(f"{c:>5} | {lam_X:>9.4f} | {r2_X:>8.4f} | {lam_b:>9.4f} | {r2_b:>8.4f} | "
          f"{lam_X/np.sqrt(c):>8.4f} | {lam_X/np.log(c):>8.4f}")

lambdas_X = np.array(lambdas_X)
lambdas_b = np.array(lambdas_b)
c_arr = np.array(c_values)

# ============================================================
# 3. Лог-лог анализ: какой показатель степени?
# ============================================================

print()
print("=" * 80)
print("ЛОГ-ЛОГ АНАЛИЗ: λ(c) ~ c^α")
print("=" * 80)

mask = (c_arr >= 3)  # исключаем c=2 (близко к порогу)
log_c = np.log(c_arr[mask])
log_lam = np.log(lambdas_X[mask])

slope_ll, intercept_ll, r_ll, _, _ = stats.linregress(log_c, log_lam)
print(f"log λ = {slope_ll:.4f} · log c + {intercept_ll:.4f}")
print(f"R² = {r_ll**2:.4f}")
print(f"α = {slope_ll:.4f}  (ожидаем 0.5 для √c)")
print(f"Если α ≈ 0.5 → λ ~ √c")
print(f"Если α ≈ 1.0 → λ ~ c")
print(f"Если α ≈ 0.0 → λ ~ const")

# ============================================================
# 4. Подгонка кандидатов
# ============================================================

print()
print("=" * 80)
print("ПОДГОНКА КАНДИДАТОВ")
print("=" * 80)

# Кандидат A: λ = α √c
def cand_A(c, alpha):
    return alpha * np.sqrt(c)

# Кандидат B: λ = α log c
def cand_B(c, alpha):
    return alpha * np.log(c)

# Кандидат C: λ = α √(c-1)
def cand_C(c, alpha):
    return alpha * np.sqrt(c - 1)

# Кандидат D: λ = α (c-1)/√c
def cand_D(c, alpha):
    return alpha * (c - 1) / np.sqrt(c)

# Кандидат E: λ = α √c / (1 + β/√c)
def cand_E(c, alpha, beta):
    return alpha * np.sqrt(c) / (1 + beta / np.sqrt(c))

# Кандидат F: λ = α √c + β
def cand_F(c, alpha, beta):
    return alpha * np.sqrt(c) + beta

candidates = {
    'A: α√c':       (cand_A, 1),
    'B: α log c':   (cand_B, 1),
    'C: α√(c-1)':   (cand_C, 1),
    'D: α(c-1)/√c': (cand_D, 1),
    'E: α√c/(1+β/√c)': (cand_E, 2),
    'F: α√c + β':   (cand_F, 2),
}

results = []
for name, (func, n_params) in candidates.items():
    try:
        popt, pcov = curve_fit(func, c_arr, lambdas_X, p0=[1.0]*n_params, maxfev=10000)
        residuals = lambdas_X - func(c_arr, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((lambdas_X - lambdas_X.mean())**2)
        r2 = 1 - ss_res / ss_tot
        # AIC
        aic = len(c_arr) * np.log(ss_res / len(c_arr)) + 2 * n_params
        results.append((name, popt, r2, aic, ss_res))
    except Exception as e:
        print(f"{name}: ошибка {e}")

# Сортируем по AIC (меньше = лучше)
results.sort(key=lambda x: x[3])

print(f"\n{'Кандидат':>20} | {'Параметры':>30} | {'R²':>8} | {'AIC':>10} | {'RMSE':>8}")
print("-" * 90)
for name, popt, r2, aic, ss_res in results:
    params_str = ", ".join([f"{p:.4f}" for p in popt])
    rmse = np.sqrt(ss_res / len(c_arr))
    print(f"{name:>20} | {params_str:>30} | {r2:>8.4f} | {aic:>10.4f} | {rmse:>8.4f}")

# ============================================================
# 5. Графики
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) λ vs c
ax = axes[0]
ax.plot(c_arr, lambdas_X, 'o-', color='steelblue', label='λ(X_e)', ms=6)
ax.plot(c_arr, lambdas_b, 's--', color='coral', label='λ(b_v)', ms=5, alpha=0.7)
ax.set_xlabel('c')
ax.set_ylabel('λ(c)')
ax.set_title('(a) λ(c) vs c')
ax.legend()
ax.grid(True, alpha=0.3)

# (b) log λ vs log c
ax = axes[1]
ax.loglog(c_arr, lambdas_X, 'o', color='steelblue', ms=6, label='λ(X_e)')
ax.loglog(c_arr, lambdas_b, 's', color='coral', ms=5, alpha=0.7, label='λ(b_v)')

# Референсные линии
c_ref = np.logspace(np.log10(c_arr.min()), np.log10(c_arr.max()), 100)
ax.loglog(c_ref, c_ref**0.5 * lambdas_X[0] / c_arr[0]**0.5, 'k--',
          alpha=0.5, label='~ √c')
ax.loglog(c_ref, c_ref**1.0 * lambdas_X[0] / c_arr[0]**1.0, 'k:',
          alpha=0.5, label='~ c')
ax.loglog(c_ref, c_ref**0.0 * lambdas_X[0], 'k-.',
          alpha=0.5, label='~ const')

ax.set_xlabel('c')
ax.set_ylabel('λ(c)')
ax.set_title(f'(b) log-log: наклон α = {slope_ll:.3f}')
ax.legend()
ax.grid(True, alpha=0.3)

# (c) Остатки лучшего кандидата
best_name, best_popt, best_r2, _, _ = results[0]
best_func = candidates[best_name][0]
ax = axes[2]
ax.plot(c_arr, lambdas_X, 'o', color='steelblue', ms=6, label='Данные')
ax.plot(c_arr, best_func(c_arr, *best_popt), 'r-', lw=2,
        label=f'Лучший: {best_name} (R²={best_r2:.4f})')
ax.set_xlabel('c')
ax.set_ylabel('λ(c)')
ax.set_title('(c) Лучший кандидат')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lambda_c_analysis.png', dpi=150)
plt.show()

# ============================================================
# 6. Итог
# ============================================================

print()
print("=" * 80)
print("ИТОГ")
print("=" * 80)
print(f"Лучший кандидат: {best_name}")
print(f"Параметры: {best_popt}")
print(f"R² = {best_r2:.4f}")
print()
print("Если α ≈ 0.5 → λ(c) ~ √c")
print("Если α ≈ 1.0 → λ(c) ~ c")
print("Если α ≈ 0.0 → λ(c) ~ const")
print(f"Эмпирически: α = {slope_ll:.4f}")