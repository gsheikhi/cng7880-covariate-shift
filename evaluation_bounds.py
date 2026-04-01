"""
Evaluation Bounds for Distribution Shift - Numerical Toy Example
=================================================================
CNG7880 Week 6 

Key formulas implemented:

1. Total Variation Distance:
   TV(P,Q) = ∫_{X×Y} |q(x,y) - p(x,y)| · dx · dy

2. Wasserstein Distance - Kantorovich-Rubinstein dual:
   W(P,Q) = sup_{f: K_f ≤ 1} ∫_{X×Y} f(x,y) · (q(x,y) - p(x,y)) · dx · dy
          = sup_{f: K_f ≤ 1} { E_Q[f(x)] - E_P[f(x)] }

   For paired samples X~P, Y~Q in 1D, the optimal transport plan pairs sorted samples:
   W(P,Q) = ( (1/n) Σ ||X_i - Y_i||^p )^{1/p}

   What is f(x,y)?  It is a 1-Lipschitz "witness function" that maximally
   separates the two distributions. NOT a loss function — it is the test
   function in the dual formulation of optimal transport.

3. TV Bound:
   E_Q[l(θ;x,y)] ≤ E_P[l(θ;x,y)] + l_max · TV(P,Q)
   where l_max = sup_{x,y} l(θ;x,y)

4. Wasserstein Bound:
   E_Q[l(θ;x,y)] ≤ E_P[l(θ;x,y)] + K_l · W(P,Q)
   where K_l is the Lipschitz constant of the loss function l(θ;·,·)

5. Covariate Shift Bound:
   E_Q[l(θ;x,y)] ≤ E_P[l(θ;x,y)] + K_l̃ · W(P(x), Q(x))
   where l̃(θ;x) = ∫_Y l(θ;x,y)·p(y|x)·dy  (marginalized loss)
   and K_l̃ is the Lipschitz constant of l̃
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import os

np.random.seed(42)
output_dir = "./outputs"

# ============================================================
# STEP 1: Create P and Q where support of Q is NOT fully
#         contained in support of P
# ============================================================
# P: x ~ N(0, 1)       → support roughly [-3, 3]
# Q: x ~ N(3, 0.8)     → support roughly [0.6, 5.4]
# Q has mass in [3, 5.4] where P has almost zero density

mu_p, sig_p = 0.0, 1.0
mu_q, sig_q = 3.0, 0.8

n_train = 20
n_test = 20

x_train = np.random.normal(mu_p, sig_p, n_train)
x_test = np.random.normal(mu_q, sig_q, n_test)

# Same labeling: p(y|x) = q(y|x), y = sign(x - 1.5) with noise
def label_fn(x, threshold=1.5, noise_prob=0.1):
    y = (x > threshold).astype(int)
    flip = np.random.random(len(x)) < noise_prob
    y[flip] = 1 - y[flip]
    return y

y_train = label_fn(x_train)
y_test = label_fn(x_test)

print("=" * 65)
print("STEP 1: Distributions with partial support overlap")
print("=" * 65)
print(f"P: x ~ N({mu_p}, {sig_p}²)  → bulk in [{mu_p-3*sig_p:.1f}, {mu_p+3*sig_p:.1f}]")
print(f"Q: x ~ N({mu_q}, {sig_q}²)  → bulk in [{mu_q-3*sig_q:.1f}, {mu_q+3*sig_q:.1f}]")
print(f"Q extends beyond P's support: region [{mu_p+3*sig_p:.1f}, {mu_q+3*sig_q:.1f}]")
print(f"\nTrain samples (P): {x_train[:5].round(3)}")
print(f"Test  samples (Q): {x_test[:5].round(3)}")

# ============================================================
# STEP 2: Total Variation Distance 
# ============================================================
# Slide formula: TV(P,Q) = ∫_{X×Y} |q(x,y) - p(x,y)| · dx · dy
#
# For continuous distributions with densities p(x), q(x):
#   TV = (1/2) ∫ |p(x) - q(x)| dx
#
# Connection to slide integral over X×Y:
#   Since p(y|x) = q(y|x) (covariate shift), we have
#   q(x,y) - p(x,y) = p(y|x)·q(x) - p(y|x)·p(x) = p(y|x)·(q(x)-p(x))
#   So ∫∫ |q(x,y)-p(x,y)| dx dy = ∫ |q(x)-p(x)| · (∫ p(y|x) dy) dx
#                                 = ∫ |q(x)-p(x)| dx = 2·TV
#
# Sample-based estimation: discretize x into bins and approximate

# --- Exact TV via numerical integration ---
x_grid = np.linspace(-6, 8, 10000)
dx = x_grid[1] - x_grid[0]
p_density = norm.pdf(x_grid, mu_p, sig_p)
q_density = norm.pdf(x_grid, mu_q, sig_q)
TV_exact = 0.5 * np.sum(np.abs(q_density - p_density)) * dx

# --- Sample-based TV estimation using histogram ---
n_bins = 30
bin_edges = np.linspace(-5, 7, n_bins + 1)
hist_p, _ = np.histogram(x_train, bins=bin_edges, density=True)
hist_q, _ = np.histogram(x_test, bins=bin_edges, density=True)
bin_width = bin_edges[1] - bin_edges[0]
TV_sample = 0.5 * np.sum(np.abs(hist_q - hist_p)) * bin_width

print("\n" + "=" * 65)
print("STEP 2: Total Variation Distance")
print("=" * 65)
print(f"\nSlide formula: TV(P,Q) = ∫_{{X×Y}} |q(x,y) - p(x,y)| dx dy")
print(f"For marginals: TV(P,Q) = (1/2) ∫ |p(x) - q(x)| dx")
print(f"\nExact TV (numerical integration):  {TV_exact:.4f}")
print(f"Sample-based TV (histogram, n=20): {TV_sample:.4f}")
print(f"\nNote: TV ∈ [0, 1]. TV={TV_exact:.2f} indicates substantial shift.")

# ============================================================
# STEP 3: Wasserstein Distance - Kantorovich-Rubinstein dual
# ============================================================
#   W(P,Q) = sup_{f: K_f ≤ 1} ∫_{X×Y} f(x,y)·(q(x,y) - p(x,y)) · dx dy
#          = sup_{f: K_f ≤ 1} { E_Q[f(x,y)] - E_P[f(x,y)] }
#
# What is f(x,y)?
#   f is a "witness function" (NOT the loss function l).
#   It must be 1-Lipschitz: |f(a) - f(b)| ≤ ||a - b|| for all a,b.
#   The sup finds the f that best separates the two distributions.
#   Think of f as a "critic" that gives high values to Q-like points
#   and low values to P-like points, constrained by smoothness.
#
# Sample formula for paired 1D samples:
#   W_p(P,Q) = ( (1/n) Σ_{i=1}^{n} ||X_i - Y_i||^p )^{1/p}
#
#   For p=1 (W1): sort both samples, pair them up, measure average distance.
#   This works because in 1D, the optimal transport plan pairs
#   the i-th smallest of P with the i-th smallest of Q.

# --- W1 exact for 1D Gaussians ---
# W1(N(μ1,σ1²), N(μ2,σ2²)) can be computed numerically
# We use the sorted-sample approach

# --- Sample-based W1 (1D optimal: sort and pair) ---
x_train_sorted = np.sort(x_train)
x_test_sorted = np.sort(x_test)
W1_sample = np.mean(np.abs(x_train_sorted - x_test_sorted))

# --- W1 via numerical integration of CDFs ---
# W1 = ∫ |F_P(x) - F_Q(x)| dx  (equivalent in 1D)
F_p = norm.cdf(x_grid, mu_p, sig_p)
F_q = norm.cdf(x_grid, mu_q, sig_q)
W1_exact = np.sum(np.abs(F_p - F_q)) * dx

# --- Show the witness function f* ---
# For 1D W1 between Gaussians, the optimal witness is related to
# f*(x) = sign(F_Q(x) - F_P(x)) clipped to be 1-Lipschitz.
# In practice, f*(x) ≈ x (identity) works when distributions are shifted.

print("\n" + "=" * 65)
print("STEP 3: Wasserstein Distance")
print("=" * 65)
print(f"\n--- What is f(x,y) in the Wasserstein formula? ---")
print(f"f(x,y) is a 1-Lipschitz 'witness function' (NOT the loss l).")
print(f"It satisfies |f(a)-f(b)| ≤ ||a-b|| for all a,b.")
print(f"The supremum finds f that best separates Q from P.")
print(f"Think of it as a smooth 'critic' constrained by Lipschitz.")
print(f"\n--- Dual formula: ---")
print(f"W(P,Q) = sup_{{f: K_f≤1}} {{ E_Q[f(x)] - E_P[f(x)] }}")
print(f"\n--- Sample formula (1D, p=1): ---")
print(f"W1 = (1/n) Σ |X_(i) - Y_(i)|  (sort both, pair, avg distance)")
print(f"\nConnection: In 1D, sorting gives the optimal transport plan.")
print(f"The i-th smallest of P is paired with i-th smallest of Q.")
print(f"\nW1 exact (CDF integration):   {W1_exact:.4f}")
print(f"W1 sample (sort-and-pair):    {W1_sample:.4f}")

print(f"\n--- Sorted pairs and distances: ---")
print(f"{'X_(i) (P)':>12s} {'Y_(i) (Q)':>12s} {'|X-Y|':>10s}")
print("-" * 38)
for i in range(n_train):
    d = abs(x_train_sorted[i] - x_test_sorted[i])
    print(f"{x_train_sorted[i]:12.3f} {x_test_sorted[i]:12.3f} {d:10.3f}")
print(f"{'':>24s} {'Mean':>10s} = {W1_sample:.3f}")

# ============================================================
# STEP 4: TV Bound
# E_Q[l] ≤ E_P[l] + l_max · TV(P,Q)
# ============================================================
# l_max = sup_{x,y} l(θ;x,y)
# For log loss: l_max can be large (unbounded in theory)
# We clip probabilities to get a finite l_max

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)
model.fit(x_train.reshape(-1, 1), y_train)

def log_loss_per_sample(model, X, y, clip=1e-3):
    probs = model.predict_proba(X)
    losses = np.zeros(len(y))
    for i in range(len(y)):
        p = np.clip(probs[i, int(y[i])], clip, 1.0)
        losses[i] = -np.log(p)
    return losses

losses_train = log_loss_per_sample(model, x_train.reshape(-1, 1), y_train)
losses_test = log_loss_per_sample(model, x_test.reshape(-1, 1), y_test)

E_P = losses_train.mean()
E_Q = losses_test.mean()
l_max = -np.log(1e-3)  # clipped maximum loss

TV_bound = E_P + l_max * TV_exact

print("\n" + "=" * 65)
print("STEP 4: TV Bound")
print("=" * 65)
print(f"\nE_Q[l(θ;x,y)] ≤ E_P[l(θ;x,y)] + l_max · TV(P,Q)")
print(f"\nE_P[l] (train loss):    {E_P:.4f}")
print(f"E_Q[l] (test loss):     {E_Q:.4f}")
print(f"l_max (clipped):        {l_max:.4f}")
print(f"TV(P,Q):                {TV_exact:.4f}")
print(f"l_max · TV(P,Q):        {l_max * TV_exact:.4f}")
print(f"\nTV bound:  E_P + l_max·TV = {TV_bound:.4f}")
print(f"Actual:    E_Q            = {E_Q:.4f}")
print(f"Bound holds: {TV_bound >= E_Q - 0.001}  (TV_bound ≥ E_Q)")
print(f"\nNote: l_max is the SUPREMUM of the loss over all (x,y).")
print(f"This makes the TV bound loose — it ignores smoothness of l.")

# ============================================================
# STEP 5: Wasserstein Bound
# E_Q[l] ≤ E_P[l] + K_l · W(P,Q)
# ============================================================
# K_l is the Lipschitz constant of the loss function l(θ;x,y).
# |l(θ;a,y_a) - l(θ;b,y_b)| ≤ K_l · ||(a,y_a) - (b,y_b)||
#
# For logistic regression with weight vector θ:
#   l(θ;x,y) = -y·log(σ(θx)) - (1-y)·log(1-σ(θx))
#   The gradient w.r.t. x is bounded by ||θ||, so K_l ≈ ||θ||.
#
# Covariate shift refinement:
#   Define l̃(θ;x) = E_{y|x}[l(θ;x,y)] = ∫_Y l(θ;x,y)·p(y|x) dy
#   Then: E_Q[l] ≤ E_P[l] + K_l̃ · W(P(x), Q(x))
#   K_l̃ is the Lipschitz constant of the MARGINALIZED loss l̃.

theta_norm = np.sqrt(model.coef_[0][0]**2)  # ||θ|| for 1D
K_l = theta_norm  # Lipschitz constant of log loss w.r.t. x

W_bound = E_P + K_l * W1_exact

print("\n" + "=" * 65)
print("STEP 5: Wasserstein Bound")
print("=" * 65)
print(f"\nE_Q[l(θ;x,y)] ≤ E_P[l(θ;x,y)] + K_l · W(P,Q)")
print(f"\n--- What is K_l? ---")
print(f"K_l is the Lipschitz constant of the loss l(θ;x,y).")
print(f"|l(θ;a) - l(θ;b)| ≤ K_l · ||a - b||  for all a, b")
print(f"For logistic regression: K_l ≈ ||θ|| = {K_l:.4f}")
print(f"\nE_P[l]:          {E_P:.4f}")
print(f"K_l:             {K_l:.4f}")
print(f"W(P,Q):          {W1_exact:.4f}")
print(f"K_l · W(P,Q):    {K_l * W1_exact:.4f}")
print(f"\nWasserstein bound: E_P + K_l·W = {W_bound:.4f}")
print(f"Actual:            E_Q          = {E_Q:.4f}")
print(f"Bound holds: {W_bound >= E_Q - 0.001}")

print(f"\n--- Comparison of bounds ---")
print(f"TV bound:          {TV_bound:.4f}  (uses l_max={l_max:.2f})")
print(f"Wasserstein bound: {W_bound:.4f}  (uses K_l={K_l:.4f})")
print(f"Actual E_Q:        {E_Q:.4f}")
print(f"\nThe Wasserstein bound is tighter because it uses the")
print(f"Lipschitz constant K_l (smoothness) instead of l_max (worst case).")

# ============================================================
# STEP 6: Covariate Shift Bound
# ============================================================
print("\n" + "=" * 65)
print("STEP 6: Covariate Shift Refinement")
print("=" * 65)
print(f"\nUnder covariate shift p(y|x) = q(y|x):")
print(f"  E_Q[l] ≤ E_P[l] + K_l̃ · W(P(x), Q(x))")
print(f"\nwhere l̃(θ;x) = E_{{y|x}}[l(θ;x,y)] = ∫ l(θ;x,y)·p(y|x) dy")
print(f"is the expected loss at x, averaged over y given x.")
print(f"\nK_l̃ is the Lipschitz constant of this marginalized loss.")
print(f"Since we only need W over marginals P(x),Q(x) (not joint),")
print(f"this bound is tighter when the shift is only in covariates.")
print(f"\nIn our example: W(P(x),Q(x)) = W1_exact = {W1_exact:.4f}")
print(f"Same as W(P,Q) here since shift is only in x.")

# ============================================================
# FIGURES
# ============================================================

# --- Figure 1: Distributions with non-overlapping support ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

x_range = np.linspace(-5, 7, 300)
p_pdf = norm.pdf(x_range, mu_p, sig_p)
q_pdf = norm.pdf(x_range, mu_q, sig_q)

# (a) Densities + overlap region
axes[0].plot(x_range, p_pdf, 'b-', lw=2, label=f'P: N({mu_p},{sig_p}²)')
axes[0].plot(x_range, q_pdf, 'r--', lw=2, label=f'Q: N({mu_q},{sig_q}²)')
# Shade non-overlap region of Q
mask_no_overlap = x_range > (mu_p + 2.5 * sig_p)
axes[0].fill_between(x_range, 0, q_pdf, where=mask_no_overlap,
                      alpha=0.3, color='red', label='Q outside P support')
axes[0].scatter(x_train, -0.02 * np.ones(n_train), c='blue', marker='|', s=80)
axes[0].scatter(x_test, -0.04 * np.ones(n_test), c='red', marker='|', s=80)
axes[0].set_xlabel('x')
axes[0].set_ylabel('Density')
axes[0].set_title('(a) P and Q: Partial Support Overlap')
axes[0].legend(fontsize=8)

# (b) |p(x) - q(x)| for TV
diff = np.abs(p_pdf - q_pdf)
axes[1].fill_between(x_range, 0, diff, alpha=0.4, color='purple')
axes[1].plot(x_range, diff, 'purple', lw=1.5)
axes[1].set_xlabel('x')
axes[1].set_ylabel('|p(x) - q(x)|')
axes[1].set_title(f'(b) TV(P,Q) = ½∫|p-q|dx = {TV_exact:.3f}')

# (c) |F_P - F_Q| for Wasserstein
F_p_plot = norm.cdf(x_range, mu_p, sig_p)
F_q_plot = norm.cdf(x_range, mu_q, sig_q)
axes[2].fill_between(x_range, F_p_plot, F_q_plot, alpha=0.3, color='green')
axes[2].plot(x_range, F_p_plot, 'b-', lw=2, label='F_P(x)')
axes[2].plot(x_range, F_q_plot, 'r--', lw=2, label='F_Q(x)')
axes[2].set_xlabel('x')
axes[2].set_ylabel('CDF')
axes[2].set_title(f'(c) W₁ = ∫|F_P - F_Q|dx = {W1_exact:.3f}')
axes[2].legend(fontsize=8)

plt.tight_layout()
fig1_path = os.path.join(output_dir, "fig_eval_bounds_distances.pdf")
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {fig1_path}")

# --- Figure 2: Wasserstein optimal pairing ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) Sort-and-pair visualization
for i in range(n_train):
    axes[0].plot([x_train_sorted[i], x_test_sorted[i]], [0, 1],
                 'gray', alpha=0.4, lw=0.8)
axes[0].scatter(x_train_sorted, np.zeros(n_train), c='blue', s=50,
                zorder=5, label='P samples (sorted)')
axes[0].scatter(x_test_sorted, np.ones(n_test), c='red', s=50,
                zorder=5, label='Q samples (sorted)')
axes[0].set_xlabel('x')
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(['P', 'Q'])
axes[0].set_title(f'(a) Optimal Transport Pairing\n'
                   f'W₁ = (1/n)Σ|X_{{(i)}} - Y_{{(i)}}| = {W1_sample:.3f}')
axes[0].legend(fontsize=8)

# (b) Comparison of bounds
methods = ['E_P[l]\n(train)', 'E_Q[l]\n(actual)', 'TV Bound', 'W Bound']
values = [E_P, E_Q, TV_bound, W_bound]
colors = ['#4472C4', '#C00000', '#BF8F00', '#548235']
bars = axes[1].bar(methods, values, color=colors, width=0.6, edgecolor='black')
axes[1].axhline(y=E_Q, color='red', ls='--', alpha=0.5, label='Actual E_Q')
axes[1].set_ylabel('Loss')
axes[1].set_title('(b) Evaluation Bounds Comparison')

# Add value labels
for bar, val in zip(bars, values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.2f}', ha='center', fontsize=9)

plt.tight_layout()
fig2_path = os.path.join(output_dir, "fig_eval_bounds_comparison.pdf")
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fig2_path}")

# --- Figure 3: Summary table ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')

table_data = [
    ['TV(P,Q)', '½∫|p(x)-q(x)|dx', f'{TV_exact:.4f}',
     'How much density disagrees'],
    ['W₁(P,Q)', '∫|F_P(x)-F_Q(x)|dx', f'{W1_exact:.4f}',
     'Min cost to move P→Q'],
    ['TV Bound', 'E_P[l] + l_max·TV', f'{TV_bound:.4f}',
     f'l_max={l_max:.2f} (worst-case loss)'],
    ['W Bound', 'E_P[l] + K_l·W', f'{W_bound:.4f}',
     f'K_l={K_l:.4f} (Lipschitz of l)'],
    ['Actual E_Q', 'E_Q[l(θ;x,y)]', f'{E_Q:.4f}',
     'True test loss'],
]
col_labels = ['Quantity', 'Formula', 'Value', 'Meaning']

table = ax.table(cellText=table_data, colLabels=col_labels,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 1.5)

for j in range(len(col_labels)):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Highlight bound rows
for i in [3, 4]:
    for j in range(len(col_labels)):
        table[i, j].set_facecolor('#E2EFDA')

ax.set_title('Evaluation Bounds Summary',
             fontsize=13, fontweight='bold', pad=20)

fig3_path = os.path.join(output_dir, "fig_eval_bounds_table.pdf")
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fig3_path}")

# --- Figure 4: Witness function f(x) illustration ---
fig, ax = plt.subplots(figsize=(8, 5))

# The optimal 1-Lipschitz witness for W1 in 1D
# f*(x) relates to sign(F_Q - F_P), but constrained to be 1-Lipschitz
# For shifted Gaussians, a good approximation is a clipped linear function
x_range_f = np.linspace(-5, 7, 300)

# Numerical approximation of optimal witness:
# f'(x) = sign(F_Q(x) - F_P(x)), constrained |f'| ≤ 1
# Since F_Q > F_P almost everywhere (Q is shifted right), f ≈ x + const
cdf_diff = norm.cdf(x_range_f, mu_q, sig_q) - norm.cdf(x_range_f, mu_p, sig_p)
# Integrate the sign to get f (1-Lipschitz)
f_witness = np.cumsum(np.sign(-cdf_diff)) * (x_range_f[1] - x_range_f[0])
# Normalize to have Lipschitz constant = 1
grad = np.diff(f_witness) / np.diff(x_range_f)
max_grad = np.max(np.abs(grad))
if max_grad > 0:
    f_witness = f_witness / max_grad

ax.plot(x_range_f, f_witness, 'k-', lw=2, label='Witness f*(x) (1-Lipschitz)')
ax.fill_between(x_range_f, 0, norm.pdf(x_range_f, mu_p, sig_p) * 3,
                alpha=0.2, color='blue', label='P(x) scaled')
ax.fill_between(x_range_f, 0, norm.pdf(x_range_f, mu_q, sig_q) * 3,
                alpha=0.2, color='red', label='Q(x) scaled')
ax.axhline(y=0, color='gray', ls='-', alpha=0.3)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Witness Function f*(x) in Wasserstein Distance\n'
             'W = sup_{f: K_f≤1} { E_Q[f] - E_P[f] }')
ax.legend(fontsize=9)
ax.text(0.02, 0.02,
        'f*(x) maximally separates Q from P\n'
        'while remaining 1-Lipschitz smooth.',
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig4_path = os.path.join(output_dir, "fig_witness_function.pdf")
plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fig4_path}")

print("\n" + "=" * 65)
print("All figures saved.")
print("=" * 65)