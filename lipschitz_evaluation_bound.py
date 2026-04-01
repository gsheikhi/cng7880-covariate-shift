"""
Evaluating Wasserstein Distance via Lipschitz Neural Network
=============================================================
CNG7880 Week 6

KEY DIFFERENCE FROM SECTION 2:
-------------------------------
Section 2 computed W1 using the CLOSED-FORM 1D trick:
  W1 = (1/n) Σ |X_(i) - Y_(i)|   (sort and pair)
  W1 = ∫ |F_P(x) - F_Q(x)| dx    (CDF integration)

These only work in 1D. In high dimensions, we have no closed form.

This section estimates W1 using the DUAL FORMULATION of Wasserstein distance:
  W(P,Q) = sup_{f: K_f ≤ 1} { E_Q[f(x)] - E_P[f(x)] }

We train a neural network f (the discriminator/critic) to MAXIMIZE
E_Q[f(x)] - E_P[f(x)], while constraining its Lipschitz constant ≤ 1.

This is the same idea as the WGAN critic (Wasserstein GAN).

HOW THE LIPSCHITZ CONSTRAINT IS ENFORCED:
---------------------------------------------------------
For a network g = g_m ∘ g_{m-1} ∘ ... ∘ g_1:
  K_g ≤ K_{g_m} · K_{g_{m-1}} · ... · K_{g_1}

For a linear layer g_j(x) = W_j x:  K_{g_j} = ||W_j||_op
For ReLU:  K_ReLU = 1

Strategy: After each gradient step, PROJECT each weight matrix
  W_j ← W_j / ||W_j||_op    (spectral normalization / projection)
This ensures K_{g_j} ≤ 1 for each layer, so K_g ≤ 1.

The training procedure:
  For t = 1, ..., T:
    For j = 1, ..., m:
      W_j ← W_j - α · ∇_{W_j} L(W_j; Z)     # gradient step
      W_j ← W_j / ||W_j||_op                  # project to K ≤ 1

Loss L maximizes: (1/n) Σ_{(x,1)∈X'} f(x) - (1/n) Σ_{(x,0)∈X'} f(x)
i.e., E_Q[f(x)] - E_P[f(x)]  (the Wasserstein dual objective)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)
output_dir = "./outputs"

# ============================================================
# STEP 0: Reproduce same data from Section 2
# ============================================================
from scipy.stats import norm

mu_p, sig_p = 0.0, 1.0
mu_q, sig_q = 3.0, 0.8
n_train = 20
n_test = 20

np.random.seed(42)
x_train = np.random.normal(mu_p, sig_p, n_train)
x_test = np.random.normal(mu_q, sig_q, n_test)

# Section 2 closed-form W1 for reference
x_grid = np.linspace(-6, 8, 10000)
dx = x_grid[1] - x_grid[0]
F_p = norm.cdf(x_grid, mu_p, sig_p)
F_q = norm.cdf(x_grid, mu_q, sig_q)
W1_exact = np.sum(np.abs(F_p - F_q)) * dx

x_train_sorted = np.sort(x_train)
x_test_sorted = np.sort(x_test)
W1_sample_sort = np.mean(np.abs(x_train_sorted - x_test_sorted))

print("=" * 65)
print("STEP 0: Same data as Section 2")
print("=" * 65)
print(f"P: N({mu_p},{sig_p}²), Q: N({mu_q},{sig_q}²), n=20 each")
print(f"Section 2 W1 (CDF):        {W1_exact:.4f}")
print(f"Section 2 W1 (sort-pair):  {W1_sample_sort:.4f}")

# ============================================================
# STEP 1: Build a Lipschitz-constrained neural network
#         from scratch (no PyTorch/TF needed)
# ============================================================
# Network: f(x) = W3 · ReLU(W2 · ReLU(W1 · x + b1) + b2) + b3
# 3 layers: input(1) → hidden(16) → hidden(16) → output(1)
#
# After each gradient step, we project:
#   W_j ← W_j / max(1, ||W_j||_op)
# This ensures K_{W_j} ≤ 1, and since K_ReLU = 1:
#   K_f ≤ K_{W3} · 1 · K_{W2} · 1 · K_{W1} ≤ 1·1·1 = 1

class LipschitzMLP:
    """Simple 3-layer MLP with spectral norm projection."""

    def __init__(self, hidden=16):
        # Xavier init
        self.W1 = np.random.randn(hidden, 1) * np.sqrt(2.0 / 1)
        self.b1 = np.zeros((hidden, 1))
        self.W2 = np.random.randn(hidden, hidden) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros((hidden, 1))
        self.W3 = np.random.randn(1, hidden) * np.sqrt(2.0 / hidden)
        self.b3 = np.zeros((1, 1))
        # Project weights immediately
        self._project_weights()

    def _spectral_norm(self, W):
        """||W||_op = largest singular value."""
        return np.linalg.svd(W, compute_uv=False)[0]

    def _project_weights(self):
        """Project W_j ← W_j / max(1, ||W_j||_op)."""
        for attr in ['W1', 'W2', 'W3']:
            W = getattr(self, attr)
            sn = self._spectral_norm(W)
            if sn > 1.0:
                setattr(self, attr, W / sn)

    def forward(self, x):
        """Forward pass, store activations for backprop."""
        # x: (1, n)
        self.x_input = x
        self.z1 = self.W1 @ x + self.b1          # (hidden, n)
        self.a1 = np.maximum(0, self.z1)           # ReLU
        self.z2 = self.W2 @ self.a1 + self.b2     # (hidden, n)
        self.a2 = np.maximum(0, self.z2)           # ReLU
        self.out = self.W3 @ self.a2 + self.b3     # (1, n)
        return self.out  # (1, n)

    def backward(self, d_out):
        """Backprop for gradients."""
        n = d_out.shape[1]

        # Layer 3
        self.dW3 = d_out @ self.a2.T / n
        self.db3 = d_out.mean(axis=1, keepdims=True)
        d_a2 = self.W3.T @ d_out

        # ReLU 2
        d_z2 = d_a2 * (self.z2 > 0)

        # Layer 2
        self.dW2 = d_z2 @ self.a1.T / n
        self.db2 = d_z2.mean(axis=1, keepdims=True)
        d_a1 = self.W2.T @ d_z2

        # ReLU 1
        d_z1 = d_a1 * (self.z1 > 0)

        # Layer 1
        self.dW1 = d_z1 @ self.x_input.T / n
        self.db1 = d_z1.mean(axis=1, keepdims=True)

    def step(self, lr):
        """Gradient ASCENT (we maximize), then project."""
        # Ascent: W ← W + lr · ∇W  (maximize the objective)
        self.W1 += lr * self.dW1
        self.b1 += lr * self.db1
        self.W2 += lr * self.dW2
        self.b2 += lr * self.db2
        self.W3 += lr * self.dW3
        self.b3 += lr * self.db3
        # Project: W_j ← W_j / max(1, ||W_j||_op)
        self._project_weights()

    def lipschitz_constant(self):
        """K_f ≤ Π ||W_j||_op."""
        k1 = self._spectral_norm(self.W1)
        k2 = self._spectral_norm(self.W2)
        k3 = self._spectral_norm(self.W3)
        return k1 * k2 * k3

# ============================================================
# STEP 2: Train the Lipschitz network to estimate W1
# ============================================================
# Objective: max_{f: K_f ≤ 1} { E_Q[f(x)] - E_P[f(x)] }
#   maximize  E_Q[f(x)] - E_P[f(x)]
#           ≈ (1/n) Σ_{x∈X_Q} f(x) - (1/n) Σ_{x∈X_P} f(x)
#
# This is exactly the Wasserstein dual: the maximum value
# achieved by a 1-Lipschitz f equals W1(P,Q).

print("\n" + "=" * 65)
print("STEP 1-2: Train Lipschitz Network to Estimate W1")
print("=" * 65)
print(f"\nObjective: max_{{f: K_f≤1}} {{ E_Q[f(x)] - E_P[f(x)] }}")
print(f"Training procedure:")
print(f"  1. Gradient ascent on W_j")
print(f"  2. Project: W_j ← W_j / max(1, ||W_j||_op)")
print(f"  This ensures K_f ≤ 1 (1-Lipschitz).\n")

net = LipschitzMLP(hidden=16)
lr = 0.01
n_epochs = 2000

# Prepare data: (1, n) format
X_P = x_train.reshape(1, -1)    # source samples
X_Q = x_test.reshape(1, -1)     # target samples

history = {'epoch': [], 'W_est': [], 'K_f': []}

for epoch in range(n_epochs):
    # Forward pass on both sets
    f_P = net.forward(X_P)   # f(x) for P samples
    f_Q = net.forward(X_Q)   # f(x) for Q samples -- need separate forward

    # We need to do them separately for proper backprop
    # Objective = mean(f_Q) - mean(f_P), maximize this

    # Forward P
    f_P = net.forward(X_P)
    # Gradient of -mean(f_P) w.r.t. output = -1/n for each
    d_out_P = -np.ones((1, n_train)) / n_train
    net.backward(d_out_P)
    dW1_P, db1_P = net.dW1.copy(), net.db1.copy()
    dW2_P, db2_P = net.dW2.copy(), net.db2.copy()
    dW3_P, db3_P = net.dW3.copy(), net.db3.copy()

    # Forward Q
    f_Q = net.forward(X_Q)
    # Gradient of +mean(f_Q) w.r.t. output = +1/n for each
    d_out_Q = np.ones((1, n_test)) / n_test
    net.backward(d_out_Q)

    # Combine gradients
    net.dW1 = net.dW1 + dW1_P
    net.db1 = net.db1 + db1_P
    net.dW2 = net.dW2 + dW2_P
    net.db2 = net.db2 + db2_P
    net.dW3 = net.dW3 + dW3_P
    net.db3 = net.db3 + db3_P

    # Step with projection
    net.step(lr)

    # Compute current W estimate
    f_P_val = net.forward(X_P)
    f_Q_val = net.forward(X_Q)
    W_est = f_Q_val.mean() - f_P_val.mean()
    K_f = net.lipschitz_constant()

    if epoch % 200 == 0 or epoch == n_epochs - 1:
        history['epoch'].append(epoch)
        history['W_est'].append(W_est)
        history['K_f'].append(K_f)
        if epoch % 400 == 0:
            print(f"  Epoch {epoch:4d}: W_est = {W_est:.4f}, "
                  f"K_f = {K_f:.4f}")

W_est_final = history['W_est'][-1]
K_f_final = history['K_f'][-1]

print(f"\n--- Final Results ---")
print(f"W1 estimated (Lipschitz net):  {W_est_final:.4f}")
print(f"W1 exact (CDF integration):    {W1_exact:.4f}")
print(f"W1 sample (sort-and-pair):     {W1_sample_sort:.4f}")
print(f"Network Lipschitz constant:    {K_f_final:.4f}  (should be ≤ 1)")

# ============================================================
# STEP 3: Key differences from Section 2
# ============================================================
print("\n" + "=" * 65)
print("STEP 3: Section 2 vs Section 3 — Key Differences")
print("=" * 65)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│              SECTION 2                SECTION 3             │
│         (Closed-form W1)      (Lipschitz Network W1)       │
├─────────────────────────────────────────────────────────────┤
│ Method:  Sort samples,        Train neural net f to        │
│          pair i-th order       maximize E_Q[f]-E_P[f]      │
│          statistics            with K_f ≤ 1                │
│                                                             │
│ Formula: W1 = (1/n)Σ|X(i)-Y(i)|   W = sup E_Q[f]-E_P[f]  │
│                                      dual form      │
│                                                             │
│ Works in: 1D ONLY              ANY dimension               │
│           (sorting trick)      (learned witness f)         │
│                                                             │
│ Output:  A number (distance)   A number + the witness f    │
│                                                             │
│ Constraint: None needed        ||W_j||_op ≤ 1 per layer   │
│                                (spectral normalization)    │
│                                                             │
│ Result:  {W1_sample_sort:.4f}                 {W_est_final:.4f}                    │
│ Exact:   {W1_exact:.4f}                 {W1_exact:.4f}                    │
└─────────────────────────────────────────────────────────────┘

WHY use the neural network approach?
- In high dimensions (images, text), sorting doesn't work.
- The dual formulation generalizes to ANY dimension.
- The learned f also tells us WHERE the distributions differ.
- This is the same idea behind WGANs (Wasserstein GANs).
""")

# ============================================================
# STEP 4: Use estimated W1 in the evaluation bound
# ============================================================
from sklearn.linear_model import LogisticRegression

def label_fn(x, threshold=1.5, noise_prob=0.1):
    y = (x > threshold).astype(int)
    flip = np.random.random(len(x)) < noise_prob
    y[flip] = 1 - y[flip]
    return y

np.random.seed(42)
_ = np.random.normal(mu_p, sig_p, n_train)  # consume same random state
_ = np.random.normal(mu_q, sig_q, n_test)
np.random.seed(42)
x_tr = np.random.normal(mu_p, sig_p, n_train)
x_te = np.random.normal(mu_q, sig_q, n_test)
y_tr = label_fn(x_tr)
y_te = label_fn(x_te)

model = LogisticRegression(random_state=42)
model.fit(x_tr.reshape(-1, 1), y_tr)

def log_loss_per_sample(mdl, X, y, clip=1e-3):
    probs = mdl.predict_proba(X)
    losses = np.zeros(len(y))
    for i in range(len(y)):
        p = np.clip(probs[i, int(y[i])], clip, 1.0)
        losses[i] = -np.log(p)
    return losses

losses_tr = log_loss_per_sample(model, x_tr.reshape(-1, 1), y_tr)
losses_te = log_loss_per_sample(model, x_te.reshape(-1, 1), y_te)
E_P = losses_tr.mean()
E_Q = losses_te.mean()
K_l = abs(model.coef_[0][0])

W_bound_sec2 = E_P + K_l * W1_exact
W_bound_sec3 = E_P + K_l * W_est_final

print("=" * 65)
print("STEP 4: Evaluation Bound using both W1 estimates")
print("=" * 65)
print(f"\nBound: E_Q[l] ≤ E_P[l] + K_l · W(P,Q)")
print(f"\nE_P[l] = {E_P:.4f},  K_l = {K_l:.4f}")
print(f"\nUsing Section 2 W1 (sort):   {W1_sample_sort:.4f}  → bound = {E_P + K_l * W1_sample_sort:.4f}")
print(f"Using Section 2 W1 (exact):  {W1_exact:.4f}  → bound = {W_bound_sec2:.4f}")
print(f"Using Section 3 W1 (net):    {W_est_final:.4f}  → bound = {W_bound_sec3:.4f}")
print(f"Actual E_Q:                                  {E_Q:.4f}")

# ============================================================
# FIGURES
# ============================================================

# --- Figure 1: Training convergence + learned witness ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) W estimate convergence
axes[0].plot(history['epoch'], history['W_est'], 'b-o', ms=4, lw=1.5)
axes[0].axhline(y=W1_exact, color='red', ls='--', lw=1.5, label=f'Exact W₁={W1_exact:.2f}')
axes[0].axhline(y=W1_sample_sort, color='green', ls=':', lw=1.5,
                label=f'Sort-pair W₁={W1_sample_sort:.2f}')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('W₁ estimate')
axes[0].set_title('(a) Convergence of W₁ Estimate')
axes[0].legend(fontsize=8)

# (b) Lipschitz constant over training
axes[1].plot(history['epoch'], history['K_f'], 'r-o', ms=4, lw=1.5)
axes[1].axhline(y=1.0, color='gray', ls='--', alpha=0.5, label='K_f = 1 (constraint)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('K_f (Lipschitz constant)')
axes[1].set_title('(b) Network Lipschitz Constant\n'
                   'K_f = Π ||W_j||_op ≤ 1')
axes[1].legend(fontsize=8)
axes[1].set_ylim(0, 1.5)

# (c) Learned witness function f(x) vs data
x_plot = np.linspace(-5, 7, 300).reshape(1, -1)
f_plot = net.forward(x_plot).flatten()

axes[2].plot(x_plot.flatten(), f_plot, 'k-', lw=2, label='Learned f(x)')
# Show P and Q densities scaled
x_range = x_plot.flatten()
axes[2].fill_between(x_range, 0, norm.pdf(x_range, mu_p, sig_p) * 2,
                      alpha=0.2, color='blue', label='P(x) scaled')
axes[2].fill_between(x_range, 0, norm.pdf(x_range, mu_q, sig_q) * 2,
                      alpha=0.2, color='red', label='Q(x) scaled')
axes[2].scatter(x_train, net.forward(x_train.reshape(1, -1)).flatten(),
                c='blue', s=30, zorder=5, alpha=0.7, label='f(x_P)')
axes[2].scatter(x_test, net.forward(x_test.reshape(1, -1)).flatten(),
                c='red', s=30, zorder=5, alpha=0.7, label='f(x_Q)')
axes[2].set_xlabel('x')
axes[2].set_ylabel('f(x)')
axes[2].set_title(f'(c) Learned Witness Function\n'
                   f'W₁ = E_Q[f]-E_P[f] = {W_est_final:.3f}')
axes[2].legend(fontsize=7, loc='upper left')

plt.tight_layout()
fig1_path = os.path.join(output_dir, "fig_lipschitz_net_training.pdf")
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {fig1_path}")

# --- Figure 2: Projection step illustration ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Weight projection diagram
ax = axes[0]
ax.axis('off')
text = (
    "Training Loop\n"
    "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "For t = 1, ..., T:\n"
    "  For j = 1, ..., m (each layer):\n\n"
    "    (1) Gradient ascent:\n"
    "        W_j <- W_j + a * grad_{W_j} L(W_j; Z)\n\n"
    "    (2) Spectral projection:\n"
    "        W_j <- W_j / max(1, ||W_j||_op)\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━\n"
    "This ensures K_{gⱼ} ≤ 1 per layer.\n"
    "Since K_ReLU = 1:\n"
    "  K_f ≤ K_{W₃}·1·K_{W₂}·1·K_{W₁} ≤ 1"
)
ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
ax.set_title('Training Procedure', fontsize=12, fontweight='bold')

# (b) Comparison bar chart
ax = axes[1]
methods = ['Sort-Pair\n(Sec. 2)', 'CDF\n(Sec. 2)', 'Lipschitz Net\n(Sec. 3)', 'Exact']
values = [W1_sample_sort, W1_exact, W_est_final, W1_exact]
colors = ['#4472C4', '#548235', '#C00000', 'gray']
bars = ax.bar(methods, values, color=colors, width=0.6, edgecolor='black')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('W₁ estimate')
ax.set_title('W₁ Estimation Methods Compared')
ax.set_ylim(0, max(values) * 1.2)

plt.tight_layout()
fig2_path = os.path.join(output_dir, "fig_lipschitz_net_comparison.pdf")
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fig2_path}")

# --- Figure 3: Lipschitz verification ---
fig, ax = plt.subplots(figsize=(8, 5))

# Verify Lipschitz: |f(a) - f(b)| / |a - b| ≤ 1 for all pairs
x_check = np.linspace(-4, 7, 100).reshape(1, -1)
f_check = net.forward(x_check).flatten()
x_flat = x_check.flatten()

# Compute pairwise ratios (subsample for speed)
ratios = []
pairs_x = []
pairs_r = []
step = 3
for i in range(0, len(x_flat), step):
    for j in range(i + 1, len(x_flat), step):
        dx = abs(x_flat[i] - x_flat[j])
        df = abs(f_check[i] - f_check[j])
        if dx > 1e-6:
            r = df / dx
            ratios.append(r)
            pairs_x.append((x_flat[i] + x_flat[j]) / 2)
            pairs_r.append(r)

ax.scatter(pairs_x, pairs_r, s=5, alpha=0.3, c='blue')
ax.axhline(y=1.0, color='red', ls='--', lw=2,
           label='Lipschitz bound K=1')
ax.set_xlabel('Midpoint of pair')
ax.set_ylabel('|f(a)-f(b)| / |a-b|')
ax.set_title('Lipschitz Verification: All Pairwise Slopes ≤ 1')
ax.legend()
ax.set_ylim(0, 1.5)
ax.text(0.02, 0.95, f'Max slope: {max(ratios):.4f}',
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat'))

fig3_path = os.path.join(output_dir, "fig_lipschitz_verification.pdf")
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fig3_path}")

print("\n" + "=" * 65)
print("All figures saved.")
print("=" * 65)