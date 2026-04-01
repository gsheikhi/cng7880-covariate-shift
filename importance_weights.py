"""
Importance Weights for Covariate Shift - Numerical Toy Example
==============================================================
CNG7880 Week 6

Key formulas implemented:
- E_Q[l(θ;x,y)] = E_P[l(θ;x,y) · w(x)]  where w(x) = q(x)/p(x)
- Estimating w(x) via discriminator:
    1. Construct X' = {(x,0) | (x,y) ∈ Z} ∪ {(x,1) | x ∈ X}
    2. Train discriminator ĝ on X' to predict b given x
    3. w(x) = 1/ĝ(b=0|x) - 1  
- Weighted ERM: θ̂ = argmin_θ Σ_{(x,y)∈Z} l(θ;x,y)·w(x)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# ============================================================
# STEP 1: Create distributions P (train) and Q (test)
# Covariate shift: p(x) ≠ q(x), but p(y|x) = q(y|x)
# ============================================================

n_train = 20
n_test = 20

# P: training distribution - x ~ N(0, 1)
x_train = np.random.normal(loc=0.0, scale=1.0, size=n_train)

# Q: test distribution - x ~ N(2, 1)  (shifted mean)
x_test = np.random.normal(loc=2.0, scale=1.0, size=n_test)

# Shared labeling function: p(y|x) = q(y|x)
# y = 1 if x > 1 else 0, with some noise
def label_fn(x, noise_prob=0.1):
    """Same labeling rule for both distributions."""
    y = (x > 1.0).astype(int)
    # Add label noise
    flip = np.random.random(len(x)) < noise_prob
    y[flip] = 1 - y[flip]
    return y

y_train = label_fn(x_train)
y_test = label_fn(x_test)

print("=" * 60)
print("STEP 1: Sampled Data")
print("=" * 60)
print(f"Training (P): x ~ N(0,1), n={n_train}")
print(f"  x_train[:5] = {x_train[:5].round(3)}")
print(f"  y_train[:5] = {y_train[:5]}")
print(f"Test (Q):      x ~ N(2,1), n={n_test}")
print(f"  x_test[:5]  = {x_test[:5].round(3)}")
print(f"  y_test[:5]  = {y_test[:5]}")

# ============================================================
# STEP 2: Estimate importance weights w(x) via discriminator
# ============================================================

# --- Step 2a: Construct X' ---
# X' = {(x, 0) | (x,y) ∈ Z} ∪ {(x, 1) | x ∈ X}
# b=0 → training (source P), b=1 → test (target Q)
x_disc = np.concatenate([x_train, x_test]).reshape(-1, 1)
b_disc = np.concatenate([np.zeros(n_train), np.ones(n_test)])

print("\n" + "=" * 60)
print("STEP 2a: Construct X' for discriminator")
print("=" * 60)
print(f"X' has {len(x_disc)} samples: {n_train} with b=0 (train), {n_test} with b=1 (test)")

# --- Step 2b: Train discriminator ĝ ---
# ĝ predicts b given x → estimates r(b|x)
discriminator = LogisticRegression(random_state=42)
discriminator.fit(x_disc, b_disc)

# Discriminator accuracy on training data
disc_acc = discriminator.score(x_disc, b_disc)
print(f"\nDiscriminator accuracy on X': {disc_acc:.3f}")
print(f"  If acc ≈ 0.5 → small shift")
print(f"  If acc >> 0.5 → large shift")

# --- Step 2c: Compute importance weights ---
# w(x) = 1/r(b=0|x) - 1   where r(b=0|x) = ĝ(b=0|x)
# Equivalently: w(x) = r(b=1|x) / r(b=0|x) = q(x)/p(x)

# Get P(b=0|x) for training samples
prob_b0_train = discriminator.predict_proba(x_train.reshape(-1, 1))[:, 0]

# Clip to avoid division by zero
prob_b0_train = np.clip(prob_b0_train, 0.01, 0.99)

# w(x) = 1/r(b=0|x) - 1 
w_x = 1.0 / prob_b0_train - 1.0

print("\n" + "=" * 60)
print("STEP 2c: Importance Weights w(x) = 1/r(b=0|x) - 1")
print("=" * 60)
print(f"{'x_train':>10s} {'r(b=0|x)':>10s} {'w(x)':>10s}")
print("-" * 35)
for i in range(n_train):
    print(f"{x_train[i]:10.3f} {prob_b0_train[i]:10.4f} {w_x[i]:10.4f}")

print(f"\nMean w(x): {w_x.mean():.4f}")
print(f"Std  w(x): {w_x.std():.4f}")

# ============================================================
# STEP 3: Compare standard vs weighted ERM
# θ̂ = argmin_θ Σ_{(x,y)∈Z} l(θ;x,y)·w(x)
# ============================================================

# Standard ERM (no weighting)
model_standard = LogisticRegression(random_state=42)
model_standard.fit(x_train.reshape(-1, 1), y_train)

# Weighted ERM using importance weights
model_weighted = LogisticRegression(random_state=42)
model_weighted.fit(x_train.reshape(-1, 1), y_train, sample_weight=w_x)

# Evaluate on TEST distribution Q
acc_standard = model_standard.score(x_test.reshape(-1, 1), y_test)
acc_weighted = model_weighted.score(x_test.reshape(-1, 1), y_test)

print("\n" + "=" * 60)
print("STEP 3: Standard vs Weighted ERM")
print("=" * 60)
print(f"Standard ERM accuracy on Q: {acc_standard:.3f}")
print(f"Weighted ERM accuracy on Q: {acc_weighted:.3f}")
print(f"θ_standard: coef={model_standard.coef_[0][0]:.4f}, "
      f"intercept={model_standard.intercept_[0]:.4f}")
print(f"θ_weighted: coef={model_weighted.coef_[0][0]:.4f}, "
      f"intercept={model_weighted.intercept_[0]:.4f}")

# ============================================================
# STEP 4: Verify the identity E_Q[l] = E_P[l · w(x)] with finite samples
# ============================================================

def log_loss_per_sample(model, X, y):
    """Compute per-sample log loss."""
    probs = model.predict_proba(X)
    losses = np.zeros(len(y))
    for i in range(len(y)):
        p = np.clip(probs[i, int(y[i])], 1e-10, 1.0)
        losses[i] = -np.log(p)
    return losses

# Use the standard model for verification
losses_test = log_loss_per_sample(model_standard, x_test.reshape(-1, 1), y_test)
losses_train = log_loss_per_sample(model_standard, x_train.reshape(-1, 1), y_train)

E_Q = losses_test.mean()
E_P_weighted = (losses_train * w_x).mean()
E_P_unweighted = losses_train.mean()

print("\n" + "=" * 60)
print("STEP 4: Verify E_Q[l] ≈ E_P[l·w(x)]  with finite samples")
print("=" * 60)
print(f"E_Q[l(θ;x,y)]           = {E_Q:.4f}")
print(f"E_P[l(θ;x,y)·w(x)]     = {E_P_weighted:.4f}")
print(f"E_P[l(θ;x,y)] (no wt)  = {E_P_unweighted:.4f}")
print(f"\nNote: E_Q ≈ E_P[l·w] holds approximately with finite samples.")

# ============================================================
# FIGURES
# ============================================================

output_dir = "./outputs"

# --- Figure 1: Distribution shift visualization ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel (a): P(x) vs Q(x)
x_range = np.linspace(-4, 6, 200)
from scipy.stats import norm
axes[0].plot(x_range, norm.pdf(x_range, 0, 1), 'b-', lw=2, label='P(x): N(0,1)')
axes[0].plot(x_range, norm.pdf(x_range, 2, 1), 'r--', lw=2, label='Q(x): N(2,1)')
axes[0].scatter(x_train, np.zeros_like(x_train) - 0.02, c='blue', marker='|', s=100, label='Train samples')
axes[0].scatter(x_test, np.zeros_like(x_test) - 0.05, c='red', marker='|', s=100, label='Test samples')
axes[0].set_xlabel('x')
axes[0].set_ylabel('Density')
axes[0].set_title('(a) Covariate Shift: P(x) vs Q(x)')
axes[0].legend(fontsize=8)

# Panel (b): Importance weights
sorted_idx = np.argsort(x_train)
axes[1].bar(range(n_train), w_x[sorted_idx], color='green', alpha=0.7)
axes[1].set_xlabel('Training sample (sorted by x)')
axes[1].set_ylabel('w(x) = q(x)/p(x)')
axes[1].set_title('(b) Importance Weights')
# Add x values as labels
for i, idx in enumerate(sorted_idx):
    if i % 4 == 0:
        axes[1].text(i, w_x[idx] + 0.05, f'{x_train[idx]:.1f}', ha='center', fontsize=7)

# Panel (c): Discriminator probabilities
x_plot = np.linspace(-4, 6, 200).reshape(-1, 1)
prob_test_plot = discriminator.predict_proba(x_plot)[:, 1]
axes[2].plot(x_plot, prob_test_plot, 'k-', lw=2, label='r(b=1|x)')
axes[2].plot(x_plot, 1 - prob_test_plot, 'gray', lw=2, ls='--', label='r(b=0|x)')
axes[2].axhline(y=0.5, color='orange', ls=':', alpha=0.5)
axes[2].set_xlabel('x')
axes[2].set_ylabel('Probability')
axes[2].set_title('(c) Discriminator r(b|x)')
axes[2].legend(fontsize=8)

plt.tight_layout()
fig1_path = os.path.join(output_dir, "fig_importance_weights.pdf")
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {fig1_path}")

# --- Figure 2: Weighted vs Standard ERM decision boundaries ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel (a): Standard ERM
x_plot = np.linspace(-4, 6, 200).reshape(-1, 1)
prob_std = model_standard.predict_proba(x_plot)[:, 1]
prob_wt = model_weighted.predict_proba(x_plot)[:, 1]

for ax, prob, title, model_name in [
    (axes[0], prob_std, '(a) Standard ERM', 'Standard'),
    (axes[1], prob_wt, '(b) Weighted ERM (w(x))', 'Weighted')
]:
    ax.plot(x_plot, prob, 'k-', lw=2, label=f'{model_name} P(y=1|x)')
    # Training data
    ax.scatter(x_train[y_train == 0], np.zeros(sum(y_train == 0)) - 0.05,
               c='blue', marker='o', s=50, label='Train y=0', alpha=0.7)
    ax.scatter(x_train[y_train == 1], np.ones(sum(y_train == 1)) + 0.05,
               c='blue', marker='^', s=50, label='Train y=1', alpha=0.7)
    # Test data
    ax.scatter(x_test[y_test == 0], np.zeros(sum(y_test == 0)) - 0.1,
               c='red', marker='o', s=50, label='Test y=0', alpha=0.7)
    ax.scatter(x_test[y_test == 1], np.ones(sum(y_test == 1)) + 0.1,
               c='red', marker='^', s=50, label='Test y=1', alpha=0.7)
    ax.axhline(y=0.5, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('P(y=1|x)')
    ax.set_title(title)
    ax.legend(fontsize=7, loc='center left')
    ax.set_ylim(-0.2, 1.2)

plt.tight_layout()
fig2_path = os.path.join(output_dir, "fig_weighted_vs_standard_erm.pdf")
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fig2_path}")

# --- Figure 3: Weight table ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Create table data
sorted_idx = np.argsort(x_train)
table_data = []
for i in sorted_idx:
    table_data.append([
        f"{i}", f"{x_train[i]:.3f}", f"{y_train[i]}",
        f"{prob_b0_train[i]:.4f}", f"{w_x[i]:.4f}"
    ])

col_labels = ['Sample', 'x', 'y', 'r(b=0|x)', 'w(x)']
table = ax.table(cellText=table_data, colLabels=col_labels,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.3)

# Color header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')

ax.set_title('Importance Weights per Training Sample\n'
             'w(x) = 1/r(b=0|x) - 1', fontsize=12, fontweight='bold')

fig3_path = os.path.join(output_dir, "fig_weight_table.pdf")
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fig3_path}")

print("\n" + "=" * 60)
print("All figures saved. Check the 'outputs' directory.")
print("=" * 60)