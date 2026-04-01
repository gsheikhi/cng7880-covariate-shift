"""
Covariate Shift Detection — Provable Bounds on Test Accuracy
==============================================================
CNG7880 Week 6

The idea: Train a discriminator to distinguish P from Q samples.
If its accuracy significantly exceeds 0.5, there is covariate shift.

KEY FORMULAS:
-------------
1. Train discriminator ĝ on X' = {(x,0)|x∈X_P} ∪ {(x,1)|x∈X_Q}

2. Accuracy(ĝ, X'') = (1/n) Σ 1(ĝ(x_i) = b_i) 

3. Under H0: P = Q, each prediction is a fair coin flip:
   z_i := 1(ĝ(x_i) = b_i)  is Bernoulli(1/2)
   S = Σ z_i ~ Binomial(n, 1/2)           

4. Detect shift if Accuracy ≥ 1/2 + ε      
   i.e., S ≥ ⌈n(1/2 + ε)⌉                  

5. Choose ε so that false positive rate ≤ α: 
   P[Detector = 1 | P = Q] = Σ_{i=⌈n(1/2+ε)⌉}^{n} Binomial(i; n, 1/2) ≤ α

6. Given α (e.g., 0.05), find the smallest ε such that this holds.
"""

import numpy as np
from scipy.stats import binom, norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

np.random.seed(42)
output_dir = "./outputs"

# ============================================================
# STEP 0: Same data as Sections 2-3
# ============================================================
mu_p, sig_p = 0.0, 1.0
mu_q, sig_q = 3.0, 0.8
n_source = 20
n_target = 20

x_source = np.random.normal(mu_p, sig_p, n_source)
x_target = np.random.normal(mu_q, sig_q, n_target)

print("=" * 65)
print("STEP 0: Same data as Sections 2-3")
print("=" * 65)
print(f"P: N({mu_p},{sig_p}²), n={n_source}")
print(f"Q: N({mu_q},{sig_q}²), n={n_target}")

# ============================================================
# STEP 1: Construct X' and split into train/test 
# ============================================================
# X' = {(x, 0) | x ∈ X_P} ∪ {(x, 1) | x ∈ X_Q}
x_all = np.concatenate([x_source, x_target])
b_all = np.concatenate([np.zeros(n_source), np.ones(n_target)])

# Split into X' (train discriminator) and X'' (evaluate)
x_train, x_held, b_train, b_held = train_test_split(
    x_all, b_all, test_size=0.5, random_state=42, stratify=b_all
)

n_held = len(x_held)  # this is n in the formulas

print(f"\n" + "=" * 65)
print("STEP 1: Construct X' and X'', split into train/test")
print("=" * 65)
print(f"X' (train discriminator): {len(x_train)} samples")
print(f"X'' (held-out test):      {n_held} samples")
print(f"  of which {int(sum(b_held==0))} from P (b=0), "
      f"{int(sum(b_held==1))} from Q (b=1)")

# ============================================================
# STEP 2: Train discriminator ĝ on X'
# ============================================================
disc = LogisticRegression(random_state=42)
disc.fit(x_train.reshape(-1, 1), b_train)

train_acc = disc.score(x_train.reshape(-1, 1), b_train)

print(f"\n" + "=" * 65)
print("STEP 2: Train discriminator ĝ on X'")
print("=" * 65)
print(f"Discriminator train accuracy: {train_acc:.4f}")

# ============================================================
# STEP 3: Evaluate on held-out X''
# ============================================================
# Accuracy(ĝ, X'') = (1/n) Σ_{i=1}^{n} 1(ĝ(x_i) = b_i)
predictions = disc.predict(x_held.reshape(-1, 1))
correct = (predictions == b_held).astype(int)
S = int(correct.sum())                    # number of correct predictions
acc_held = S / n_held                     # held-out accuracy

print(f"\n" + "=" * 65)
print("STEP 3: Evaluate on X'' and compute accuracy")
print("=" * 65)
print(f"\nAccuracy(ĝ, X'') = (1/n) Σ 1(ĝ(x_i) = b_i)")
print(f"\nPer-sample results on X'':")
print(f"{'i':>3s} {'x_i':>8s} {'b_i':>5s} {'ĝ(x_i)':>7s} {'correct':>8s}")
print("-" * 35)
for i in range(n_held):
    print(f"{i:3d} {x_held[i]:8.3f} {int(b_held[i]):5d} "
          f"{int(predictions[i]):7d} {correct[i]:8d}")
print(f"\nS = Σ correct = {S}")
print(f"n = {n_held}")
print(f"Accuracy = S/n = {S}/{n_held} = {acc_held:.4f}")

# ============================================================
# STEP 4: Under H0 (P=Q), S ~ Binomial(n, 1/2)
# ============================================================
print(f"\n" + "=" * 65)
print("STEP 4: Null distribution of S under H0: P = Q")
print("=" * 65)
print(f"\nUnder H0: P = Q")
print(f"  z_i = 1(ĝ(x_i) = b_i) ~ Bernoulli(1/2)")
print(f"  (random guessing, each prediction is a coin flip)")
print(f"\n  S = Σ z_i ~ Binomial(n={n_held}, p=1/2)")
print(f"  E[S] = n/2 = {n_held/2:.1f}")
print(f"  Std[S] = sqrt(n/4) = {np.sqrt(n_held/4):.2f}")

# ============================================================
# STEP 5: Find ε given α = 0.05
# ============================================================
# We need the smallest ε such that:
#   Σ_{i=⌈n(1/2+ε)⌉}^{n} Binomial(i; n, 1/2) ≤ α
#
# Equivalently, find the smallest threshold k such that:
#   P[S ≥ k | S ~ Bin(n, 0.5)] ≤ α
# Then ε = k/n - 1/2

alpha = 0.05

# Find critical value k
for k in range(n_held + 1):
    # P[S ≥ k] = 1 - P[S ≤ k-1] = 1 - F(k-1)
    tail_prob = 1 - binom.cdf(k - 1, n_held, 0.5)
    if tail_prob <= alpha:
        k_critical = k
        tail_prob_at_k = tail_prob
        break

epsilon = k_critical / n_held - 0.5
acc_threshold = 0.5 + epsilon

print(f"\n" + "=" * 65)
print(f"STEP 5: Find ε for α = {alpha}")
print("=" * 65)
print(f"\nGoal: Σ_{{i=⌈n(1/2+ε)⌉}}^{{n}} Binomial(i; n, 1/2) ≤ α")
print(f"\nSearch over possible thresholds k:")
print(f"{'k':>4s} {'P[S≥k]':>10s} {'≤ α?':>6s} {'ε = k/n - 0.5':>14s}")
print("-" * 38)
for k_try in range(max(0, n_held // 2 - 2), min(n_held + 1, n_held // 2 + 8)):
    tp = 1 - binom.cdf(k_try - 1, n_held, 0.5)
    is_valid = "YES" if tp <= alpha else "no"
    eps_try = k_try / n_held - 0.5
    marker = " <-- chosen" if k_try == k_critical else ""
    print(f"{k_try:4d} {tp:10.6f} {is_valid:>6s} {eps_try:14.4f}{marker}")

print(f"\nResult:")
print(f"  k_critical = {k_critical}")
print(f"  ε = {k_critical}/{n_held} - 0.5 = {epsilon:.4f}")
print(f"  Accuracy threshold = 1/2 + ε = {acc_threshold:.4f}")
print(f"  P[S ≥ {k_critical} | H0] = {tail_prob_at_k:.6f} ≤ α = {alpha}")

# ============================================================
# STEP 6: Apply the detector to our observed accuracy
# ============================================================
shift_detected = acc_held >= acc_threshold

print(f"\n" + "=" * 65)
print(f"STEP 6: Apply detector to observed accuracy")
print("=" * 65)
print(f"\nDecision rule: Detect shift if Accuracy(ĝ, X'') ≥ 1/2 + ε")
print(f"\n  Observed accuracy: {acc_held:.4f}")
print(f"  Threshold (1/2+ε): {acc_threshold:.4f}")
print(f"  S = {S},  k_critical = {k_critical}")
print(f"\n  S ≥ k_critical?  {S} ≥ {k_critical}?  {shift_detected}")
print(f"\n  RESULT: Covariate shift {'DETECTED' if shift_detected else 'NOT detected'}"
      f" at {(1-alpha)*100:.0f}% confidence")

# ============================================================
# STEP 7: Provable bound on TRUE discriminator accuracy
# ============================================================
# From the test set accuracy, we can bound the true accuracy.
# Using Clopper-Pearson exact binomial confidence interval:
#   P[true_acc ≥ lower_bound] ≥ 1 - α

from scipy.stats import beta as beta_dist

# Clopper-Pearson lower bound
if S > 0:
    cp_lower = beta_dist.ppf(alpha / 2, S, n_held - S + 1)
else:
    cp_lower = 0.0
if S < n_held:
    cp_upper = beta_dist.ppf(1 - alpha / 2, S + 1, n_held - S)
else:
    cp_upper = 1.0

print(f"\n" + "=" * 65)
print(f"STEP 7: Provable bound on true accuracy")
print("=" * 65)
print(f"\nObserved: S = {S} correct out of n = {n_held}")
print(f"Test accuracy = {acc_held:.4f}")
print(f"\n{(1-alpha)*100:.0f}% Clopper-Pearson confidence interval:")
print(f"  P[ {cp_lower:.4f} ≤ true_accuracy ≤ {cp_upper:.4f} ] ≥ {1-alpha:.2f}")
print(f"\nInterpretation:")
print(f"  We can provably state that the discriminator's true accuracy")
print(f"  is at least {cp_lower:.4f} with {(1-alpha)*100:.0f}% confidence.")
if cp_lower > 0.5:
    print(f"  Since {cp_lower:.4f} > 0.5, this is provable evidence of shift.")
else:
    print(f"  Since {cp_lower:.4f} ≤ 0.5, we cannot provably claim shift.")

# ============================================================
# STEP 8: Full summary table
# ============================================================
print(f"\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"""
┌───────────────────────────────────────────────────────────┐
│  Covariate Shift Detection               │
├───────────────────────────────────────────────────────────┤
│  α (significance level):          {alpha}                    │
│  n (held-out samples):            {n_held}                    │
│  S (correct predictions):         {S}                    │
│  Observed accuracy:               {acc_held:.4f}               │
│                                                           │
│  ε (computed from α):             {epsilon:.4f}               │
│  Accuracy threshold (1/2 + ε):    {acc_threshold:.4f}               │
│  k_critical:                      {k_critical}                    │
│  P[S ≥ k | H0] (p-value):        {tail_prob_at_k:.6f}           │
│                                                           │
│  Shift detected?                  {'YES' if shift_detected else 'NO':3s}                  │
│  95% CI for true accuracy:        [{cp_lower:.4f}, {cp_upper:.4f}]    │
└───────────────────────────────────────────────────────────┘
""")

# ============================================================
# FIGURES
# ============================================================

# --- Figure 1: Null distribution + observed S ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) Binomial null distribution
k_vals = np.arange(0, n_held + 1)
pmf_vals = binom.pmf(k_vals, n_held, 0.5)

colors_bar = ['#C00000' if k >= k_critical else '#4472C4' for k in k_vals]
axes[0].bar(k_vals, pmf_vals, color=colors_bar, edgecolor='black', linewidth=0.5)
axes[0].axvline(x=S, color='black', ls='-', lw=2, label=f'Observed S={S}')
axes[0].axvline(x=k_critical, color='red', ls='--', lw=1.5,
                label=f'k_crit={k_critical} (ε={epsilon:.3f})')
axes[0].set_xlabel('S (# correct predictions)')
axes[0].set_ylabel('P[S = k | H0: P=Q]')
axes[0].set_title(f'(a) Null Distribution: S ~ Bin({n_held}, 0.5)\n'
                   f'Red region: P[S≥{k_critical}] = {tail_prob_at_k:.4f} ≤ α={alpha}')
axes[0].legend(fontsize=8)

# (b) Tail probability as function of k
tail_probs = []
for k in k_vals:
    tp = 1 - binom.cdf(k - 1, n_held, 0.5) if k > 0 else 1.0
    tail_probs.append(tp)

axes[1].plot(k_vals, tail_probs, 'b-o', ms=3, lw=1.5)
axes[1].axhline(y=alpha, color='red', ls='--', lw=1.5, label=f'α = {alpha}')
axes[1].axvline(x=k_critical, color='gray', ls=':', alpha=0.5)
axes[1].set_xlabel('k (threshold)')
axes[1].set_ylabel('P[S ≥ k | H0]')
axes[1].set_title(f'(b) Tail Probability vs Threshold\n'
                   f'Find smallest k where P[S≥k] ≤ α')
axes[1].legend(fontsize=8)
axes[1].set_ylim(-0.05, 1.05)

# Annotate the critical point
axes[1].annotate(f'k={k_critical}\nP={tail_prob_at_k:.4f}',
                 xy=(k_critical, tail_prob_at_k),
                 xytext=(k_critical + 2, tail_prob_at_k + 0.15),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=9, color='red')

# (c) Epsilon as function of alpha
alphas_range = np.arange(0.01, 0.50, 0.01)
epsilons_range = []
for a in alphas_range:
    for kk in range(n_held + 1):
        tp = 1 - binom.cdf(kk - 1, n_held, 0.5)
        if tp <= a:
            epsilons_range.append(kk / n_held - 0.5)
            break
    else:
        epsilons_range.append(0.5)

axes[2].plot(alphas_range, epsilons_range, 'g-', lw=2)
axes[2].axvline(x=alpha, color='red', ls='--', label=f'α={alpha}')
axes[2].axhline(y=epsilon, color='gray', ls=':', alpha=0.5)
axes[2].scatter([alpha], [epsilon], c='red', s=100, zorder=5,
                label=f'ε={epsilon:.3f}')
axes[2].set_xlabel('α (significance level)')
axes[2].set_ylabel('ε (accuracy margin)')
axes[2].set_title(f'(c) Required ε vs α for n={n_held}\n'
                   f'Smaller α → larger ε needed')
axes[2].legend(fontsize=8)

plt.tight_layout()
fig1_path = os.path.join(output_dir, "fig_shift_detection.pdf")
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fig1_path}")

# --- Figure 2: Decision visualization ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) Discriminator decision on X''
x_plot = np.linspace(-5, 7, 300).reshape(-1, 1)
prob_q = disc.predict_proba(x_plot)[:, 1]

axes[0].plot(x_plot, prob_q, 'k-', lw=2, label='ĝ(b=1|x)')
axes[0].axhline(y=0.5, color='gray', ls=':', alpha=0.3)

# Mark held-out samples
for i in range(n_held):
    c = 'green' if correct[i] else 'red'
    m = 'o' if b_held[i] == 0 else '^'
    axes[0].scatter(x_held[i], disc.predict_proba(x_held[i].reshape(-1, 1))[0, 1],
                    c=c, marker=m, s=60, edgecolors='black', linewidth=0.5,
                    zorder=5)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
           markersize=8, label='Correct (b=0, P)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
           markersize=8, label='Correct (b=1, Q)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markersize=8, label='Wrong'),
]
axes[0].legend(handles=legend_elements, fontsize=8, loc='center left')
axes[0].set_xlabel('x')
axes[0].set_ylabel('P(b=1|x)')
axes[0].set_title(f'(a) Discriminator on X\'\'\n'
                   f'Accuracy = {S}/{n_held} = {acc_held:.3f}')

# (b) Summary diagram
ax = axes[1]
ax.axis('off')

summary_text = (
    f"Covariate Shift Detection\n"
    f"{'='*40}\n\n"
    f"Step 1: Train g on X'\n"
    f"  X' = {{(x,0)|x in X_P}} U {{(x,1)|x in X_Q}}\n\n"
    f"Step 2: Find epsilon for alpha = {alpha}\n"
    f"  Sum_{{i=ceil(n(1/2+eps))}}^n Bin(i;n,1/2) <= alpha\n"
    f"  epsilon = {epsilon:.4f}\n\n"
    f"Step 3: Test on held-out X''\n"
    f"  Accuracy = {acc_held:.4f}\n"
    f"  Threshold = 1/2 + eps = {acc_threshold:.4f}\n\n"
    f"  {acc_held:.4f} >= {acc_threshold:.4f}? "
    f"{'YES -> SHIFT DETECTED' if shift_detected else 'NO -> No shift'}\n\n"
    f"Step 4: Provable bound\n"
    f"  95% CI: [{cp_lower:.4f}, {cp_upper:.4f}]\n"
    f"  True acc >= {cp_lower:.4f} with 95% confidence"
)

ax.text(0.05, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
ax.set_title('(b) Detection Summary', fontsize=12, fontweight='bold')

plt.tight_layout()
fig2_path = os.path.join(output_dir, "fig_shift_detection_decision.pdf")
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fig2_path}")

# --- Figure 3: Provable accuracy bound ---
fig, ax = plt.subplots(figsize=(8, 5))

# Show binomial PMF under H0, shade rejection region and mark observation
ax.bar(k_vals, pmf_vals, color='lightblue', edgecolor='gray', linewidth=0.5,
       label='Bin(n, 0.5) under H0')

# Shade rejection region
reject_mask = k_vals >= k_critical
ax.bar(k_vals[reject_mask], pmf_vals[reject_mask], color='#C00000',
       edgecolor='gray', linewidth=0.5, label=f'Rejection region (S>={k_critical})')

# Mark observed S
ax.axvline(x=S, color='black', ls='-', lw=2.5, label=f'Observed S={S}')

# Shade acceptance region text
ax.annotate(f'Cannot reject H0\n(accuracy ~ chance)',
            xy=(n_held // 2, max(pmf_vals) * 0.8),
            fontsize=9, ha='center', color='blue')

if S >= k_critical:
    ax.annotate(f'S={S} is in\nrejection region\n-> Shift detected!',
                xy=(S, binom.pmf(S, n_held, 0.5)),
                xytext=(S - 3, max(pmf_vals) * 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')

ax.set_xlabel('S (number of correct predictions)', fontsize=11)
ax.set_ylabel('Probability under H0', fontsize=11)
ax.set_title(f'Classifier Test for Covariate Shift\n'
             f'n={n_held}, α={alpha}, ε={epsilon:.3f}, '
             f'threshold={k_critical}', fontsize=12)
ax.legend(fontsize=9)

fig3_path = os.path.join(output_dir, "fig_classifier_test.pdf")
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fig3_path}")

print("\n" + "=" * 65)
print("All figures saved.")
print("=" * 65)