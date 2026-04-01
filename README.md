# Covariate Shift — Toy Examples

Numerical toy examples for the covariate shift topics covered in CNG7880 Week 6. These are meant to clarify the formulas from the lecture notes using small, concrete numbers (not deep learning). Each script produces structured console output and PDF figures.

All scripts use, 20 train and 20 test samples, shared labelling rule p(y|x) = q(y|x). So the shift is only in the covariates, not the labels.

## Scripts

### `importance_weights.py`
Importance weighting for robust training when Q is contained in P. Trains a discriminator to estimate w(x) = q(x)/p(x) via the formula w(x) = 1/r(b=0|x) − 1, then compares standard ERM vs weighted ERM on the test distribution.

**Figures:** `fig_importance_weights.pdf`, `fig_weighted_vs_standard_erm.pdf`, `fig_weight_table.pdf`

### `evaluation_bounds.py`
Evaluation bounds using TV and Wasserstein distance. P and Q have partial support overlap here — Q extends into regions where P has almost no mass. Computes TV and W₁ from samples, then shows the two bounds:
- TV bound: E_Q[l] ≤ E_P[l] + l_max · TV (loose, uses worst-case loss)
- W bound: E_Q[l] ≤ E_P[l] + K_l · W (tighter, uses Lipschitz constant of the loss)

Also clarifies what f(x,y) actually is in the Wasserstein dual — it is a 1-Lipschitz witness function, not the loss.

**Figures:** `fig_eval_bounds_distances.pdf`, `fig_eval_bounds_comparison.pdf`, `fig_eval_bounds_table.pdf`, `fig_witness_function.pdf`

### `lipschitz_evaluation_bound.py`
Estimates W₁ by training a Lipschitz-constrained neural network (the dual formulation approach). Built from scratch in NumPy — no PyTorch needed. After each gradient step, weight matrices are projected via spectral normalisation to enforce K_f ≤ 1. This is the same idea behind WGAN critics, and it generalises to any dimension unlike the sort-and-pair trick from the previous script.

**Figures:** `fig_lipschitz_net_training.pdf`, `fig_lipschitz_net_comparison.pdf`, `fig_lipschitz_verification.pdf`

### `shift_detection.py`
Classifier-based statistical test for detecting covariate shift. Splits data into train/held-out, trains a discriminator, then checks whether accuracy on the held-out set significantly exceeds 0.5 using a binomial test. Given α = 0.05 and n = 20, finds ε = 0.25 (threshold accuracy = 0.75). Observed accuracy of 0.95 is well above — shift detected at 95% confidence.

**Figures:** `fig_shift_detection.pdf`, `fig_shift_detection_decision.pdf`, `fig_classifier_test.pdf`

## Other Files

- `CNG7880_week6.pdf` — Original lecture slides for reference

## How to Run

```bash
python importance_weights.py
python evaluation_bounds.py
python lipschitz_evaluation_bound.py
python shift_detection.py
```

All figures are saved as PDFs in the outputs folder. The console output has the step-by-step numerical breakdowns. It is worth reading alongside the figures.

## Notes

- These are deliberately small (n=20) so you can trace every number by hand.
- The Lipschitz network underestimates W₁ a bit which is expected with a small net and limited epochs. The point is to show the method, not to get a perfect estimate.
- The Wasserstein bound slightly underestimates E_Q in the evaluation bounds script, Again, K_l is approximate for logistic regression. With 20 samples, things are noisy.
- Random seed is fixed (42) throughout, so results are reproducible.