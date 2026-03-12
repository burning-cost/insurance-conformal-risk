# insurance-conformal-risk

Conformal Risk Control for UK insurance pricing. Distribution-free bounds on expected financial shortfall — not just coverage probability.

```bash
pip install insurance-conformal-risk
```

## The problem

Your GBM predicts a pure premium of £450 for a motor policy. The actual claim comes in at £1,200. That's an underpricing event.

Standard conformal prediction (see [insurance-conformal](https://github.com/burning-cost/insurance-conformal)) tells you: "the claim will fall below your upper bound on 90% of policies." It says nothing about what happens in the other 10% — the shortfall could be £10 or £10,000.

Conformal Risk Control (CRC) controls the magnitude directly:

> E[max(claim - upper_bound, 0) / premium] ≤ α

This bounds the expected underpricing shortfall as a fraction of premium income. With α = 0.05, you are guaranteeing that expected shortfall from underpriced policies is at most 5% of expected premium income — no parametric assumptions, finite-sample valid.

This is what actuaries actually want to know. Not "how often am I wrong?" but "how much does being wrong cost me?"

## Background

This library implements Conformal Risk Control from [Angelopoulos et al. (2024), ICLR](https://arxiv.org/abs/2208.02814), applied specifically to insurance pricing problems.

CRC extends split conformal prediction from coverage control to expected-loss control for any bounded monotone loss function. The algorithm is five lines of numpy. The value is in the correct insurance-specific loss functions, the finite-sample correction that naive implementations get wrong, and the workflow that maps onto how pricing teams actually work.

No Python package on PyPI implements general regression risk control with user-defined monotone losses. This is the first.

## Three controllers

### 1. Premium Sufficiency (main use case)

Find the smallest loading factor such that expected shortfall is bounded:

```python
from insurance_conformal_risk import PremiumSufficiencyController
import numpy as np

# y_cal: observed claims on held-out calibration set (n=1000-5000)
# premium_cal: model-predicted pure premiums for same policies
psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
psc.calibrate(y_cal, premium_cal)

print(psc.lambda_hat_)  # e.g., 1.34 — load all premiums by 34%

# Apply to next year's book
result = psc.predict(premium_new)
# result["upper_bound"] = 1.34 * premium_new (risk-controlled bound)
```

The guarantee: E[max(claim - 1.34 × premium, 0) / premium] ≤ 0.05 on any exchangeable test set.

**On setting B:** B is the maximum possible normalised shortfall (max claim / min premium). For a policy limit of £50,000 and minimum premium of £200, B = 250. If you normalise by premium and your claims are bounded by the sum insured, B is well-defined. The default B=1 is only valid if claims never exceed premium — rarely true. Inspect your data.

### 2. Interval Width Control

Find the tightest prediction intervals that still keep expected width below a budget:

```python
from insurance_conformal_risk import IntervalWidthController
import numpy as np

# widths_cal[i, j] = interval width for observation i at quantile level lambda_grid[j]
# (generate this by calling your conformal predictor at each lambda value)
lambda_grid = np.linspace(0.50, 0.995, 100)
controller = IntervalWidthController(width_target=800.0, scale=2000.0, lambda_grid=lambda_grid)
controller.calibrate_from_widths(widths_cal)

print(controller.lambda_hat_)  # e.g., 0.82 — use 82nd percentile intervals
```

### 3. Selective Underwriting

Accept only risks where expected loss on the accepted book is bounded:

```python
from insurance_conformal_risk import SelectiveRiskController
import numpy as np

def large_claim_loss(y, scores):
    """Binary: 1 if claim exceeds £5,000."""
    return (y > 5000).astype(float)

src = SelectiveRiskController(alpha=0.08, loss_fn=large_claim_loss, xi_min=0.60)
src.calibrate(y_cal, scores_cal)
# src.threshold_: accept iff risk_score >= threshold

decisions = src.predict(scores_new)
# decisions["accept"]: True/False per policy
```

The guarantee: among accepted risks, E[large_claim_loss] ≤ 0.08, provided at least 60% of risks are accepted.

## Integration with insurance-conformal

These two libraries work together. Use `insurance-conformal` to generate coverage-controlled intervals, then use `insurance-conformal-risk` to verify premium sufficiency:

```python
from insurance_conformal import InsuranceConformalPredictor
from insurance_conformal_risk import PremiumSufficiencyController

# Step 1: standard conformal intervals (coverage control)
cp = InsuranceConformalPredictor(model=fitted_gbm, nonconformity="pearson_weighted")
cp.calibrate(X_cal, y_cal)
intervals_cal = cp.predict_interval(X_cal, alpha=0.10)

# Step 2: risk control on top of conformal upper bounds
psc = PremiumSufficiencyController(alpha=0.04, B=8.0)
psc.calibrate(y_cal, intervals_cal["upper"].to_numpy())

# The conformal upper bound is both coverage-controlled AND shortfall-controlled
intervals_new = cp.predict_interval(X_new, alpha=0.10)
bounds = psc.predict(intervals_new["upper"].to_numpy())
```

## Regulatory framing

For Solvency II (Article 105) and Solvency UK model validation:

```python
from insurance_conformal_risk.reporting import (
    premium_sufficiency_report,
    solvency_ii_model_error_note,
)

report = premium_sufficiency_report(
    lambda_hat=psc.lambda_hat_,
    alpha=psc.alpha,
    n_calibration=psc.n_calibration_,
    B=psc.B,
    corrected_risk=psc.risk_summary()["corrected_risk_at_lambda"],
    portfolio_gwp=45_000_000,  # £45m GWP
)

note = solvency_ii_model_error_note(psc.alpha, psc.lambda_hat_, psc.n_calibration_)
print(note)
```

## Core algorithm

The CRC algorithm (Algorithm 1 of Angelopoulos et al. 2024):

1. Compute empirical risk: R̂(λ) = (1/n) Σ L_i(λ)
2. Apply finite-sample correction: (n/(n+1)) × R̂(λ) + B/(n+1) ≤ α
3. Find λ* = smallest λ satisfying the corrected inequality

The finite-sample correction is not optional. It accounts for the unseen test point. For n=500, B=5, the correction adds B/(n+1) ≈ 0.01 to the risk threshold — small but load-bearing for tight alpha values.

```python
from insurance_conformal_risk import conformal_risk_calibration
import numpy as np

# losses[i, j] = loss for observation i at lambda_grid[j]
# Must be non-increasing in j (larger lambda = lower loss)
losses = np.random.rand(500, 200) * (1 - np.linspace(0, 1, 200))

lambdas = np.linspace(0, 2, 200)
lambda_hat, idx, risk_curve = conformal_risk_calibration(losses, lambdas, alpha=0.05, B=1.0)
```

## Limitations (be explicit about these)

- **Marginal guarantee only.** CRC controls the average over the calibration distribution. A particular segment (young drivers, high-value properties) may have higher shortfall than alpha. Check `shortfall_report()` for segment diagnostics.
- **Exchangeability required.** Calibration and test data must be exchangeable. For insurance, this means same underwriting year, same distribution mix. Year-on-year deployment with changing book mix violates this. There are extensions for non-exchangeable data (arXiv:2310.01262) — not implemented here.
- **B must be set correctly.** Setting B too small produces invalid guarantees (the algorithm will still run; the guarantee will not hold). B is the maximum possible loss value. For the shortfall loss, this is max_claim / min_premium in your data. For unlimited policies, cap using a policy limit.
- **Not a replacement for Solvency II SCR.** CRC controls expected shortfall. Solvency II requires 99.5% VaR. These are different quantities. CRC bounds model error, not the full underwriting risk capital charge.

## What this is not

This is not conformal prediction (coverage control) — that's [insurance-conformal](https://github.com/burning-cost/insurance-conformal). Use standard conformal prediction when you want P(Y in interval) ≥ 1 - α. Use this library when you want E[financial_loss] ≤ α.

## Installation

```bash
pip install insurance-conformal-risk
```

With scikit-learn for SelectiveRiskController integration:

```bash
pip install insurance-conformal-risk[sklearn]
```

## References

- Angelopoulos, Bates, Fisch, Lei & Schuster (2024). *Conformal Risk Control*. ICLR 2024. [arXiv:2208.02814](https://arxiv.org/abs/2208.02814)
- Selective CRC: [arXiv:2512.12844](https://arxiv.org/abs/2512.12844) (2025)
- Hong (2025). *Conformal prediction of future insurance claims*. [arXiv:2503.03659](https://arxiv.org/abs/2503.03659)

## License

MIT. Built by [Burning Cost](https://github.com/burning-cost).
