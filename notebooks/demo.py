# Databricks notebook source
# insurance-conformal-risk: Full Workflow Demo
#
# Three use cases on synthetic UK motor portfolio data:
# 1. Premium sufficiency control (main use case)
# 2. Interval width optimisation
# 3. Selective risk acceptance
#
# This notebook runs on Databricks serverless. It generates synthetic data
# with a known DGP so the CRC guarantees can be verified empirically.

# COMMAND ----------
# MAGIC %pip install insurance-conformal-risk numpy scipy polars

# COMMAND ----------

import numpy as np
import polars as pl

# COMMAND ----------
# MAGIC ## Synthetic Motor Portfolio Data
# MAGIC
# MAGIC DGP: 5 rating factors (age, vehicle_value, postcode_risk, ncb, annual_mileage).
# MAGIC Pure premium = exp(linear combination). Claims ~ Tweedie(p=1.5).
# MAGIC 40% zero claims (Poisson frequency, Gamma severity).

def generate_motor_portfolio(n: int, seed: int = 42) -> dict:
    """
    Synthetic UK motor portfolio with known data-generating process.

    Returns dict with X (features), y (claims), premium_model (GBM predictions),
    exposure (years on risk).
    """
    rng = np.random.default_rng(seed)

    # Rating factors (standardised)
    age = rng.uniform(17, 75, n)
    vehicle_value = rng.uniform(3000, 40000, n)
    postcode_risk = rng.uniform(0, 1, n)
    ncb = rng.integers(0, 6, n).astype(float)  # No-claims bonus years
    annual_mileage = rng.uniform(3000, 25000, n)

    # True log-premium (known DGP)
    log_premium = (
        5.5
        + 0.02 * np.maximum(25 - age, 0)       # Young driver loading
        + 0.008 * np.log(vehicle_value)          # Vehicle value
        + 0.6 * postcode_risk                    # Area risk
        - 0.08 * ncb                             # NCB discount
        + 0.0001 * annual_mileage               # Mileage
        + rng.normal(0, 0.15, n)                # Unexplained variation
    )
    true_premium = np.exp(log_premium)

    # Model predictions (imperfect — GBM with ~20% RMSE on log scale)
    log_model_noise = rng.normal(0, 0.20, n)
    premium_model = np.exp(log_premium + log_model_noise)

    # Claims: 55% frequency, Gamma severity
    has_claim = rng.binomial(1, 0.55, n)
    severity = rng.gamma(shape=2.0, scale=true_premium * 0.6 / 2.0, size=n)
    y = has_claim * severity

    # Exposure (partial years)
    exposure = rng.uniform(0.3, 1.0, n)

    # Risk score (negative log-loss proxy — higher = better risk)
    risk_score = 1.0 - np.clip(y / (true_premium * 3), 0, 1) + rng.normal(0, 0.1, n)
    risk_score = np.clip(risk_score, 0, 1)

    X = np.column_stack([age, vehicle_value, postcode_risk, ncb, annual_mileage])

    return {
        "X": X,
        "y": y,
        "premium_model": premium_model,
        "true_premium": true_premium,
        "exposure": exposure,
        "risk_score": risk_score,
    }

# Generate train/calibration/test split
data_cal = generate_motor_portfolio(n=3000, seed=42)
data_test = generate_motor_portfolio(n=10000, seed=99)

print(f"Calibration set: {len(data_cal['y'])} policies")
print(f"  Claim frequency: {(data_cal['y'] > 0).mean():.1%}")
print(f"  Mean claim (non-zero): £{data_cal['y'][data_cal['y'] > 0].mean():.0f}")
print(f"  Mean model premium: £{data_cal['premium_model'].mean():.0f}")
print(f"  Mean true premium: £{data_cal['true_premium'].mean():.0f}")

# COMMAND ----------
# MAGIC ## Use Case 1: Premium Sufficiency Control
# MAGIC
# MAGIC Question: 'Given my GBM's predicted premiums, what loading factor lambda* ensures
# MAGIC that expected shortfall from underpriced policies is at most 5% of premium income?'
# MAGIC
# MAGIC Guarantee: E[max(claim - lambda* * GBM_premium, 0) / GBM_premium] <= 0.05

from insurance_conformal_risk import PremiumSufficiencyController

# Set B based on your data
# B = max reasonable (claim / premium) ratio
# Here: max_claim / min_premium in calibration set
B = np.percentile(
    data_cal["y"] / np.clip(data_cal["premium_model"], 50, np.inf),
    99.9
)
print(f"Setting B = {B:.2f} (99.9th percentile of claim/premium ratio)")

psc = PremiumSufficiencyController(
    alpha=0.05,
    lambda_grid=np.linspace(0.5, 6.0, 600),
    B=B,
)
psc.calibrate(data_cal["y"], data_cal["premium_model"])

print(f"\nCalibration result:")
print(f"  lambda_hat = {psc.lambda_hat_:.4f}")
print(f"  Safety loading = {(psc.lambda_hat_ - 1) * 100:.1f}%")
print(f"  Risk summary: {psc.risk_summary()}")

# COMMAND ----------
# Verify guarantee on test set
upper_test = psc.lambda_hat_ * data_test["premium_model"]
shortfall = np.maximum(data_test["y"] - upper_test, 0) / data_test["premium_model"]
empirical_risk = shortfall.mean()

print(f"Guarantee verification on test set (n={len(data_test['y'])})")
print(f"  Target alpha:      {psc.alpha:.4f} ({psc.alpha:.1%})")
print(f"  Empirical risk:    {empirical_risk:.4f} ({empirical_risk:.1%})")
print(f"  Guarantee holds:   {empirical_risk <= psc.alpha}")
print(f"  % underpriced:     {(data_test['y'] > upper_test).mean():.1%}")

# COMMAND ----------
# Decile analysis — where is the shortfall concentrated?
shortfall_report = psc.shortfall_report(data_cal["y"], data_cal["premium_model"])
print("\nShortfall by premium decile:")
print(shortfall_report)

# COMMAND ----------
# Regulatory reporting
from insurance_conformal_risk.reporting import (
    premium_sufficiency_report,
    solvency_ii_model_error_note,
)

PORTFOLIO_GWP = 50_000_000  # £50m gross written premium

reg_report = premium_sufficiency_report(
    lambda_hat=psc.lambda_hat_,
    alpha=psc.alpha,
    n_calibration=psc.n_calibration_,
    B=psc.B,
    corrected_risk=psc.risk_summary()["corrected_risk_at_lambda"],
    portfolio_gwp=PORTFOLIO_GWP,
)
print("\nSolvency II model error report:")
print(reg_report)

print("\nNarrative for ORSA/SFCR:")
print(solvency_ii_model_error_note(psc.alpha, psc.lambda_hat_, psc.n_calibration_))

# COMMAND ----------
# MAGIC ## Use Case 2: Interval Width Control
# MAGIC
# MAGIC Question: 'What is the tightest prediction interval (quantile level lambda*) that
# MAGIC keeps expected interval width below £1,200?'
# MAGIC
# MAGIC This requires computing intervals at multiple quantile levels and feeding
# MAGIC the width matrix to IntervalWidthController.

from insurance_conformal_risk import IntervalWidthController

# Build width matrix: width[i, j] = interval width for policy i at quantile level lambda_grid[j]
# We use a simple empirical quantile approach (no model needed for this demo)
lambda_grid = np.linspace(0.50, 0.98, 80)
n_cal = len(data_cal["y"])

# Simulate quantile-based intervals using calibration set residuals
# In practice this would come from your conformal predictor
residuals_cal = data_cal["y"] - data_cal["premium_model"]
widths_cal = np.empty((n_cal, len(lambda_grid)))

for j, q in enumerate(lambda_grid):
    # Interval: [premium + q_lower * |residual|, premium + q_upper * |residual|]
    q_lo = np.quantile(residuals_cal, (1 - q) / 2)
    q_hi = np.quantile(residuals_cal, 1 - (1 - q) / 2)
    width = (q_hi - q_lo) * np.ones(n_cal)  # Same width for all (marginal intervals)
    widths_cal[:, j] = np.maximum(width, 0)

WIDTH_TARGET = 1200.0  # Max expected interval width in £
SCALE = widths_cal.max()  # Normalisation constant

controller = IntervalWidthController(
    width_target=WIDTH_TARGET,
    scale=SCALE,
    lambda_grid=lambda_grid,
)
controller.calibrate_from_widths(widths_cal)

print(f"Interval width control:")
print(f"  Width target:    £{WIDTH_TARGET:.0f}")
print(f"  lambda_hat:      {controller.lambda_hat_:.4f}")
print(f"  Interpretation:  Use {controller.lambda_hat_:.1%} quantile level for intervals")
print(f"  Expected width at lambda_hat: £{widths_cal.mean(axis=0)[np.searchsorted(lambda_grid, controller.lambda_hat_)]:.0f}")

# COMMAND ----------
# MAGIC ## Use Case 3: Selective Risk Acceptance
# MAGIC
# MAGIC Question: 'If we use our risk score to decline high-risk policies, can we guarantee
# MAGIC that the accepted book has at most 10% large-claim frequency?'
# MAGIC
# MAGIC A large claim is defined as: claim > 3x the model premium.

from insurance_conformal_risk import SelectiveRiskController

def large_claim_loss(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """1 if claim > 3x model premium (passed via scores context)."""
    # In real usage, you'd pass model premium separately via closure or wrapper
    threshold = data_cal["premium_model"][:len(y)] * 3
    return (y > threshold[:len(y)]).astype(float)

# Note: in practice, wrap loss_fn with a closure to capture premium_model
def make_loss_fn(premium_arr):
    def loss_fn(y, scores):
        return (y > 3 * premium_arr[:len(y)]).astype(float)
    return loss_fn

loss_fn = make_loss_fn(data_cal["premium_model"])

src = SelectiveRiskController(
    alpha=0.10,             # Max 10% large-claim frequency on accepted book
    loss_fn=loss_fn,
    xi_min=0.50,            # Must accept at least 50% of risks
    B=1.0,                  # Binary loss is bounded by 1
    delta=0.05,             # DKW confidence on selection rate
)
src.calibrate(data_cal["y"], data_cal["risk_score"])

print(f"Selective risk control calibration:")
print(f"  Acceptance threshold:  {src.threshold_:.4f}")
print(f"  Selection rate (cal):  {src.selection_rate_:.1%}")
print(f"  N accepted (cal):      {int(src.selection_rate_ * src.n_calibration_):,}")

# Apply to test set
decisions = src.predict(data_test["risk_score"])
accepted_mask = decisions["accept"].to_numpy()

print(f"\nTest set results:")
print(f"  N accepted:            {accepted_mask.sum():,}")
print(f"  Selection rate:        {accepted_mask.mean():.1%}")

# Compute loss on accepted test policies
loss_fn_test = make_loss_fn(data_test["premium_model"])
test_losses = loss_fn_test(data_test["y"], data_test["risk_score"])
empirical_large_claim_rate = test_losses[accepted_mask].mean()

print(f"  Large-claim frequency (accepted): {empirical_large_claim_rate:.1%}")
print(f"  Target alpha:                     {src.alpha:.1%}")
print(f"  Guarantee holds:                  {empirical_large_claim_rate <= src.alpha}")

# COMMAND ----------
# MAGIC ## Comparison: CRC vs Naive Threshold
# MAGIC
# MAGIC Show that the naive approach (thresholding at median risk score) does NOT
# MAGIC control the expected loss, while CRC does.

naive_threshold = np.median(data_cal["risk_score"])
naive_accepted = data_test["risk_score"] >= naive_threshold
naive_risk = test_losses[naive_accepted].mean()
crc_risk = test_losses[accepted_mask].mean()

print("Comparison: CRC vs naive median threshold")
print(f"{'Method':<25} {'Threshold':>10} {'Selection Rate':>15} {'Expected Loss':>15}")
print("-" * 70)
print(f"{'Naive (median)':<25} {naive_threshold:>10.4f} {naive_accepted.mean():>14.1%} {naive_risk:>14.1%}")
print(f"{'CRC (alpha=0.10)':<25} {src.threshold_:>10.4f} {accepted_mask.mean():>14.1%} {crc_risk:>14.1%}")
print(f"{'Target (alpha)':<25} {'—':>10} {'>=50%':>15} {'<=10%':>15}")

# COMMAND ----------
# MAGIC ## Summary
# MAGIC
# MAGIC Results across all three use cases:

print("=" * 60)
print("insurance-conformal-risk: Summary")
print("=" * 60)
print()
print("1. PREMIUM SUFFICIENCY")
print(f"   lambda_hat = {psc.lambda_hat_:.3f} (load all premiums by {(psc.lambda_hat_ - 1)*100:.0f}%)")
print(f"   Guarantee: E[shortfall] <= {psc.alpha:.0%} of premium income")
print(f"   Verified on test set: {empirical_risk:.2%} (target {psc.alpha:.0%})")
print()
print("2. INTERVAL WIDTH")
print(f"   lambda_hat = {controller.lambda_hat_:.3f} (use {controller.lambda_hat_:.1%} quantile)")
print(f"   Guarantee: E[interval width] <= £{WIDTH_TARGET:.0f}")
print()
print("3. SELECTIVE UNDERWRITING")
print(f"   Threshold = {src.threshold_:.4f}")
print(f"   Accepts {src.selection_rate_:.0%} of risks")
print(f"   Guarantee: E[large_claim | accepted] <= {src.alpha:.0%}")
print(f"   Verified on test set: {crc_risk:.1%} (target {src.alpha:.0%})")
