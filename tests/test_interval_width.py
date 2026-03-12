"""
Tests for IntervalWidthController.
"""

import numpy as np
import pytest

from insurance_conformal_risk import IntervalWidthController


def make_synthetic_widths(n: int, m: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic interval widths for calibration testing.
    widths[i, j] = base_width[i] * (1 + lambdas[j]) where lambda is a quantile level.
    Widths increase with lambda (higher quantile = wider interval).
    """
    rng = np.random.default_rng(seed)
    lambdas = np.linspace(0.5, 0.99, m)
    base_widths = rng.uniform(100, 1000, size=n)
    # Wider at higher lambda (monotone increasing in lambda)
    widths = np.outer(base_widths, 0.5 + lambdas)
    return widths, lambdas


class TestIntervalWidthControllerBasic:

    def test_calibrate_returns_self(self):
        widths, lambdas = make_synthetic_widths(300, 50)
        controller = IntervalWidthController(
            width_target=0.5, scale=1.0, lambda_grid=lambdas
        )
        result = controller.calibrate_from_widths(widths)
        assert result is controller

    def test_is_calibrated_flag(self):
        widths, lambdas = make_synthetic_widths(200, 30)
        controller = IntervalWidthController(
            width_target=0.5, scale=1.0, lambda_grid=lambdas
        )
        assert not controller.is_calibrated_
        controller.calibrate_from_widths(widths)
        assert controller.is_calibrated_

    def test_lambda_hat_in_grid(self):
        widths, lambdas = make_synthetic_widths(400, 50)
        controller = IntervalWidthController(
            width_target=600.0, scale=1.0, lambda_grid=lambdas
        )
        controller.calibrate_from_widths(widths)
        # lambda_hat should be close to a value in the grid
        assert controller.lambda_hat_ >= lambdas[0]
        assert controller.lambda_hat_ <= lambdas[-1]

    def test_tighter_target_gives_smaller_lambda(self):
        """A tighter width constraint should select a smaller lambda (narrower intervals)."""
        widths, lambdas = make_synthetic_widths(500, 50, seed=7)

        scale = 1.0
        c_tight = IntervalWidthController(width_target=300.0, scale=scale, lambda_grid=lambdas)
        c_tight.calibrate_from_widths(widths)

        c_loose = IntervalWidthController(width_target=800.0, scale=scale, lambda_grid=lambdas)
        c_loose.calibrate_from_widths(widths)

        # Tighter width target => smaller lambda (narrower intervals)
        assert c_tight.lambda_hat_ <= c_loose.lambda_hat_ + 0.05

    def test_n_calibration_recorded(self):
        n = 250
        widths, lambdas = make_synthetic_widths(n, 40)
        controller = IntervalWidthController(width_target=500.0, scale=1.0, lambda_grid=lambdas)
        controller.calibrate_from_widths(widths)
        assert controller.n_calibration_ == n

    def test_repr_after_calibrate(self):
        widths, lambdas = make_synthetic_widths(200, 30)
        controller = IntervalWidthController(width_target=400.0, scale=1.0, lambda_grid=lambdas)
        controller.calibrate_from_widths(widths)
        r = repr(controller)
        assert "lambda_hat" in r

    def test_repr_before_calibrate(self):
        controller = IntervalWidthController(width_target=400.0)
        assert "not calibrated" in repr(controller)


class TestIntervalWidthControllerEdgeCases:

    def test_negative_widths_raises(self):
        widths, lambdas = make_synthetic_widths(100, 20)
        widths[0, 0] = -1.0
        controller = IntervalWidthController(width_target=500.0, scale=1.0, lambda_grid=lambdas)
        with pytest.raises(ValueError, match="negative"):
            controller.calibrate_from_widths(widths)

    def test_wrong_column_count_raises(self):
        widths = np.ones((100, 10))
        lambdas = np.linspace(0.5, 0.99, 20)  # Mismatched
        controller = IntervalWidthController(width_target=0.5, scale=1.0, lambda_grid=lambdas)
        with pytest.raises(ValueError):
            controller.calibrate_from_widths(widths)

    def test_1d_input_raises(self):
        widths = np.ones(100)
        lambdas = np.linspace(0.5, 0.99, 50)
        controller = IntervalWidthController(width_target=0.5, scale=1.0, lambda_grid=lambdas)
        with pytest.raises(ValueError):
            controller.calibrate_from_widths(widths)

    def test_zero_width_target_raises(self):
        with pytest.raises(ValueError, match="width_target"):
            IntervalWidthController(width_target=0.0)

    def test_zero_scale_raises(self):
        with pytest.raises(ValueError, match="scale"):
            IntervalWidthController(width_target=0.5, scale=0.0)

    def test_non_increasing_grid_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            IntervalWidthController(
                width_target=0.5,
                lambda_grid=np.array([0.9, 0.8, 0.7])
            )

    def test_target_too_tight_raises(self):
        """Width target tighter than minimum achievable should raise RuntimeError."""
        n, m = 100, 30
        lambdas = np.linspace(0.5, 0.99, m)
        # Widths are always > 1000
        widths = 1000 + np.random.default_rng(42).uniform(0, 100, (n, m))

        controller = IntervalWidthController(
            width_target=0.001,  # Impossibly tight
            scale=1.0,
            lambda_grid=lambdas
        )
        with pytest.raises(RuntimeError):
            controller.calibrate_from_widths(widths)

    def test_risk_summary_keys(self):
        widths, lambdas = make_synthetic_widths(200, 30)
        controller = IntervalWidthController(width_target=400.0, scale=1.0, lambda_grid=lambdas)
        controller.calibrate_from_widths(widths)
        summary = controller.risk_summary()
        assert "lambda_hat" in summary
        assert "width_target" in summary
        assert "n_calibration" in summary
