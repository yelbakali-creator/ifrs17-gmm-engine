"""Validation tests: the GMM roll-forward reconciles every year and over the lifetime of the group."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.cashflows import Basis, Product, RiskAdjustment, bel_per_policy
from src.gmm import run_group

ROOT = Path(__file__).parents[1]
pop = pd.read_csv(ROOT / "data" / "mortality_ew_male_2011.csv", index_col="age")["q_population_2011"]
Q = 0.7 * pop.loc[40:49].values


def _run(premium):
    prod = Product("test", 40, 10, 250_000, premium, 75.0, 0.02, 0.5 * premium, 2_000)
    mult = [1.0] * 4 + [0.9] * 7
    rates = [0.03] * 3 + [0.02] * 8
    return run_group(prod, Q, mult, rates, 0.9, RiskAdjustment(n_scenarios=1000), np.random.default_rng(1))


def test_annual_reconciliation_and_lifetime_identity():
    for premium in (600.0, 350.0):                 # profitable and onerous
        initial, df = _run(premium)
        assert df["reconciliation_gap"].abs().max() < 1e-4
        lifetime_profit = df["profit"].sum() - initial["day1_loss"]
        assert abs(lifetime_profit - df["net_cash_flow"].sum()) < 1e-3


def test_csm_and_loss_component_run_off():
    for premium in (600.0, 350.0):
        _, df = _run(premium)
        assert abs(df["csm_close"].iloc[-1]) < 1e-6
        assert abs(df["loss_component_close"].iloc[-1]) < 1e-6


def test_onerous_group_has_day_one_loss():
    initial, _ = _run(350.0)
    assert initial["day1_loss"] > 0 and initial["csm"] == 0


def test_bel_is_zero_at_maturity():
    prod = Product("test", 40, 10, 250_000, 600.0, 75.0, 0.02, 300.0, 1)
    assert bel_per_policy(prod, Basis(Q), 10, 0.03) == 0.0
