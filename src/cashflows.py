"""Cash-flow projection for a group of level-premium term life contracts.

Timing within each policy year: premium and maintenance expense at the start,
death claims at the end, lapses at the end among survivors. The insurance
acquisition cash flow (commission) is paid once, at inception.

All present values are per policy in force at the valuation date, so the
best estimate liability (BEL) of a group is N x v(t). BEL = PV(outflows) - PV(inflows).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Product:
    name: str
    issue_age: int
    term: int
    sum_assured: float
    premium: float
    expense: float            # maintenance expense per policy in year 0
    expense_inflation: float
    acquisition: float        # commission per policy, paid at inception
    n_policies: int


@dataclass
class Basis:
    q: np.ndarray             # best-estimate annual mortality by policy year (before multiplier)
    mort_mult: float = 1.0
    lapse: float = 0.05


def project(product, basis, t, rate, mort_mult=None, lapse_mult=None, include_acquisition=False):
    """PV at time t (per policy in force at t) of future outflows minus inflows.

    mort_mult and lapse_mult can be arrays (scenarios) for the risk adjustment.
    Returns (pv, expected survivors by future year, per policy).
    """
    mm = basis.mort_mult if mort_mult is None else mort_mult
    lm = 1.0 if lapse_mult is None else lapse_mult
    mm, lm = np.broadcast_arrays(np.asarray(mm, float), np.asarray(lm, float))
    pv = np.zeros(mm.shape)
    n = np.ones(mm.shape)
    in_force = []
    for s in range(t, product.term):
        q = np.minimum(basis.q[s] * mm, 1.0)
        w = np.minimum(basis.lapse * lm, 1.0)
        v_start = (1 + rate) ** -(s - t)
        v_end = (1 + rate) ** -(s - t + 1)
        expense = product.expense * (1 + product.expense_inflation) ** s
        pv += n * (expense - product.premium) * v_start + n * q * product.sum_assured * v_end
        in_force.append(n.copy())
        n = n * (1 - q) * (1 - w)
    if include_acquisition and t == 0:
        pv += product.acquisition
    return pv, in_force


def bel_per_policy(product, basis, t, rate, include_acquisition=False):
    if t >= product.term:
        return 0.0
    return float(project(product, basis, t, rate, include_acquisition=include_acquisition)[0])


class RiskAdjustment:
    """Confidence-level risk adjustment: 75th percentile of the PV distribution minus its mean.

    Uncertainty comes from the levels of mortality and lapse (lognormal multipliers).
    The same scenarios are reused at every valuation date (common random numbers).
    """

    def __init__(self, n_scenarios=4000, sigma_mort=0.10, sigma_lapse=0.20, level=0.75, seed=17):
        rng = np.random.default_rng(seed)
        self.mort = np.exp(sigma_mort * rng.standard_normal(n_scenarios) - sigma_mort ** 2 / 2)
        self.lapse = np.exp(sigma_lapse * rng.standard_normal(n_scenarios) - sigma_lapse ** 2 / 2)
        self.level = level

    def per_policy(self, product, basis, t, rate, include_acquisition=False):
        if t >= product.term:
            return 0.0
        pv, _ = project(product, basis, t, rate, basis.mort_mult * self.mort, self.lapse,
                        include_acquisition)
        return float(np.quantile(pv, self.level) - pv.mean())


def expected_coverage_units(product, basis, t):
    """Expected sum assured in force for each remaining year, per policy in force at t."""
    if t >= product.term:
        return np.array([])
    _, in_force = project(product, basis, t, 0.0)
    return np.array([float(n) for n in in_force]) * product.sum_assured
