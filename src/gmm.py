"""IFRS 17 General Measurement Model (GMM) for one group of contracts, year by year.

Liability for remaining coverage = BEL + risk adjustment (RA) + CSM.

Year t, from valuation date t to t + 1 (analysis of change):
  1. Expected run-off of the BEL with interest at the current rate.
  2. Experience: actual in-force differs from expected (deaths, lapses). The change in
     future cash flows adjusts the CSM, measured at the locked-in rate (IFRS 17.B96).
  3. Assumption change for future years: adjusts the CSM, at the locked-in rate.
  4. Discount rate change: insurance finance expense, not CSM.
  The gap between current-rate and locked-in-rate measures of steps 2 and 3 is
  insurance finance expense. The CSM accretes interest at the locked-in rate and is
  released in proportion to coverage units (sum assured in force, IFRS 17.B119).
  If the CSM would turn negative, the excess is a loss on an onerous group and
  builds a loss component; favourable changes reverse the loss component first.

Presentation (IFRS 17.B120-B125):
  Insurance revenue  = expected claims + expected expenses + RA release + CSM release
                       + recovery of acquisition cash flows - loss component allocation
  Service expenses   = actual claims + actual expenses + amortisation of acquisition
                       cash flows + losses on onerous groups - reversals - loss component allocation
  Finance expense    = BEL interest + rate effects + CSM accretion
"""
import numpy as np
import pandas as pd

from .cashflows import Basis, bel_per_policy, expected_coverage_units


def run_group(product, base_q, mort_mult_by_t, rate_by_t, true_mort_mult, ra, rng, lapse=0.05):
    T, SA = product.term, product.sum_assured
    r0 = rate_by_t[0]                                              # locked-in rate
    basis = lambda t: Basis(q=base_q, mort_mult=mort_mult_by_t[min(t, T)], lapse=lapse)
    v = lambda t, b, r, acq=False: bel_per_policy(product, b, t, r, include_acquisition=acq)
    ra_pp = lambda t, b, r, acq=False: ra.per_policy(product, b, t, r, include_acquisition=acq)

    # Initial recognition (before any cash flow)
    N = product.n_policies
    bel = N * v(0, basis(0), r0, acq=True)
    ra_bal = N * ra_pp(0, basis(0), r0, acq=True)
    fcf0 = bel + ra_bal
    csm = max(-fcf0, 0.0)
    lc = max(fcf0, 0.0)
    acq_total = N * product.acquisition
    cu0 = N * expected_coverage_units(product, basis(0), 0)
    acq_schedule = acq_total * cu0 / cu0.sum()
    initial = {"bel": bel, "ra": ra_bal, "fcf": fcf0, "csm": csm, "day1_loss": lc}

    rows = []
    for t in range(T):
        b_t, b_n = basis(t), basis(t + 1)
        r_t, r_n = rate_by_t[t], rate_by_t[t + 1]
        bel_o, ra_o, csm_o, lc_o, N_o = bel, ra_bal, csm, lc, N

        # Cash flows of the year
        prem = N * product.premium
        exp_ = N * product.expense * (1 + product.expense_inflation) ** t
        acq = acq_total if t == 0 else 0.0
        q_be = min(base_q[t] * b_t.mort_mult, 1.0)
        claims_exp = N * q_be * SA
        deaths = rng.binomial(N, min(base_q[t] * true_mort_mult, 1.0))
        lapses = rng.binomial(N - deaths, lapse)
        claims_act = deaths * SA
        N_exp = N * (1 - q_be) * (1 - lapse)
        N_new = N - deaths - lapses

        # 1. Expected run-off with interest at the current rate
        bel_acc = r_t * (bel_o + prem - exp_ - acq)
        bel_E = (bel_o + prem - exp_ - acq) * (1 + r_t) - claims_exp
        assert np.isclose(bel_E, N_exp * v(t + 1, b_t, r_t), rtol=1e-9, atol=1e-6), "BEL recursion fails"

        # 2-4. Experience, assumption change, rate change
        d_exp_cur = (N_new - N_exp) * v(t + 1, b_t, r_t)
        d_exp_lock = (N_new - N_exp) * v(t + 1, b_t, r0)
        d_ass_cur = N_new * (v(t + 1, b_n, r_t) - v(t + 1, b_t, r_t))
        d_ass_lock = N_new * (v(t + 1, b_n, r0) - v(t + 1, b_t, r0))
        d_rate = N_new * (v(t + 1, b_n, r_n) - v(t + 1, b_n, r_t))
        bel = N_new * v(t + 1, b_n, r_n)

        ra_E = N_exp * ra_pp(t + 1, b_t, r_t)
        ra_release = ra_o - ra_E
        ra_fs = N_new * ra_pp(t + 1, b_n, r_t) - ra_E
        ra_rate = N_new * (ra_pp(t + 1, b_n, r_n) - ra_pp(t + 1, b_n, r_t))
        ra_bal = N_new * ra_pp(t + 1, b_n, r_n)

        # CSM and loss component
        csm_acc = r0 * csm_o
        fs = d_exp_lock + d_ass_lock + ra_fs                       # > 0 means unfavourable
        new_loss = reversal = 0.0
        if fs >= 0:
            csm_pre = csm_o + csm_acc - fs
            if csm_pre < 0:
                new_loss, csm_pre = -csm_pre, 0.0
                lc += new_loss
        else:
            reversal = min(lc, -fs)
            lc -= reversal
            csm_pre = csm_o + csm_acc + (-fs - reversal)
        cu_now = N_o * SA
        cu_future = N_new * expected_coverage_units(product, b_n, t + 1).sum() if t + 1 < T else 0.0
        share = cu_now / (cu_now + cu_future)
        csm_release = csm_pre * share
        csm = csm_pre - csm_release
        lc_alloc = lc * share                                      # presentation only (IFRS 17.50(a))
        lc -= lc_alloc

        revenue = claims_exp + exp_ + ra_release + csm_release + acq_schedule[t] - lc_alloc
        service_exp = claims_act + exp_ + acq_schedule[t] + new_loss - reversal - lc_alloc
        finance_exp = (bel_acc + (d_exp_cur - d_exp_lock) + (d_ass_cur - d_ass_lock) + d_rate
                       + ra_rate + csm_acc)
        isr = revenue - service_exp
        profit = isr - finance_exp

        # Reconciliation: change in liability explained by cash and P&L
        net_cash = prem - exp_ - acq - claims_act
        d_liab = (bel + ra_bal + csm) - (bel_o + ra_o + csm_o)
        recon_gap = net_cash - d_liab - profit

        rows.append({
            "year": t + 1, "policies_start": N_o, "deaths": deaths, "lapses": lapses,
            "premiums": prem, "claims_actual": claims_act, "claims_expected": claims_exp,
            "expenses": exp_, "acquisition_paid": acq, "net_cash_flow": net_cash,
            "bel_open": bel_o, "bel_close": bel, "ra_open": ra_o, "ra_close": ra_bal,
            "csm_open": csm_o, "csm_accretion": csm_acc,
            "csm_future_service_adj": -fs,
            "fcf_change_experience": d_exp_lock, "fcf_change_assumptions": d_ass_lock, "ra_future_service": ra_fs,
            "csm_release": csm_release, "csm_close": csm,
            "loss_component_close": lc, "new_onerous_loss": new_loss, "loss_reversal": reversal,
            "rev_expected_claims": claims_exp, "rev_expected_expenses": exp_, "rev_ra_release": ra_release,
            "rev_csm_release": csm_release, "rev_acquisition_recovery": acq_schedule[t],
            "rev_lc_allocation": -lc_alloc, "insurance_revenue": revenue,
            "insurance_service_expenses": service_exp, "insurance_service_result": isr,
            "fin_bel_interest": bel_acc, "fin_locked_in_vs_current": (d_exp_cur - d_exp_lock) + (d_ass_cur - d_ass_lock),
            "fin_rate_change": d_rate + ra_rate, "fin_csm_accretion": csm_acc,
            "insurance_finance_expense": finance_exp, "profit": profit, "reconciliation_gap": recon_gap,
        })
        N = N_new
    df = pd.DataFrame(rows)
    return initial, df
