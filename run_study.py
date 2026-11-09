"""IFRS 17 GMM study: a profitable and an onerous group of 10-year term life contracts."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cashflows import Basis, Product, RiskAdjustment, project
from src.gmm import run_group

ROOT = Path(__file__).parent
TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
for folder in (TABLES, FIGS):
    folder.mkdir(parents=True, exist_ok=True)
NAVY, RED, GREY, GOLD, GREEN = "#1F3864", "#C0392B", "#7F8C8D", "#D4A017", "#2E7D32"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

TERM, SA, N0 = 10, 250_000, 10_000
INSURED_FACTOR = 0.70          # best estimate = 70% of 2011 population mortality
TRUE_MULT = 0.90               # actual mortality turns out 10% lighter than best estimate
REVISION_AT = 4                # valuation at end of year 4: best estimate revised to 90%
RATE_DROP_AT = 3               # valuation at end of year 3: discount rate falls from 3% to 2%
LOCKED_RATE, NEW_RATE = 0.03, 0.02
EXPENSE, INFLATION, ACQ_SHARE = 75.0, 0.02, 0.50


def base_q(issue_age):
    pop = pd.read_csv(ROOT / "data" / "mortality_ew_male_2011.csv", index_col="age")["q_population_2011"]
    return INSURED_FACTOR * pop.loc[issue_age:issue_age + TERM - 1].values


def breakeven_premium(issue_age):
    """Premium with PV(premiums) = PV(claims + expenses + commission), commission = 50% of premium."""
    q = base_q(issue_age)
    out = Product("x", issue_age, TERM, SA, 0.0, EXPENSE, INFLATION, 0.0, 1)
    pv_out = float(project(out, Basis(q), 0, LOCKED_RATE)[0])
    annuity = -float(project(Product("x", issue_age, TERM, 0.0, 1.0, 0.0, 0.0, 0.0, 1), Basis(q), 0, LOCKED_RATE)[0])
    return pv_out / (annuity - ACQ_SHARE)


def make_product(name, issue_age, margin):
    p = breakeven_premium(issue_age) * (1 + margin)
    return Product(name, issue_age, TERM, SA, p, EXPENSE, INFLATION, ACQ_SHARE * p, N0)


def mc_check_bel(product, q, rate, n_sims=4000, seed=5):
    """Independent check: simulate whole portfolios policy by policy and average the PV."""
    rng = np.random.default_rng(seed)
    n = np.full(n_sims, product.n_policies)
    pv = np.full(n_sims, product.n_policies * product.acquisition, dtype=float)
    for s in range(product.term):
        exp_ = product.expense * (1 + product.expense_inflation) ** s
        pv += n * (exp_ - product.premium) * (1 + rate) ** -s
        d = rng.binomial(n, q[s])
        pv += d * product.sum_assured * (1 + rate) ** -(s + 1)
        n = n - d - rng.binomial(n - d, 0.05)
    return pv.mean(), pv.std(ddof=1) / np.sqrt(n_sims)


def main():
    ra = RiskAdjustment()
    mort_mult = [1.0 if t < REVISION_AT else TRUE_MULT for t in range(TERM + 1)]
    rates = [LOCKED_RATE if t < RATE_DROP_AT else NEW_RATE for t in range(TERM + 1)]
    groups = {"A_profitable": make_product("Group A (profitable)", 40, 0.25),
              "B_onerous": make_product("Group B (onerous)", 50, -0.05)}

    summary, results = [], {}
    for key, prod in groups.items():
        rng = np.random.default_rng(2026)
        initial, df = run_group(prod, base_q(prod.issue_age), mort_mult, rates, TRUE_MULT, ra, rng)
        results[key] = (initial, df)
        df.to_csv(TABLES / f"annual_results_{key}.csv", index=False)
        mc_mean, mc_se = mc_check_bel(prod, base_q(prod.issue_age), LOCKED_RATE)
        total_profit = df["profit"].sum() - initial["day1_loss"]
        summary.append({
            "group": prod.name, "issue_age": prod.issue_age, "annual_premium": prod.premium,
            "bel_at_inception": initial["bel"], "bel_independent_mc": mc_mean, "mc_se": mc_se,
            "risk_adjustment": initial["ra"], "csm_at_inception": initial["csm"], "day1_loss": initial["day1_loss"],
            "lifetime_profit_ifrs17": total_profit, "lifetime_net_cash_flow": df["net_cash_flow"].sum(),
            "max_abs_reconciliation_gap": df["reconciliation_gap"].abs().max(),
            "csm_at_maturity": df["csm_close"].iloc[-1], "loss_component_at_maturity": df["loss_component_close"].iloc[-1],
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(TABLES / "summary_and_validation.csv", index=False)

    # Charts
    for key, (initial, df) in results.items():
        name = groups[key].name
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        yrs = df["year"]
        rev = df[["rev_expected_claims", "rev_expected_expenses", "rev_ra_release", "rev_csm_release",
                  "rev_acquisition_recovery", "rev_lc_allocation"]] / 1e6
        labels = ["Expected claims", "Expected expenses", "RA release", "CSM release", "Acquisition recovery",
                  "Loss component allocation"]
        colors = [NAVY, GREY, GOLD, GREEN, "#8E44AD", RED]
        bottom_pos, bottom_neg = np.zeros(len(yrs)), np.zeros(len(yrs))
        for col, lab, c in zip(rev.columns, labels, colors):
            vals = rev[col].values
            if np.allclose(vals, 0):
                continue
            pos, neg = np.clip(vals, 0, None), np.clip(vals, None, 0)
            axes[0].bar(yrs, pos, bottom=bottom_pos, color=c, label=lab)
            axes[0].bar(yrs, neg, bottom=bottom_neg, color=c)
            bottom_pos += pos; bottom_neg += neg
        axes[0].set_title("Insurance revenue by component")
        axes[0].set_xlabel("Year"); axes[0].set_ylabel("Million")
        axes[0].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)

        axes[1].bar(yrs - 0.2, df["insurance_service_result"] / 1e6, width=0.4, color=NAVY, label="Insurance service result")
        axes[1].bar(yrs + 0.2, -df["insurance_finance_expense"] / 1e6, width=0.4, color=GOLD, label="Insurance finance result")
        axes[1].plot(yrs, df["profit"] / 1e6, color=RED, marker="o", label="Profit")
        if initial["day1_loss"] > 0:
            axes[1].bar([0], [-initial["day1_loss"] / 1e6], width=0.4, color=RED, alpha=0.6, label="Day-one loss (onerous)")
        axes[1].axhline(0, color="black", lw=0.6)
        top = axes[1].get_ylim()[1]
        for x, lab, ha in [(RATE_DROP_AT + 0.5, "Rates 3% to 2%  ", "right"), (REVISION_AT + 0.5, "  Mortality revised", "left")]:
            axes[1].axvline(x, color=GREY, ls=":", lw=1)
            axes[1].text(x, top * 0.92, lab, fontsize=7, ha=ha, color=GREY)
        axes[1].set_title("Profit emergence")
        axes[1].set_xlabel("Year"); axes[1].set_ylabel("Million")
        axes[1].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        fig.suptitle(f"IFRS 17 profit or loss, {name}")
        fig.tight_layout(); fig.savefig(FIGS / f"pnl_{key}.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for key, col in [("A_profitable", GREEN), ("B_onerous", RED)]:
        initial, df = results[key]
        ax.plot(np.r_[0, df["year"]], np.r_[initial["csm"], df["csm_close"]] / 1e6, color=col, marker="o",
                label=f"CSM, {groups[key].name}")
        ax.plot(np.r_[0, df["year"]], np.r_[initial["day1_loss"], df["loss_component_close"]] / 1e6, color=col,
                ls="--", label=f"Loss component, {groups[key].name}")
    ax.axvline(REVISION_AT, color=GREY, ls=":", lw=1)
    ax.text(REVISION_AT, ax.get_ylim()[1] * 0.95, " Mortality revised (end of year 4)", fontsize=7, color=GREY)
    ax.set_xlabel("Year"); ax.set_ylabel("Million")
    ax.set_title("Contractual service margin and loss component")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "csm_and_loss_component.png", dpi=150); plt.close(fig)

    # Console summary
    pd.set_option("display.width", 220)
    print(summary.round(2).T.to_string())
    for key, (initial, df) in results.items():
        cols = ["year", "deaths", "lapses", "insurance_revenue", "insurance_service_expenses",
                "insurance_service_result", "insurance_finance_expense", "profit", "csm_open", "csm_accretion",
                "csm_future_service_adj", "csm_release", "csm_close", "loss_component_close", "reconciliation_gap"]
        print(f"\n{groups[key].name}: initial {', '.join(f'{k}={v:,.0f}' for k, v in initial.items())}")
        print((df[cols].set_index("year") / np.r_[1, 1, [1e3] * 11, 1]).round(1).to_string())


if __name__ == "__main__":
    main()
