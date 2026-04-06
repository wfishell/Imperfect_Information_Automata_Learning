"""
Visualise the three stochastic-dominance comparison outcomes:
  DOMINATES / EQUIVALENT / INCOMPARABLE

Each panel shows two prefixes (P1, P2) with their future rank sets
and the resulting CDF curves.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── colour palette ──────────────────────────────────────────────────────────
C1  = "#2563EB"   # blue  – P1
C2  = "#DC2626"   # red   – P2
BG  = "#F8FAFC"
GRID= "#E2E8F0"

# ── helpers ─────────────────────────────────────────────────────────────────
def cdf(ranks, k, total):
    return sum(1 for r in ranks if r <= k) / total

def make_cdf_curve(ranks, n_traces):
    ks  = list(range(0, n_traces + 2))
    vals = [cdf(frozenset(ranks), k, len(ranks)) for k in ks]
    return ks, vals

# ── three cases ─────────────────────────────────────────────────────────────
N = 4   # total traces / rank ceiling

cases = [
    {
        "title":    "DOMINATES",
        "subtitle": "P1's CDF ≥ P2's at every threshold (strict somewhere)",
        "p1_ranks": [1, 2],
        "p2_ranks": [3, 4],
        "color":    "#16A34A",   # green label
    },
    {
        "title":    "EQUIVALENT",
        "subtitle": "P1's CDF = P2's at every threshold",
        "p1_ranks": [1, 3],
        "p2_ranks": [1, 3],
        "color":    "#7C3AED",   # purple label
    },
    {
        "title":    "INCOMPARABLE",
        "subtitle": "CDFs cross — neither dominates the other",
        "p1_ranks": [1, 4],
        "p2_ranks": [2, 3],
        "color":    "#EA580C",   # orange label
    },
]

# ── figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=BG)
fig.suptitle("Stochastic Dominance: Three Comparison Outcomes",
             fontsize=14, fontweight="bold", y=1.01)

for ax, case in zip(axes, cases):
    p1, p2 = case["p1_ranks"], case["p2_ranks"]
    ks1, v1 = make_cdf_curve(p1, N)
    ks2, v2 = make_cdf_curve(p2, N)

    ax.set_facecolor(BG)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    # ── CDF step curves ──────────────────────────────────────────────────────
    ax.step(ks1, v1, where="post", color=C1, linewidth=2.5,
            label="P1", zorder=3)
    ax.step(ks2, v2, where="post", color=C2, linewidth=2.5,
            linestyle="--", label="P2", zorder=3)

    # shade the region between CDFs to make crossing visible
    k_fine = np.linspace(0, N + 1, 500)
    v1_fine = np.array([cdf(frozenset(p1), k, len(p1)) for k in k_fine])
    v2_fine = np.array([cdf(frozenset(p2), k, len(p2)) for k in k_fine])
    ax.fill_between(k_fine, v1_fine, v2_fine,
                    where=(v1_fine >= v2_fine), alpha=0.12, color=C1)
    ax.fill_between(k_fine, v1_fine, v2_fine,
                    where=(v1_fine <  v2_fine), alpha=0.12, color=C2)

    # ── future rank markers along the x-axis ────────────────────────────────
    for r in p1:
        ax.axvline(r, color=C1, linewidth=1, linestyle=":", alpha=0.5)
    for r in p2:
        ax.axvline(r, color=C2, linewidth=1, linestyle=":", alpha=0.5)

    # ── future-set annotations ───────────────────────────────────────────────
    ax.text(0.97, 0.18,
            f"futures(P1) = {set(p1)}\nfutures(P2) = {set(p2)}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, alpha=0.9))

    # ── axis labels / limits ─────────────────────────────────────────────────
    ax.set_xlim(-0.2, N + 1.2)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xticks(range(1, N + 1))
    ax.set_xticklabels([f"rank {k}" for k in range(1, N + 1)], fontsize=8)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_xlabel("Rank threshold  k", fontsize=9)
    ax.set_ylabel("CDF  F(k)", fontsize=9)

    # ── title block ──────────────────────────────────────────────────────────
    ax.set_title(case["title"], fontsize=13, fontweight="bold",
                 color=case["color"], pad=10)
    ax.text(0.5, 1.01, case["subtitle"],
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, style="italic", color="#475569")

    # ── legend ───────────────────────────────────────────────────────────────
    p1_patch = mpatches.Patch(color=C1, label="P1")
    p2_patch = mpatches.Patch(color=C2, label="P2")
    ax.legend(handles=[p1_patch, p2_patch], loc="upper left",
              fontsize=9, framealpha=0.9)

plt.tight_layout()
out = "dominance_cases.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
