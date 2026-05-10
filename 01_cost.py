"""01_cost.py — Position-based cost per paper, cost figures.

Loads all 288,368 papers directly from archive/. No sampling needed
since position-based priors are instant (no web lookup).

Salary assignment:
  First author  → Graduate researcher (PhD/Postdoc)
  Last author   → PI/Professor
  Middle        → distribution-weighted mix
  Single author → Professor
"""
import os, sys, json
sys.stdout.reconfigure(encoding='utf-8')
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from config import (SEED, COST_FRACTION, MONTHS_PER_PAPER,
                    OUTPUT_DIR, INPUT_DIR, DATA_PATH, FILES,
                    SALARY_MULTIPLIER, RESOURCE_COST,
                    SALARY_PROFESSOR, SALARY_GRAD, SALARY_MASTERS,
                    SALARY_UNDERGRAD, SALARY_OTHER_ACADEMIC,
                    DIST_PROFESSOR, DIST_GRAD, DIST_MASTERS,
                    DIST_OTHER, DIST_UNDERGRAD)

warnings.filterwarnings("ignore")
np.random.seed(SEED)
sns.set_style("whitegrid")

# Academic-unknown = distribution-weighted average of all non-undergrad roles (middle-author prior)
_acad_total = DIST_PROFESSOR + DIST_GRAD + DIST_MASTERS + DIST_OTHER
SALARY_ACADEMIC_UNKNOWN = int(
    (DIST_PROFESSOR * SALARY_PROFESSOR +
     DIST_GRAD      * SALARY_GRAD      +
     DIST_MASTERS   * SALARY_MASTERS   +
     DIST_OTHER     * SALARY_OTHER_ACADEMIC) / max(_acad_total, 0.01)
)


def salary_by_position(pos, n_authors, role_med):
    if n_authors == 1:           return role_med["professor"],        3
    if pos == 0:                 return role_med["grad"],             3
    if pos == n_authors - 1:     return role_med["professor"],        3
    return                              role_med["academic_unknown"],  3


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR,  exist_ok=True)
    print("=== 01_cost.py ===")

    # Load ALL papers directly — no sample needed
    print("\n[1/4] Loading all papers from archive/...")
    records = []
    for f in FILES:
        with open(DATA_PATH + f, "r", encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))
    df = pd.DataFrame(records).drop_duplicates(subset="id").reset_index(drop=True)
    df["published_dt"] = pd.to_datetime(df["published"], utc=True)
    df["year"]  = df["published_dt"].dt.year
    df["month"] = df["published_dt"].dt.month
    print(f"  {len(df):,} unique papers loaded")

    # Salary table
    print(f"\n[2/4] Building salary table (multiplier: {SALARY_MULTIPLIER}×)...")
    _scale = lambda d: {k: int(v * SALARY_MULTIPLIER) for k, v in d.items()}
    role_medians = _scale({
        "professor":        SALARY_PROFESSOR,
        "grad":             SALARY_GRAD,
        "masters":          SALARY_MASTERS,
        "academic_unknown": SALARY_ACADEMIC_UNKNOWN,
        "undergrad":        SALARY_UNDERGRAD,
        "default":          SALARY_ACADEMIC_UNKNOWN,
    })
    for role, sal in role_medians.items():
        print(f"  {role:<20} ${sal:>7,}")

    # Cost per paper
    print("\n[3/4] Computing cost per paper...")
    cost_records = []
    for _, row in df.iterrows():
        authors    = row["authors"]
        n_authors  = len(authors)
        salaries   = [salary_by_position(pos, n_authors, role_medians)[0]
                      for pos in range(n_authors)]
        salary_cost = sum((s / 12) * COST_FRACTION * MONTHS_PER_PAPER for s in salaries)
        total_cost  = salary_cost + RESOURCE_COST
        cost_records.append({
            "paper_id":           row["id"],
            "total_cost":         total_cost,
            "mean_author_salary": float(np.mean(salaries)),
            "author_count":       n_authors,
            "primary_category":   row["primary_category"],
            "year":               row["year"],
            "month":              row["month"],
        })

    cost_df = pd.DataFrame(cost_records)
    df = df.merge(cost_df.drop(columns=["primary_category","year","month"]),
                  left_on="id", right_on="paper_id", how="left")

    print(f"  Cost computed for {len(cost_df):,} papers")
    print(cost_df["total_cost"].describe().apply(lambda x: f"${x:,.0f}").to_string())

    low_thresh  = cost_df["total_cost"].quantile(1/3)
    high_thresh = cost_df["total_cost"].quantile(2/3)

    def assign_bucket(cost):
        if cost <= low_thresh:  return "Low"
        if cost <= high_thresh: return "Medium"
        return "High"

    cost_df["cost_bucket"] = cost_df["total_cost"].apply(assign_bucket)
    df["cost_bucket"]      = df["total_cost"].apply(assign_bucket)
    print(f"  Low ≤ ${low_thresh:,.0f}  |  Medium ≤ ${high_thresh:,.0f}  |  High > ${high_thresh:,.0f}")
    print(cost_df["cost_bucket"].value_counts().to_string())

    # Figures
    print("\n[4/4] Generating figures...")

    # ── Author distribution histogram with Gaussian KDE (from scratch) ─────────
    auth_counts = cost_df["author_count"].values
    clip_95     = int(np.percentile(auth_counts, 95))
    mean_a, median_a, std_a = np.mean(auth_counts), np.median(auth_counts), np.std(auth_counts)

    h_bw   = 1.06 * std_a * len(auth_counts)**(-0.2)
    x_kde  = np.linspace(1, clip_95, 400)
    rng    = np.random.RandomState(SEED)
    s_kde  = auth_counts if len(auth_counts) <= 30000 else rng.choice(auth_counts, 30000, replace=False)
    diff   = x_kde[:, None] - s_kde[None, :]
    kde_v  = np.exp(-0.5*(diff/h_bw)**2).mean(axis=1) / (h_bw*np.sqrt(2*np.pi))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(auth_counts[auth_counts <= clip_95], bins=range(1, clip_95+2),
            density=True, alpha=0.65, color="#4472C4", edgecolor="white", linewidth=0.5, label="Histogram")
    ax.plot(x_kde, kde_v, color="#C00000", linewidth=2.5, label=f"KDE (h={h_bw:.3f})")
    ax.axvline(mean_a,   color="#FF6B00", linewidth=2, linestyle="--", label=f"Mean = {mean_a:.2f}")
    ax.axvline(median_a, color="#00B050", linewidth=2, linestyle=":",  label=f"Median = {int(median_a)}")
    stats_txt = f"n = {len(auth_counts):,}\nMean = {mean_a:.2f}\nMedian = {int(median_a)}\nStd = {std_a:.2f}"
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes, fontsize=10,
            va="top", ha="right", bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="gray"))
    ax.set_xlabel("Authors per Paper", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"Distribution of Authors per Paper (All {len(auth_counts):,} Papers)", fontsize=14)
    ax.set_xlim(0, clip_95+0.5)
    ax.legend(fontsize=11)
    pct_shown = 100*np.sum(auth_counts <= clip_95)/len(auth_counts)
    ax.text(0.5, -0.12, f"Histogram clipped at {clip_95} authors (95th percentile, {pct_shown:.1f}% of papers); KDE uses full dataset",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="#555555")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_authors_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Category cost distribution (y-axis capped at 99th percentile) ──────────
    top_cats   = ["cs.CV", "cs.LG", "cs.CL", "cs.AI", "cs.IR", "cs.NE"]
    top_cats   = [c for c in top_cats if c in cost_df["primary_category"].values]
    cat_data   = [cost_df[cost_df["primary_category"]==c]["total_cost"].values for c in top_cats]
    cat_colors = ["#4472C4", "#ED7D31", "#70AD47", "#FF0000", "#7030A0", "#00B050"]
    p99        = np.percentile(np.concatenate(cat_data), 99)
    n_out      = int(np.sum(np.concatenate(cat_data) > p99))

    fig, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot(cat_data, patch_artist=True, labels=top_cats,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5, linestyle="--"),
                    capprops=dict(linewidth=2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.25, markeredgecolor="gray"))
    for patch, color in zip(bp["boxes"], cat_colors):
        patch.set_facecolor(color); patch.set_alpha(0.78)
    for i, (data, color) in enumerate(zip(cat_data, cat_colors)):
        mean_v = np.mean(data)
        ax.scatter(i+1, min(mean_v, p99*0.98), marker="D", color="white",
                   edgecolor="black", s=60, zorder=5)
        ax.text(i+1, p99*1.025, f"n={len(data):,}",       ha="center", fontsize=9,  color="#333")
        ax.text(i+1, p99*1.085, f"μ=${mean_v:,.0f}",      ha="center", fontsize=8,  color=color, fontweight="bold")
    ax.set_ylim(-p99*0.02, p99*1.13)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("Primary Category", fontsize=13)
    ax.set_ylabel("Estimated Cost per Paper ($)", fontsize=13)
    ax.set_title("Research Cost Distribution by Category", fontsize=15)
    ax.text(0.5, -0.12, f"◆ = mean  ·  Y-axis capped at 99th percentile (${p99:,.0f}); {n_out:,} extreme outliers not shown",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_cost_by_category_box.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Monthly cost trend with zoom and linear trend overlay ──────────────────
    trend_df = cost_df[~((cost_df["year"] == 2026) & (cost_df["month"] == 3))]
    monthly  = (trend_df
                .assign(date=pd.to_datetime(
                    trend_df["year"].astype(str) + "-" + trend_df["month"].astype(str).str.zfill(2) + "-01"))
                .groupby("date")["total_cost"].mean().sort_index().reset_index())
    x_num   = np.arange(len(monthly))
    coef    = np.polyfit(x_num, monthly["total_cost"].values, 1)
    trend_v = np.polyval(coef, x_num)
    y_lo    = monthly["total_cost"].min() * 0.975
    y_hi    = monthly["total_cost"].max() * 1.030

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(monthly["date"], monthly["total_cost"], y_lo, alpha=0.12, color="steelblue")
    ax.plot(monthly["date"], monthly["total_cost"], "-", color="steelblue", linewidth=2)
    ax.plot(monthly["date"], monthly["total_cost"], "o", color="steelblue", markersize=4, alpha=0.7)
    ax.plot(monthly["date"], trend_v, "--", color="#C00000", linewidth=1.8, alpha=0.85, label="Linear trend")
    ax.annotate(f"${monthly['total_cost'].iloc[0]:,.0f}",
                xy=(monthly["date"].iloc[0], monthly["total_cost"].iloc[0]),
                xytext=(8, 7), textcoords="offset points", fontsize=9, color="steelblue", fontweight="bold")
    pct_growth = (monthly["total_cost"].iloc[-1] - monthly["total_cost"].iloc[0]) / monthly["total_cost"].iloc[0] * 100
    ax.annotate(f"${monthly['total_cost'].iloc[-1]:,.0f}  ({pct_growth:+.1f}%)",
                xy=(monthly["date"].iloc[-1], monthly["total_cost"].iloc[-1]),
                xytext=(-95, 7), textcoords="offset points", fontsize=9, color="steelblue", fontweight="bold")
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Mean Estimated Cost ($)", fontsize=12)
    ax.set_title("Mean Estimated Research Cost per Paper by Month\n(December 2022–February 2026)", fontsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_cost_trend_month.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Preserve tokens column added by 02_nlp.py so 03_analysis.py PCA still works
    existing_pkl = os.path.join(INPUT_DIR, "sample_df_with_costs.pkl")
    if os.path.exists(existing_pkl):
        try:
            old = pd.read_pickle(existing_pkl)
            if "tokens" in old.columns:
                token_map = old.set_index("id")["tokens"]
                df["tokens"] = df["id"].map(token_map)
        except Exception:
            pass

    # Save
    cost_df.to_pickle(os.path.join(INPUT_DIR, "cost_df.pkl"))
    df.to_pickle(os.path.join(INPUT_DIR, "sample_df_with_costs.pkl"))
    np.save(os.path.join(INPUT_DIR, "low_thresh.npy"),  np.array([low_thresh]))
    np.save(os.path.join(INPUT_DIR, "high_thresh.npy"), np.array([high_thresh]))
    print("Saved: cost_df.pkl, sample_df_with_costs.pkl, thresholds")
    print("=== 01_cost.py complete ===")


if __name__ == "__main__":
    main()
