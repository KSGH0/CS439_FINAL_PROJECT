"""03_analysis.py — Feature importance, PCA, K-Means, temporal analysis, category analysis."""
import os
import sys
import types
sys.stdout.reconfigure(encoding='utf-8')
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import re as _re
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from config import SEED, OUTPUT_DIR, INPUT_DIR

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def compute_tfidf(token_lists, vocab, idf=None, training=True):
    N, V = len(token_lists), len(vocab)
    tf = np.zeros((N, V), dtype=np.float32)
    for i, tokens in enumerate(token_lists):
        for tok in tokens:
            if tok in vocab:
                tf[i, vocab[tok]] += 1
        s = tf[i].sum()
        if s > 0:
            tf[i] /= s
    if training:
        df  = (tf > 0).sum(axis=0).astype(float)
        idf = np.log((N + 1) / (df + 1)) + 1.0
    tfidf = tf * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf /= norms
    return tfidf, idf


def cluster_purity(true_labels, cluster_labels):
    lmap = {"Low": 0, "Medium": 1, "High": 2}
    true_enc  = np.array([lmap[l] for l in true_labels])
    correct   = 0
    for c in np.unique(cluster_labels):
        m = cluster_labels == c
        correct += (true_enc[m] == np.bincount(true_enc[m]).argmax()).sum()
    return correct / len(true_labels)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)

    print("=== 03_analysis.py ===")

    # Load
    print("\n[1/5] Loading inputs...")
    sample_df  = pd.read_pickle(os.path.join(INPUT_DIR, "sample_df_with_costs.pkl"))
    cost_df    = pd.read_pickle(os.path.join(INPUT_DIR, "cost_df.pkl"))
    X_test     = np.load(os.path.join(INPUT_DIR, "X_test.npy"))
    y_test     = np.load(os.path.join(INPUT_DIR, "y_test.npy"))
    nb_pred    = np.load(os.path.join(INPUT_DIR, "nb_pred.npy"))
    mlp_pred   = np.load(os.path.join(INPUT_DIR, "mlp_pred.npy"))
    lr_weights = np.load(os.path.join(INPUT_DIR, "lr_weights.npy"))
    idf        = np.load(os.path.join(INPUT_DIR, "idf.npy"))
    vocab_df   = pd.read_csv(os.path.join(INPUT_DIR, "vocab.csv"))
    vocab      = dict(zip(vocab_df["word"], vocab_df["idx"]))
    inv_vocab  = {i: w for w, i in vocab.items()}
    lr_model   = types.SimpleNamespace(weights_=lr_weights)
    print(f"Loaded {len(sample_df):,} papers, vocab size {len(vocab):,}")

    # ── Confusion matrices: NB and MLP only (2-panel) ─────────────────────────
    print("\n[2/5] Confusion matrices...")
    cls_names = ["Low", "Medium", "High"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (name, pred) in zip(axes, [("Naive Bayes", nb_pred), ("Neural Network (MLP)", mlp_pred)]):
        cm      = confusion_matrix(y_test, pred, labels=[0, 1, 2])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        annot   = np.array([[f"{cm[r,c]:,}\n({cm_norm[r,c]*100:.0f}%)" for c in range(3)] for r in range(3)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax,
                    xticklabels=cls_names, yticklabels=cls_names,
                    linewidths=0.6, linecolor="white", annot_kws={"size": 9})
        acc = accuracy_score(y_test, pred)
        f1  = f1_score(y_test, pred, average="weighted")
        ax.set_title(f"{name}\nAcc = {acc:.3f}  |  F1 = {f1:.3f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
    plt.suptitle("Confusion Matrices — Cost Bucket Classification\n(cell = count, row % in parentheses)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Rebuild tokens from abstract if 02_nlp.py hasn't been run or pickle was refreshed
    if "tokens" not in sample_df.columns:
        print("  'tokens' column missing — retokenizing from abstract...")
        _stops = set(ENGLISH_STOP_WORDS) | {
            "model", "paper", "propose", "method", "approach", "show", "result",
            "experiment", "dataset", "performance", "based", "learning", "neural",
            "network", "train",
        }
        sample_df = sample_df.copy()
        sample_df["tokens"] = (sample_df["title"].fillna("") + " " + sample_df["abstract"].fillna("")).apply(
            lambda t: [w for w in _re.sub(r"[^a-z0-9]+", " ", str(t).lower()).split()
                       if w not in _stops and len(w) >= 3]
        )
        print("  Retokenization complete.")

    focus_cats = ["cs.CV", "cs.LG", "cs.CL", "cs.AI", "cs.IR", "cs.NE"]

    # PCA visualization
    print("\n[3/5] PCA visualization...")
    sub_idx  = np.random.RandomState(SEED).choice(len(sample_df), 3000, replace=False)
    X_sub, _ = compute_tfidf(sample_df.iloc[sub_idx]["tokens"].tolist(),
                              vocab, idf=idf, training=False)
    sub_meta = sample_df.iloc[sub_idx].reset_index(drop=True)

    svd   = TruncatedSVD(n_components=50, random_state=SEED)
    X_50  = svd.fit_transform(X_sub)
    pca   = PCA(n_components=2, random_state=SEED)
    X_2d  = pca.fit_transform(X_50)

    bucket_col = sub_meta["cost_bucket"].values
    cat_col    = sub_meta["primary_category"].values

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    bkt_colors = {"Low": "#2196F3", "Medium": "#4CAF50", "High": "#F44336"}
    for bkt, color in bkt_colors.items():
        m = bucket_col == bkt
        axes[0].scatter(X_2d[m, 0], X_2d[m, 1], c=color, s=18, alpha=0.52, label=bkt, edgecolors="none")
    # Add centroids
    for bkt, color in bkt_colors.items():
        m = bucket_col == bkt
        cx, cy = X_2d[m, 0].mean(), X_2d[m, 1].mean()
        axes[0].scatter(cx, cy, marker="*", s=280, color=color, edgecolors="black", linewidths=0.8, zorder=5)
    axes[0].set_title("PCA of TF-IDF — Colored by Cost Bucket\n(★ = cluster centroid)", fontsize=13)
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    axes[0].legend(fontsize=10, markerscale=2, framealpha=0.85)
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    top6 = [c for c in focus_cats if c in cat_col]
    pal  = plt.cm.tab10(np.linspace(0, 1, 6))
    for cat, color in zip(top6, pal):
        m = cat_col == cat
        axes[1].scatter(X_2d[m, 0], X_2d[m, 1], c=[color], s=18, alpha=0.52, label=cat, edgecolors="none")
    other_mask = ~np.isin(cat_col, top6)
    if other_mask.any():
        axes[1].scatter(X_2d[other_mask, 0], X_2d[other_mask, 1],
                        c="lightgray", s=8, alpha=0.25, label="Other")
    axes[1].set_title("PCA of TF-IDF — Colored by Category", fontsize=13)
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    axes[1].legend(fontsize=9, markerscale=2, loc="upper right", framealpha=0.85)
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_pca_visualization.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"SVD explains {svd.explained_variance_ratio_.sum()*100:.1f}% variance in 50 components")

    # K-Means clustering
    print("\n[4/5] K-Means clustering and temporal/category analysis...")
    kmeans         = KMeans(n_clusters=3, random_state=SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(X_sub)

    purity = cluster_purity(bucket_col, cluster_labels)
    print(f"K-Means cluster purity (k=3): {purity:.4f}")
    print(f"Random-assignment baseline  : {1/3:.4f}")
    print()
    print("Cluster vs Cost Bucket (row-normalised):")
    ct = pd.crosstab(cluster_labels, bucket_col, normalize="index").round(3)
    print(ct.to_string())

    # Temporal analysis — monthly trend (left) + full-year category comparison (right)
    # Dec 2022 (dataset start) and Jan-Feb 2026 are included; March 2026 excluded (5 days only).
    # Category right panel uses only full years 2023-2025 for fair comparison.

    # Exclude March 2026 — only 5 days of data, not a full month
    monthly_df = sample_df[~((sample_df["year"] == 2026) & (sample_df["month"] == 3))][
        ["year", "month", "total_cost"]].copy()
    monthly_df["date"] = pd.to_datetime(
        monthly_df["year"].astype(str) + "-" + monthly_df["month"].astype(str).str.zfill(2) + "-01"
    )
    monthly_mean = monthly_df.groupby("date")["total_cost"].mean().sort_index()

    full_years  = [2023, 2024, 2025]
    yearly_cat  = (cost_df[cost_df["primary_category"].isin(focus_cats) & cost_df["year"].isin(full_years)]
                   .groupby(["primary_category", "year"])["total_cost"].mean()
                   .reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: monthly trend zoomed to data range with linear overlay
    x_num    = np.arange(len(monthly_mean))
    coef_m   = np.polyfit(x_num, monthly_mean.values, 1)
    trend_m  = np.polyval(coef_m, x_num)
    y_lo_m   = monthly_mean.values.min() * 0.975
    y_hi_m   = monthly_mean.values.max() * 1.030

    axes[0].fill_between(monthly_mean.index, monthly_mean.values, y_lo_m, alpha=0.12, color="steelblue")
    axes[0].plot(monthly_mean.index, monthly_mean.values, "-", color="steelblue", linewidth=2)
    axes[0].plot(monthly_mean.index, monthly_mean.values, "o", color="steelblue", markersize=4, alpha=0.7)
    axes[0].plot(monthly_mean.index, trend_m, "--", color="#C00000", linewidth=1.8, alpha=0.85, label="Linear trend")
    axes[0].annotate(f"${monthly_mean.values[0]:,.0f}",
                     xy=(monthly_mean.index[0], monthly_mean.values[0]),
                     xytext=(6, 6), textcoords="offset points", fontsize=8.5, color="steelblue", fontweight="bold")
    pct_m = (monthly_mean.values[-1] - monthly_mean.values[0]) / monthly_mean.values[0] * 100
    axes[0].annotate(f"${monthly_mean.values[-1]:,.0f}  ({pct_m:+.1f}%)",
                     xy=(monthly_mean.index[-1], monthly_mean.values[-1]),
                     xytext=(-95, 6), textcoords="offset points", fontsize=8.5, color="steelblue", fontweight="bold")
    axes[0].set_ylim(y_lo_m, y_hi_m)
    axes[0].set_title("Mean Estimated Cost per Paper by Month\n(Dec 2022–Feb 2026)", fontsize=13)
    axes[0].set_xlabel("Month", fontsize=12)
    axes[0].set_ylabel("Mean Cost ($)", fontsize=12)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    axes[0].legend(fontsize=10)
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Right: category growth with end-of-line labels
    tc_palette = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#E91E63", "#795548"]
    for cat, color in zip(focus_cats, tc_palette):
        sub = yearly_cat[yearly_cat["primary_category"] == cat]
        if len(sub) > 1:
            axes[1].plot(sub["year"], sub["total_cost"], "o-", label=cat,
                         color=color, linewidth=2.2, markersize=8)
            last_row = sub.iloc[-1]
            axes[1].text(last_row["year"] + 0.05, last_row["total_cost"],
                         f"  {cat}", fontsize=8.5, color=color, va="center", fontweight="bold")
    axes[1].set_title("Cost Growth by Category (2023–2025, full years only)", fontsize=13)
    axes[1].set_xlabel("Year", fontsize=12)
    axes[1].set_ylabel("Mean Cost ($)", fontsize=12)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_temporal_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    pct_chg   = (monthly_mean.iloc[-1] - monthly_mean.iloc[0]) / monthly_mean.iloc[0] * 100
    first_lbl = monthly_mean.index[0].strftime("%b %Y")
    last_lbl  = monthly_mean.index[-1].strftime("%b %Y")
    print(f"Overall cost change ({first_lbl} → {last_lbl}): {pct_chg:+.1f}%")

    # Category analysis
    print("\n[5/5] Category analysis...")
    tc_palette = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#E91E63", "#795548"]
    fc_data    = cost_df[cost_df["primary_category"].isin(focus_cats)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cat_boxes = [fc_data[fc_data["primary_category"] == c]["total_cost"].values
                 for c in focus_cats]
    bp = axes[0].boxplot(cat_boxes, patch_artist=True, labels=focus_cats,
                          medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], tc_palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[0].set_title("Cost Distribution by Primary Category", fontsize=13)
    axes[0].set_ylabel("Estimated Cost ($)", fontsize=12)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    auth_trend = (fc_data[fc_data["year"].isin(full_years)]
                  .groupby(["primary_category", "year"])["author_count"]
                  .mean().reset_index())
    for cat, color in zip(focus_cats, tc_palette):
        sub = auth_trend[auth_trend["primary_category"] == cat]
        axes[1].plot(sub["year"], sub["author_count"], "o-", label=cat,
                     color=color, linewidth=2, markersize=7)
    axes[1].set_title("Mean Author Count by Category (2023–2025, full years only)", fontsize=13)
    axes[1].set_xlabel("Year", fontsize=12)
    axes[1].set_ylabel("Mean Authors per Paper", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_category_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    summary = (fc_data.groupby("primary_category")
               .agg(papers=("total_cost", "count"),
                    mean_cost=("total_cost", "mean"),
                    median_cost=("total_cost", "median"),
                    mean_authors=("author_count", "mean"))
               .round(0))
    print("Human Capital Investment Summary (Focus Categories):")
    print(summary.to_string())

    print("=== 03_analysis.py complete ===")


if __name__ == "__main__":
    main()
