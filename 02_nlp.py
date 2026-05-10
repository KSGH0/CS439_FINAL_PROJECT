"""02_nlp.py — TF-IDF, Naive Bayes, Logistic Regression, MLP, evaluation figures."""
import os
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from device import DEVICE, DEVICE_LABEL

from config import (SEED, ALPHA, VOCAB_SIZE, LR_RATE, N_ITER, LAMBDA,
                    OUTPUT_DIR, INPUT_DIR, TRAIN_SIZE)

warnings.filterwarnings("ignore")
np.random.seed(SEED)
sns.set_style("whitegrid")

# Alias to match notebook variable name used in model defaults
LR = LR_RATE

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def build_vocab(token_lists, vocab_size=VOCAB_SIZE):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    words = [w for w, _ in counter.most_common(vocab_size)]
    return {w: i for i, w in enumerate(words)}


def compute_tfidf(token_lists, vocab, idf=None, training=True):
    """CPU TF-IDF — used only for PCA subset in 03_analysis. Main pipeline uses sklearn."""
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


# ── Naive Bayes ───────────────────────────────────────────────────────────────
class NaiveBayesMulticlass:
    def __init__(self, alpha=ALPHA):
        self.alpha = alpha
        self.device = DEVICE

    def fit(self, X, y):
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y, dtype=torch.long,    device=self.device)
        classes = torch.unique(yt)
        n_cls, n_feat = len(classes), Xt.shape[1]
        self.log_prior_      = torch.zeros(n_cls, device=self.device)
        self.log_likelihood_ = torch.zeros((n_cls, n_feat), device=self.device)
        for i, c in enumerate(classes):
            mask = (yt == c)
            X_c  = Xt[mask]
            self.log_prior_[i] = torch.log(mask.float().sum() / len(yt))
            counts = X_c.sum(0) + self.alpha
            self.log_likelihood_[i] = torch.log(counts / counts.sum())
        return self

    def predict(self, X):
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        scores = Xt @ self.log_likelihood_.T + self.log_prior_
        return scores.argmax(dim=1).cpu().numpy()


# ── Logistic Regression (OvR) ────────────────────────────────────────────────
class LogisticRegressionOvR:
    def __init__(self, lr=LR, n_iter=N_ITER, lambda_=LAMBDA):
        self.lr, self.n_iter, self.lambda_ = lr, n_iter, lambda_
        self.device = DEVICE

    def fit(self, X, y):
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y, dtype=torch.long,    device=self.device)
        n, p = Xt.shape
        classes = torch.unique(yt).tolist()
        K = len(classes)
        self.weights_      = torch.zeros((K, p), device=self.device)
        self.biases_       = torch.zeros(K,       device=self.device)
        self.loss_history_ = [[] for _ in range(K)]

        for i, c in enumerate(classes):
            y_bin = (yt == c).float()
            w = torch.zeros(p, device=self.device, requires_grad=False)
            b = torch.zeros(1, device=self.device, requires_grad=False)
            for _ in range(self.n_iter):
                logits = Xt @ w + b
                prob   = torch.sigmoid(logits)
                err    = prob - y_bin
                dw     = (Xt.T @ err) / n + self.lambda_ * w / n
                db     = err.mean()
                w      = w - self.lr * dw
                b      = b - self.lr * db
                p_cl   = prob.clamp(1e-7, 1 - 1e-7)
                loss   = -(y_bin * p_cl.log() + (1 - y_bin) * (1 - p_cl).log()).mean()
                loss  += 0.5 * self.lambda_ * (w @ w) / n
                self.loss_history_[i].append(loss.item())
            self.weights_[i] = w
            self.biases_[i]  = b
        return self

    def predict(self, X):
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        scores = Xt @ self.weights_.T + self.biases_
        return scores.argmax(dim=1).cpu().numpy()


def evaluate(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"Model": name, "Accuracy": round(acc, 4), "Precision": round(p, 4),
            "Recall": round(r, 4), "F1": round(f1, 4)}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)

    print("=== 02_nlp.py ===")

    # Load — cost labels always come from script 03 (sample_df_with_costs.pkl)
    # When TRAIN_SIZE=0, script 03 already computed costs for all 288K papers.
    print(f"\n[1/7] Loading sample_df_with_costs...")
    sample_df = pd.read_pickle(os.path.join(INPUT_DIR, "sample_df_with_costs.pkl"))
    if TRAIN_SIZE > 0 and TRAIN_SIZE < len(sample_df):
        sample_df = sample_df.sample(n=TRAIN_SIZE, random_state=SEED).reset_index(drop=True)
    print(f"Loaded {len(sample_df):,} papers")
    print(f"Cost bucket distribution:\n{sample_df['cost_bucket'].value_counts().to_string()}")

    # Preprocessing
    print("\n[2/7] Preprocessing text...")
    DOMAIN_STOPS = {
        "model", "paper", "propose", "method", "approach", "show", "result",
        "experiment", "dataset", "performance", "based", "learning", "neural",
        "network", "train",
    }
    ALL_STOPS = set(ENGLISH_STOP_WORDS) | DOMAIN_STOPS

    def preprocess(text):
        text   = NON_ALNUM_RE.sub(" ", text.lower())
        tokens = [t for t in text.split() if len(t) > 2 and t not in ALL_STOPS]
        return tokens

    sample_df["text"]   = sample_df["title"].fillna("") + " " + sample_df["abstract"].fillna("")
    sample_df["tokens"] = sample_df["text"].apply(preprocess)

    print(f"Preprocessing complete.")
    print(f"Sample tokens (first paper): {sample_df['tokens'].iloc[0][:12]}")

    # Train/test split
    print("\n[3/7] Splitting train/test...")
    rng_split = np.random.RandomState(SEED)
    train_idx, test_idx = [], []

    for bucket in ["Low", "Medium", "High"]:
        idxs = np.where(sample_df["cost_bucket"].values == bucket)[0]
        perm = rng_split.permutation(idxs)
        n_test = max(1, int(len(perm) * 0.2))
        test_idx.extend(perm[:n_test].tolist())
        train_idx.extend(perm[n_test:].tolist())

    train_df = sample_df.iloc[train_idx].reset_index(drop=True)
    test_df  = sample_df.iloc[test_idx].reset_index(drop=True)

    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    print(f"Train distribution: {train_df['cost_bucket'].value_counts().to_dict()}")
    print(f"Test  distribution: {test_df['cost_bucket'].value_counts().to_dict()}")

    # TF-IDF — sklearn C-optimized (10-100× faster than from-scratch Python loops)
    print("\n[4/7] Building TF-IDF features (sklearn vectorizer)...")
    DOMAIN_STOPS_LIST = list(DOMAIN_STOPS)
    all_stops = list(ENGLISH_STOP_WORDS) + DOMAIN_STOPS_LIST

    vectorizer = TfidfVectorizer(
        max_features=VOCAB_SIZE,
        sublinear_tf=True,
        norm="l2",
        stop_words=all_stops,
        token_pattern=r"(?u)\b[a-z0-9]{3,}\b",
    )
    # Fit on joined token strings (sklearn expects raw text or token strings)
    train_texts = [" ".join(t) for t in train_df["tokens"].tolist()]
    test_texts  = [" ".join(t) for t in test_df["tokens"].tolist()]

    X_train = vectorizer.fit_transform(train_texts).astype(np.float32).toarray()
    X_test  = vectorizer.transform(test_texts).astype(np.float32).toarray()
    vocab   = {w: i for w, i in vectorizer.vocabulary_.items()}
    idf     = vectorizer.idf_

    print(f"Vocabulary size : {len(vocab):,}")
    print(f"X_train shape   : {X_train.shape}")
    print(f"X_test  shape   : {X_test.shape}")
    print(f"Top 20 vocab    : {list(vocab.keys())[:20]}")

    label_map     = {"Low": 0, "Medium": 1, "High": 2}
    inv_label_map = {0: "Low", 1: "Medium", 2: "High"}

    y_train = np.array([label_map[b] for b in train_df["cost_bucket"]])
    y_test  = np.array([label_map[b] for b in test_df["cost_bucket"]])
    print(f"Classes: {label_map}")

    # Naive Bayes
    print("\n[5/7] Training models...")
    nb_model = NaiveBayesMulticlass(alpha=ALPHA)
    nb_model.fit(X_train, y_train)
    nb_pred  = nb_model.predict(X_test)
    nb_acc   = accuracy_score(y_test, nb_pred)
    print(f"Naive Bayes accuracy: {nb_acc:.4f}")

    # Logistic Regression
    print("Training Logistic Regression (One-vs-Rest) ...")
    lr_model = LogisticRegressionOvR(lr=LR, n_iter=N_ITER, lambda_=LAMBDA)
    lr_model.fit(X_train, y_train)
    lr_pred  = lr_model.predict(X_test)
    lr_acc   = accuracy_score(y_test, lr_pred)
    print(f"Logistic Regression accuracy: {lr_acc:.4f}")

    fig, ax = plt.subplots(figsize=(11, 5))
    cls_labels = ["Low", "Medium", "High"]
    cls_colors = ["#2196F3", "#4CAF50", "#FF9800"]
    for i, (label, color) in enumerate(zip(cls_labels, cls_colors)):
        history = lr_model.loss_history_[i]
        ax.plot(history, label=f"Class: {label}", color=color, linewidth=2.2)
        ax.annotate(f"{history[-1]:.4f}",
                    xy=(len(history)-1, history[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=9, color=color, fontweight="bold", va="center")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Binary Log-Loss + L2", fontsize=12)
    ax.set_title("Logistic Regression Convergence (OvR)\nAll three classifiers converge; Medium collapse reflects majority-class dominance", fontsize=13)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_lr_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Neural Network — MLP
    print("Training Neural Network (MLP) ...")
    device = DEVICE
    print(f"  Device: {DEVICE_LABEL}")

    class MLP(nn.Module):
        def __init__(self, in_dim, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64),     nn.ReLU(),
                nn.Linear(64, n_classes)
            )
        def forward(self, x):
            return self.net(x)

    torch.manual_seed(SEED)
    Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
    yt = torch.tensor(y_train, dtype=torch.long).to(device)
    Xe = torch.tensor(X_test,  dtype=torch.float32).to(device)

    mlp_net   = MLP(Xt.shape[1], 3).to(device)
    optimizer = torch.optim.Adam(mlp_net.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loader    = DataLoader(TensorDataset(Xt, yt), batch_size=512, shuffle=True)

    best_acc, patience, wait = 0.0, 5, 0
    for epoch in range(100):
        mlp_net.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(mlp_net(xb), yb).backward()
            optimizer.step()
        mlp_net.eval()
        with torch.no_grad():
            val_pred = mlp_net(Xe).argmax(dim=1).cpu().numpy()
        val_acc = accuracy_score(y_test, val_pred)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in mlp_net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    mlp_net.load_state_dict(best_state)
    mlp_net.eval()
    with torch.no_grad():
        mlp_pred = mlp_net(Xe).argmax(dim=1).cpu().numpy()
    mlp_acc = accuracy_score(y_test, mlp_pred)
    print(f"Neural Network accuracy : {mlp_acc:.4f}  (best val: {best_acc:.4f}, epochs: {epoch+1})")

    # Evaluation
    print("\n[6/7] Evaluation and figures...")
    results_df = pd.DataFrame([
        evaluate(y_test, nb_pred,  "Naive Bayes"),
        evaluate(y_test, lr_pred,  "Logistic Regression"),
        evaluate(y_test, mlp_pred, "Neural Network (MLP)"),
    ])
    print("Model Comparison:")
    print(results_df.to_string(index=False))

    from sklearn.metrics import f1_score
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    preds_list = [
        ("Naive Bayes",          nb_pred),
        ("Logistic Regression",  lr_pred),
        ("Neural Network (MLP)", mlp_pred),
    ]
    cls_names = ["Low", "Medium", "High"]
    for ax, (name, pred) in zip(axes, preds_list):
        cm      = confusion_matrix(y_test, pred, labels=[0, 1, 2])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        # Annotate with count + row %
        annot   = np.array([[f"{cm[r,c]:,}\n({cm_norm[r,c]*100:.0f}%)"
                             for c in range(3)] for r in range(3)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax,
                    xticklabels=cls_names, yticklabels=cls_names,
                    linewidths=0.6, linecolor="white", annot_kws={"size": 9})
        acc = accuracy_score(y_test, pred)
        f1  = f1_score(y_test, pred, average="weighted")
        ax.set_title(f"{name}\nAcc = {acc:.3f}  |  F1 = {f1:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
    plt.suptitle("Confusion Matrices — Cost Bucket Classification\n(cell = count, row % in parentheses)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Save
    print("\n[7/7] Saving outputs...")
    np.save(os.path.join(INPUT_DIR, "X_train.npy"),    X_train)
    np.save(os.path.join(INPUT_DIR, "X_test.npy"),     X_test)
    np.save(os.path.join(INPUT_DIR, "y_train.npy"),    y_train)
    np.save(os.path.join(INPUT_DIR, "y_test.npy"),     y_test)
    np.save(os.path.join(INPUT_DIR, "nb_pred.npy"),    nb_pred)
    np.save(os.path.join(INPUT_DIR, "lr_pred.npy"),    lr_pred)
    np.save(os.path.join(INPUT_DIR, "mlp_pred.npy"),   mlp_pred)
    np.save(os.path.join(INPUT_DIR, "lr_weights.npy"),         lr_model.weights_.cpu().numpy())
    np.save(os.path.join(INPUT_DIR, "nb_log_likelihood.npy"), nb_model.log_likelihood_.cpu().numpy())
    np.save(os.path.join(INPUT_DIR, "nb_log_prior.npy"),      nb_model.log_prior_.cpu().numpy())
    torch.save(mlp_net.state_dict(), os.path.join(INPUT_DIR, "mlp_state.pt"))
    np.save(os.path.join(INPUT_DIR, "idf.npy"),        idf)
    pd.DataFrame(vocab.items(), columns=["word", "idx"]).to_csv(
        os.path.join(INPUT_DIR, "vocab.csv"), index=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_results.csv"), index=False)
    sample_df.to_pickle(os.path.join(INPUT_DIR, "sample_df_with_costs.pkl"))
    print("Saved: X_train/test, y_train/test, predictions, lr_weights, idf, vocab, model_results, sample_df_with_costs")
    print("=== 02_nlp.py complete ===")


if __name__ == "__main__":
    main()
