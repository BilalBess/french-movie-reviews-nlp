"""Sentiment analysis of French movie reviews (Allocine, 200k reviews).

Run:  python src/run.py
"""
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REP = ROOT / "reports"
FIG = REP / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(exist_ok=True)
BASE = "https://huggingface.co/datasets/tblard/allocine/resolve/refs%2Fconvert%2Fparquet/allocine/{}/0000.parquet"


def load(split):
    f = DATA / f"{split}.parquet"
    if not f.exists():
        f.write_bytes(requests.get(BASE.format(split), timeout=300).content)
    return pd.read_parquet(f)


def clean(t):
    t = t.lower()
    t = re.sub(r"http\S+", " ", t)
    return t


def timed(name, fn):
    t0 = time.time()
    out = fn()
    print(f"{name}: {time.time() - t0:.0f}s")
    return out


def main():
    tr, va, te = load("train"), load("validation"), load("test")
    for d in (tr, va, te):
        d["text"] = d["review"].map(clean)
    print(len(tr), len(va), len(te))

    # ---- features: word 1-2 grams, optionally + character n-grams
    # (fit on train only)
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=400_000, sublinear_tf=True)
    Xw_tr = timed("word tfidf", lambda: word.fit_transform(tr["text"]))
    Xw_va, Xw_te = word.transform(va["text"]), word.transform(te["text"])

    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=400_000, sublinear_tf=True)
    Xc_tr = timed("char tfidf", lambda: char.fit_transform(tr["text"]))
    Xc_va, Xc_te = char.transform(va["text"]), char.transform(te["text"])

    Xwc_tr, Xwc_va, Xwc_te = hstack([Xw_tr, Xc_tr]).tocsr(), hstack([Xw_va, Xc_va]).tocsr(), hstack([Xw_te, Xc_te]).tocsr()
    y_tr, y_va, y_te = tr["label"].values, va["label"].values, te["label"].values

    candidates = {
        "Majority class": (DummyClassifier(strategy="most_frequent"), Xw_tr, Xw_va, Xw_te),
        "Naive Bayes (word)": (MultinomialNB(alpha=0.3), Xw_tr, Xw_va, Xw_te),
        "Logistic regression (word)": (LogisticRegression(C=10, max_iter=1000), Xw_tr, Xw_va, Xw_te),
        "Linear SVM (word)": (LinearSVC(C=0.5), Xw_tr, Xw_va, Xw_te),
        "Linear SVM (word + char)": (LinearSVC(C=0.3), Xwc_tr, Xwc_va, Xwc_te),
    }
    rows, fitted = [], {}
    for name, (m, a, b, c) in candidates.items():
        timed(name, lambda m=m, a=a: m.fit(a, y_tr))
        p = m.predict(b)
        rows.append({"model": name, "val_accuracy": accuracy_score(y_va, p), "val_f1": f1_score(y_va, p)})
        fitted[name] = (m, b, c)
    val_table = pd.DataFrame(rows).round(4)
    print(val_table.to_string(index=False))
    val_table.to_csv(REP / "validation_results.csv", index=False)

    # ---- pick the best model on validation, evaluate ONCE on test
    best = val_table.sort_values("val_accuracy", ascending=False).iloc[0]["model"]
    m, _, Xte = fitted[best]
    pred = m.predict(Xte)
    acc, f1 = accuracy_score(y_te, pred), f1_score(y_te, pred)
    print(f"BEST = {best} | TEST accuracy {acc:.4f} f1 {f1:.4f}")
    cm = confusion_matrix(y_te, pred)
    print(cm)

    # ---- error analysis
    te = te.assign(pred=pred, ok=(pred == y_te), n_words=te["review"].str.split().str.len())
    te["len_bucket"] = pd.cut(te["n_words"], [0, 20, 50, 100, 200, 1000], labels=["<20", "20-50", "50-100", "100-200", ">200"])
    by_len = te.groupby("len_bucket", observed=True)["ok"].agg(["mean", "size"]).round(3)
    print(by_len)
    neg_words = te["text"].str.contains(r"\bpas\b|\bjamais\b|\bne\b.*\bplus\b", regex=True)
    print("accuracy with negation words:", round(te.loc[neg_words, "ok"].mean(), 3),
          "| without:", round(te.loc[~neg_words, "ok"].mean(), 3))
    wrong = te[~te["ok"]].copy()
    wrong["true"] = wrong["label"].map({0: "negative", 1: "positive"})
    wrong[["review", "true"]].sample(12, random_state=1).to_csv(REP / "misclassified_examples.csv", index=False)
    confident = wrong.assign(score=np.abs(m.decision_function(Xwc_te if "char" in best else Xw_te)[wrong.index])) \
        .sort_values("score", ascending=False).head(10)[["review", "true", "score"]]
    confident.to_csv(REP / "most_confident_errors.csv", index=False)

    # ---- most important words (from the logistic regression, easier to read)
    lr = fitted["Logistic regression (word)"][0]
    names = np.array(word.get_feature_names_out())
    order = np.argsort(lr.coef_[0])
    top_neg, top_pos = names[order[:20]], names[order[-20:][::-1]]
    print("NEGATIVE:", list(top_neg))
    print("POSITIVE:", list(top_pos))
    pd.DataFrame({"negative": top_neg, "positive": top_pos}).to_csv(REP / "top_words.csv", index=False)

    # ---- learning curve: how much data do we need?
    sizes = [1000, 5000, 20000, 80000, 160000]
    accs = []
    for n in sizes:
        idx = np.random.RandomState(0).choice(len(tr), n, replace=False)
        w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=400_000, sublinear_tf=True)
        a = w.fit_transform(tr["text"].iloc[idx])
        mm = LogisticRegression(C=10, max_iter=1000).fit(a, y_tr[idx])
        accs.append(accuracy_score(y_va, mm.predict(w.transform(va["text"]))))
    print(dict(zip(sizes, np.round(accs, 4))))

    # ---- figures
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, accs, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Number of training reviews")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Learning curve (TF-IDF + logistic regression)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "learning_curve.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, words, coef_sign, col, ttl in [(axes[0], top_neg[:15], -1, "#c53030", "Words pushing to NEGATIVE"),
                                           (axes[1], top_pos[:15], 1, "#2f855a", "Words pushing to POSITIVE")]:
        idx = [list(names).index(w) for w in words]
        a.barh(words[::-1], np.abs(lr.coef_[0][idx])[::-1], color=col)
        a.set_title(ttl)
    fig.tight_layout()
    fig.savefig(FIG / "top_words.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(by_len.index.astype(str), by_len["mean"], color="#2b6cb0")
    ax.set_ylim(0.8, 1)
    ax.set_xlabel("Review length (words)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy by review length")
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_by_length.png", dpi=150)
    plt.close(fig)

    (REP / "metrics.json").write_text(json.dumps({
        "best_model": best, "test_accuracy": round(acc, 4), "test_f1": round(f1, 4),
        "confusion_matrix": cm.tolist(), "accuracy_by_length": by_len["mean"].to_dict() | {},
        "accuracy_with_negation_words": round(float(te.loc[neg_words, "ok"].mean()), 4),
        "accuracy_without_negation_words": round(float(te.loc[~neg_words, "ok"].mean()), 4),
        "learning_curve": dict(zip(map(str, sizes), np.round(accs, 4).tolist())),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
