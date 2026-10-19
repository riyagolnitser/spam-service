# -*- coding: utf-8 -*-
"""
Обучение бинарного спам-классификатора.

Вход: data/processed/train.csv, val.csv, test.csv
Выход:
- spam_model.joblib
- metrics.json
- classification_report.txt
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42
DATA_DIR = Path("data/processed")
MODEL_PATH = Path("spam_model.joblib")
METRICS_PATH = Path("metrics.json")
REPORT_PATH = Path("classification_report.txt")

MIN_RECALL = 0.90
C_GRID = [0.1, 1.0, 3.0, 10.0]


def add_manual_features(texts):
    features = []

    for text in texts:
        text = text if isinstance(text, str) else ""

        subject = text.split("\n", 1)[0]
        body = text.split("\n", 1)[1] if "\n" in text else ""

        features.append([
            len(subject),
            sum(1 for c in subject if c.isupper()),
            subject.count("!"),
            len(body),
            len(re.findall(r"http[s]?://", text)),
            int(bool(re.search(r"<html|<div|<table", text, re.I))),
            text.count("!") / max(len(text), 1) * 1000,
        ])

    return np.array(features, dtype=float)


import re


def find_best_threshold(y_true, probs, min_recall=MIN_RECALL):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)

    valid = np.where(recall[:-1] >= min_recall)[0]

    if len(valid) > 0:
        best = valid[np.argmax(precision[:-1][valid])]
    else:
        f1 = 2 * precision[:-1] * recall[:-1] / np.clip(
            precision[:-1] + recall[:-1], 1e-12, None
        )
        best = int(np.argmax(f1))

    return {
        "threshold": float(thresholds[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }


def evaluate(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(int)

    return {
        "precision_spam": precision_score(y_true, y_pred),
        "recall_spam": recall_score(y_true, y_pred),
        "f1_spam": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(
            y_true,
            y_pred,
            target_names=["ham", "spam"],
            digits=4,
        ),
    }


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("Test:", test.shape)

    print("\nБаланс train:")
    print(train["label_spam"].value_counts(normalize=True))

    print("\n[1/6] Векторизация...")

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        sublinear_tf=True,
    )

    X_train_text = vectorizer.fit_transform(train["text"])
    X_val_text = vectorizer.transform(val["text"])
    X_test_text = vectorizer.transform(test["text"])

    X_train_manual = csr_matrix(add_manual_features(train["text"]))
    X_val_manual = csr_matrix(add_manual_features(val["text"]))
    X_test_manual = csr_matrix(add_manual_features(test["text"]))

    X_train = hstack([X_train_text, X_train_manual]).tocsr()
    X_val = hstack([X_val_text, X_val_manual]).tocsr()
    X_test = hstack([X_test_text, X_test_manual]).tocsr()

    y_train = train["label_spam"].values
    y_val = val["label_spam"].values
    y_test = test["label_spam"].values

    print("\n[2/6] Подбор C на validation...")

    best_c = None
    best_result = None
    best_probs = None

    for c in C_GRID:
        model = LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]

        result = find_best_threshold(y_val, probs)
        f1 = 2 * result["precision"] * result["recall"] / max(
            result["precision"] + result["recall"], 1e-12
        )

        print(
            f"C={c}: threshold={result['threshold']:.4f}, "
            f"precision={result['precision']:.4f}, "
            f"recall={result['recall']:.4f}, F1={f1:.4f}"
        )

        if best_result is None or f1 > (
            2 * best_result["precision"] * best_result["recall"]
            / max(best_result["precision"] + best_result["recall"], 1e-12)
        ):
            best_c = c
            best_result = result
            best_probs = probs

    print(f"\nЛучший C: {best_c}")

    print("\n[3/6] Кросс-валидация на train...")

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_f1 = []
    cv_precision = []
    cv_recall = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train), 1):
        fold_model = LogisticRegression(
            C=best_c,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )

        fold_model.fit(X_train[train_idx], y_train[train_idx])
        fold_probs = fold_model.predict_proba(X_train[valid_idx])[:, 1]

        fold_result = find_best_threshold(y_train[valid_idx], fold_probs)

        cv_precision.append(fold_result["precision"])
        cv_recall.append(fold_result["recall"])
        cv_f1.append(
            2 * fold_result["precision"] * fold_result["recall"]
            / max(fold_result["precision"] + fold_result["recall"], 1e-12)
        )

        print(
            f"Fold {fold}: P={fold_result['precision']:.4f}, "
            f"R={fold_result['recall']:.4f}, F1={cv_f1[-1]:.4f}"
        )

    print("\n[4/6] Финальное обучение на train...")

    final_model = LogisticRegression(
        C=best_c,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    final_model.fit(X_train, y_train)

    threshold = best_result["threshold"]

    print("\n[5/6] Оценка на validation...")
    val_metrics = evaluate(y_val, best_probs, threshold)
    print(val_metrics["report"])

    print("\n[6/6] Финальная оценка на изолированном test...")
    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_probs, threshold)
    print(test_metrics["report"])

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": final_model,
            "threshold": threshold,
            "best_C": best_c,
            "min_recall": MIN_RECALL,
        },
        MODEL_PATH,
    )

    metrics = {
        "model": "TF-IDF + manual features + LogisticRegression",
        "best_C": best_c,
        "threshold": threshold,
        "threshold_rule": f"max Precision(spam) при Recall(spam) >= {MIN_RECALL}",
        "validation": {
            "precision_spam": val_metrics["precision_spam"],
            "recall_spam": val_metrics["recall_spam"],
            "f1_spam": val_metrics["f1_spam"],
            "confusion_matrix": val_metrics["confusion_matrix"],
        },
        "test": {
            "precision_spam": test_metrics["precision_spam"],
            "recall_spam": test_metrics["recall_spam"],
            "f1_spam": test_metrics["f1_spam"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
        "cross_validation": {
            "n_folds": 5,
            "precision_mean": float(np.mean(cv_precision)),
            "precision_std": float(np.std(cv_precision)),
            "recall_mean": float(np.mean(cv_recall)),
            "recall_std": float(np.std(cv_recall)),
            "f1_mean": float(np.mean(cv_f1)),
            "f1_std": float(np.std(cv_f1)),
        },
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    REPORT_PATH.write_text(
        "VALIDATION\n"
        + val_metrics["report"]
        + "\n\nTEST\n"
        + test_metrics["report"],
        encoding="utf-8",
    )

    print(f"\nМодель сохранена: {MODEL_PATH}")
    print(f"Метрики сохранены: {METRICS_PATH}")
    print(f"Отчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    main()