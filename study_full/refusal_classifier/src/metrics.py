"""Shared metrics module. Every candidate is scored through exactly these functions.

Min-n discipline (A5): any subgroup claim with support < MIN_N is reported as
`insufficient_evidence` with n shown, never as a verdict.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LABELS3 = ["NORMAL", "DISCLAIMER", "REFUSAL"]
LABELS5 = ["NORMAL", "DISCLAIMER_TECHNICAL", "DISCLAIMER_ETHICAL",
           "REFUSAL_TECHNICAL", "REFUSAL_ETHICAL"]
MIN_N = 30


def evidence(n, value):
    """Wrap a subgroup statistic with its support and an evidence verdict."""
    return {"n": int(n), "value": value,
            "evidence": "ok" if n >= MIN_N else "insufficient_evidence"}


def classification_metrics(y_true, y_pred, y_prob=None, labels=LABELS3):
    """y_true / y_pred are integer class ids indexed into `labels`."""
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 log_loss, precision_recall_fscore_support)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ids = list(range(len(labels)))

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=ids,
                                                 zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=ids)
    with np.errstate(invalid="ignore", divide="ignore"):
        cmn = cm / cm.sum(axis=1, keepdims=True)
    cmn = np.nan_to_num(cmn)

    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=ids,
                                      average="weighted", zero_division=0)),
        "per_class": {},
        "confusion_matrix": {"labels": labels, "counts": cm.tolist(),
                             "row_normalised": cmn.tolist()},
    }
    for i, lab in enumerate(labels):
        out["per_class"][lab] = {
            "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
            "support": int(s[i]),
            "evidence": "ok" if s[i] >= MIN_N else "insufficient_evidence",
        }
    # headline
    out["disclaimer_f1"] = out["per_class"]["DISCLAIMER"]["f1"]
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        finite = bool(np.isfinite(y_prob).all())
        out["probs_finite"] = finite
        # A collapsed/diverged run can produce NaN/inf probabilities; log_loss()
        # raises ValueError on those rather than returning a value. macro_f1 /
        # accuracy above are computed from y_pred (already-taken argmax), so they
        # stay meaningful and the run still gets a real metrics.json instead of
        # dying here -- see README's "read the numbers" section on collapsed runs.
        out["log_loss"] = float(log_loss(y_true, y_prob, labels=ids)) if finite else None
    return out


def label_geometry(emb, labels, label_names=LABELS3, pooling=None):
    """Centroid geometry of a labelled embedding matrix.

    Reports, per label pair, the Euclidean distance between the two class centroids
    and the cosine similarity between them. Pair numbers alone are not readable --
    a distance of 3.0 means nothing without knowing how wide the classes are -- so
    each class also carries its intra-class spread, and each pair carries a
    `separation` ratio of centroid distance to the mean of the two spreads. Ratio > 1
    means the centroids are further apart than the classes are wide.

    Min-n discipline (A5) applies: a class with support < MIN_N is marked
    `insufficient_evidence`, and any pair touching such a class inherits that mark.
    """
    emb = np.asarray(emb, dtype=np.float64)
    labels = np.asarray(labels)
    if emb.ndim != 2:
        raise ValueError(f"emb must be 2-D [n, dim], got shape {emb.shape}")
    if len(emb) != len(labels):
        raise ValueError(f"emb has {len(emb)} rows but {len(labels)} labels")

    cents, per_label = {}, {}
    for i, name in enumerate(label_names):
        rows = emb[labels == i]
        n = len(rows)
        if n == 0:
            per_label[name] = {"n": 0, "evidence": "insufficient_evidence"}
            continue
        c = rows.mean(axis=0)
        cents[name] = c
        d = np.linalg.norm(rows - c, axis=1)
        cn, rn = np.linalg.norm(c), np.linalg.norm(rows, axis=1)
        denom = rn * cn
        cos = np.divide(rows @ c, denom, out=np.zeros(n), where=denom > 0)
        per_label[name] = {
            "n": int(n),
            "centroid_norm": float(cn),
            "mean_dist_to_centroid": float(d.mean()),
            "mean_cos_to_centroid": float(cos.mean()),
            "evidence": "ok" if n >= MIN_N else "insufficient_evidence",
        }

    pairs = {}
    for a_i, a in enumerate(label_names):
        for b in label_names[a_i + 1:]:
            key = f"{a}|{b}"
            if a not in cents or b not in cents:
                pairs[key] = {"evidence": "insufficient_evidence",
                              "n_a": per_label[a]["n"], "n_b": per_label[b]["n"]}
                continue
            ca, cb = cents[a], cents[b]
            na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
            spread = (per_label[a]["mean_dist_to_centroid"]
                      + per_label[b]["mean_dist_to_centroid"]) / 2
            ok = (per_label[a]["evidence"] == "ok"
                  and per_label[b]["evidence"] == "ok")
            dist = float(np.linalg.norm(ca - cb))
            pairs[key] = {
                "euclidean": dist,
                "cosine": float(ca @ cb / (na * nb)) if na > 0 and nb > 0 else 0.0,
                "separation": float(dist / spread) if spread > 0 else float("inf"),
                "n_a": per_label[a]["n"], "n_b": per_label[b]["n"],
                "evidence": "ok" if ok else "insufficient_evidence",
            }

    return {"pooling": pooling, "dim": int(emb.shape[1]), "n": int(len(emb)),
            "per_label": per_label, "pairs": pairs}


def save_confusion_png(metrics, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(metrics["confusion_matrix"]["counts"])
    cmn = np.array(metrics["confusion_matrix"]["row_normalised"])
    labels = metrics["confusion_matrix"]["labels"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, mat, name, fmt in ((axes[0], cm, "counts", "d"),
                               (axes[1], cmn, "row-normalised", ".2f")):
        im = ax.imshow(mat, cmap="Blues", vmin=0,
                       vmax=mat.max() if mat.max() else 1)
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(name)
        thr = (mat.max() if mat.max() else 1) * 0.6
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        fontsize=9, color="white" if v > thr else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title or "confusion matrix")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return Path(path)


def majority_floor(y_true, labels=LABELS3, majority_id=0):
    y_pred = np.full(len(y_true), majority_id, dtype=int)
    return classification_metrics(y_true, y_pred, labels=labels)


def aggregate_seeds(runs):
    """runs: list of metrics dicts -> mean/std over the scalar headline numbers."""
    keys = ["macro_f1", "weighted_f1", "accuracy", "disclaimer_f1"]
    agg = {}
    for k in keys:
        v = np.array([r[k] for r in runs], dtype=float)
        agg[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=0)),
                  "values": [float(x) for x in v]}
    for lab in runs[0]["per_class"]:
        v = np.array([r["per_class"][lab]["f1"] for r in runs], dtype=float)
        agg[f"f1_{lab}"] = {"mean": float(v.mean()), "std": float(v.std(ddof=0)),
                            "values": [float(x) for x in v]}
    agg["n_seeds"] = len(runs)
    return agg


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path
