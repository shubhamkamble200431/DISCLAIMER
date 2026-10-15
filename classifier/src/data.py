from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
_WS = re.compile(r"\s+")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(rel):
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def download_raw(cfg):
    d = cfg["dataset"]
    return Path(hf_hub_download(repo_id=d["repo_id"], filename=d["filename"],
                                repo_type=d["repo_type"]))


def _unwrap_json_list(df, col, id_col="id"):
    out, bad_parse, bad_len = [], [], []
    for rid, raw in zip(df[id_col], df[col]):
        try:
            parsed = json.loads(raw)
        except Exception:
            bad_parse.append(rid)
            out.append(None)
            continue
        if not isinstance(parsed, list) or len(parsed) != 1:
            bad_len.append((rid, None if not isinstance(parsed, list) else len(parsed)))
            out.append(None)
            continue
        out.append(parsed[0])
    if bad_parse:
        raise AssertionError(
            f"{col}: {len(bad_parse)} rows failed json.loads. ids={bad_parse[:20]}")
    if bad_len:
        raise AssertionError(
            f"{col}: {len(bad_len)} rows are not length-1 lists. (id, len)={bad_len[:20]}")
    return pd.Series(out, index=df.index, dtype="object")


def melt_pairs(raw):
    prompt = _unwrap_json_list(raw, "prompt")
    resp = {s: _unwrap_json_list(raw, "response_" + s) for s in ("a", "b")}

    frames = []
    for side in ("a", "b"):
        frames.append(pd.DataFrame({
            "prompt_id": raw["id"].astype("int64"),
            "side": side,
            "response_text": resp[side],
            "label_5_raw": raw["refusal_label_" + side],
            "prompt": prompt,
            "model": raw["model_" + side],
            "model_a": raw["model_a"],
            "model_b": raw["model_b"],
            "winner_model_a": raw["winner_model_a"].astype("int64"),
            "winner_model_b": raw["winner_model_b"].astype("int64"),
            "winner_tie": raw["winner_tie"].astype("int64"),
        }))
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["prompt_id", "side"], kind="stable").reset_index(drop=True)


def drop_empty(df):
    text = df["response_text"]
    null_mask = text.isna()
    blank_mask = (~null_mask) & (text.fillna("").astype(str).str.strip() == "")
    keep = ~(null_mask | blank_mask)
    stats = {"n_before": int(len(df)), "n_null": int(null_mask.sum()),
             "n_blank": int(blank_mask.sum()), "n_after": int(keep.sum())}
    return df[keep].reset_index(drop=True), stats


def collapse_labels(df, cfg):
    mapping = cfg["labels"]["collapse"]
    df = df.copy()
    df["label_5"] = df["label_5_raw"].astype(str).str.strip().str.upper()
    unmapped = sorted(set(df["label_5"]) - set(mapping))
    if unmapped:
        counts = df.loc[df["label_5"].isin(unmapped), "label_5"].value_counts().to_dict()
        raise AssertionError("unmapped 5-class labels: " + str(counts))
    df["label_3"] = df["label_5"].map(mapping)
    assert df["label_3"].notna().all(), "label_3 has nulls after mapping"
    order = cfg["labels"]["label_order"]
    df["label_id"] = df["label_3"].map({l: i for i, l in enumerate(order)}).astype("int64")
    return df.drop(columns=["label_5_raw"])


def normalise_text(s):
    return _WS.sub(" ", str(s).strip()).casefold()


def add_text_hash(df):
    df = df.copy()
    norm = df["response_text"].map(normalise_text)
    df["text_norm_len"] = norm.str.len().astype("int64")
    df["text_hash"] = norm.map(
        lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()[:16])
    return df


def duplicate_report(df, top_k=10):
    g = df.groupby("text_hash")
    sizes = g.size()
    dup_hashes = sizes[sizes > 1]

    rep = {
        "n_rows": int(len(df)),
        "n_unique_texts": int(sizes.size),
        "n_duplicate_clusters": int(dup_hashes.size),
        "n_rows_in_duplicate_clusters": int(dup_hashes.sum()),
        "n_redundant_rows": int(dup_hashes.sum() - dup_hashes.size),
        "largest_cluster_size": int(sizes.max()),
    }

    n_pid = g["prompt_id"].nunique()
    cross = n_pid[(sizes > 1) & (n_pid > 1)]
    rep["n_clusters_spanning_multiple_prompt_ids"] = int(cross.size)
    rep["n_rows_in_cross_prompt_clusters"] = int(sizes.loc[cross.index].sum()) if cross.size else 0

    n_lab3 = g["label_3"].nunique()
    conflict3 = n_lab3[(sizes > 1) & (n_lab3 > 1)]
    n_lab5 = g["label_5"].nunique()
    conflict5 = n_lab5[(sizes > 1) & (n_lab5 > 1)]
    rep["n_clusters_conflicting_label_3"] = int(conflict3.size)
    rep["n_clusters_conflicting_label_5"] = int(conflict5.size)

    def cluster_rows(hashes, k=None):
        out = []
        hashes = list(hashes)
        if not hashes:
            return out
        sub = sizes.loc[hashes].sort_values(ascending=False)
        idx = sub.index[:k] if k else sub.index
        for h in idx:
            rows = df[df["text_hash"] == h]
            out.append({
                "text_hash": h,
                "count": int(len(rows)),
                "n_prompt_ids": int(rows["prompt_id"].nunique()),
                "labels_3": {str(k2): int(v) for k2, v in rows["label_3"].value_counts().items()},
                "labels_5": {str(k2): int(v) for k2, v in rows["label_5"].value_counts().items()},
                "text_200": str(rows["response_text"].iloc[0])[:200],
            })
        return out

    rep["top_duplicates"] = cluster_rows(dup_hashes.index, top_k)
    rep["conflicting_clusters"] = cluster_rows(conflict5.index)
    return rep


def length_report(df, cfg):
    from transformers import AutoTokenizer

    pcts = cfg["length_report"]["percentiles"]
    texts = df["response_text"].astype(str).tolist()
    out = {"percentiles": pcts, "n": len(texts), "series": {}}

    chars = np.array([len(t) for t in texts])
    d = {"p" + str(p): float(np.percentile(chars, p)) for p in pcts}
    d["mean"] = float(chars.mean())
    out["series"]["chars"] = d

    for name in cfg["length_report"]["tokenizers"]:
        tok = AutoTokenizer.from_pretrained(name)
        enc = tok(texts, add_special_tokens=True, truncation=False,
                  padding=False)["input_ids"]
        n = np.array([len(e) for e in enc])
        d = {"p" + str(p): float(np.percentile(n, p)) for p in pcts}
        d["mean"] = float(n.mean())
        for cap in (512, 1024, 2048, 4096):
            d["frac_le_" + str(cap)] = float((n <= cap).mean())
        out["series"]["tokens::" + name] = d
    return out


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(df):
    uf = _UnionFind()
    for pid, h in zip(df["prompt_id"], df["text_hash"]):
        uf.union(("P", int(pid)), ("H", h))
    raw_groups = pd.Series([uf.find(("P", int(p))) for p in df["prompt_id"]],
                           index=df.index)
    codes = {g: i for i, g in enumerate(sorted(set(raw_groups), key=str))}
    groups = raw_groups.map(codes).astype("int64")

    n_pid_groups = int(df["prompt_id"].nunique())
    n_groups = int(groups.nunique())
    moved = int(df.assign(_g=groups).groupby("_g")["prompt_id"]
                .transform("nunique").gt(1).sum())
    stats = {
        "n_groups_prompt_id_only": n_pid_groups,
        "n_groups_prompt_id_plus_text_hash": n_groups,
        "n_prompt_id_groups_merged_by_text_hash": n_pid_groups - n_groups,
        "n_rows_in_merged_groups": moved,
    }
    return groups, stats


def make_splits(df, groups, cfg):
    from sklearn.model_selection import StratifiedGroupKFold

    sc = cfg["split"]
    y = df[sc["stratify_on"]].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=sc["n_splits"], shuffle=True,
                                random_state=sc["seed"])
    folds = np.empty(len(df), dtype=int)
    for i, (_, idx) in enumerate(sgkf.split(np.zeros(len(df)), y, groups.to_numpy())):
        folds[idx] = i

    n_test, n_val = sc["test_folds"], sc["val_folds"]
    assign = np.where(folds < n_test, "test",
                      np.where(folds < n_test + n_val, "val", "train"))
    return pd.Series(assign, index=df.index, name="split")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
