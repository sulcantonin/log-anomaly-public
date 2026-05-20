#!/usr/bin/env python3
"""
Benchmark experiment: Word2Vec + HMM anomaly detection on public loghub datasets.

Produces quantitative metrics (Precision, Recall, F1, MCC, AUC-ROC) and ablation
results to address reviewer feedback requesting comparative evaluation on labeled
benchmark data.

Datasets used:
  - HDFS_v1 (block-level labels, pre-parsed event traces)
  - BGL      (line-level labels, raw log messages)

Reproducibility notes:
  - All random operations are seeded (see SEED constant below).
  - Word2Vec is run with workers=1; multi-worker training is non-deterministic
    even with a fixed seed due to thread-scheduling variance in gensim.
  - Run with:  PYTHONHASHSEED=0 python benchmark_experiment.py
    (PYTHONHASHSEED=0 disables Python's per-process hash randomisation so that
    set/dict iteration order is consistent across runs.)
"""

import os
import csv
import re
import ast
import json
import time
import warnings
import numpy as np

# ─── global seed — change this one value to replicate with a different seed ──
SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
np.random.seed(SEED)
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from gensim.models import Word2Vec
from hmmlearn import hmm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, precision_recall_curve,
)

warnings.filterwarnings("ignore")

torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)

HDFS_DIR = "/global/home/users/asulc/projects/log-anomaly/data/HDFS_v1/preprocessed"
BGL_LOG   = "/global/home/users/asulc/projects/log-anomaly/data/BGL/BGL.log"

# ─── reproducibility ────────────────────────────────────────────────────────
RNG = np.random.default_rng(SEED)

# ─── hyper-parameters ───────────────────────────────────────────────────────
W2V_DIM     = 16       # embedding dimension (matches paper)
W2V_WINDOW  = 3
W2V_EPOCHS  = 10
HMM_ITER    = 100
N_TRAIN     = 5_000    # normal blocks used for HMM training (HDFS)
N_TEST_EACH = 1_000    # normal + anomaly blocks in test set  (HDFS)
BGL_LINES   = 200_000  # lines to load from BGL
BGL_WIN     = 20       # sliding window size (BGL)
BGL_STRIDE  = 5        # sliding window stride (BGL)
TBIRD_LOG    = "/global/home/users/asulc/projects/log-anomaly/data/Thunderbird/Thunderbird.log"
TBIRD_LINES  = 2_000_000  # lines to load from Thunderbird (first anomaly at ~877k)
TBIRD_STRIDE = 20          # larger stride than BGL to keep window count tractable

# ─── neural baseline hyper-parameters ───────────────────────────────────────
NEURAL_EPOCHS  = 50
NEURAL_LR      = 1e-3
NEURAL_WD      = 1e-4
NEURAL_BATCH   = 32

SMALL_CFG = dict(d_model=16, nhead=2,  ff_dim=32,  n_tf=1, gru_h=32,  gru_l=1, out_dim=16)
BIG_CFG   = dict(d_model=64, nhead=4,  ff_dim=256, n_tf=2, gru_h=128, gru_l=2, out_dim=32)

# training sizes for the data-efficiency scaling experiment
LOW_DATA_SIZES = [200, 500, 5_000]


# ════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def best_threshold_f1(y_true, scores):
    """Return predictions at the F1-optimal threshold."""
    prec, rec, thresh = precision_recall_curve(y_true, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    idx = np.argmax(f1)
    # precision_recall_curve returns len(thresh) = len(prec) - 1
    t = thresh[min(idx, len(thresh) - 1)]
    return (scores >= t).astype(int)


def compute_metrics(y_true, scores):
    """Compute all requested metrics at the F1-optimal threshold."""
    y_pred = best_threshold_f1(y_true, scores)
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "MCC":       matthews_corrcoef(y_true, y_pred),
        "AUC-ROC":   roc_auc_score(y_true, scores),
    }


def score_sequences_hmm(model, seqs):
    """
    Score each sequence with -log P(seq) / len(seq).
    Higher = more anomalous (lower probability under the normal model).
    """
    scores = []
    for s in seqs:
        if len(s) == 0:
            scores.append(0.0)
            continue
        try:
            lp = model.score(s)
            scores.append(-lp / len(s))
        except Exception:
            scores.append(0.0)
    return np.array(scores)


def train_hmm(seqs, n_components):
    """Fit a Gaussian HMM on a list of embedding arrays."""
    X      = np.concatenate(seqs, axis=0)
    lengths = [len(s) for s in seqs]
    model  = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=HMM_ITER,
        random_state=SEED,
    )
    model.fit(X, lengths)
    return model


def print_results_table(title, results):
    cols = ["Method", "Precision", "Recall", "F1", "MCC", "AUC-ROC"]
    widths = [max(len(c), max(len(r["Method"]) for r in results)) for c in cols]
    widths[0] = max(len("Method"), max(len(r["Method"]) for r in results))
    for i, c in enumerate(cols[1:], 1):
        widths[i] = max(len(c), 6)

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols)) + " |"

    print(f"\n{'─'*len(sep)}")
    print(f"  {title}")
    print(sep)
    print(hdr)
    print(sep)
    for r in results:
        row = "| " + r["Method"].ljust(widths[0])
        for i, c in enumerate(cols[1:], 1):
            row += " | " + f"{r[c]:.4f}".ljust(widths[i])
        row += " |"
        print(row)
    print(sep)


# ════════════════════════════════════════════════════════════════════════════
#  NEURAL BASELINES: Transformer + GRU with Deep SVDD loss
# ════════════════════════════════════════════════════════════════════════════

class _PosEnc(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerGRUEncoder(nn.Module):
    def __init__(self, in_dim, d_model, nhead, ff_dim, n_tf, gru_h, gru_l, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        self.pos  = _PosEnc(d_model)
        layer     = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True,
        )
        self.tf  = nn.TransformerEncoder(layer, num_layers=n_tf)
        self.gru = nn.GRU(d_model, gru_h, num_layers=gru_l, batch_first=True)
        self.head = nn.Linear(gru_h, out_dim, bias=False)  # no bias: SVDD requirement

    def forward(self, x, pad_mask=None):
        h = self.proj(x)
        h = self.pos(h)
        h = self.tf(h, src_key_padding_mask=pad_mask)
        _, hn = self.gru(h)
        return self.head(hn[-1])


def _make_model(cfg, in_dim):
    torch.manual_seed(SEED)
    return TransformerGRUEncoder(in_dim=in_dim, **cfg)


def _pad_batch(emb_list):
    """List of (L, D) np arrays → padded tensor (B, L_max, D) + bool mask."""
    tensors = [torch.tensor(e, dtype=torch.float32) for e in emb_list]
    padded  = nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    mask    = torch.zeros(len(tensors), padded.size(1), dtype=torch.bool)
    for i, t in enumerate(tensors):
        mask[i, t.size(0):] = True
    return padded, mask


def train_svdd(model, train_embs):
    """Deep SVDD: fix center c after first forward pass, then minimize ||φ(x)-c||²."""
    model.eval()
    with torch.no_grad():
        outs = []
        for i in range(0, len(train_embs), NEURAL_BATCH):
            p, m = _pad_batch(train_embs[i:i + NEURAL_BATCH])
            outs.append(model(p, m))
        c = torch.cat(outs).mean(0)
    # anti-collapse: nudge near-zero dimensions
    c = torch.where(c.abs() < 1e-6, torch.full_like(c, 1e-6), c)

    model.train()
    opt = optim.Adam(model.parameters(), lr=NEURAL_LR, weight_decay=NEURAL_WD)
    idx = list(range(len(train_embs)))
    for _ in range(NEURAL_EPOCHS):
        np.random.shuffle(idx)
        for i in range(0, len(idx), NEURAL_BATCH):
            batch = [train_embs[j] for j in idx[i:i + NEURAL_BATCH]]
            p, m  = _pad_batch(batch)
            opt.zero_grad()
            out  = model(p, m)
            loss = ((out - c) ** 2).sum(1).mean()
            loss.backward()
            opt.step()
    return c.detach()


def score_svdd(model, c, test_embs):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(test_embs), NEURAL_BATCH):
            p, m = _pad_batch(test_embs[i:i + NEURAL_BATCH])
            dist = ((model(p, m) - c) ** 2).sum(1)
            scores.extend(dist.numpy().tolist())
    return np.array(scores)


# ════════════════════════════════════════════════════════════════════════════
#  HDFS_v1 EXPERIMENT
# ════════════════════════════════════════════════════════════════════════════

def load_hdfs_templates():
    """Return {EventId: [token, ...]} from HDFS.log_templates.csv."""
    templates = {}
    with open(f"{HDFS_DIR}/HDFS.log_templates.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid  = row["EventId"]
            tmpl = row["EventTemplate"]
            # remove wildcards, lowercase, split on non-alpha
            tmpl = re.sub(r"\[\*\]", " ", tmpl).lower()
            tokens = [t for t in re.findall(r"[a-z]+", tmpl) if len(t) > 1]
            templates[eid] = tokens if tokens else ["unknown"]
    return templates


def load_hdfs_data():
    """Load pre-parsed HDFS sequences and binary labels from .npz."""
    d = np.load(f"{HDFS_DIR}/HDFS.npz", allow_pickle=True)
    x_data = d["x_data"]   # array of lists of event IDs
    y_data = d["y_data"]   # 0=normal, 1=anomaly
    return x_data, y_data


def load_hdfs_count_matrix():
    """Load event-count feature matrix (for Isolation Forest baseline)."""
    rows, labels = [], []
    with open(f"{HDFS_DIR}/Event_occurrence_matrix.csv") as f:
        reader = csv.DictReader(f)
        event_cols = None
        for row in reader:
            if event_cols is None:
                event_cols = [k for k in row if k.startswith("E")]
            rows.append([float(row[k]) for k in event_cols])
            labels.append(1 if row["Label"] == "Fail" else 0)
    return np.array(rows), np.array(labels)


def make_w2v_embeddings(templates, seqs):
    """
    Train Word2Vec on event tokens drawn from block sequences,
    then return per-event embedding dict.
    """
    corpus = []
    for seq in seqs:
        for eid in seq:
            corpus.append(templates.get(eid, ["unknown"]))

    model = Word2Vec(
        sentences=corpus,
        vector_size=W2V_DIM,
        window=W2V_WINDOW,
        min_count=1,
        workers=1,      # workers=1 required for deterministic output with fixed seed
        epochs=W2V_EPOCHS,
        sg=0,   # CBOW
        seed=SEED,
    )

    event_emb = {}
    for eid, toks in templates.items():
        vecs = [model.wv[t] for t in toks if t in model.wv]
        event_emb[eid] = np.mean(vecs, axis=0) if vecs else np.zeros(W2V_DIM)
    return event_emb


def make_onehot_embeddings(templates):
    """One-hot encode each event ID (ablation baseline)."""
    eids = sorted(templates.keys())
    idx  = {e: i for i, e in enumerate(eids)}
    dim  = len(eids)
    event_emb = {}
    for eid in eids:
        v = np.zeros(dim)
        v[idx[eid]] = 1.0
        event_emb[eid] = v
    return event_emb, dim


def make_random_embeddings(templates, dim=16):
    """Random fixed embeddings (control condition)."""
    rng = np.random.default_rng(SEED)
    return {eid: rng.standard_normal(dim) for eid in templates}


def seqs_to_embeddings(seqs, event_emb):
    """Convert lists of event IDs to lists of embedding arrays."""
    zero = np.zeros(len(next(iter(event_emb.values()))))
    result = []
    for seq in seqs:
        arr = np.stack([event_emb.get(eid, zero) for eid in seq])
        result.append(arr)
    return result


def run_hdfs():
    print("\n" + "="*70)
    print("  HDFS_v1 Benchmark")
    print("="*70)

    # ── load ─────────────────────────────────────────────────────────────
    t0 = time.time()
    print("Loading data ...", end=" ", flush=True)
    templates     = load_hdfs_templates()
    x_data, y_data = load_hdfs_data()
    count_X, count_y = load_hdfs_count_matrix()
    print(f"done ({time.time()-t0:.1f}s)")

    normal_idx  = np.where(y_data == 0)[0]
    anomaly_idx = np.where(y_data == 1)[0]
    print(f"  Total={len(y_data):,}  Normal={len(normal_idx):,}  "
          f"Anomaly={len(anomaly_idx):,}")

    # ── split ────────────────────────────────────────────────────────────
    RNG.shuffle(normal_idx)
    RNG.shuffle(anomaly_idx)

    train_idx    = normal_idx[:N_TRAIN]
    test_norm_idx = normal_idx[N_TRAIN: N_TRAIN + N_TEST_EACH]
    test_anom_idx = anomaly_idx[:N_TEST_EACH]
    test_idx     = np.concatenate([test_norm_idx, test_anom_idx])
    test_labels  = np.concatenate([
        np.zeros(len(test_norm_idx)),
        np.ones(len(test_anom_idx))
    ]).astype(int)

    train_seqs = [list(x_data[i]) for i in train_idx]
    test_seqs  = [list(x_data[i]) for i in test_idx]

    print(f"  Train (normal)={len(train_seqs):,}  "
          f"Test normal={len(test_norm_idx):,}  "
          f"Test anomaly={len(test_anom_idx):,}")

    # ── Word2Vec embeddings ───────────────────────────────────────────────
    print("Training Word2Vec ...", end=" ", flush=True)
    t0 = time.time()
    w2v_emb = make_w2v_embeddings(templates, train_seqs)
    print(f"done ({time.time()-t0:.1f}s)")

    onehot_emb, oh_dim = make_onehot_embeddings(templates)
    rand_emb = make_random_embeddings(templates, dim=W2V_DIM)

    results = []

    # ── Main method: Word2Vec + HMM (varying N states) ───────────────────
    for n_states in [2, 4, 8]:
        label = f"Word2Vec + HMM (N={n_states})"
        print(f"  {label} ...", end=" ", flush=True)
        t0 = time.time()

        train_emb = seqs_to_embeddings(train_seqs, w2v_emb)
        test_emb  = seqs_to_embeddings(test_seqs,  w2v_emb)

        model  = train_hmm(train_emb, n_states)
        scores = score_sequences_hmm(model, test_emb)
        m      = compute_metrics(test_labels, scores)
        m["Method"] = label
        results.append(m)
        print(f"F1={m['F1']:.4f}  ({time.time()-t0:.1f}s)")

    # ── Ablation: one-hot embeddings ──────────────────────────────────────
    label = "One-hot + HMM (N=4)"
    print(f"  {label} ...", end=" ", flush=True)
    t0 = time.time()

    train_oh = seqs_to_embeddings(train_seqs, onehot_emb)
    test_oh  = seqs_to_embeddings(test_seqs,  onehot_emb)
    model_oh = train_hmm(train_oh, 4)
    scores   = score_sequences_hmm(model_oh, test_oh)
    m        = compute_metrics(test_labels, scores)
    m["Method"] = label
    results.append(m)
    print(f"F1={m['F1']:.4f}  ({time.time()-t0:.1f}s)")

    # ── Ablation: random embeddings ───────────────────────────────────────
    label = "Random + HMM (N=4)"
    print(f"  {label} ...", end=" ", flush=True)
    t0 = time.time()

    train_rand = seqs_to_embeddings(train_seqs, rand_emb)
    test_rand  = seqs_to_embeddings(test_seqs,  rand_emb)
    model_rand = train_hmm(train_rand, 4)
    scores     = score_sequences_hmm(model_rand, test_rand)
    m          = compute_metrics(test_labels, scores)
    m["Method"] = label
    results.append(m)
    print(f"F1={m['F1']:.4f}  ({time.time()-t0:.1f}s)")

    # ── Baseline: Isolation Forest on event counts ────────────────────────
    label = "Isolation Forest (event counts)"
    print(f"  {label} ...", end=" ", flush=True)
    t0 = time.time()

    # Use the same train/test split on the count matrix
    # count_y rows correspond to the same order as x_data rows
    train_count = count_X[train_idx]
    test_count  = count_X[test_idx]

    clf = IsolationForest(n_estimators=100, contamination="auto", random_state=SEED)
    clf.fit(train_count)
    # decision_function: higher = more normal → negate for anomaly score
    if_scores = -clf.decision_function(test_count)
    m = compute_metrics(test_labels, if_scores)
    m["Method"] = label
    results.append(m)
    print(f"F1={m['F1']:.4f}  ({time.time()-t0:.1f}s)")

    # ── Neural baselines: Small and Big Transformer+GRU with SVDD ────────────
    train_emb_w2v = seqs_to_embeddings(train_seqs, w2v_emb)
    test_emb_w2v  = seqs_to_embeddings(test_seqs,  w2v_emb)

    for label, cfg in [("Small TF+GRU SVDD", SMALL_CFG), ("Big TF+GRU SVDD", BIG_CFG)]:
        print(f"  {label} ...", end=" ", flush=True)
        t0    = time.time()
        model = _make_model(cfg, in_dim=W2V_DIM)
        c     = train_svdd(model, train_emb_w2v)
        sc    = score_svdd(model, c, test_emb_w2v)
        m     = compute_metrics(test_labels, sc)
        m["Method"] = label
        results.append(m)
        print(f"F1={m['F1']:.4f}  ({time.time()-t0:.1f}s)")

    print_results_table("HDFS_v1 Results", results)
    return results


# ════════════════════════════════════════════════════════════════════════════
#  BGL EXPERIMENT
# ════════════════════════════════════════════════════════════════════════════

def _preprocess_message(tokens):
    """Lowercase; replace hex/numeric tokens with $num; drop single-char tokens."""
    out = []
    for t in tokens:
        t = t.lower()
        if re.fullmatch(r"0x[0-9a-f]+|[0-9]+", t):
            t = "$num"
        if len(t) > 1:
            out.append(t)
    return out if out else ["empty"]


def _load_sliding_log(path, max_lines):
    """
    Parse BGL/Thunderbird-format log files.
    Column 0: label ("-" = normal, anything else = anomaly).
    Columns 9+: message tokens.
    Returns list of (label:int, tokens:list[str]).
    """
    entries = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            label    = 0 if parts[0] == "-" else 1
            msg_toks = _preprocess_message(parts[9:])
            entries.append((label, msg_toks))
    return entries


def sliding_windows(entries, win_size, stride):
    """Sliding window over log entries; window label = 1 if any entry is anomalous."""
    labels_raw = [e[0] for e in entries]
    toks_raw   = [e[1] for e in entries]
    wins, win_labels = [], []
    for start in range(0, len(entries) - win_size + 1, stride):
        end = start + win_size
        wins.append(toks_raw[start:end])
        win_labels.append(int(any(labels_raw[start:end])))
    return wins, np.array(win_labels)


def _run_sliding_window_log(log_path, name, n_lines, win_size, stride, max_train_wins=3_000):
    """Shared pipeline for BGL-format datasets (BGL, Thunderbird)."""
    print("\n" + "="*70)
    print(f"  {name} Benchmark (sliding window, first {n_lines:,} lines)")
    print("="*70)

    t0 = time.time()
    print(f"Loading {name} log ...", end=" ", flush=True)
    entries = _load_sliding_log(log_path, n_lines)
    print(f"done ({time.time()-t0:.1f}s)  lines={len(entries):,}")

    wins, win_labels = sliding_windows(entries, win_size, stride)
    n_anom = win_labels.sum()
    print(f"  Windows={len(wins):,}  Anomalous={n_anom:,} ({100*n_anom/len(wins):.1f}%)")

    norm_idx = np.where(win_labels == 0)[0]
    anom_idx = np.where(win_labels == 1)[0]
    split     = int(0.7 * len(norm_idx))
    train_pool = norm_idx[:split]
    test_norm  = norm_idx[split:]
    test_anom  = anom_idx

    rng_local = np.random.default_rng(SEED)
    # cap training windows so HMM/neural training stays tractable
    if len(train_pool) > max_train_wins:
        rng_local.shuffle(train_pool)
        train_idx = train_pool[:max_train_wins]
    else:
        train_idx = train_pool

    n_test = min(len(test_norm), len(test_anom), 500)
    rng_local.shuffle(test_norm); rng_local.shuffle(test_anom)
    test_idx    = np.concatenate([test_norm[:n_test], test_anom[:n_test]])
    test_labels = np.concatenate([np.zeros(n_test), np.ones(n_test)]).astype(int)

    print(f"  Train={len(train_idx):,}  Test normal={n_test:,}  Test anomaly={n_test:,}")

    # ── Word2Vec ──────────────────────────────────────────────────────────────
    print("Training Word2Vec ...", end=" ", flush=True)
    t0 = time.time()
    corpus = [tok_list for i in train_idx for tok_list in wins[i]]
    w2v = Word2Vec(
        sentences=corpus,
        vector_size=W2V_DIM,
        window=W2V_WINDOW,
        min_count=2,
        workers=1,
        epochs=W2V_EPOCHS,
        sg=0,
        seed=SEED,
    )
    print(f"done ({time.time()-t0:.1f}s)  vocab={len(w2v.wv):,}")

    zero = np.zeros(W2V_DIM)

    def embed_line(tok_list):
        vecs = [w2v.wv[t] for t in tok_list if t in w2v.wv]
        return np.mean(vecs, axis=0) if vecs else zero.copy()

    def window_to_emb_seq(win):
        return np.stack([embed_line(tl) for tl in win])

    train_seqs = [window_to_emb_seq(wins[i]) for i in train_idx]
    test_seqs  = [window_to_emb_seq(wins[i]) for i in test_idx]

    results = []

    # ── W2V + HMM (N=2,4,8) ──────────────────────────────────────────────────
    for n_states in [2, 4, 8]:
        lbl = f"Word2Vec + HMM (N={n_states})"
        print(f"  {lbl} ...", end=" ", flush=True)
        t0    = time.time()
        model = train_hmm(train_seqs, n_states)
        sc    = score_sequences_hmm(model, test_seqs)
        m     = compute_metrics(test_labels, sc)
        m["Method"] = lbl
        results.append(m)
        print(f"F1={m['F1']:.4f}  ({time.time()-t0:.1f}s)")

    # ── Isolation Forest (BoW) ────────────────────────────────────────────────
    vocab   = {w: i for i, w in enumerate(w2v.wv.index_to_key)}
    n_vocab = len(vocab)

    def win_bow(win):
        v = np.zeros(n_vocab)
        for tl in win:
            for t in tl:
                if t in vocab:
                    v[vocab[t]] += 1
        return v

    train_bow = np.stack([win_bow(wins[i]) for i in train_idx])
    test_bow  = np.stack([win_bow(wins[i]) for i in test_idx])
    clf = IsolationForest(n_estimators=100, contamination="auto", random_state=SEED)
    clf.fit(train_bow)
    m_if = compute_metrics(test_labels, -clf.decision_function(test_bow))
    m_if["Method"] = "Isolation Forest (BoW)"
    results.append(m_if)

    # ── Neural baselines ──────────────────────────────────────────────────────
    for lbl, cfg in [("Small TF+GRU SVDD", SMALL_CFG), ("Big TF+GRU SVDD", BIG_CFG)]:
        print(f"  {lbl} ...", end=" ", flush=True)
        t0    = time.time()
        net   = _make_model(cfg, in_dim=W2V_DIM)
        c     = train_svdd(net, train_seqs)
        sc    = score_svdd(net, c, test_seqs)
        mn    = compute_metrics(test_labels, sc)
        mn["Method"] = lbl
        results.append(mn)
        print(f"F1={mn['F1']:.4f}  ({time.time()-t0:.1f}s)")

    print_results_table(f"{name} Results", results)
    return results


def run_bgl():
    return _run_sliding_window_log(BGL_LOG, "BGL", BGL_LINES, BGL_WIN, BGL_STRIDE)


def run_thunderbird():
    return _run_sliding_window_log(TBIRD_LOG, "Thunderbird", TBIRD_LINES, BGL_WIN, TBIRD_STRIDE)


# ════════════════════════════════════════════════════════════════════════════
#  DATA-EFFICIENCY SCALING EXPERIMENT (HDFS)
# ════════════════════════════════════════════════════════════════════════════

def run_hdfs_neural_scaling():
    """
    Train W2V+HMM(N=8), Small TF+GRU, and Big TF+GRU at N_TRAIN in
    LOW_DATA_SIZES on HDFS_v1.  Test set is always fixed at the same
    1000 normal + 1000 anomaly blocks to ensure comparability.
    """
    print("\n" + "="*70)
    print("  HDFS_v1 — Data-Efficiency Scaling (neural vs HMM)")
    print("="*70)

    templates       = load_hdfs_templates()
    x_data, y_data  = load_hdfs_data()
    normal_idx  = np.where(y_data == 0)[0]
    anomaly_idx = np.where(y_data == 1)[0]

    rng_local = np.random.default_rng(SEED)
    rng_local.shuffle(normal_idx)
    rng_local.shuffle(anomaly_idx)

    # Fixed test set — always taken from beyond the largest training split
    N_RESERVE   = LOW_DATA_SIZES[-1]   # 5000
    test_norm   = normal_idx[N_RESERVE: N_RESERVE + N_TEST_EACH]
    test_anom   = anomaly_idx[:N_TEST_EACH]
    test_idx    = np.concatenate([test_norm, test_anom])
    test_labels = np.concatenate([np.zeros(len(test_norm)),
                                  np.ones(len(test_anom))]).astype(int)
    test_seqs   = [list(x_data[i]) for i in test_idx]

    scaling_rows = []   # list of {Method, N, ...metrics}

    for n_train in LOW_DATA_SIZES:
        print(f"\n  ── N_train = {n_train:,} ──")
        train_idx  = normal_idx[:n_train]
        train_seqs = [list(x_data[i]) for i in train_idx]

        # W2V embeddings retrained on this subset
        w2v_emb = make_w2v_embeddings(templates, train_seqs)
        train_emb = seqs_to_embeddings(train_seqs, w2v_emb)
        test_emb  = seqs_to_embeddings(test_seqs,  w2v_emb)

        # W2V + HMM (N=8) — reference method
        label = "W2V+HMM (N=8)"
        print(f"    {label} ...", end=" ", flush=True)
        t0    = time.time()
        model = train_hmm(train_emb, 8)
        sc    = score_sequences_hmm(model, test_emb)
        m     = compute_metrics(test_labels, sc)
        m.update({"Method": label, "N": n_train})
        scaling_rows.append(m)
        print(f"F1={m['F1']:.4f}  ({time.time()-t0:.1f}s)")

        # Neural baselines
        for net_label, cfg in [("Small TF+GRU SVDD", SMALL_CFG),
                                ("Big TF+GRU SVDD",   BIG_CFG)]:
            print(f"    {net_label} ...", end=" ", flush=True)
            t0    = time.time()
            net   = _make_model(cfg, in_dim=W2V_DIM)
            c     = train_svdd(net, train_emb)
            sc    = score_svdd(net, c, test_emb)
            m     = compute_metrics(test_labels, sc)
            m.update({"Method": net_label, "N": n_train})
            scaling_rows.append(m)
            print(f"F1={m['F1']:.4f}  ({time.time()-t0:.1f}s)")

    # Print compact scaling table (F1 + AUC-ROC columns per N)
    methods = ["W2V+HMM (N=8)", "Small TF+GRU SVDD", "Big TF+GRU SVDD"]
    header  = f"{'Method':<22}" + "".join(
        f"  N={n:<5} F1    AUC " for n in LOW_DATA_SIZES
    )
    print(f"\n{'─'*len(header)}")
    print("  HDFS_v1 — Data-Efficiency Scaling")
    print(f"{'─'*len(header)}")
    print(header)
    print("─" * len(header))
    row_by = {(r["Method"], r["N"]): r for r in scaling_rows}
    for method in methods:
        row = f"{method:<22}"
        for n in LOW_DATA_SIZES:
            r    = row_by.get((method, n), {})
            f1   = r.get("F1",      float("nan"))
            auc  = r.get("AUC-ROC", float("nan"))
            row += f"  {f1:.4f}  {auc:.4f}   "
        print(row)
    print("─" * len(header))

    return scaling_rows


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\nWord2Vec + HMM — Benchmark Evaluation")
    print("Datasets: HDFS_v1 (block-level) | BGL (sliding window)")

    hdfs_results    = run_hdfs()
    bgl_results     = run_bgl()
    tbird_results   = run_thunderbird()
    scaling_results = run_hdfs_neural_scaling()

    all_results = {
        "HDFS_v1":      hdfs_results,
        "BGL":          bgl_results,
        "Thunderbird":  tbird_results,
        "HDFS_scaling": scaling_results,
    }
    out_path = "/global/home/users/asulc/projects/log-anomaly/benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
