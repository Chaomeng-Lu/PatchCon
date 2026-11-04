"""Shared helpers for the analysis scripts: loading per-sample predictions, pairing them, and computing metrics.

A prediction file has one row per function with the columns
    id          pair id (the vulnerable and the patched function of a pair share the same id)
    true_label  1 = vulnerable, 0 = patched
    pred_label  predicted label
    prob_1      predicted probability of the vulnerable class
(`sample_type` and `prob_0` may also be present and are ignored.)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score


def load_predictions(path):
    d = pd.read_csv(path)
    d = d.rename(columns={"id": "pair_id"})
    return d[["pair_id", "true_label", "pred_label", "prob_1"]].astype({"true_label": int, "pred_label": int})


def pair_table(d):
    """One row per vulnerable--patched pair: predicted labels and scores of both versions."""
    v = d[d.true_label == 1].set_index("pair_id")
    p = d[d.true_label == 0].set_index("pair_id")
    ids = v.index.intersection(p.index)
    return pd.DataFrame({"pred_v": v.loc[ids, "pred_label"], "pred_p": p.loc[ids, "pred_label"],
                         "s_v": v.loc[ids, "prob_1"], "s_p": p.loc[ids, "prob_1"]})


def function_metrics(d):
    y, p, s = d.true_label, d.pred_label, d.prob_1
    return {"ACC": accuracy_score(y, p), "Precision": precision_score(y, p, zero_division=0),
            "Recall": recall_score(y, p, zero_division=0), "F1": f1_score(y, p, zero_division=0),
            "MCC": matthews_corrcoef(y, p), "AUPRC": average_precision_score(y, s)}


def pairwise_metrics(P):
    """PACC (= P-C), MCG and the three failure modes P-V, P-B, P-R."""
    return {"PACC": ((P.pred_v == 1) & (P.pred_p == 0)).mean(), "MCG": (P.s_v - P.s_p).mean(),
            "P-V": ((P.pred_v == 1) & (P.pred_p == 1)).mean(), "P-B": ((P.pred_v == 0) & (P.pred_p == 0)).mean(),
            "P-R": ((P.pred_v == 0) & (P.pred_p == 1)).mean(), "n_pairs": len(P)}


def all_metrics(d):
    return {**function_metrics(d), **pairwise_metrics(pair_table(d))}


def r4(x):
    """Round half up to four decimals, as in the paper."""
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(repr(float(x))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def bootstrap_ci(model, base=None, reps=1000, seed=42):
    """95% bootstrap CIs as in the paper (Table 8).

    MCC CI: resampling functions; PACC CI and the CIs of the differences (model minus base): paired resampling of patch pairs.
    Both resamplings start from a generator seeded with `seed`.
    `model` and `base` are prediction frames; if `base` is None, the CIs of `model` itself are returned.
    """
    base = model if base is None else base
    ids = pd.unique(base.pair_id)

    def ordered(d):  # rows: vulnerable, patched for each pair, in the order of `ids`
        return d.set_index(["pair_id", "true_label"]).loc[[(i, y) for i in ids for y in (1, 0)]].reset_index()

    A, B = ordered(model), ordered(base)
    rng = np.random.default_rng(seed)
    n = len(B)
    mcc = [matthews_corrcoef(B.true_label.values[i], B.pred_label.values[i]) for i in (rng.integers(0, n, n) for _ in range(reps))]
    ok = lambda d: ((d.pred_label.values[0::2] == 1) & (d.pred_label.values[1::2] == 0)).astype(float)
    oa, ob = ok(A), ok(B)
    pi = np.random.default_rng(seed).integers(0, len(ids), (reps, len(ids)))
    d_mcc = []
    for s in pi:
        rows = np.r_[2 * s, 2 * s + 1]
        d_mcc.append(matthews_corrcoef(A.true_label.values[rows], A.pred_label.values[rows])
                     - matthews_corrcoef(B.true_label.values[rows], B.pred_label.values[rows]))
    q = lambda x: tuple(np.percentile(x, [2.5, 97.5]))
    return {"MCC_CI": q(mcc), "PACC_CI": q(ob[pi].mean(1)), "dMCC_CI": q(d_mcc), "dPACC_CI": q(oa[pi].mean(1) - ob[pi].mean(1))}
