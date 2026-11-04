"""PACC and MCG on all test pairs, on pairs whose CVE appears in the training set (Seen), and on pairs whose CVE appears in
neither the training nor the validation set (Unseen) (paper, Section "Robustness to CVE Overlap" / Table 10).

Usage:
    python analysis/cve_overlap.py results/pairvul_linux/patchcon.csv
"""
import argparse
import os

import pandas as pd

from common import load_predictions, pair_table, pairwise_metrics, r4

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "linux")


def cve_subsets(data_dir=DATA):
    tr, va, te = (pd.read_csv(os.path.join(data_dir, f"linux_{s}.csv"), usecols=["pair_id", "cve_id"]) for s in ("train", "val", "test"))
    cve = te.drop_duplicates("pair_id").set_index("pair_id").cve_id
    seen = set(cve[cve.isin(set(tr.cve_id))].index)
    unseen = set(cve[~cve.isin(set(tr.cve_id) | set(va.cve_id))].index)
    return seen, unseen, len(cve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--data-dir", default=DATA)
    args = ap.parse_args()
    seen, unseen, n = cve_subsets(args.data_dir)
    print(f"test pairs: {n}  seen: {len(seen)}  unseen: {len(unseen)}  CVE only in validation: {n - len(seen) - len(unseen)}")
    P = pair_table(load_predictions(args.predictions))
    for name, sel in [("All", P.index), ("Seen", P.index.intersection(seen)), ("Unseen", P.index.intersection(unseen))]:
        m = pairwise_metrics(P.loc[sel])
        print(f"{name:<7} n={m['n_pairs']:<4} PACC {r4(m['PACC']):.4f}  MCG {r4(m['MCG']):.4f}")


if __name__ == "__main__":
    main()
