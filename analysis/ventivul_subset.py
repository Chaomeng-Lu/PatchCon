"""Recompute the `in_paper_subset` flag of data/ventivul/ventivul_linux_pairs.csv.

A VentiVul pair is excluded when its two versions are identical after whitespace normalization or when one of its
functions also appears (after whitespace normalization) in PairVul-Linux. This yields 111 of the 116 pairs.

Usage:
    python analysis/ventivul_subset.py
"""
import hashlib
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
canon = lambda t: hashlib.sha1(" ".join(str(t).split()).encode()).hexdigest()


def main():
    v = pd.read_csv(os.path.join(ROOT, "ventivul", "ventivul_linux_pairs.csv"))
    pv = pd.concat([pd.read_csv(os.path.join(ROOT, "linux", f"linux_{s}.csv"), usecols=["Function"]) for s in ("train", "val", "test")])
    seen = set(pv.Function.map(canon))
    v["h"] = v.Function.map(canon)
    identical = v.groupby("pair_id").h.nunique().eq(1)
    overlap = v[v.h.isin(seen)].pair_id
    drop = set(identical[identical].index) | set(overlap)
    keep = v.pair_id.map(lambda p: p not in drop)
    print(f"pairs: {v.pair_id.nunique()}  excluded: {sorted(drop)}  kept: {v[keep].pair_id.nunique()}")
    print("flag matches the released file:", bool((keep == v.in_paper_subset).all()))


if __name__ == "__main__":
    main()
