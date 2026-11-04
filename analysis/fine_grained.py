"""PACC by CWE type, by function length, and by patch size; PACC on pairs that fit into a single 512-token chunk
(paper, RQ2 and RQ4).

Usage:
    python analysis/fine_grained.py results/pairvul_linux/patchcon.csv
    python analysis/fine_grained.py results/pairvul_linux/patchcon.csv --single-chunk   # needs `transformers`

- CWE: the 12 most frequent CWE types; "train pairs" counts training pairs labelled with exactly this CWE type.
- Function length: lines of the vulnerable function, bins <=20, 21-40, 41-80, >80.
- Patch size: changed lines of a line-level diff (difflib) between the two versions, bins <=2, 3-5, 6-10, >10.
- Single chunk: both versions have at most 510 CodeBERT tokens, so max pooling does not change their representations.
"""
import argparse
import difflib
import os

import pandas as pd

from common import load_predictions, pair_table, r4

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "linux")
TOP12 = ["CWE-416", "CWE-476", "CWE-362", "CWE-20", "CWE-787", "CWE-119", "CWE-200", "CWE-125", "CWE-401", "CWE-264", "CWE-400", "CWE-399"]


def changed_lines(a, b):
    sm = difflib.SequenceMatcher(None, a.splitlines(), b.splitlines(), autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--single-chunk", action="store_true", help="also report PACC on pairs within one 512-token chunk")
    args = ap.parse_args()

    te = pd.read_csv(os.path.join(args.data_dir, "linux_test.csv"))
    tr = pd.read_csv(os.path.join(args.data_dir, "linux_train.csv")).drop_duplicates("pair_id")
    P = pair_table(load_predictions(args.predictions))
    ok = (P.pred_v == 1) & (P.pred_p == 0)
    code = te.set_index(["pair_id", "Vulnerable"]).Function.astype(str)

    print("PACC by CWE type (train pairs, test pairs):")
    cwe = te.drop_duplicates("pair_id").set_index("pair_id").CWE_type.astype(str)
    for w in sorted(TOP12, key=lambda c: -(tr.CWE_type.astype(str) == c).sum()):
        ids = ok.index.intersection(cwe[cwe == w].index)
        print(f"  {w:<8} train={int((tr.CWE_type.astype(str) == w).sum()):<4} n={len(ids):<3} PACC {r4(ok.loc[ids].mean()):.4f}")

    lines = pd.Series({i: code.loc[(i, 1)].count("\n") + 1 for i in ok.index})
    chg = pd.Series({i: changed_lines(code.loc[(i, 1)], code.loc[(i, 0)]) for i in ok.index})
    for title, ser, bins, labels in [("function length (lines of the vulnerable function)", lines, [0, 20, 40, 80, 10**9], ["<=20", "21-40", "41-80", ">80"]),
                                     ("patch size (changed lines)", chg, [-1, 2, 5, 10, 10**9], ["<=2", "3-5", "6-10", ">10"])]:
        print(f"PACC by {title}:")
        cat = pd.cut(ser, bins=bins, labels=labels)
        for lab in labels:
            ids = cat[cat == lab].index
            print(f"  {lab:<6} n={len(ids):<4} PACC {r4(ok.loc[ids].mean()):.4f}")

    if args.single_chunk:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        n_tok = te.assign(t=te.Function.astype(str).map(lambda c: len(tok.tokenize(c)))).groupby("pair_id").t.max()
        single = ok.index.intersection(n_tok[n_tok <= 510].index)
        short = ok.index.intersection(lines[lines <= 20].index)
        print(f"pairs whose two versions both have <= 510 tokens: {len(single)} of {len(ok)} ({len(single) / len(ok):.1%}), "
              f"PACC {r4(ok.loc[single].mean()):.4f}")
        print(f"max tokens among pairs of at most 20 lines: {int(n_tok.loc[short].max())}")


if __name__ == "__main__":
    main()
