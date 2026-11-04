"""Function-level and pairwise metrics of one prediction file.

Usage:
    python analysis/metrics.py results/pairvul_linux/patchcon.csv

AUPRC is computed from prob_1. The AUPRC printed in the training/test logs of patchcon/ is computed from the predicted
labels and therefore differs; the paper uses the score-based value reported here.
"""
import argparse

from common import all_metrics, load_predictions, r4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="per-sample prediction CSV (id, true_label, pred_label, prob_1)")
    args = ap.parse_args()
    m = all_metrics(load_predictions(args.predictions))
    for k, v in m.items():
        print(f"{k:<10} {v if k == 'n_pairs' else r4(v)}")


if __name__ == "__main__":
    main()
