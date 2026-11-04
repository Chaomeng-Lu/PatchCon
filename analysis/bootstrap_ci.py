"""95% bootstrap confidence intervals (paper, RQ5 / Table 8).

MCC: resampling of functions. PACC and the model differences: paired resampling of patch pairs. 1,000 repetitions, seed 42.

Usage:
    python analysis/bootstrap_ci.py results/pairvul_linux/patchcon.csv
    python analysis/bootstrap_ci.py results/pairvul_linux/patchcon.csv --baseline other_model_predictions.csv

With --baseline, the MCC/PACC CIs refer to the baseline and the difference CIs to PatchCon minus the baseline; the
baseline file must use the same format and pair ids (e.g., predictions of a baseline on data/linux/linux_test.csv).
"""
import argparse

from common import bootstrap_ci, load_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    model = load_predictions(args.predictions)
    base = load_predictions(args.baseline) if args.baseline else None
    ci = bootstrap_ci(model, base, reps=args.reps, seed=args.seed)
    for k, (lo, hi) in ci.items():
        if base is None and k.startswith("d"):
            continue
        print(f"{k:<9} [{lo:+.4f}, {hi:+.4f}]" if k.startswith("d") else f"{k:<9} [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
