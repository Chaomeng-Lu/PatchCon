# === Datasets (with function-level / pair_id-level clean split) ===
import pandas as pd
import hashlib
import os
from datetime import datetime
import numpy as np
from trainer import logger

def _canon(text: str) -> str:
    """Minimal canonicalization for function text: strip spaces & newlines"""
    return " ".join(str(text).split())

def _func_id(code: str) -> str:
    """Generate SHA1 hash for canonicalized function code"""
    return hashlib.sha1(_canon(code).encode("utf-8")).hexdigest()

def make_val_splits(train_csv: str,
                    val_ratio: float = 0.1,
                    mode: str = "func",   # "func" or "func+pair"
                    seed: int = 42,
                    out_dir: str = ".cache_splits"):
    """
    Returns: train_split_csv, val_split_csv
    Split strategy:
        - For repair pairs (pair_id present in >=2 rows): split at pair_id level
        - For missing / single functions: split at func_id level
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(train_csv).copy()
    if "pair_id" not in df.columns:
        df["pair_id"] = "missing"
    else:
        df["pair_id"] = df["pair_id"].fillna("missing").astype(str)

    # build func_id
    df["func_id"] = df["Function"].apply(_func_id)

    rng = np.random.default_rng(seed)

    # === 1. Separate repair pairs and missing/single functions ===
    pid_counts = df["pair_id"].value_counts()
    repair_pids = set(pid_counts[pid_counts >= 2].index) - {"missing"}
    missing_rows = df[df["pair_id"] == "missing"]
    repair_rows = df[df["pair_id"].isin(repair_pids)]

    # === 2. Split repair pairs at pair_id level ===
    repair_pid_list = list(repair_pids)
    rng.shuffle(repair_pid_list)
    n_val_pid = max(1, int(len(repair_pid_list) * val_ratio))
    val_repair_pids = set(repair_pid_list[:n_val_pid])
    train_repair_pids = set(repair_pid_list[n_val_pid:])

    val_repair = repair_rows[repair_rows["pair_id"].isin(val_repair_pids)]
    train_repair = repair_rows[repair_rows["pair_id"].isin(train_repair_pids)]

    # === 3. Split missing functions at func_id level ===
    missing_func_ids = list(missing_rows["func_id"].unique())
    rng.shuffle(missing_func_ids)
    n_val_func = max(1, int(len(missing_func_ids) * val_ratio))
    val_missing_ids = set(missing_func_ids[:n_val_func])
    train_missing_ids = set(missing_func_ids[n_val_func:])

    val_missing = missing_rows[missing_rows["func_id"].isin(val_missing_ids)]
    train_missing = missing_rows[missing_rows["func_id"].isin(train_missing_ids)]

    # === 4. Merge train/val sets ===
    df_train = pd.concat([train_repair, train_missing], ignore_index=True)
    df_val = pd.concat([val_repair, val_missing], ignore_index=True)

    # === 5. Optional "func+pair" strict mode ===
    if mode == "func+pair":
        # Ensure no func_id leakage between train and val
        before_t = len(df_train)
        before_v = len(df_val)
        val_funcs = set(df_val["func_id"].unique())
        df_train = df_train[~df_train["func_id"].isin(val_funcs)]
        train_funcs = set(df_train["func_id"].unique())
        df_val = df_val[~df_val["func_id"].isin(train_funcs)]
        logger.info(f"[Strict mode] Removed {before_t - len(df_train)} train rows and {before_v - len(df_val)} val rows due to func_id overlap.")

    # === 6. Output CSV ===
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_out = os.path.join(out_dir, f"train_split_{mode}_{tag}.csv")
    val_out = os.path.join(out_dir, f"val_split_{mode}_{tag}.csv")
    cols = [c for c in df.columns if c != "func_id"]
    df_train[cols].to_csv(train_out, index=False)
    df_val[cols].to_csv(val_out, index=False)

    # === 7. Log stats ===
    logger.info(f"[Split:{mode}] repair_pairs(train/val) = {len(train_repair_pids)}/{len(val_repair_pids)} | "
          f"missing_funcs(train/val) = {len(train_missing_ids)}/{len(val_missing_ids)} | "
          f"rows(train/val) = {len(df_train)}/{len(df_val)}")

    return train_out, val_out