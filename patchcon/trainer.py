
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
from tqdm import tqdm
import logging
import os
import yaml
import csv
import time
import re
import itertools

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


class Trainer:
    def __init__(self, model, tokenizer, device, optimizer,
                 supcon_loss_fn, scheduler=None,
                 config_path=None):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.supcon_loss = supcon_loss_fn
        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.alpha_supcon = 0.3
        self.alpha_cls = 1.0
        self.alpha_single = 1.0
        self.use_supcon = True
        self.use_cls = True
        self.use_single = True
        self.verbose = 1
        self.checkpoint_path = "best_model.pt"

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.alpha_supcon = config.get("alpha_supcon", self.alpha_supcon)
            self.alpha_cls = config.get("alpha_cls", self.alpha_cls)
            self.alpha_single = config.get("alpha_single", self.alpha_single)
            self.use_supcon = config.get("use_supcon", self.use_supcon)
            self.use_cls = config.get("use_cls", self.use_cls)
            self.use_single = config.get("use_single", self.use_single)
            self.verbose = config.get("verbose", self.verbose)
            self.checkpoint_path = config.get("checkpoint_path", self.checkpoint_path)

        self.best_f1 = 0.0
        self.best_auprc = 0.0

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        logger.info("📦 [Train] Starting epoch...")
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="🧠 Training")

        for step, batch in pbar:
            encoded_a, encoded_b, label_a, label_b, pair_labels, pair_ids, pair_types = batch

            encoded_a = {k: v.to(self.device) for k, v in encoded_a.items()}
            encoded_b = {k: v.to(self.device) for k, v in encoded_b.items()}
            label_a = label_a.to(self.device)
            label_b = label_b.to(self.device)
            pair_labels = pair_labels.to(self.device)
            pair_ids = pair_ids.to(self.device)

            loss = torch.tensor(0.0, device=self.device)
            logs = {}

            if self.use_supcon:
                proj_a, proj_b = self.model(
                    input_ids_a=encoded_a['input_ids'],
                    attention_mask_a=encoded_a['attention_mask'],
                    input_ids_b=encoded_b['input_ids'],
                    attention_mask_b=encoded_b['attention_mask'],
                    mode='supcon'
                )
                B = proj_a.size(0)
                d = proj_a.size(1)
                features = torch.empty((2 * B, d), device=proj_a.device, dtype=proj_a.dtype)
                features[0::2] = proj_a
                features[1::2] = proj_b
                pair_types_interleaved = [pt for p in pair_types for pt in (p, p)]
                loss_supcon = self.supcon_loss(features, pair_types_interleaved)
                loss = loss + self.alpha_supcon * loss_supcon
                logs['supcon_loss'] = loss_supcon.item()

            if self.use_cls:
                valid_mask = pair_labels >= 0
                if valid_mask.sum() == 0:
                    if self.verbose >= 2:
                        logger.debug(f"[Step {step}] No valid pairs for cls_loss, skipping.")
                else:
                    input_ids_a = encoded_a['input_ids'][valid_mask]
                    attention_mask_a = encoded_a['attention_mask'][valid_mask]
                    input_ids_b = encoded_b['input_ids'][valid_mask]
                    attention_mask_b = encoded_b['attention_mask'][valid_mask]
                    valid_labels = pair_labels[valid_mask]

                    cls_loss, _ = self.model(
                        input_ids_a=input_ids_a,
                        attention_mask_a=attention_mask_a,
                        input_ids_b=input_ids_b,
                        attention_mask_b=attention_mask_b,
                        labels=valid_labels,
                        mode='cls'
                    )
                    loss = loss + self.alpha_cls * cls_loss
                    logs['cls_loss'] = cls_loss.item()

            if self.use_single:
                loss_single_a, _ = self.model(
                    input_ids_a=encoded_a['input_ids'],
                    attention_mask_a=encoded_a['attention_mask'],
                    labels=label_a,
                    mode='single'
                )
                loss_single_b, _ = self.model(
                    input_ids_a=encoded_b['input_ids'],
                    attention_mask_a=encoded_b['attention_mask'],
                    labels=label_b,
                    mode='single'
                )
                single_loss = 0.5 * (loss_single_a + loss_single_b)
                loss = loss + self.alpha_single * single_loss
                logs['single_loss'] = single_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            logs['total_loss'] = loss.item()
            pbar.set_postfix({k: f"{v:.3f}" for k, v in logs.items()})

        avg_loss = total_loss / max(1, len(dataloader))
        if self.verbose >= 1:
            logger.info(f"✅ [Train] Epoch complete | Avg Loss = {avg_loss:.4f}")
        return avg_loss

    def evaluate(self, dataloader, name="Val"):
        self.model.eval()
        total_loss = 0.0
        all_preds_single, all_labels_single = [], []
        all_pair_preds, all_pair_labels = [], []
        logger.info(f"🔍 [{name}] Evaluating...")

        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataloader, desc=f"🧪 {name}")):
                encoded_a, encoded_b, label_a, label_b, pair_labels, pair_ids, pair_types = batch
                encoded_a = {k: v.to(self.device) for k, v in encoded_a.items()}
                encoded_b = {k: v.to(self.device) for k, v in encoded_b.items()}
                label_a = label_a.to(self.device)
                label_b = label_b.to(self.device)
                pair_labels = pair_labels.to(self.device)

                loss = torch.tensor(0.0, device=self.device)

                if self.use_single:
                    loss_a, logits_a = self.model(
                        input_ids_a=encoded_a['input_ids'],
                        attention_mask_a=encoded_a['attention_mask'],
                        labels=label_a,
                        mode='single'
                    )
                    loss_b, logits_b = self.model(
                        input_ids_a=encoded_b['input_ids'],
                        attention_mask_a=encoded_b['attention_mask'],
                        labels=label_b,
                        mode='single'
                    )
                    loss = loss + 0.5 * (loss_a + loss_b)
                    all_preds_single.extend(torch.argmax(logits_a, dim=1).cpu().tolist())
                    all_preds_single.extend(torch.argmax(logits_b, dim=1).cpu().tolist())
                    all_labels_single.extend(label_a.cpu().tolist())
                    all_labels_single.extend(label_b.cpu().tolist())

                if self.use_cls:
                    valid_mask = pair_labels >= 0
                    if valid_mask.sum() == 0:
                        if self.verbose >= 2:
                            logger.debug(f"[Eval Step {step}] No valid pairs for cls_loss")
                    else:
                        input_ids_a = encoded_a['input_ids'][valid_mask]
                        attention_mask_a = encoded_a['attention_mask'][valid_mask]
                        input_ids_b = encoded_b['input_ids'][valid_mask]
                        attention_mask_b = encoded_b['attention_mask'][valid_mask]
                        valid_labels = pair_labels[valid_mask]

                        cls_loss, logits_pair = self.model(
                            input_ids_a=input_ids_a,
                            attention_mask_a=attention_mask_a,
                            input_ids_b=input_ids_b,
                            attention_mask_b=attention_mask_b,
                            labels=valid_labels,
                            mode='cls'
                        )
                        loss = loss + cls_loss
                        all_pair_preds.extend(torch.argmax(logits_pair, dim=1).cpu().tolist())
                        all_pair_labels.extend(valid_labels.cpu().tolist())

                total_loss += loss.item()

        result = {'avg_loss': total_loss / max(1, len(dataloader))}
        if self.use_single and all_preds_single:
            result.update(self._compute_metrics(all_labels_single, all_preds_single, prefix='single'))
        if self.use_cls and all_pair_preds:
            result.update(self._compute_metrics(all_pair_labels, all_pair_preds, prefix='pair'))

        # f1_key = 'single_f1' if 'single_f1' in result else 'pair_f1'
        # f1_score_val = result.get(f1_key, 0.0)
        # if f1_score_val > self.best_f1:
        #     self.best_f1 = f1_score_val
        #     torch.save(self.model.state_dict(), self.checkpoint_path)
        #     logger.info(f"💾 Saved best model to {self.checkpoint_path} (f1={f1_score_val:.4f})")
        
        auprc_key = 'single_auprc' if 'single_auprc' in result else 'pair_auprc'
        auprc_val = result.get(auprc_key, 0.0)
        if auprc_val > self.best_auprc:
            self.best_auprc = auprc_val
            torch.save(self.model.state_dict(), self.checkpoint_path)
            logger.info(f"💾 Saved best model to {self.checkpoint_path} (auprc={auprc_val:.4f})")

        for k, v in result.items():
            logger.info(f"    {k:<12}: {v:.4f}")
        return result

    def test_single(self, dataloader, save_path: str = None, save_logits: bool = False,
                    threshold=None, test_path=None, compute_embedding_metrics=True,
                    embedding_space="projection", positive_scope="cwe",
                    max_positive_pairs_per_type=1000, random_seed=42):
        """
        Test single-function classification and optionally compute pairwise + embedding metrics.

        Args:
            dataloader: SingleFunctionDataset DataLoader. Expected batch output:
                encoded, labels, pair_id, sample_type
            save_path: output CSV for prediction scores.
            save_logits: whether to save raw logits.
            threshold: optional threshold for class 1 probability.
            test_path: original test CSV. Needed for CWE_type mapping.
            compute_embedding_metrics:
                If True, compute Mean Repair Cosine, Mean Positive Cosine,
                Positive/Repair Cosine Gap, etc.
            embedding_space:
                "projection" uses projection_head(encoder_repr), matching SupCon space.
                "encoder" uses encoder pooled representation before projection_head.
            positive_scope:
                "cwe" computes same-label positives within the same CWE_type.
                "global" computes same-label positives globally.
            max_positive_pairs_per_type:
                Upper bound for sampled same-label positive pairs to avoid O(N^2) explosion.
            random_seed: seed for positive-pair sampling.
        """
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        rows = []
        embedding_rows = []
        global_row_idx = 0

        ts = time.strftime("%Y%m%d-%H%M%S")
        if save_path is None:
            save_path = f"test_results/single_test_scores_{ts}.csv"

        logger.info("🔍 [Test] Testing single-function performance...")
        with torch.no_grad():
            for encoded, labels, *extras in tqdm(dataloader, desc="🧪 Test-Single"):
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                labels = labels.to(self.device)

                sample_ids = None
                sample_types = None
                for e in extras:
                    if isinstance(e, (list, tuple)) and len(e) == labels.size(0):
                        if isinstance(e[0], (str, int)):
                            # In your SingleFunctionDataset, this is normally pair_id first,
                            # then sample_type second.
                            if sample_ids is None:
                                sample_ids = list(e)
                            elif sample_types is None:
                                sample_types = list(e)

                logits = self.model(
                    input_ids_a=encoded['input_ids'],
                    attention_mask_a=encoded['attention_mask'],
                    mode='single'
                )
                if isinstance(logits, tuple):
                    logits = logits[1]

                loss = F.cross_entropy(logits, labels)
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size

                probs = torch.softmax(logits, dim=1)
                if threshold is None:
                    preds = torch.argmax(logits, dim=1)
                else:
                    preds = (probs[:, 1] >= threshold).long()

                batch_embeddings = None
                if compute_embedding_metrics:
                    batch_embeddings = self._extract_single_embeddings(
                        encoded=encoded,
                        embedding_space=embedding_space
                    ).detach().cpu().numpy()

                C = probs.size(1)
                for i in range(batch_size):
                    sample_id = sample_ids[i] if sample_ids is not None else global_row_idx
                    sample_type = sample_types[i] if sample_types is not None else None

                    row = {
                        "id": str(sample_id),
                        "true_label": int(labels[i].item()),
                        "pred_label": int(preds[i].item()),
                    }
                    if sample_type is not None:
                        row["sample_type"] = str(sample_type)
                    for c in range(C):
                        row[f"prob_{c}"] = float(probs[i, c].item())
                    if save_logits:
                        for c in range(C):
                            row[f"logits_{c}"] = float(logits[i, c].item())
                    rows.append(row)

                    if compute_embedding_metrics and batch_embeddings is not None:
                        embedding_rows.append({
                            "id": str(sample_id),
                            "true_label": int(labels[i].item()),
                            "embedding": batch_embeddings[i],
                        })

                    global_row_idx += 1

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        result = self._compute_metrics(all_labels, all_preds, prefix='single')
        result['avg_loss'] = total_loss / max(1, len(all_labels))
        for k, v in result.items():
            logger.info(f"    {k:<12}: {v:.4f}")

        if rows:
            fieldnames = list(rows[0].keys())
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"📝 Per-sample scores saved to: {save_path}")

        results_df = pd.DataFrame(rows)
        if len(results_df) == 0:
            logger.warning("No test rows collected; skip pairwise and embedding metrics.")
            return result

        # Optional metadata merge: needed for CWE-level metrics.
        eval_df = self._attach_test_metadata(results_df, test_path=test_path)
        
        # =========================
        # CWE-level regular metrics
        # =========================
        cwe_cls_summary = self.calculate_cwe_classification_metrics(eval_df)
        self.log_cwe_classification_metrics(cwe_cls_summary)

        # =========================
        # Pairwise prediction metrics
        # =========================
        pairwise_metrics = self.calculate_pairwise_and_cwe_metrics(eval_df)
        self.log_pairwise_metrics(*pairwise_metrics)

        # Add pairwise metrics to returned result with safe keys.
        overall_pairwise, _, _ = pairwise_metrics
        for k, v in overall_pairwise.items():
            result[self._metric_key(k)] = v

        # =========================
        # Embedding-level metrics
        # =========================
        if compute_embedding_metrics and embedding_rows:
            emb_df = pd.DataFrame(embedding_rows)
            emb_df = self._attach_test_metadata(emb_df, test_path=test_path)
            embedding_metrics = self.calculate_embedding_pair_metrics(
                emb_df,
                positive_scope=positive_scope,
                max_positive_pairs_per_type=max_positive_pairs_per_type,
                random_seed=random_seed,
            )
            self.log_embedding_metrics(*embedding_metrics)

            overall_emb, _, _ = embedding_metrics
            for k, v in overall_emb.items():
                result[self._metric_key(k)] = v

        return result


    def calculate_cwe_classification_metrics(self, df):
        """
        Compute regular classification metrics for each CWE type.

        Required columns:
            - CWE_type
            - true_label
            - pred_label
        """

        records = []

        for cwe, group in df.groupby('CWE_type'):
            if pd.isna(cwe):
                continue

            y_true = group['true_label'].astype(int).tolist()
            y_pred = group['pred_label'].astype(int).tolist()

            if len(y_true) == 0:
                continue

            records.append({
                'CWE': cwe,
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Recall': recall_score(y_true, y_pred, zero_division=0),
                'F1': f1_score(y_true, y_pred, zero_division=0),
                'MCC': matthews_corrcoef(y_true, y_pred),
                'ROC_AUC': roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else float('nan'),
                'AUPRC': average_precision_score(y_true, y_pred) if 1 in y_true else float('nan'),
                'Support': len(y_true),
                'Vuln_Count': int(sum(y_true)),
                'Safe_Count': int(len(y_true) - sum(y_true)),
            })

        cwe_summary = pd.DataFrame(records)

        if len(cwe_summary) == 0:
            return cwe_summary

        cwe_summary = cwe_summary.sort_values(
            by='Support',
            ascending=False
        )

        return cwe_summary


    def _extract_single_embeddings(self, encoded, embedding_space="projection"):
        """Extract embeddings for single-function samples."""
        if not hasattr(self.model, "encode_with_pooling"):
            raise AttributeError(
                "Model must expose encode_with_pooling() to compute embedding metrics."
            )

        cls_vec = self.model.encode_with_pooling(
            encoded['input_ids'],
            encoded['attention_mask']
        )

        if embedding_space == "projection":
            if not hasattr(self.model, "projection_head"):
                raise AttributeError(
                    "Model must expose projection_head to compute projection embeddings."
                )
            emb = self.model.projection_head(cls_vec)
        elif embedding_space == "encoder":
            emb = cls_vec
        else:
            raise ValueError(
                f"embedding_space must be 'projection' or 'encoder', got {embedding_space}"
            )
        return F.normalize(emb, dim=1)

    def _attach_test_metadata(self, df, test_path=None):
        """Attach pair_id and cleaned CWE_type to a result dataframe."""
        df = df.copy()
        df["id"] = df["id"].astype(str)

        # In SingleFunctionDataset, id is normally pair_id.
        if "pair_id" not in df.columns:
            df["pair_id"] = df["id"]

        if test_path is not None and os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            if "pair_id" in test_df.columns:
                test_df["pair_id"] = test_df["pair_id"].astype(str)
                meta_cols = ["pair_id"]
                if "CWE_type" in test_df.columns:
                    meta_cols.append("CWE_type")

                meta = test_df[meta_cols].drop_duplicates("pair_id")

                df = df.drop(columns=["pair_id"], errors="ignore").merge(
                    meta,
                    left_on="id",
                    right_on="pair_id",
                    how="left"
                )
                if "pair_id" not in df.columns:
                    df["pair_id"] = df["id"]

        if "CWE_type" not in df.columns:
            df["CWE_type"] = np.nan
        df["CWE_type"] = df["CWE_type"].apply(self.clean_cwe)
        return df

    @staticmethod
    def _metric_key(name):
        return name.lower().replace("/", "_").replace(" ", "_")

    @staticmethod
    def _cosine(u, v, eps=1e-12):
        u = np.asarray(u, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        denom = np.linalg.norm(u) * np.linalg.norm(v)
        if denom < eps:
            return np.nan
        return float(np.dot(u, v) / denom)

    @staticmethod
    def clean_cwe(s):
        if pd.isna(s):
            return np.nan
        s = str(s).strip()
        if not s:
            return np.nan
        matches = re.findall(r"CWE-\d+", s)
        return matches[0] if len(matches) == 1 else np.nan

    def calculate_pairwise_and_cwe_metrics(self, df):
        """
        Prediction-level pairwise metrics.

        Required columns:
            - pair_id
            - CWE_type
            - true_label
            - pred_label
            - prob_1
        """
        required = {"pair_id", "CWE_type", "true_label", "pred_label", "prob_1"}
        missing = required - set(df.columns)
        if missing:
            logger.warning(f"Skip pairwise metrics; missing columns: {sorted(missing)}")
            empty = pd.DataFrame(columns=[
                "pair_id", "CWE", "pairwise_accuracy", "confidence_gap", "contrastive_win"
            ])
            return {
                "Pairwise Accuracy": np.nan,
                "Mean Confidence Gap": np.nan,
                "Contrastive Win Rate": np.nan,
                "Total Pairs": 0,
            }, empty, pd.DataFrame()

        pairwise_records = []
        for (pair_id, cwe), group in df.groupby(['pair_id', 'CWE_type'], dropna=False):
            if len(group) < 2:
                continue
            pos_sample = group[group['true_label'] == 1]
            neg_sample = group[group['true_label'] == 0]
            if len(pos_sample) == 0 or len(neg_sample) == 0:
                continue

            pos_sample = pos_sample.iloc[0]
            neg_sample = neg_sample.iloc[0]
            is_correct = int(pos_sample['pred_label'] == 1 and neg_sample['pred_label'] == 0)
            conf_gap = float(pos_sample['prob_1'] - neg_sample['prob_1'])
            is_win = int(pos_sample['prob_1'] > neg_sample['prob_1'])

            pairwise_records.append({
                'pair_id': pair_id,
                'CWE': cwe,
                'pairwise_accuracy': is_correct,
                'confidence_gap': conf_gap,
                'contrastive_win': is_win
            })

        pairwise_df = pd.DataFrame(pairwise_records)
        if pairwise_df.empty:
            return {
                "Pairwise Accuracy": np.nan,
                "Mean Confidence Gap": np.nan,
                "Contrastive Win Rate": np.nan,
                "Total Pairs": 0,
            }, pairwise_df, pd.DataFrame()

        overall_metrics = {
            'Pairwise Accuracy': float(pairwise_df['pairwise_accuracy'].mean()),
            'Mean Confidence Gap': float(pairwise_df['confidence_gap'].mean()),
            'Contrastive Win Rate': float(pairwise_df['contrastive_win'].mean()),
            'Total Pairs': int(len(pairwise_df))
        }

        cwe_summary = (
            pairwise_df
            .groupby('CWE', dropna=False)
            .agg({
                'pairwise_accuracy': 'mean',
                'confidence_gap': 'mean',
                'contrastive_win': 'mean',
                'pair_id': 'count'
            })
            .rename(columns={
                'pairwise_accuracy': 'Pairwise_Accuracy',
                'confidence_gap': 'Mean_Confidence_Gap',
                'contrastive_win': 'Contrastive_Win_Rate',
                'pair_id': 'Pair_Count'
            })
            .sort_values(by='Pair_Count', ascending=False)
        )
        return overall_metrics, pairwise_df, cwe_summary

    def calculate_embedding_pair_metrics(self, emb_df, positive_scope="cwe",
                                         max_positive_pairs_per_type=1000,
                                         random_seed=42):
        """
        Embedding-level metrics.

        Required columns:
            - id
            - pair_id
            - CWE_type
            - true_label
            - embedding

        Metrics:
            - Mean Repair Cosine:
                cosine(vulnerable embedding, fixed embedding) within the same repair pair_id.
            - Mean Positive Cosine:
                average same-label cosine. By default, positives are sampled within the same CWE.
            - Positive/Repair Cosine Gap:
                Mean Positive Cosine - Mean Repair Cosine.
        """
        required = {"pair_id", "CWE_type", "true_label", "embedding"}
        missing = required - set(emb_df.columns)
        if missing:
            logger.warning(f"Skip embedding metrics; missing columns: {sorted(missing)}")
            return self._empty_embedding_metrics(), pd.DataFrame(), pd.DataFrame()

        records = []

        # 1. Repair cosine: vulnerable vs fixed under the same pair_id.
        for (pair_id, cwe), group in emb_df.groupby(["pair_id", "CWE_type"], dropna=False):
            vuln = group[group["true_label"] == 1]
            fixed = group[group["true_label"] == 0]
            if len(vuln) == 0 or len(fixed) == 0:
                continue
            cos = self._cosine(vuln.iloc[0]["embedding"], fixed.iloc[0]["embedding"])
            if not np.isnan(cos):
                records.append({
                    "pair_id": pair_id,
                    "CWE": cwe,
                    "pair_type": "repair",
                    "cosine": cos,
                })

        # 2. Same-label positive cosine.
        rng = np.random.default_rng(random_seed)
        if positive_scope == "cwe":
            group_keys = ["CWE_type", "true_label"]
        elif positive_scope == "global":
            group_keys = ["true_label"]
        else:
            raise ValueError("positive_scope must be 'cwe' or 'global'")

        for key, group in emb_df.groupby(group_keys, dropna=False):
            label = int(group["true_label"].iloc[0])
            pair_type = "same_vuln" if label == 1 else "same_safe"
            items = group[["pair_id", "CWE_type", "embedding"]].reset_index(drop=True)
            n = len(items)
            if n < 2:
                continue

            all_pairs = list(itertools.combinations(range(n), 2))
            if len(all_pairs) > max_positive_pairs_per_type:
                selected = rng.choice(len(all_pairs), size=max_positive_pairs_per_type, replace=False)
                pair_indices = [all_pairs[idx] for idx in selected]
            else:
                pair_indices = all_pairs

            for i, j in pair_indices:
                # Match training positives: same-label but not the exact same repair pair.
                if str(items.iloc[i]["pair_id"]) == str(items.iloc[j]["pair_id"]):
                    continue
                cos = self._cosine(items.iloc[i]["embedding"], items.iloc[j]["embedding"])
                if not np.isnan(cos):
                    records.append({
                        "pair_id": f"{items.iloc[i]['pair_id']}__{items.iloc[j]['pair_id']}",
                        "CWE": items.iloc[i]["CWE_type"],
                        "pair_type": pair_type,
                        "cosine": cos,
                    })

        metric_df = pd.DataFrame(records)
        if metric_df.empty:
            return self._empty_embedding_metrics(), metric_df, pd.DataFrame()

        mean_by_type = metric_df.groupby("pair_type")["cosine"].mean()
        mean_repair = float(mean_by_type.get("repair", np.nan))
        mean_same_safe = float(mean_by_type.get("same_safe", np.nan))
        mean_same_vuln = float(mean_by_type.get("same_vuln", np.nan))

        pos_df = metric_df[metric_df["pair_type"].isin(["same_safe", "same_vuln"])]
        mean_positive = float(pos_df["cosine"].mean()) if len(pos_df) else np.nan
        gap = mean_positive - mean_repair if not np.isnan(mean_positive) and not np.isnan(mean_repair) else np.nan

        overall = {
            "Mean Repair Cosine": mean_repair,
            "Mean Same Safe Cosine": mean_same_safe,
            "Mean Same Vuln Cosine": mean_same_vuln,
            "Mean Positive Cosine": mean_positive,
            "Positive/Repair Cosine Gap": float(gap) if not np.isnan(gap) else np.nan,
            "Total Repair Pairs Emb": int((metric_df["pair_type"] == "repair").sum()),
            "Total Positive Pairs Emb": int(metric_df["pair_type"].isin(["same_safe", "same_vuln"]).sum()),
        }

        cwe_summary = (
            metric_df
            .groupby(["CWE", "pair_type"], dropna=False)
            .agg(cosine=("cosine", "mean"), pair_id=("pair_id", "count"))
            .rename(columns={"cosine": "Mean_Cosine", "pair_id": "Pair_Count"})
            .reset_index()
            .sort_values(by=["CWE", "pair_type"])
        )
        return overall, metric_df, cwe_summary

    @staticmethod
    def _empty_embedding_metrics():
        return {
            "Mean Repair Cosine": np.nan,
            "Mean Same Safe Cosine": np.nan,
            "Mean Same Vuln Cosine": np.nan,
            "Mean Positive Cosine": np.nan,
            "Positive/Repair Cosine Gap": np.nan,
            "Total Repair Pairs Emb": 0,
            "Total Positive Pairs Emb": 0,
        }

    def log_pairwise_metrics(self, overall, pairwise, cwe):
        logging.info("\n" + "Pairwise Prediction Results:" + "\n" + "=" * 25 + " OVERALL METRICS " + "=" * 25)
        for key, val in overall.items():
            formatted_val = f"{val:.4f}" if isinstance(val, float) and not np.isnan(val) else val
            logging.info(f"{key:<25}: {formatted_val}")

        logging.info("\n" + "=" * 26 + " CWE SUMMARY " + "=" * 26)
        if cwe is not None and not cwe.empty:
            logging.info("\n" + cwe.to_string(float_format=lambda x: f"{x:.4f}"))
        else:
            logging.info("No CWE-level pairwise metrics available.")

        logging.info("\n" + "=" * 23 + " SAMPLE PAIR RECORDS " + "=" * 23)
        if pairwise is not None and not pairwise.empty:
            logging.info("\n" + pairwise.head(20).to_string(index=False))
        else:
            logging.info("No valid pairwise records.")
        logging.info("=" * 65 + "\n")

    def log_embedding_metrics(self, overall, metric_df, cwe):
        logging.info("\n" + "Embedding Contrastive Results:" + "\n" + "=" * 25 + " OVERALL METRICS " + "=" * 25)
        for key, val in overall.items():
            formatted_val = f"{val:.4f}" if isinstance(val, float) and not np.isnan(val) else val
            logging.info(f"{key:<32}: {formatted_val}")

        logging.info("\n" + "=" * 26 + " CWE / PAIR-TYPE SUMMARY " + "=" * 26)
        if cwe is not None and not cwe.empty:
            logging.info("\n" + cwe.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        else:
            logging.info("No CWE-level embedding metrics available.")

        logging.info("\n" + "=" * 23 + " SAMPLE EMBEDDING PAIR RECORDS " + "=" * 23)
        if metric_df is not None and not metric_df.empty:
            logging.info("\n" + metric_df.head(20).to_string(index=False))
        else:
            logging.info("No valid embedding pair records.")
        logging.info("=" * 65 + "\n")

    def log_cwe_classification_metrics(self, cwe_summary):
        """
        Log CWE-level regular classification metrics.
        """

        logging.info("\n" + "=" * 25 + " CWE CLASSIFICATION METRICS " + "=" * 25)

        if cwe_summary is None or cwe_summary.empty:
            logging.info("No CWE-level classification metrics available.")
            logging.info("=" * 80 + "\n")
            return

        logging.info(
            "\n" + cwe_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        logging.info("=" * 80 + "\n")

    def _compute_metrics(self, y_true, y_pred, prefix=""):
        return {
            f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
            f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
            f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
            f"{prefix}_mcc": matthews_corrcoef(y_true, y_pred),
            f"{prefix}_roc_auc": roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else float('nan'),
            f"{prefix}_auprc": average_precision_score(y_true, y_pred) if 1 in y_true else float('nan')
        }