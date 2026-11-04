import pandas as pd
import random
import torch
import re
from collections import defaultdict
from torch.utils.data import Dataset
from trainer import logger

# Supported pair types
PAIR_TYPES = {
    'repair':        {'cls': 1, 'supcon': 'neg'},   # real fix pair
    'random_neg':    {'cls': 0, 'supcon': 'neg'},   # random safe vs vuln
    'same_safe':     {'cls': None, 'supcon': 'pos'},# unrelated safe functions
    'same_vuln':     {'cls': None, 'supcon': 'pos'},# unrelated vuln functions
}


class ContrastiveFunctionDataset(Dataset):
    def __init__(self, csv_path,
                 max_length=512,
                 max_pairs_per_type=50000,
                 balance_cls_pairs=True,
                 balance_all_pairs=True,
                 seed=42,
                 cwe_filter=None):
        random.seed(seed)
        self.max_length = max_length
        self.max_pairs_per_type = max_pairs_per_type
        self.balance_cls_pairs = balance_cls_pairs
        self.balance_all_pairs = balance_all_pairs

        self.df = pd.read_csv(csv_path)
        self.df.fillna({'pair_id': 'missing'}, inplace=True)

        # === get specified CWE type ===
        if cwe_filter is not None and "CWE_type" in self.df.columns:
            allowed = set(cwe_filter) if isinstance(cwe_filter, (list, set)) else {cwe_filter}
            before = len(self.df)
            self.df = self.df[self.df["CWE_type"].isin(allowed)]
            logger.info(f"[Filter] Keep CWE_types={allowed}, rows {before} -> {len(self.df)}")

        self.label_map = defaultdict(list)      # label -> list of (code, label, pair_id)
        self.pair_map = defaultdict(list)       # pair_id -> list of (code, label, pair_id)

        for _, row in self.df.iterrows():
            code = row["Function"]
            label = int(row["Vulnerable"])
            pair_id = str(row["pair_id"]) if pd.notna(row["pair_id"]) else "missing"
            item = (code, label, pair_id)
            self.label_map[label].append(item)
            self.pair_map[pair_id].append(item)

        # List of all usable pairs with their type
        self.pairs = []

        self._generate_all_pairs()

    def _generate_all_pairs(self):
        # === Type 1: Real repair pairs ===
        repair_pairs = []
        for pair_id, items in self.pair_map.items():
            if len(items) == 2:
                a, b = items
                if a[1] != b[1]:  # label mismatch
                    repair_pairs.append((a, b, 'repair'))
        repair_pairs = repair_pairs[:self.max_pairs_per_type]

        rnp_sn = 1
        if self.balance_cls_pairs:
            rnp_sn = 2

        # ======================================
        # =========balance all 4 types =========
        # ======================================
        if self.balance_all_pairs:
            self.max_pairs_per_type = len(repair_pairs)

        # === Type 2: Random negative (vuln vs safe, not a repair) ===
        random_neg_pairs = []
        vuln_items = self.label_map[1]
        safe_items = self.label_map[0]
        for _ in range(self.max_pairs_per_type * rnp_sn):
            a = random.choice(vuln_items)
            b = random.choice(safe_items)
            if a[2] != b[2]:  # skip actual repair pairs
                random_neg_pairs.append((a, b, 'random_neg'))

        # === Type 3: Same-label vuln-vuln pairs ===
        same_vuln_pairs = []
        vuln_pool = random.sample(vuln_items, min(len(vuln_items), int((2 * self.max_pairs_per_type) ** 2)))
        for i in range(len(vuln_pool)):
            for j in range(i + 1, len(vuln_pool)):
                a, b = vuln_pool[i], vuln_pool[j]
                if a[2] != b[2]:  # skip if same pair_id
                    same_vuln_pairs.append((a, b, 'same_vuln'))
                if len(same_vuln_pairs) >= self.max_pairs_per_type:
                    break
            if len(same_vuln_pairs) >= self.max_pairs_per_type:
                break

        # === Type 4: Same-label safe-safe pairs ===
        same_safe_pairs = []
        safe_pool = random.sample(safe_items, min(len(safe_items), int((2 * self.max_pairs_per_type) ** 2)))
        for i in range(len(safe_pool)):
            for j in range(i + 1, len(safe_pool)):
                a, b = safe_pool[i], safe_pool[j]
                if a[2] != b[2]:
                    same_safe_pairs.append((a, b, 'same_safe'))
                if len(same_safe_pairs) >= self.max_pairs_per_type:
                    break
            if len(same_safe_pairs) >= self.max_pairs_per_type:
                break

        # Optional: balance cls-loss pairs (repair vs random_neg)
        if self.balance_cls_pairs:
            n = min(len(repair_pairs), len(random_neg_pairs))
            repair_pairs = random.sample(repair_pairs, n)
            random_neg_pairs = random.sample(random_neg_pairs, n)

        # Combine all
        self.pairs = repair_pairs + random_neg_pairs + same_vuln_pairs + same_safe_pairs
        random.shuffle(self.pairs)

        logger.info(f"[Pair Summary]")
        logger.info(f"  max_pairs_per_type: {self.max_pairs_per_type}")
        logger.info(f"  repair_pairs:       {len(repair_pairs)}")
        logger.info(f"  random_neg_pairs:   {len(random_neg_pairs)}")
        logger.info(f"  same_vuln_pairs:    {len(same_vuln_pairs)}")
        logger.info(f"  same_safe_pairs:    {len(same_safe_pairs)}")
        logger.info(f"  total pairs:  {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, pair_type = self.pairs[idx]
        code_a, label_a, pair_id_a = a
        code_b, label_b, pair_id_b = b

        # Assign binary pair label only for cls_loss-supported types
        if pair_type in PAIR_TYPES and PAIR_TYPES[pair_type]['cls'] is not None:
            pair_label = PAIR_TYPES[pair_type]['cls']
        else:
            pair_label = -1  # Not for classification

        # Assign unique pair ID
        if pair_id_a == pair_id_b:
            pair_id = pair_id_a
        else:
            pair_id = f"{pair_id_a}_{pair_id_b}"
        
        return code_a, code_b, label_a, label_b, pair_label, pair_id, pair_type



class SingleFunctionDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512, chunk_stride=480, max_chunks=4, cwe_filter=None):
        """
        Dataset for single-function classification (used in testing or training).
        :param csv_path: path to .csv file
        :param tokenizer: tokenizer (e.g., RobertaTokenizer)
        :param max_length: max token length for CodeBERT
        """
        self.df = pd.read_csv(csv_path)
        self.chunk_stride = chunk_stride
        self.max_chunks = max_chunks

        if cwe_filter is not None and "CWE_type" in self.df.columns:
            allowed = set(cwe_filter) if isinstance(cwe_filter, (list, set)) else {cwe_filter}
            before = len(self.df)
            self.df = self.df[self.df["CWE_type"].isin(allowed)]
            logger.info(f"[Filter] Keep CWE_types={allowed}, rows {before} -> {len(self.df)}")

        self.tokenizer = tokenizer
        self.max_length = max_length

        # Optional columns
        if 'pair_id' not in self.df.columns:
            self.df['pair_id'] = 'none'
        if 'sample_type' not in self.df.columns:
            self.df['sample_type'] = 'unspecified'

        # Build list of (code, label, pair_id, sample_type)
        self.samples = []
        for _, row in self.df.iterrows():
            code = row['Function']
            label = int(row['Vulnerable'])
            pair_id = str(row.get('pair_id', 'none'))
            sample_type = str(row.get('sample_type', 'unspecified'))
            self.samples.append((code, label, pair_id, sample_type))

    def __len__(self):
        return len(self.samples)

    def chunk_code(self, code):
        """
        Split code into overlapping token chunks (max_chunks × max_length).
        """
        tokens = self.tokenizer.tokenize(str(code))
        chunks = []
        start = 0

        while start < len(tokens) and len(chunks) < self.max_chunks:
            end = min(start + self.max_length - 2, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)

            if end == len(tokens):
                break
            start += self.chunk_stride

        return chunks or [""]
    
    def __getitem__(self, idx):
        code, label, pair_id, sample_type = self.samples[idx]

        # === Split code into chunks ===
        chunks = self.chunk_code(code)

        # === Tokenize all chunks ===
        encoded_chunks = self.tokenizer(
            chunks,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # encoded_chunks['input_ids']: (C, L)
        # Pad to fixed number of chunks (max_chunks)
        num_chunks = encoded_chunks["input_ids"].size(0)
        if num_chunks < self.max_chunks:
            pad_size = self.max_chunks - num_chunks
            pad_tensor = torch.ones((pad_size, self.max_length), dtype=torch.long) * self.tokenizer.pad_token_id
            encoded_chunks["input_ids"] = torch.cat([encoded_chunks["input_ids"], pad_tensor], dim=0)
            pad_tensor_mask = torch.zeros((pad_size, self.max_length), dtype=torch.long)
            encoded_chunks["attention_mask"] = torch.cat([encoded_chunks["attention_mask"], pad_tensor_mask], dim=0)

        # Return (C, L) tensors and metadata
        return encoded_chunks, label, pair_id, sample_type