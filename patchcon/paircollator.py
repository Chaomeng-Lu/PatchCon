import torch
import re

class PairCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        """
        batch: list of tuples (code_a, code_b, label_a, label_b, pair_label, pair_id, pair_type)

        Returns:
            encoded_a: dict of tensors (input_ids, attention_mask, ...)
            encoded_b: dict of tensors (input_ids, attention_mask, ...)
            label_a: tensor (B,)
            label_b: tensor (B,)
            pair_label: tensor (B,)
            pair_ids: tensor (B,)
            pair_types: list of strings (B,)
        """
        codes_a, codes_b, labels_a, labels_b, pair_labels, pair_ids, pair_types = zip(*batch)

        encoded_a = self.tokenizer(
            list(codes_a),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        encoded_b = self.tokenizer(
            list(codes_b),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        label_a_tensor = torch.tensor(labels_a, dtype=torch.long)
        label_b_tensor = torch.tensor(labels_b, dtype=torch.long)
        pair_label_tensor = torch.tensor(pair_labels, dtype=torch.long)

        def parse_pair_id(pid):
            if isinstance(pid, str) and re.fullmatch(r"\d+", pid):
                return int(pid)
            return -1

        pair_ids_tensor = torch.tensor([parse_pair_id(pid) for pid in pair_ids], dtype=torch.long)

        return encoded_a, encoded_b, label_a_tensor, label_b_tensor, pair_label_tensor, pair_ids_tensor, list(pair_types)

class PairCollatorPool:
    def __init__(self, tokenizer, max_length=512, chunk_stride=480):
        """
        Args:
            tokenizer: CodeBERT/RobertaTokenizer
            max_length: Maximum token length in each chunk
            chunk_stride: Chunk stride (< max_length, overlap is allowed)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_stride = chunk_stride

    def chunk_code(self, code):
        """Divide a single code string into chunks by token."""
        tokens = self.tokenizer.tokenize(code)
        chunks = []
        start = 0
        max_chunks = 4  # limit to avoid GPU memory overflow

        while start < len(tokens) and len(chunks) < max_chunks:
        # while start < len(tokens):
            end = min(start + self.max_length - 2, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
            start += self.chunk_stride
        return chunks or [""]

    def encode_with_chunks(self, codes):
        """
        Encode and align multiple code samples in blocks.
        Return:
            input_ids: (B, num_chunks, max_length)
            attention_mask: (B, num_chunks, max_length)
            chunk_counts: The number of blocks per sample, used for subsequent max-pooling.
        """
        all_input_ids, all_masks, chunk_counts = [], [], []

        for code in codes:
            chunks = self.chunk_code(code)
            encoded = self.tokenizer(
                chunks,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            all_input_ids.append(encoded["input_ids"])       # (num_chunks, L)
            all_masks.append(encoded["attention_mask"])
            chunk_counts.append(encoded["input_ids"].size(0))

        # Align to the same number of chunks
        max_chunks = max(chunk_counts)
        pad_id = self.tokenizer.pad_token_id
        for i in range(len(all_input_ids)):
            num_chunks = all_input_ids[i].size(0)
            if num_chunks < max_chunks:
                pad_shape = (max_chunks - num_chunks, self.max_length)
                all_input_ids[i] = torch.cat(
                    [all_input_ids[i], torch.full(pad_shape, pad_id, dtype=torch.long)], dim=0)
                all_masks[i] = torch.cat(
                    [all_masks[i], torch.zeros(pad_shape, dtype=torch.long)], dim=0)

        return {
            "input_ids": torch.stack(all_input_ids),         # (B, C, L)
            "attention_mask": torch.stack(all_masks),        # (B, C, L)
            "chunk_counts": torch.tensor(chunk_counts)       # (B,)
        }

    def __call__(self, batch):
        codes_a, codes_b, labels_a, labels_b, pair_labels, pair_ids, pair_types = zip(*batch)

        encoded_a = self.encode_with_chunks(list(codes_a))
        encoded_b = self.encode_with_chunks(list(codes_b))

        label_a_tensor = torch.tensor(labels_a, dtype=torch.long)
        label_b_tensor = torch.tensor(labels_b, dtype=torch.long)
        pair_label_tensor = torch.tensor(pair_labels, dtype=torch.long)

        def parse_pair_id(pid):
            if isinstance(pid, str) and re.fullmatch(r"\d+", pid):
                return int(pid)
            return -1

        pair_ids_tensor = torch.tensor([parse_pair_id(pid) for pid in pair_ids], dtype=torch.long)

        return encoded_a, encoded_b, label_a_tensor, label_b_tensor, pair_label_tensor, pair_ids_tensor, list(pair_types)