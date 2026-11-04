import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from transformers import AutoModel


class SupConModel(nn.Module):
    def __init__(self, pretrained_model="microsoft/codebert-base", hidden_dim=768, projection_dim=128):
        super(SupConModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.encoder.gradient_checkpointing_enable()
        self.hidden_size = hidden_dim

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, projection_dim)
        )

        self.single_cls_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 2)
        )

        # Pair classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 2)
        )

    # ========= Helper function: Encode with Max-Pooling support =========
    def encode_with_pooling(self, input_ids, attention_mask):
        """
        Handles both:
            - (B, L): normal single-chunk input
            - (B, C, L): multi-chunk input for long code

        Returns:
            Tensor of shape (B, hidden_dim)
        """
        if input_ids is None:
            return None

        if input_ids.dim() == 3:
            B, C, L = input_ids.size()

            # Mark a real chunk: at least one token has an attention_mask of 1.
            chunk_valid = attention_mask.sum(dim=2) > 0  # (B, C)

            flat_input_ids = input_ids.view(B * C, L)
            flat_attention_mask = attention_mask.view(B * C, L)

            outputs = self.encoder(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask
            )
            cls_vec = outputs.last_hidden_state[:, 0, :]  # (B*C, H)
            cls_vec = cls_vec.view(B, C, -1)              # (B, C, H)

            # Invalid padded chunks are masked to prevent them from being selected by max.
            cls_vec = cls_vec.masked_fill(
                ~chunk_valid.unsqueeze(-1),
                torch.finfo(cls_vec.dtype).min
            )

            cls_vec, _ = cls_vec.max(dim=1)

        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_vec = outputs.last_hidden_state[:, 0, :]

        return cls_vec

    # ========= Main forward pass =========
    def forward(self, input_ids_a=None, attention_mask_a=None,
                      input_ids_b=None, attention_mask_b=None,
                      labels=None, mode='supcon'):
        """
        Args:
            input_ids_a, attention_mask_a: tokens for sample A
            input_ids_b, attention_mask_b: tokens for sample B
            labels: optional tensor for classification
            mode: one of ['supcon', 'cls', 'single']

        Returns:
            Depends on mode:
                'supcon' -> (proj_a, proj_b)
                'cls'    -> (loss, logits) or logits
                'single' -> (loss, logits) or logits
        """

        # ----- Encode inputs (with pooling if needed) -----
        cls_a = self.encode_with_pooling(input_ids_a, attention_mask_a)
        cls_b = None
        if input_ids_b is not None:
            cls_b = self.encode_with_pooling(input_ids_b, attention_mask_b)

        # ----- Mode selection -----
        if mode == 'supcon':
            proj_a = F.normalize(self.projection_head(cls_a), dim=1)
            proj_b = F.normalize(self.projection_head(cls_b), dim=1)
            return proj_a, proj_b

        elif mode == 'cls':
            combined = torch.cat([cls_a, cls_b], dim=1)
            logits = self.classifier(combined)
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)
                return loss, logits
            else:
                return logits

        elif mode == 'single':
            logits = self.single_cls_head(cls_a)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, logits
            else:
                return logits