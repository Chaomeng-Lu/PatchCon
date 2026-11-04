import torch
import torch.nn as nn
import torch.nn.functional as F


# Mapping from pair_type to weight for contrastive learning.
#
# same_safe / same_vuln:
#   positive pairs, should be pulled closer.
#
# random_neg:
#   ordinary negative pair, should be pushed apart.
#
# repair:
#   hard negative pair, usually highly similar before/after fix,
#   but semantically/security-wise different, so it receives larger weight.
SUPCON_PAIR_WEIGHTS = {
    "same_safe": 1.0,
    "same_vuln": 1.0,
    "random_neg": 1.0,
    "repair": 2.0,
}


class PairSupConLoss(nn.Module):
    """
    Pair-aware supervised contrastive loss.

    Expected feature layout:
        features = [a0, b0, a1, b1, ..., a_{B-1}, b_{B-1}]
        shape: [2B, D]

    Expected pair_types:
        Either length 2B:
            [p0, p0, p1, p1, ..., p_{B-1}, p_{B-1}]
        or length B:
            [p0, p1, ..., p_{B-1}]

    Behavior:
        - same_safe / same_vuln:
            treated as positive pairs in supervised contrastive loss.
        - repair / random_neg:
            treated as explicit pair-level negatives.
            Their cosine similarity is penalized directly.
        - repair receives larger weight via SUPCON_PAIR_WEIGHTS.
    """

    def __init__(
        self,
        temperature=0.07,
        hard_neg_margin=0.0,
        eps=1e-12,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.temperature = float(temperature)
        self.hard_neg_margin = float(hard_neg_margin)
        self.eps = float(eps)

    @staticmethod
    def _normalize_pair_types(pair_types, n_features):
        """
        Convert pair_types to one type per pair.

        Supports:
            len(pair_types) == n_features:
                [p0, p0, p1, p1, ...]
            len(pair_types) == n_features // 2:
                [p0, p1, ...]
        """
        if pair_types is None:
            raise ValueError("pair_types must not be None")

        pair_types = list(pair_types)

        if n_features % 2 != 0:
            raise ValueError(
                f"features must be interleaved pairs and have even length, "
                f"got n_features={n_features}"
            )

        n_pairs = n_features // 2

        if len(pair_types) == n_features:
            # Current trainer.py passes duplicated/interleaved pair types:
            # [p0, p0, p1, p1, ...]
            return [pair_types[2 * k] for k in range(n_pairs)]

        if len(pair_types) == n_pairs:
            # Also support one pair type per pair:
            # [p0, p1, ...]
            return pair_types

        raise ValueError(
            f"pair_types length must be either n_pairs={n_pairs} "
            f"or n_features={n_features}, got {len(pair_types)}"
        )

    def forward(self, features, pair_types):
        """
        Args:
            features:
                Tensor of shape [2B, D], interleaved as
                [a0, b0, a1, b1, ...].

            pair_types:
                List[str] of length 2B or B.
                Supported values:
                    same_safe, same_vuln, random_neg, repair

        Returns:
            Scalar loss.
        """
        if features.dim() != 2:
            raise ValueError(
                f"features must be a 2D tensor [2B, D], "
                f"got shape={tuple(features.shape)}"
            )

        device = features.device
        dtype = features.dtype
        n_features = features.size(0)

        pair_types_per_pair = self._normalize_pair_types(
            pair_types=pair_types,
            n_features=n_features,
        )

        # Use a dtype-safe epsilon for mixed precision.
        if dtype.is_floating_point:
            dtype_eps = torch.finfo(dtype).tiny
            eps = max(self.eps, dtype_eps)
        else:
            eps = self.eps

        # Normalize embeddings before cosine similarity.
        z = F.normalize(features, dim=1)

        # Cosine similarity in [-1, 1].
        cosine = torch.matmul(z, z.T)

        # Temperature-scaled logits for supervised contrastive positive loss.
        logits = cosine / self.temperature

        # Numerical stability for exp/logsumexp style computation.
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # Exclude self-comparison.
        logits_mask = torch.ones((n_features, n_features), device=device, dtype=dtype)
        logits_mask.fill_diagonal_(0.0)

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(eps)
        )

        # Positive mask for same-label pairs.
        pos_mask = torch.zeros((n_features, n_features), device=device, dtype=dtype)

        # Anchor weights for positive SupCon part.
        sample_weights = torch.ones(n_features, device=device, dtype=dtype)

        # Explicit hard/ordinary negative pair losses.
        hard_neg_losses = []

        for k, pair_type in enumerate(pair_types_per_pair):
            i = 2 * k
            j = i + 1

            weight = float(SUPCON_PAIR_WEIGHTS.get(pair_type, 1.0))
            weight_tensor = torch.as_tensor(weight, device=device, dtype=dtype)

            sample_weights[i] = weight
            sample_weights[j] = weight

            if pair_type in ("same_safe", "same_vuln"):
                # Positive pair: pull together.
                pos_mask[i, j] = 1.0
                pos_mask[j, i] = 1.0

            elif pair_type in ("repair", "random_neg"):
                # Negative pair: push apart.
                #
                # Penalize high cosine similarity.
                # If cosine is high, softplus((cos - margin) / T) is large.
                # If cosine is low/negative, the penalty becomes small.
                #
                # repair receives larger weight, e.g. 3.0.
                pair_cosine = cosine[i, j]
                neg_loss = F.softplus(
                    (pair_cosine - self.hard_neg_margin) / self.temperature
                )
                hard_neg_losses.append(weight_tensor * neg_loss)

            else:
                # Unknown pair types are ignored as explicit pair constraints,
                # but they still remain in the batch denominator.
                continue

        # ----------------------------
        # 1. Positive SupCon component
        # ----------------------------
        pos_count = pos_mask.sum(dim=1)
        has_pos = pos_count > 0

        if has_pos.any():
            mean_log_prob_pos = (
                (pos_mask * log_prob).sum(dim=1)[has_pos]
                / pos_count[has_pos]
            )

            loss_pos = -(
                sample_weights[has_pos] * mean_log_prob_pos
            ).mean()
        else:
            # Keep graph connected.
            loss_pos = features.sum() * 0.0

        # ----------------------------------
        # 2. Explicit negative pair component
        # ----------------------------------
        if len(hard_neg_losses) > 0:
            loss_neg = torch.stack(hard_neg_losses).mean()
        else:
            # Keep graph connected.
            loss_neg = features.sum() * 0.0

        # Final pair-aware supervised contrastive loss.
        loss = loss_pos + loss_neg

        return loss