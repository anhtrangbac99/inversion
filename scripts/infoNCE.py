import math
import torch
import torch.nn.functional as F

def info_nce(
    x, y,                 # [1,D] or [D]
    y_neg,                # [K,D] negatives for Y (X->Y direction)
    x_neg=None,           # [K,D] negatives for X (optional; enables symmetric Y->X)
    tau=0.07,
    normalize=True,
    return_bits=True,
):
    """
    Computes InfoNCE for a single positive pair (x,y) with explicit negatives.
    Returns: loss (minimize), mi_nats (lower bound), mi_bits
    """
    # --- shape sanitation ---
    if x.dim() == 1: x = x.unsqueeze(0)  # -> [1,D]
    if y.dim() == 1: y = y.unsqueeze(0)  # -> [1,D]
    assert x.shape == y.shape and x.size(0) == 1, "x and y must be [1,D]"
    assert y_neg.dim() == 2 and y_neg.size(1) == x.size(1), "y_neg must be [K,D]"
    if x_neg is not None:
        assert x_neg.dim() == 2 and x_neg.size(1) == x.size(1), "x_neg must be [K,D]"

    # --- optional cosine normalization ---
    if normalize:
        x  = F.normalize(x,  dim=1)
        y  = F.normalize(y,  dim=1)
        y_neg = F.normalize(y_neg, dim=1)
        if x_neg is not None:
            x_neg = F.normalize(x_neg, dim=1)

    # ---------- X -> Y ----------
    cand_y = torch.cat([y, y_neg], dim=0)       # [1+K, D], positive at index 0
    logits_xy = (x @ cand_y.t()) / tau          # [1, 1+K]
    ce_xy = F.cross_entropy(logits_xy, torch.tensor([0], device=x.device))
    B_xy = cand_y.size(0)                        # = 1 + K
    mi_xy = math.log(B_xy) - ce_xy               # nats
    loss_xy = ce_xy - math.log(B_xy)

    # ---------- (optional) Y -> X ----------
    if x_neg is None:
        mi_nats = mi_xy
        loss = loss_xy
    else:
        cand_x = torch.cat([x, x_neg], dim=0)   # [1+K, D], positive at index 0
        logits_yx = (y @ cand_x.t()) / tau      # [1, 1+K]
        ce_yx = F.cross_entropy(logits_yx, torch.tensor([0], device=y.device))
        B_yx = cand_x.size(0)
        mi_yx = math.log(B_yx) - ce_yx          # nats
        loss_yx = ce_yx - math.log(B_yx)

        mi_nats = 0.5 * (mi_xy + mi_yx)
        loss = 0.5 * (loss_xy + loss_yx)

    mi_bits = mi_nats / math.log(2) if return_bits else None
    return loss, mi_nats, mi_bits
