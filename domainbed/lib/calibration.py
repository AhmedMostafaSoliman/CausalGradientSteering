import torch
import torch.nn.functional as F


def _ece_from_confidence_and_correct(confidence: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) given confidences and correctness.

    Args:
        confidence: Tensor of shape [N] with values in [0, 1].
        correct: Bool or 0/1 Tensor of shape [N].
        n_bins: Number of equal-width bins in [0,1].

    Returns:
        ECE as a Python float.
    """
    assert confidence.dim() == 1 and confidence.size(0) == correct.size(0)
    n = confidence.size(0)
    if n == 0:
        return 0.0

    # Clamp to [0,1] to be safe against small numerical issues
    conf = confidence.clamp(0.0, 1.0)
    corr = correct.float()

    # Bin edges: [0, 1], equal width
    bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1, device=conf.device)
    ece = torch.tensor(0.0, device=conf.device)

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        # Include left, exclude right except for the last bin which includes 1.0
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.any():
            conf_b = conf[mask].mean()
            acc_b = corr[mask].mean()
            w_b = mask.float().mean()  # |b| / N
            ece = ece + w_b * (acc_b - conf_b).abs()

    return float(ece.item())


def compute_ece_from_logits(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Compute classic ECE using top-1 confidence from logits.

    Handles binary (C=1) and multiclass (C>1) logits.

    Args:
        logits: Tensor of shape [N, C] (C==1 allowed for binary).
        labels: Tensor of shape [N] with class indices {0..C-1}.
        n_bins: Number of bins for calibration.

    Returns:
        ECE as a Python float.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be [N, C]")
    if labels.ndim != 1 or labels.size(0) != logits.size(0):
        raise ValueError("labels must be [N] matching logits.shape[0]")

    n, c = logits.shape
    labels = labels.to(logits.device)

    if c == 1:
        # Binary: sigmoid -> prob of class 1; top-1 confidence is max(p, 1-p)
        probs1 = torch.sigmoid(logits.view(-1))
        conf = torch.maximum(probs1, 1.0 - probs1)
        preds = (probs1 >= 0.5).long()
    else:
        probs = F.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)

    correct = (preds == labels).to(torch.float32)
    return _ece_from_confidence_and_correct(conf.detach(), correct.detach(), n_bins=n_bins)


def maxprob_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample max class probability from logits.

    Supports binary (C=1) and multiclass (C>1).
    Returns a Tensor of shape [N] with values in [0,1].
    """
    if logits.ndim != 2:
        raise ValueError("logits must be [N, C]")
    n, c = logits.shape
    if c == 1:
        p1 = torch.sigmoid(logits.view(-1))
        return torch.maximum(p1, 1.0 - p1)
    else:
        probs = F.softmax(logits, dim=1)
        return probs.max(dim=1).values


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample entropy H(p) where p = softmax(logits) for C>1 or sigmoid for C=1.

    Returns Tensor [N] with entropy in nats.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be [N, C]")
    n, c = logits.shape
    if c == 1:
        p1 = torch.sigmoid(logits.view(-1))
        p0 = 1.0 - p1
        # Clamp to avoid log(0)
        p1c = p1.clamp_min(1e-12)
        p0c = p0.clamp_min(1e-12)
        return -(p1c.log() * p1 + p0c.log() * p0)
    else:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=1)


def kl_to_uniform_from_logits(logits: torch.Tensor, num_classes: int = None) -> torch.Tensor:
    """Per-sample KL divergence KL(p || u), where u is uniform over classes.

    For C>1: KL = log(C) - H(p). For C<=1: returns zeros.
    Returns Tensor [N].
    """
    if logits.ndim != 2:
        raise ValueError("logits must be [N, C]")
    n, c = logits.shape
    c_eff = c if num_classes is None else int(num_classes)
    if c_eff <= 1:
        return torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)
    ent = entropy_from_logits(logits)
    return torch.log(torch.tensor(float(c_eff), device=logits.device)) - ent
