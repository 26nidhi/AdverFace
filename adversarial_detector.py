

import torch

def detect_adversarial(original_tensor, perturbed_tensor, threshold=0.1, return_l2=False):
    """
    Detect adversarial perturbation using per-sample L2 norm.
    original_tensor and perturbed_tensor expected shape: [B, C, H, W] (B can be 1).
    Returns boolean (True if any sample exceeds threshold). If return_l2=True, also returns tensor of L2 norms.
    """
    if original_tensor.shape != perturbed_tensor.shape:
        raise ValueError("original_tensor and perturbed_tensor must have the same shape")

    # use reshape to avoid contiguous issues
    perturb = (perturbed_tensor - original_tensor).reshape(original_tensor.size(0), -1)
    l2_norms = torch.norm(perturb, p=2, dim=1)  # per-sample L2
    flagged = (l2_norms > threshold)
    if return_l2:
        return flagged.any().item(), l2_norms
    return flagged.any().item()


























