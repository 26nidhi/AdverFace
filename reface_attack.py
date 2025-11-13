import torch
import torch.nn.functional as F

def reface_attack(model, source_tensor, target_emb, num_steps=100, lr=0.1, l2_weight=0.001, device=None):
    """
    Optimize the input image to increase cosine similarity to target_emb.
    source_tensor: [1,3,160,160] with values in [-1,1]
    target_emb: [1,512] or [512] embedding (will be reshaped)
    Returns perturbed tensor (detached).
    """
    device = device or source_tensor.device
    source_tensor = source_tensor.to(device)

    # Ensure target_emb is shape [1, D]
    if target_emb.dim() == 1:
        target_emb = target_emb.unsqueeze(0)
    target_emb = target_emb.to(device).detach()

    perturbed = source_tensor.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([perturbed], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()

        emb = model(perturbed)  # [1, D]
        cos_sim = F.cosine_similarity(emb, target_emb, dim=1)  # [1]
        loss = -cos_sim.mean()

        # use reshape instead of view to avoid non-contiguous issues
        perturb = (perturbed - source_tensor).reshape(perturbed.size(0), -1)
        l2_loss = torch.norm(perturb, p=2, dim=1).mean()
        loss = loss + l2_weight * l2_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            perturbed.clamp_(-1.0, 1.0)

        if step % 20 == 0 or step == num_steps - 1:
            print(f"Step {step}, Loss: {loss.item():.6f}, Cosine Sim: {cos_sim.item():.6f}, L2 mean: {l2_loss.item():.6f}")

        if cos_sim.item() > 0.9:
            print("Success: attack reached high similarity.")
            break

    return perturbed.detach()