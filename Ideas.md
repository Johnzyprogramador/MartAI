**CI Pipeline**

- Do unit testing 


def vae_loss(outputs, targets, loss_factor=1.0, beta=1.0):
    reconstruction, mean, logvar = outputs  # unpack
    total_losses = []

    # --- Numéricas ---
    num_preds = reconstruction.get("numericals")
    num_targets = targets.get("numericals")
    if num_preds is not None and num_targets is not None and len(num_preds) > 0:
        # média por dimensão → [B]
        num_loss = F.mse_loss(num_preds, num_targets, reduction="none").mean(dim=1)
        total_losses.append(num_loss)

    # --- Categóricas ---
    cat_preds = reconstruction.get("categoricals")
    cat_targets = targets.get("categoricals")
    if cat_preds is not None and cat_targets is not None and len(cat_preds) > 0:
        cat_losses = []
        for i, feature_preds in enumerate(cat_preds):
            feature_targets = cat_targets[..., i].long()
            # CE já dá loss por sample, média mantém escala estável
            loss = F.cross_entropy(feature_preds, feature_targets, reduction="none")
            cat_losses.append(loss)
        cat_loss = torch.stack(cat_losses, dim=1).mean(dim=1)  # média entre features
        total_losses.append(cat_loss)

    if not total_losses:
        raise ValueError("No valid features found for loss computation.")

    # Recon final: média das componentes → [B]
    all_losses = torch.stack(total_losses, dim=1).mean(dim=1)

    # --- KL: média por dimensão ---
    KL_divergence = -0.5 * torch.mean(
        1 + logvar - mean.pow(2) - logvar.exp(), dim=1
    )  # [B]

    # Total
    total_loss = all_losses * loss_factor + KL_divergence * beta
    
    return total_loss, KL_divergence