def dkt_loss(logits, target_correct, target_ids, seq_steps):
    """
    Calculate cross-entropy loss for DKT (Deep Knowledge Tracing) in PyTorch.

    Args:
        logits (Tensor): Logits (no sigmoid applied) with shape [batch_size, length, vocab_size]
        target_correct (Tensor): Targets with shape [batch_size, length]
        target_ids (Tensor): Target IDs with shape [batch_size, length]
        seq_steps (Tensor): Sequence lengths with shape [batch_size]

    Returns:
        Tensor: Cross-entropy loss
    """
    batch_size, length, vocab_size = logits.shape
    flat_logits = logits.view(-1, vocab_size)
    flat_target_correct = target_correct.view(-1)

    flat_base_target_index = torch.arange(batch_size * length, device=logits.device) * vocab_size
    flat_bias_target_id = target_ids.view(-1)
    flat_target_id = flat_base_target_index + flat_bias_target_id

    flat_target_logits = flat_logits[torch.arange(flat_logits.size(0), device=logits.device), flat_target_id]
    mask = torch.arange(length, device=logits.device).expand(batch_size, length) < seq_steps.unsqueeze(1)
    mask = mask.view(-1)

    flat_target_correct_mask = flat_target_correct[mask]
    flat_target_logits_mask = flat_target_logits[mask]

    # PyTorch's cross-entropy loss expects the targets as class indices, not binary (0 or 1)
    # Therefore, we need to use the BCE loss function and manually handle the pos_weight
    loss = F.binary_cross_entropy_with_logits(
        flat_target_logits_mask,
        flat_target_correct_mask.float(),
        pos_weight=torch.tensor([1.0], device=logits.device)  # PyTorch expects pos_weight to be a tensor
    )

    return loss