import torch
import torch.nn as nn

def transfer_sequential_weights(source_seq, target_seq):
    """Transfer weights between two nn.Sequential modules, layer by layer, where possible."""
    for src, tgt in zip(source_seq, target_seq):
        if isinstance(src, nn.Conv2d) and isinstance(tgt, nn.Conv2d):
            if src.weight.shape == tgt.weight.shape:
                tgt.weight.data.copy_(src.weight.data)
                if src.bias is not None and tgt.bias is not None:
                    tgt.bias.data.copy_(src.bias.data)
        elif isinstance(src, nn.ConvTranspose2d) and isinstance(tgt, nn.ConvTranspose2d):
            if src.weight.shape == tgt.weight.shape:
                tgt.weight.data.copy_(src.weight.data)
                if src.bias is not None and tgt.bias is not None:
                    tgt.bias.data.copy_(src.bias.data)
        elif isinstance(src, nn.BatchNorm2d) and isinstance(tgt, nn.BatchNorm2d):
            if src.weight.shape == tgt.weight.shape:
                tgt.weight.data.copy_(src.weight.data)
                tgt.bias.data.copy_(src.bias.data)
                tgt.running_mean.data.copy_(src.running_mean.data)
                tgt.running_var.data.copy_(src.running_var.data)

def transfer_gan_weights(old_model, new_model):
    """Transfer as many weights as possible from old_model to new_model (Generator or Discriminator)."""
    import torch.nn as nn
    # Generator
    if hasattr(old_model, 'initial') and hasattr(new_model, 'initial'):
        transfer_sequential_weights(old_model.initial, new_model.initial)
    if hasattr(old_model, 'upsample_blocks') and hasattr(new_model, 'upsample_blocks'):
        for old_block, new_block in zip(old_model.upsample_blocks, new_model.upsample_blocks):
            transfer_sequential_weights(old_block, new_block)
    if hasattr(old_model, 'penultimate') and hasattr(new_model, 'penultimate'):
        transfer_sequential_weights(old_model.penultimate, new_model.penultimate)
    if hasattr(old_model, 'final') and hasattr(new_model, 'final'):
        transfer_sequential_weights(old_model.final, new_model.final)
    # Discriminator
    if hasattr(old_model, 'initial') and hasattr(new_model, 'initial'):
        transfer_sequential_weights(old_model.initial, new_model.initial)
    if hasattr(old_model, 'downsample_blocks') and hasattr(new_model, 'downsample_blocks'):
        for old_block, new_block in zip(old_model.downsample_blocks, new_model.downsample_blocks):
            transfer_sequential_weights(old_block, new_block)
    if hasattr(old_model, 'final') and hasattr(new_model, 'final'):
        transfer_sequential_weights(old_model.final, new_model.final)
    # Optionally print a summary
    print("[WEIGHT TRANSFER] Weights transferred for all matching blocks.")

def get_best_practice_scheduler(optimizer):
    """
    Returns a ReduceLROnPlateau scheduler with best-practice settings for GANs.
    Patience is set to 7, min_lr to 1e-5.
    """
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    return ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=7,  # More stable, less aggressive
        verbose=True,
        min_lr=1e-5,  # Don't go too low
        threshold=0.0,
        cooldown=0
    )
