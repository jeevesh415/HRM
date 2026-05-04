import torch

def get_block_mask(t, h, w, mask_ratio=0.6):
    """
    Generate a 3D block mask for (T, H, W) patch grid.
    Returns a boolean mask of shape (t*h*w,) where True means masked.
    """
    total_patches = t * h * w
    num_masked = int(total_patches * mask_ratio)
    
    # Randomly select a starting point and block size
    # This is a simplified block mask. Official V-JEPA uses multi-block.
    mask = torch.zeros(t, h, w, dtype=torch.bool)
    
    # Simple random masking for now
    indices = torch.randperm(total_patches)[:num_masked]
    mask.view(-1)[indices] = True
    
    return mask.view(-1)

def apply_mask(x, mask):
    """
    x: (bs, seq_len, dim)
    mask: (seq_len,) or (bs, seq_len)
    Returns: (bs, num_visible, dim), (bs, num_masked, dim)
    """
    bs, seq_len, dim = x.shape
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(bs, -1)
        
    if mask.ndim == 2:
        visible_counts = (~mask).sum(dim=1)
        masked_counts = mask.sum(dim=1)
        if not torch.all(visible_counts.eq(visible_counts[0])) or not torch.all(masked_counts.eq(masked_counts[0])):
            raise ValueError("All samples must have the same number of visible/masked patches for batched stacking.")

    visible_patches = []
    masked_patches = []

    for i in range(bs):
        visible_patches.append(x[i, ~mask[i]])
        masked_patches.append(x[i, mask[i]])

    return torch.stack(visible_patches), torch.stack(masked_patches)
