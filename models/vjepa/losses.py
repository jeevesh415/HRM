import torch
import torch.nn.functional as F

def vicreg_loss(x, y, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
    """
    VICReg loss to prevent representation collapse.
    x, y: (bs, D) - latent representations from two views or context/target.
    """
    # Invariance loss (Similiarity)
    repr_loss = F.mse_loss(x, y)

    # Variance loss
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    
    std_x = torch.sqrt(x.var(dim=0) + 1e-04)
    std_y = torch.sqrt(y.var(dim=0) + 1e-04)
    
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

    # Covariance loss
    def covariance_loss(z):
        n, d = z.shape
        cov = (z.T @ z) / (n - 1)
        # Sum of squared non-diagonal elements
        off_diagonal = cov.pow(2).sum() - cov.diag().pow(2).sum()
        return off_diagonal / d

    cov_loss = covariance_loss(x) + covariance_loss(y)

    return sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
