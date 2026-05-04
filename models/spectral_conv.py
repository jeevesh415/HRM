"""
Spectral Graph Convolution for Multi-Scale Visual Reasoning.

Implements graph neural network layers that operate in the spectral domain
(frequency domain) of the graph Laplacian. This provides natural multi-scale
processing: low frequencies capture global structure, high frequencies
capture fine details.

This is analogous to how human vision processes:
  - Low frequency: scene layout, global shape
  - Mid frequency: object boundaries, textures
  - High frequency: edges, fine details

Based on:
  - "Spectral Networks and Locally Connected Networks on Graphs" (Bruna et al., 2014)
  - "Chebyshev Spectral CNN" (Defferrard et al., 2016)
  - "Graph Neural Networks with Spectral Filters" (2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLaplacian(nn.Module):
    """
    Computes the graph Laplacian from a set of node features.

    The Laplacian L = D - A where D is the degree matrix and A is
    the adjacency matrix. The eigenvectors of L form a Fourier basis
    for the graph, enabling spectral analysis.
    """

    def __init__(self, k_neighbors: int = 8):
        super().__init__()
        self.k_neighbors = k_neighbors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the normalized graph Laplacian.

        Args:
            x: (bs, num_nodes, dim) node features.

        Returns:
            L: (bs, num_nodes, num_nodes) normalized Laplacian.
        """
        bs, n, d = x.shape

        # Compute pairwise cosine similarity
        x_norm = F.normalize(x, dim=-1)
        similarity = torch.bmm(x_norm, x_norm.transpose(1, 2))  # (bs, n, n)

        # K-nearest neighbor adjacency
        k = min(self.k_neighbors, n - 1)
        topk_vals, topk_idx = similarity.topk(k, dim=-1)
        A = torch.zeros_like(similarity)
        A.scatter_(-1, topk_idx, torch.ones_like(topk_vals))

        # Symmetrize
        A = (A + A.transpose(1, 2)).clamp(max=1)

        # Degree matrix
        degree = A.sum(dim=-1)  # (bs, n)
        D_inv_sqrt = (1.0 / (degree + 1e-6).sqrt()).diag_embed()  # (bs, n, n)

        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        I = torch.eye(n, device=x.device, dtype=x.dtype).unsqueeze(0).expand(bs, -1, -1)
        L = I - torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)

        return L


class SpectralGraphConv(nn.Module):
    """
    Spectral graph convolution using Chebyshev polynomial approximation.

    Instead of computing eigenvectors (expensive), we approximate the
    spectral filter using Chebyshev polynomials of the Laplacian.
    This gives us multi-scale processing at O(n * k) cost.

    Args:
        dim: feature dimensionality.
        num_filters: number of spectral filters (frequency bands).
        polynomial_order: order of Chebyshev approximation.
    """

    def __init__(
        self,
        dim: int,
        num_filters: int = 4,
        polynomial_order: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.num_filters = num_filters
        self.polynomial_order = polynomial_order

        # Learnable spectral filter coefficients
        # Each filter is a polynomial of the Laplacian
        self.filter_coeffs = nn.Parameter(
            torch.randn(num_filters, polynomial_order + 1) * 0.02
        )

        # Per-filter projection
        self.filter_projs = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(num_filters)
        ])

        # Fusion of multi-scale features
        self.fusion = nn.Sequential(
            nn.Linear(dim * num_filters, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

        # Laplacian computer
        self.laplacian = GraphLaplacian(k_neighbors=8)

    def _chebyshev_polynomials(
        self,
        L: torch.Tensor,
        x: torch.Tensor,
    ) -> list:
        """
        Compute Chebyshev polynomials T_k(L) applied to x.

        T_0(L) x = x
        T_1(L) x = L x
        T_k(L) x = 2 L T_{k-1}(L) x - T_{k-2}(L) x

        Args:
            L: (bs, n, n) normalized Laplacian.
            x: (bs, n, dim) node features.

        Returns:
            List of T_k(L) x for k = 0, ..., polynomial_order.
        """
        # Scale L to [-1, 1] for Chebyshev stability
        # L_norm has eigenvalues in [0, 2], so L_scaled = L - I has eigenvalues in [-1, 1]
        bs, n, _ = L.shape
        I = torch.eye(n, device=L.device, dtype=L.dtype).unsqueeze(0).expand(bs, -1, -1)
        L_scaled = L - I

        polynomials = [x]  # T_0(L) x = x

        if self.polynomial_order >= 1:
            # T_1(L) x = L_scaled @ x
            T1 = torch.bmm(L_scaled, x)
            polynomials.append(T1)

        for k in range(2, self.polynomial_order + 1):
            # T_k(L) x = 2 L_scaled T_{k-1}(L) x - T_{k-2}(L) x
            Tk = 2 * torch.bmm(L_scaled, polynomials[-1]) - polynomials[-2]
            polynomials.append(Tk)

        return polynomials

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale spectral graph convolution.

        Args:
            x: (bs, num_nodes, dim) node features.

        Returns:
            (bs, num_nodes, dim) multi-scale features.
        """
        bs, n, d = x.shape

        # Compute graph Laplacian
        L = self.laplacian(x)

        # Compute Chebyshev polynomials (multi-scale)
        cheb_polys = self._chebyshev_polynomials(L, x)

        # Apply each spectral filter
        filter_outputs = []
        for f_idx in range(self.num_filters):
            # Weighted combination of polynomial terms
            filtered = sum(
                self.filter_coeffs[f_idx, k] * cheb_polys[k]
                for k in range(self.polynomial_order + 1)
            )
            # Project
            filtered = self.filter_projs[f_idx](filtered)
            filter_outputs.append(filtered)

        # Fuse multi-scale features
        multi_scale = torch.cat(filter_outputs, dim=-1)  # (bs, n, dim * num_filters)
        output = self.fusion(multi_scale)  # (bs, n, dim)

        return output
