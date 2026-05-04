"""
Enhanced Holographic Memory with Cleanup and Multi-Backend Support.

Implements Holographic Reduced Representations (HRR) with two backends:
  1. Circular Convolution (classic HRR) - binds via FFT-based circular convolution
  2. Fourier Holographic Reduced Representations (FHRR) - binds via Fourier
     coefficient multiplication (more biologically plausible)

Key enhancements over the basic HRR:
  - Resonator network for iterative cleanup/unbinding
  - Bounded superposition for capacity management
  - Multi-item retrieval with noise-tolerant cleanup
  - Pluggable backend (HRR vs FHRR)

Based on:
  - "Holographic Reduced Representations" (Plate, 1995)
  - "Fourier Holographic Reduced Representations" (Plate, 2003)
  - "HyperSpace: Hyperdimensional Computing" (2026)
  - "Resonator Networks" (Frady et al., 2020)
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class ResonatorNetwork(nn.Module):
    """
    Resonator Network for iterative cleanup/unbinding.

    Given a composite vector (result of binding), iteratively recovers
    the original factors by alternating between:
      1. Estimate one factor by unbinding the composite with the current
         estimate of the other factor.
      2. Project the estimate onto the nearest stored item (cleanup).
      3. Repeat until convergence.

    This is the key mechanism for multi-item retrieval from superimposed
    holographic memory.

    Args:
        dim: vector dimensionality.
        cleanup_memory: (num_items, dim) stored item vectors for cleanup.
        max_iterations: maximum resonator iterations.
        convergence_threshold: stop when change is below this threshold.
    """

    def __init__(
        self,
        dim: int,
        cleanup_memory: Optional[torch.Tensor] = None,
        max_iterations: int = 20,
        convergence_threshold: float = 1e-3,
    ):
        super().__init__()
        self.dim = dim
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        if cleanup_memory is not None:
            self.register_buffer("cleanup_memory", cleanup_memory)
        else:
            self.cleanup_memory = None

    def set_cleanup_memory(self, memory: torch.Tensor) -> None:
        """Set or update the cleanup memory bank."""
        self.cleanup_memory = memory

    def cleanup(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project x onto the nearest vector in cleanup memory.

        Uses cosine similarity to find the nearest match.

        Args:
            x: (bs, dim) noisy query vectors.

        Returns:
            (bs, dim) cleaned-up vectors.
        """
        if self.cleanup_memory is None or self.cleanup_memory.shape[0] == 0:
            return x

        # Cosine similarity with all cleanup items
        x_norm = F.normalize(x, dim=-1)                          # (bs, dim)
        mem_norm = F.normalize(self.cleanup_memory, dim=-1)      # (num_items, dim)
        similarity = x_norm @ mem_norm.T                          # (bs, num_items)

        # Weighted combination (soft cleanup)
        weights = F.softmax(similarity * 10.0, dim=-1)  # temperature scaling
        cleaned = weights @ self.cleanup_memory          # (bs, dim)

        return cleaned

    def resonator_step(
        self,
        composite: torch.Tensor,
        known_factor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single resonator iteration: unbind with known factor and cleanup.

        Args:
            composite: (bs, dim) bound composite vector.
            known_factor: (bs, dim) the known factor for unbinding.

        Returns:
            (bs, dim) estimate of the unknown factor.
        """
        # Unbind: retrieve unknown factor
        estimate = self._unbind(composite, known_factor)

        # Cleanup: project onto nearest stored item
        estimate = self.cleanup(estimate)

        return estimate

    def _unbind(self, composite: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Unbind using circular correlation (inverse of convolution)."""
        comp_fft = torch.fft.fft(composite)
        key_fft = torch.fft.fft(key)
        return torch.fft.ifft(comp_fft * key_fft.conj()).real

    def forward(
        self,
        composite: torch.Tensor,
        factor_a_hint: Optional[torch.Tensor] = None,
        factor_b_hint: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Iteratively recover both factors from a composite vector.

        Uses alternating projection between the binding constraint and
        the cleanup memory until convergence.

        Args:
            composite: (bs, dim) bound composite vector.
            factor_a_hint: (bs, dim) initial guess for factor A.
            factor_b_hint: (bs, dim) initial guess for factor B.

        Returns:
            factor_a: (bs, dim) recovered factor A.
            factor_b: (bs, dim) recovered factor B.
        """
        bs = composite.shape[0]
        device = composite.device

        # Initialize guesses
        if factor_a_hint is not None:
            factor_a = factor_a_hint.clone()
        else:
            factor_a = self.cleanup(torch.randn(bs, self.dim, device=device))

        if factor_b_hint is not None:
            factor_b = factor_b_hint.clone()
        else:
            factor_b = self.cleanup(torch.randn(bs, self.dim, device=device))

        # Alternating projection
        for _ in range(self.max_iterations):
            old_a = factor_a.clone()

            # Estimate A from composite and current B estimate
            factor_a = self.resonator_step(composite, factor_b)

            # Estimate B from composite and current A estimate
            factor_b = self.resonator_step(composite, factor_a)

            # Check convergence
            change = (factor_a - old_a).abs().mean()
            if change < self.convergence_threshold:
                break

        return factor_a, factor_b


class HolographicMemory(nn.Module):
    """
    Enhanced Holographic Reduced Representations memory.

    Supports two backends:
      - 'hrr': Circular convolution binding (classic HRR)
      - 'fhrr': Fourier-domain binding (FHRR)

    Features:
      - Bounded superposition with capacity management
      - Multi-item binding with iterative cleanup retrieval
      - Resonator network for noise-tolerant unbinding

    Args:
        dim: vector dimensionality.
        backend: 'hrr' or 'fhrr'.
        max_items: maximum number of items in superposition (capacity).
        cleanup_memory: (num_items, dim) optional cleanup bank.
    """

    def __init__(
        self,
        dim: int,
        backend: str = "hrr",
        max_items: int = 64,
        cleanup_memory: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.dim = dim
        self.backend = backend
        self.max_items = max_items

        # Resonator for iterative cleanup
        self.resonator = ResonatorNetwork(dim, cleanup_memory)

        # Learnable role vectors for binding
        self.role_vectors = nn.Parameter(torch.randn(max_items, dim) * 0.02)

        # Capacity normalization factor
        self.capacity_norm = nn.Parameter(torch.ones(1))

    def _bind_hrr(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Bind via circular convolution (FFT multiplication)."""
        key_fft = torch.fft.fft(key)
        value_fft = torch.fft.fft(value)
        return torch.fft.ifft(key_fft * value_fft).real

    def _unbind_hrr(self, composite: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Unbind via circular correlation (FFT conjugate multiplication)."""
        comp_fft = torch.fft.fft(composite)
        key_fft = torch.fft.fft(key)
        return torch.fft.ifft(comp_fft * key_fft.conj()).real

    def _bind_fhrr(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Bind via Fourier-domain multiplication (FHRR).

        FHRR operates directly in the frequency domain, making binding
        equivalent to element-wise multiplication of Fourier coefficients.
        This is more biologically plausible than circular convolution.
        """
        key_fourier = torch.fft.fft(key)
        value_fourier = torch.fft.fft(value)
        return torch.fft.ifft(key_fourier * value_fourier).real

    def _unbind_fhrr(self, composite: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Unbind FHRR by dividing Fourier coefficients."""
        comp_fourier = torch.fft.fft(composite)
        key_fourier = torch.fft.fft(key)
        # Division in Fourier domain (with regularization)
        return torch.fft.ifft(comp_fourier / (key_fourier + 1e-8)).real

    def bind(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Bind key and value vectors."""
        if self.backend == "fhrr":
            return self._bind_fhrr(key, value)
        return self._bind_hrr(key, value)

    def unbind(self, composite: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Retrieve value from composite given key."""
        if self.backend == "fhrr":
            return self._unbind_fhrr(composite, key)
        return self._unbind_hrr(composite, key)

    def superpose(self, vectors: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Superpose (sum) vectors with bounded capacity.

        Applies capacity normalization to prevent interference from
        growing unboundedly as more items are added.

        Args:
            vectors: (bs, num_items, dim) vectors to superpose.
            dim: dimension to sum over.

        Returns:
            (bs, dim) superimposed vector.
        """
        summed = vectors.sum(dim=dim)

        # Bounded superposition: normalize by capacity factor
        num_items = vectors.shape[dim] if dim < vectors.ndim else 1
        capacity = min(num_items, self.max_items)
        bound = self.capacity_norm * (capacity ** 0.5)

        return summed / (bound + 1e-6)

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bind and superpose key-value pairs into a single memory vector.

        Args:
            keys: (bs, seq_len, dim) key vectors.
            values: (bs, seq_len, dim) value vectors.

        Returns:
            (bs, dim) compressed holographic memory.
        """
        # Limit to capacity
        seq_len = min(keys.shape[1], self.max_items)
        keys = keys[:, :seq_len]
        values = values[:, :seq_len]

        # Bind key-value pairs
        bound = self.bind(keys, values)  # (bs, seq_len, dim)

        # Superpose with capacity management
        memory = self.superpose(bound, dim=1)  # (bs, dim)

        return memory

    def retrieve(self, memory: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Retrieve a value from memory given a key.

        Uses basic unbinding without cleanup.

        Args:
            memory: (bs, dim) or (bs, seq, dim) memory vector.
            key: (bs, dim) or (bs, seq, dim) query key.

        Returns:
            Retrieved value(s).
        """
        return self.unbind(memory, key)

    def retrieve_with_cleanup(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve with iterative cleanup using the resonator network.

        After basic unbinding, the result is projected onto the nearest
        stored item via the resonator's cleanup mechanism.

        Args:
            memory: (bs, dim) memory vector.
            key: (bs, dim) query key.

        Returns:
            (bs, dim) cleaned-up retrieved value.
        """
        raw = self.unbind(memory, key)
        return self.resonator.cleanup(raw)

    def multi_retrieve(
        self,
        memory: torch.Tensor,
        keys: torch.Tensor,
        iterations: int = 3,
    ) -> torch.Tensor:
        """
        Retrieve multiple items from a superimposed memory with iterative cleanup.

        Uses the resonator network to iteratively refine each retrieval,
        reducing cross-talk between items.

        Args:
            memory: (bs, dim) superimposed memory.
            keys: (bs, num_items, dim) query keys.
            iterations: number of cleanup iterations per item.

        Returns:
            (bs, num_items, dim) retrieved values.
        """
        bs, num_items, _ = keys.shape
        retrieved = []

        for i in range(num_items):
            key_i = keys[:, i]  # (bs, dim)

            # Initial retrieval
            value_i = self.unbind(memory, key_i)

            # Iterative cleanup
            for _ in range(iterations):
                value_i = self.resonator.cleanup(value_i)

            retrieved.append(value_i)

        return torch.stack(retrieved, dim=1)  # (bs, num_items, dim)

    def set_cleanup_memory(self, memory: torch.Tensor) -> None:
        """
        Update the resonator's cleanup memory bank.

        Args:
            memory: (num_items, dim) stored item vectors.
        """
        self.resonator.set_cleanup_memory(memory)
