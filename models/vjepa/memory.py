import torch
from torch import nn
import torch.nn.functional as F

class HolographicMemory(nn.Module):
    """
    Holographic Reduced Representations (HRR) based memory.
    Uses circular convolution for binding and circular correlation for retrieval.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def _circular_convolution(self, a, b):
        """Bind two vectors a and b."""
        a_fft = torch.fft.fft(a)
        b_fft = torch.fft.fft(b)
        return torch.fft.ifft(a_fft * b_fft).real

    def _circular_correlation(self, a, b):
        """Retrieve b given a (or vice versa)."""
        a_fft = torch.fft.fft(a)
        b_fft = torch.fft.fft(b)
        # Correlation is fft(a)* conj(fft(b))
        return torch.fft.ifft(a_fft.conj() * b_fft).real

    def bind(self, key, value):
        return self._circular_convolution(key, value)

    def retrieve(self, memory, key):
        return self._circular_correlation(key, memory)

    def forward(self, keys, values):
        """
        keys, values: (bs, seq_len, dim)
        Returns a single compressed memory vector: (bs, dim)
        """
        # Bind key-value pairs
        bound = self.bind(keys, values)
        # Superpose (sum) into a single holographic state
        memory = bound.sum(dim=1)
        return memory
