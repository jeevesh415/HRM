"""
Latent Memory Palace (LMP): Hierarchical Semantic & Episodic Memory.

Replaces flat Holographic Memory with a 3-tier hierarchical structure:
  - Wings: Global semantic categories.
  - Rooms: Contextual/Task-specific subspaces.
  - Halls: Episodic/Verbatim traces.

Uses differentiable hierarchical attention to 'walk' the palace and 
AAAK-style compression for long-range context preservation.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple

class LatentMemoryPalace(nn.Module):
    """
    Differentiable Latent Memory Palace for V-JEPA 2.1.
    Organizes latent knowledge into a spatial-hierarchical manifold.
    """
    def __init__(self, dim: int, num_wings: int = 4, num_rooms: int = 8, num_halls: int = 16):
        super().__init__()
        self.dim = dim
        self.num_wings = num_wings
        self.num_rooms = num_rooms
        self.num_halls = num_halls

        # The Palace: Hierarchical Parameter Structure
        # (Wings, Rooms, Halls, Dim)
        self.palace = nn.Parameter(torch.randn(num_wings, num_rooms, num_halls, dim) * 0.02)
        
        # Palace Navigation Heads (Differentiable Walkers)
        self.wing_head = nn.Linear(dim, num_wings)
        self.room_head = nn.Linear(dim + dim, num_rooms) # Conditional on Wing
        self.hall_head = nn.Linear(dim + dim, num_halls) # Conditional on Room

        # AAAK Compression / Consolidation Gate
        self.consolidation_gate = nn.GRUCell(dim, dim)

    def _walk(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable hierarchical navigation of the Palace.
        Returns: (bs, n, dim) retrieved memory and (bs, n, 1) confidence score.
        """
        bs, n, d = query.shape
        
        # 1. Select Wing
        wing_logits = self.wing_head(query) # (bs, n, W)
        wing_probs = F.softmax(wing_logits, dim=-1)
        
        # Soft-select wing embedding
        # (bs, n, W) @ (W, R*H*D) -> (bs, n, R*H*D)
        wing_flat = self.palace.view(self.num_wings, -1)
        wing_context = wing_probs @ wing_flat
        wing_context = wing_context.view(bs, n, self.num_rooms, self.num_halls, d)
        
        # 2. Select Room (Conditional on Wing context)
        # We pool the wing context to query rooms
        room_query = torch.cat([query, wing_context.mean(dim=(2, 3))], dim=-1)
        room_logits = self.room_head(room_query) # (bs, n, R)
        room_probs = F.softmax(room_logits, dim=-1)
        
        # Soft-select room
        # (bs, n, R) @ (bs, n, R, H*D) -> (bs, n, H*D)
        room_context = torch.einsum('bsr,bsrhd->bshd', room_probs, wing_context)
        room_context = room_context.view(bs, n, self.num_halls, d)
        
        # 3. Select Hall (The episodic trace)
        hall_query = torch.cat([query, room_context.mean(dim=2)], dim=-1)
        hall_logits = self.hall_head(hall_query) # (bs, n, H)
        hall_probs = F.softmax(hall_logits, dim=-1)
        
        # Final Retrieval: (bs, n, H) @ (bs, n, H, D) -> (bs, n, D)
        retrieved = torch.einsum('bsh,bshd->bsd', hall_probs, room_context)
        
        # Confidence is the max hall probability
        confidence = hall_probs.max(dim=-1, keepdim=True)[0]
        
        return retrieved, confidence

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the most relevant memory trace from the Palace.
        """
        memory, _ = self._walk(query)
        return memory

    def consolidate(self, sensory_input: torch.Tensor, wing_idx: int, room_idx: int, hall_idx: int):
        """
        Frontier: Differentiable update of the Palace parameters.
        Consolidates new sensory information into a specific location in the Palace.
        """
        # This is used during the backward pass or a dedicated consolidation cycle
        target_hall = self.palace[wing_idx, room_idx, hall_idx]
        updated_hall = self.consolidation_gate(sensory_input.mean(dim=(0, 1)), target_hall)
        
        # In-place update (or gradient-based if training)
        with torch.no_grad():
            self.palace[wing_idx, room_idx, hall_idx].copy_(updated_hall)
