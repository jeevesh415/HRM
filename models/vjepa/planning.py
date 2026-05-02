import math
import torch
from typing import List, Dict, Optional

class MCTSNode:
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def ucb_score(self, parent_visits, c_puct=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c_puct * math.sqrt(math.log(parent_visits) / self.visits)

class LatentPlannerMCTS:
    """
    Monte Carlo Tree Search in Latent Space.
    Uses the HRM V-JEPA Predictor as the 'World Model'.
    """
    def __init__(self, model, n_simulations=50):
        self.model = model
        self.n_simulations = n_simulations

    def plan(self, initial_state: torch.Tensor, available_actions: torch.Tensor):
        root = MCTSNode(state=initial_state)

        for _ in range(self.n_simulations):
            node = root
            
            # 1. Selection
            while node.children:
                node = max(node.children, key=lambda c: c.ucb_score(node.visits))

            # 2. Expansion & Simulation
            # Use HRM to predict the next latent state given an action
            next_state, value = self._imagine_future(node.state, available_actions)
            
            # 3. Backpropagation
            self._backpropagate(node, value)
            
        # Return action from best child
        return max(root.children, key=lambda c: c.visits).action

    @torch.no_grad()
    def _imagine_future(self, state, action):
        """
        This is where the HRM-VJEPA Predictor is called.
        It simulates the 'physics' of the world in latent space.
        """
        # predictor_output = self.model.predictor(state, action)
        # return next_state, value_estimate
        return state, 0.0 # Placeholder

    def _backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
