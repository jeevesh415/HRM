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

    @torch.no_grad()
    def _imagine_future(self, state, action):
        """
        This is where the HRM-VJEPA Predictor is called.
        It simulates the 'physics' of the world in latent space.
        state: (1, D) or (1, num_patches, D)
        action: (1, action_dim)
        """
        # We need some dummy context/queries for the predictor's signature
        # In a real planning scenario, 'state' would be the current world state
        bs = state.shape[0]
        device = state.device
        
        # We assume 'state' is the compressed world state from Holographic Memory
        # or the set of latents. For simplicity, let's treat it as the world state.
        
        # Call the physics engine directly for planning efficiency
        delta_t = 1.0 # Planning step size
        next_state = self.model.predictor.physics_engine(state, delta_t, action=action)
        
        # Estimate value of the new state
        value = self.model.value_head(next_state).item()
        
        return next_state, value

    def _backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.value += value
            node = node.parent

    def plan(self, initial_state: torch.Tensor, available_actions: torch.Tensor):
        # initial_state: (1, D)
        # available_actions: (num_actions, action_dim)
        root = MCTSNode(state=initial_state)

        # Pre-expand root
        for i in range(available_actions.shape[0]):
            action = available_actions[i:i+1]
            next_state, value = self._imagine_future(root.state, action)
            child = MCTSNode(state=next_state, action=action, parent=root)
            root.children.append(child)
            self._backpropagate(child, value)

        for _ in range(self.n_simulations):
            node = root
            
            # 1. Selection
            while node.children:
                node = max(node.children, key=lambda c: c.ucb_score(node.visits))

            # 2. Expansion & Simulation
            # For simplicity, we just simulate one step further from the selected node
            # using a random action or a simple heuristic
            random_action = available_actions[torch.randint(0, available_actions.shape[0], (1,))]
            next_state, value = self._imagine_future(node.state, random_action)
            
            # In a full MCTS, we would add this as a child
            # child = MCTSNode(state=next_state, action=random_action, parent=node)
            # node.children.append(child)
            
            # 3. Backpropagation
            self._backpropagate(node, value)
            
        # Return action from best child
        return max(root.children, key=lambda c: c.visits).action
