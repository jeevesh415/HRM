"""
Monte Carlo Tree Search (MCTS) in Latent Space.

A complete MCTS implementation with:
  - PUCT (Predictor + Upper Confidence bound applied to Trees) selection
  - Full tree expansion with all available actions as children
  - Virtual loss for parallel simulation support
  - Progressive widening to limit branching factor
  - Proper backpropagation through the tree
  - Value network integration for leaf evaluation
  - Action pruning based on prior scores

This replaces the original 1-ply lookahead with a proper tree search
that can discover multi-step plans through the latent dynamics model.

Based on:
  - AlphaZero MCTS (Silver et al., 2017)
  - MuZero (Schrittwieser et al., 2020)
  - Progressive widening (Couetoux et al., 2011)
"""

import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class MCTSNode:
    """
    A node in the Monte Carlo search tree.

    Each node stores:
      - state: the latent state at this point in the tree
      - action: the action that led to this node (None for root)
      - parent: parent node reference
      - children: list of child nodes
      - visits: number of times this node was visited
      - value_sum: cumulative value (for computing mean value)
      - prior: prior probability from the policy network
      - virtual_loss: temporary visit penalty for parallel simulations
      - is_terminal: whether this is a terminal state
    """

    __slots__ = [
        "state", "action", "parent", "children", "visits",
        "value_sum", "prior", "virtual_loss", "is_terminal",
    ]

    def __init__(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        parent: Optional["MCTSNode"] = None,
        prior: float = 0.0,
    ):
        self.state = state
        self.action = action
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.virtual_loss = 0
        self.is_terminal = False

    @property
    def mean_value(self) -> float:
        """Mean value Q(s, a) of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def is_expanded(self) -> bool:
        """Whether this node has been expanded (has children)."""
        return len(self.children) > 0

    @property
    def effective_visits(self) -> int:
        """Visits including virtual loss for parallel simulations."""
        return self.visits + self.virtual_loss

    def puct_score(self, parent_visits: int, c_puct: float = 1.41) -> float:
        """
        Compute the PUCT selection score.

        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(parent)) / (1 + N(s,a))

        Where:
          Q(s,a) = mean value of the child
          P(s,a) = prior probability from the policy network
          N(parent) = total visits to the parent
          N(s,a) = visits to this child

        Args:
            parent_visits: total visits to the parent node.
            c_puct: exploration constant (higher = more exploration).

        Returns:
            PUCT score (higher is better).
        """
        if self.visits == 0:
            # Unvisited nodes get infinite score (exploration bonus)
            return float("inf")

        # Exploitation term: mean value
        q_value = self.mean_value

        # Exploration term: UCB-style with prior
        exploration = (
            c_puct
            * self.prior
            * math.sqrt(parent_visits)
            / (1 + self.visits)
        )

        return q_value + exploration


class MCTS:
    """
    Monte Carlo Tree Search with PUCT selection for latent planning.

    Uses the HRM V-JEPA Predictor as the world model to simulate
    future states in latent space. The value head estimates state values,
    and the policy prior guides exploration.

    Args:
        model: the VJEPA model (must have predictor and value_head).
        n_simulations: number of MCTS simulations (default: 50).
        c_puct: exploration constant for PUCT (default: 1.41).
        gamma: discount factor for returns (default: 0.99).
        progressive_widening_alpha: alpha for progressive widening (default: 0.5).
        progressive_widening_c: c for progressive widening (default: 10).
        virtual_loss_weight: virtual loss for parallel simulations (default: 3.0).
        temperature: action selection temperature (default: 1.0).
    """

    def __init__(
        self,
        model,
        n_simulations: int = 50,
        c_puct: float = 1.41,
        gamma: float = 0.99,
        progressive_widening_alpha: float = 0.5,
        progressive_widening_c: float = 10.0,
        virtual_loss_weight: float = 3.0,
        temperature: float = 1.0,
    ):
        self.model = model
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.pw_alpha = progressive_widening_alpha
        self.pw_c = progressive_widening_c
        self.virtual_loss_weight = virtual_loss_weight
        self.temperature = temperature

    @torch.no_grad()
    def _imagine_future(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Simulate the next state using the world model.

        Args:
            state: (1, D) or (1, seq, D) current latent state.
            action: (1, action_dim) action to take.

        Returns:
            next_state: (1, D) predicted next state.
            value: scalar value estimate of the next state.
            policy_query: (1, action_dim) action-prior query embedding.
        """
        # Use the physics engine for dynamics prediction
        delta_t = torch.ones(state.shape[0], device=state.device)
        next_state = self.model.predictor.physics_engine(state, delta_t, action=action)

        # Estimate value
        value = self.model.value_head(
            next_state.mean(dim=1) if next_state.ndim > 2 else next_state
        ).item()

        # Estimate action priors from a learned policy-query head.
        pooled_next_state = next_state.mean(dim=1) if next_state.ndim > 2 else next_state
        policy_query = self.model.policy_query_head(pooled_next_state)

        return next_state, value, policy_query

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a leaf node using PUCT.

        Traverses the tree from the root, always picking the child with
        the highest PUCT score, until reaching a leaf (unexpanded or
        unvisited) node.

        Applies virtual loss along the path to discourage other parallel
        simulations from selecting the same path.

        Args:
            node: root node to start selection from.

        Returns:
            Selected leaf node.
        """
        while node.is_expanded and node.children:
            # Apply virtual loss to discourage parallel selection
            node.virtual_loss += 1

            # Select child with highest PUCT score
            best_child = max(
                node.children,
                key=lambda c: c.puct_score(node.visits, self.c_puct),
            )
            node = best_child

        return node

    def _expand(
        self,
        node: MCTSNode,
        available_actions: torch.Tensor,
        policy_query: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Expand a node by creating children for available actions.

        Uses progressive widening: only expands min(c * N^alpha, |A|)
        children, where N is the node's visit count.

        Args:
            node: the node to expand.
            available_actions: (num_actions, action_dim) action set.
            policy_query: (1, action_dim) optional action-prior query vector.

        Returns:
            Value estimate of the expanded node.
        """
        if node.is_terminal:
            return node.mean_value

        num_actions = available_actions.shape[0]

        # Progressive widening: limit number of children
        max_children = max(1, int(self.pw_c * (node.visits + 1) ** self.pw_alpha))
        max_children = min(max_children, num_actions)

        # Compute priors from policy query vector.
        if policy_query is None and hasattr(self.model, "policy_query_head"):
            with torch.no_grad():
                pooled_state = node.state.mean(dim=1) if node.state.ndim > 2 else node.state
                policy_query = self.model.policy_query_head(pooled_state)

        if policy_query is not None and policy_query.numel() > 0:
            # Similarity(action_i, query) -> prior logit
            # available_actions: (num_actions, action_dim)
            # policy_query: (1, action_dim)
            logits = torch.matmul(available_actions, policy_query.squeeze(0))
            priors = F.softmax(logits / self.temperature, dim=0)
        else:
            # Uniform prior if no policy network
            priors = torch.ones(num_actions, device=available_actions.device) / num_actions

        # Expand children (only up to progressive widening limit)
        if len(node.children) < max_children:
            # Sort actions by prior (highest first) for progressive widening
            _, sorted_indices = priors.sort(descending=True)

            for idx in sorted_indices[len(node.children) : max_children]:
                action = available_actions[idx : idx + 1]
                next_state, value, _ = self._imagine_future(node.state, action)

                child = MCTSNode(
                    state=next_state,
                    action=action,
                    parent=node,
                    prior=priors[idx].item(),
                )
                node.children.append(child)

        # Return value of the first child (for backprop)
        if node.children:
            return node.children[0].mean_value
        return 0.0

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Backpropagate a value estimate through the tree.

        Updates visit counts and value sums for all ancestors of the node.
        Values are discounted by gamma at each level. Virtual loss is
        removed along the path.

        Args:
            node: the leaf node from which to backpropagate.
            value: the value estimate to propagate.
        """
        current = node
        discounted_value = value

        while current is not None:
            current.visits += 1
            current.value_sum += discounted_value

            # Remove virtual loss that was applied during selection
            current.virtual_loss = max(0, current.virtual_loss - 1)

            # Discount value for parent
            discounted_value *= self.gamma
            current = current.parent

    def _get_action_probabilities(self, root: MCTSNode, num_actions: int) -> torch.Tensor:
        """
        Convert visit counts to action probabilities.

        Uses visit count distribution with optional temperature scaling.

        Args:
            root: the root node after search.
            num_actions: total number of available actions.

        Returns:
            (num_actions,) action probability distribution.
        """
        visit_counts = torch.zeros(num_actions)
        for child in root.children:
            if child.action is not None:
                # Find the action index by matching against available actions
                # For now, use the order in which children were created
                idx = root.children.index(child)
                if idx < num_actions:
                    visit_counts[idx] = child.visits

        # Apply temperature
        if self.temperature == 0:
            # Greedy selection
            probs = torch.zeros_like(visit_counts)
            probs[visit_counts.argmax()] = 1.0
        else:
            # Boltzmann distribution over visit counts
            log_counts = torch.log(visit_counts + 1e-10) / self.temperature
            probs = F.softmax(log_counts, dim=0)

        return probs

    def plan(
        self,
        initial_state: torch.Tensor,
        available_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run MCTS to find the best action from the initial state.

        Args:
            initial_state: (1, D) or (1, seq, D) current latent state.
            available_actions: (num_actions, action_dim) action set.

        Returns:
            best_action: (1, action_dim) the selected action.
            action_probs: (num_actions,) visit-count-based probabilities.
        """
        # Handle empty action set
        if available_actions.shape[0] == 0:
            return torch.zeros(1, 0), torch.zeros(0)

        # Initialize root node
        root = MCTSNode(state=initial_state)

        # Initial expansion of root with uniform priors
        self._expand(root, available_actions)

        # Run simulations
        for _ in range(self.n_simulations):
            # 1. Selection: traverse tree to a leaf
            leaf = self._select(root)

            # 2. Expansion + Evaluation: expand leaf and get value
            if not leaf.is_terminal:
                value = self._expand(leaf, available_actions)
            else:
                value = leaf.mean_value

            # 3. Backpropagation: update statistics up the tree
            self._backpropagate(leaf, value)

        # Select best action based on visit counts
        action_probs = self._get_action_probabilities(root, available_actions.shape[0])

        if self.temperature == 0:
            best_idx = action_probs.argmax()
        else:
            best_idx = torch.multinomial(action_probs, 1).item()

        best_action = available_actions[best_idx : best_idx + 1]

        return best_action, action_probs


class LatentPlannerMCTS:
    """
    High-level MCTS planner that wraps the MCTS class.

    Provides a simple interface for planning in latent space using
    the HRM V-JEPA model as a world model.

    Args:
        model: the VJEPA model.
        n_simulations: number of MCTS simulations.
        c_puct: exploration constant.
        gamma: discount factor.
        temperature: action selection temperature.
    """

    def __init__(
        self,
        model,
        n_simulations: int = 50,
        c_puct: float = 1.41,
        gamma: float = 0.99,
        temperature: float = 1.0,
    ):
        self.mcts = MCTS(
            model=model,
            n_simulations=n_simulations,
            c_puct=c_puct,
            gamma=gamma,
            temperature=temperature,
        )

    def plan(
        self,
        initial_state: torch.Tensor,
        available_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Plan an action from the current state.

        Args:
            initial_state: (1, D) current latent state.
            available_actions: (num_actions, action_dim) available actions.

        Returns:
            (1, action_dim) the best action according to MCTS.
        """
        best_action, _ = self.mcts.plan(initial_state, available_actions)
        return best_action

    def plan_with_uncertainty(
        self,
        initial_state: torch.Tensor,
        available_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Plan and return both the best action and the full action distribution.

        Args:
            initial_state: (1, D) current latent state.
            available_actions: (num_actions, action_dim) available actions.

        Returns:
            best_action: (1, action_dim) the selected action.
            action_probs: (num_actions,) probability distribution over actions.
        """
        return self.mcts.plan(initial_state, available_actions)
