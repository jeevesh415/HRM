"""
Variational Uncertainty Quantification for Vision.

Implements two types of uncertainty:
  1. Aleatoric uncertainty: inherent noise in the data (e.g., blurry image)
  2. Epistemic uncertainty: model's lack of knowledge (e.g., unfamiliar object)

By quantifying both, the model can:
  - Know when to "think harder" (high epistemic → increase depth)
  - Know when to ask for help (high aleatoric → request better data)
  - Provide confidence estimates with its predictions

Based on:
  - "What Uncertainties Do We Need in Bayesian Deep Learning?" (Kendall & Gal, 2017)
  - "Dropout as a Bayesian Approximation" (Gal & Ghahramani, 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class VariationalLinear(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.

    Instead of deterministic weights, uses Gaussian distributions:
      W ~ N(mu, sigma^2)

    During training, samples weights via reparameterization trick.
    During inference, uses mean weights (or multiple samples for
    epistemic uncertainty).

    Args:
        in_features: input dimensionality.
        out_features: output dimensionality.
        prior_sigma: prior standard deviation for weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Posterior parameters (learnable)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight_log_sigma = nn.Parameter(torch.full((out_features, in_features), -5.0))

        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_log_sigma = nn.Parameter(torch.full((out_features,), -5.0))

        # Prior
        self.prior_sigma = prior_sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with weight sampling (training) or mean weights (inference).
        """
        if self.training:
            # Sample weights via reparameterization
            weight_sigma = torch.exp(self.weight_log_sigma)
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)

            bias_sigma = torch.exp(self.bias_log_sigma)
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior.

        KL(q(W) || p(W)) where:
          q(W) = N(mu, sigma^2) (posterior)
          p(W) = N(0, prior_sigma^2) (prior)
        """
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)

        # KL for weights
        kl_weight = (
            torch.log(self.prior_sigma / weight_sigma)
            + (weight_sigma ** 2 + self.weight_mu ** 2) / (2 * self.prior_sigma ** 2)
            - 0.5
        ).sum()

        # KL for bias
        kl_bias = (
            torch.log(self.prior_sigma / bias_sigma)
            + (bias_sigma ** 2 + self.bias_mu ** 2) / (2 * self.prior_sigma ** 2)
            - 0.5
        ).sum()

        return kl_weight + kl_bias


class UncertaintyQuantification(nn.Module):
    """
    Uncertainty quantification module for visual representations.

    Provides:
      - Aleatoric uncertainty (data noise) via learned variance
      - Epistemic uncertainty (model uncertainty) via MC dropout
      - Combined uncertainty for decision making

    Args:
        dim: feature dimensionality.
        num_mc_samples: number of MC dropout samples for epistemic uncertainty.
        dropout_rate: dropout rate for MC dropout.
    """

    def __init__(
        self,
        dim: int,
        num_mc_samples: int = 10,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_mc_samples = num_mc_samples
        self.dropout_rate = dropout_rate

        # Aleatoric uncertainty head (predicts variance)
        self.aleatoric_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, dim),
            nn.Softplus(),  # ensure positive
        )

        # Epistemic uncertainty via MC dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Bayesian final layer
        self.bayesian_layer = VariationalLinear(dim, dim)

        # Uncertainty-aware gating
        self.uncertainty_gate = nn.Linear(dim * 2, dim, bias=True)
        nn.init.zeros_(self.uncertainty_gate.weight)
        nn.init.constant_(self.uncertainty_gate.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty-aware representation.

        Args:
            x: (bs, seq_len, dim) input features.

        Returns:
            Dictionary with:
              - 'output': (bs, seq_len, dim) uncertainty-aware features
              - 'aleatoric': (bs, seq_len, dim) aleatoric uncertainty
              - 'epistemic': (bs, seq_len, dim) epistemic uncertainty
              - 'total_uncertainty': (bs, seq_len, dim) combined uncertainty
              - 'kl_loss': scalar KL divergence for Bayesian layers
        """
        bs, seq_len, dim = x.shape

        # Aleatoric uncertainty (data noise)
        aleatoric = self.aleatoric_head(x)  # (bs, seq_len, dim)

        # Epistemic uncertainty (model uncertainty) via MC dropout
        if self.training:
            # During training, dropout is active — single forward pass
            epistemic = self.dropout(torch.ones_like(x)) * x
            epistemic = epistemic.var(dim=-1, keepdim=True).expand_as(x)
        else:
            # During inference, run multiple forward passes with dropout
            samples = []
            for _ in range(self.num_mc_samples):
                sample = self.dropout(x)
                samples.append(sample)
            samples = torch.stack(samples, dim=0)  # (mc, bs, seq, dim)
            epistemic = samples.var(dim=0)  # (bs, seq, dim)

        # Combined uncertainty
        total_uncertainty = aleatoric + epistemic

        # Uncertainty-aware output
        # When uncertainty is high, the representation is more cautious
        uncertainty_features = torch.cat([x, total_uncertainty], dim=-1)
        gate = torch.sigmoid(self.uncertainty_gate(uncertainty_features))
        output = gate * x + (1 - gate) * torch.zeros_like(x)  # suppress uncertain features

        # KL divergence from Bayesian layer
        kl_loss = self.bayesian_layer.kl_divergence()

        return {
            'output': output,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'total_uncertainty': total_uncertainty,
            'kl_loss': kl_loss,
        }
