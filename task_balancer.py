"""Uncertainty-weighted multitask loss balancer (Kendall et al., 2018).

Supports dynamic number of tasks: core tasks (MSE/BCE, LM, contrastive)
plus optional auxiliary tasks (descriptor regression, functional group, DTI, MoA).
KL divergence handled separately via beta-annealing.
"""

import torch
import torch.nn as nn


class MultiTaskBalancer(nn.Module):
    """Learnable multitask loss balancer with dynamic task registration."""

    def __init__(self, task_names=None, kl_max_weight: float = 0.005,
                 kl_warmup_epochs: int = 30):
        super().__init__()
        if task_names is None:
            task_names = ['primary', 'lm', 'contrastive']
        self.task_names = list(task_names)
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.0))
            for name in self.task_names
        })
        self.kl_max_weight = kl_max_weight
        self.kl_warmup_epochs = kl_warmup_epochs

    def get_kl_weight(self, epoch: int) -> float:
        if self.kl_warmup_epochs <= 0:
            return self.kl_max_weight
        return min(1.0, epoch / self.kl_warmup_epochs) * self.kl_max_weight

    def forward(self, losses: dict, epoch: int = 0) -> tuple:
        """Compute total weighted loss.

        Args:
            losses: dict with 'primary', 'lm', 'contrastive', 'kl',
                    and optional 'descriptor', 'funcgroup', 'dti', 'moa'
            epoch: current epoch for KL annealing

        Returns:
            total_loss, info_dict
        """
        info = {}
        device = next(iter(self.log_vars.values())).device
        total = torch.tensor(0.0, device=device)

        for name in self.task_names:
            if name not in losses or losses[name] is None:
                continue
            loss = losses[name]
            if name not in self.log_vars:
                # Dynamically add new task weight
                self.log_vars[name] = nn.Parameter(torch.tensor(0.0, device=device))
                self.task_names.append(name)

            log_var = self.log_vars[name]
            precision = torch.exp(-log_var)
            weighted = precision * loss + log_var
            total = total + weighted
            info[f'{name}_raw'] = loss.item()
            info[f'{name}_weighted'] = weighted.item()
            info[f'sigma_{name}'] = torch.exp(0.5 * log_var).item()

        # KL with separate beta annealing
        if 'kl' in losses and losses['kl'] is not None:
            kl_w = self.get_kl_weight(epoch)
            kl_term = kl_w * losses['kl']
            total = total + kl_term
            info['kl_raw'] = losses['kl'].item()
            info['kl_beta'] = kl_w

        info['total'] = total.item()
        return total, info
