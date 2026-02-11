import torch


class GSAM:
    """
    Surrogate Gap Guided Sharpness-Aware Minimization (GSAM).

    Finds flat minima by minimizing both the perturbed loss (SAM objective)
    and the surrogate gap between perturbed and original loss.

    Reference: Zhuang et al., "Surrogate Gap Minimization Improves
    Sharpness-Aware Training", ICLR 2022.

    Args:
        params: Model parameters.
        base_optimizer_cls: Base optimizer class (e.g., torch.optim.Adam).
        rho: Perturbation radius for SAM ascent step.
        alpha: Weight for surrogate gap minimization (0 = pure SAM).
        **kwargs: Passed to the base optimizer (lr, weight_decay, etc.).
    """

    def __init__(self, params, base_optimizer_cls, rho=0.05, alpha=0.1, **kwargs):
        self.base_optimizer = base_optimizer_cls(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.state = {}
        self.rho = rho
        self.alpha = alpha

    @torch.no_grad()
    def first_step(self):
        """Perturb weights toward the sharpest direction: w -> w + epsilon."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Save original gradient for GSAM decomposition later
                self.state[p] = {
                    'grad_orig': p.grad.clone(),
                    'e_w': p.grad / (grad_norm + 1e-12) * self.rho,
                }
                p.add_(self.state[p]['e_w'])  # ascend to perturbed point

    @torch.no_grad()
    def second_step(self):
        """Compute GSAM gradient, restore weights, and step."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p not in self.state:
                    continue

                # g1 = gradient at original weights (saved in first_step)
                # g2 = gradient at perturbed weights (current p.grad)
                g1 = self.state[p]['grad_orig']
                g2 = p.grad

                # Project g2 onto g1: proj = (g2 . g1 / ||g1||^2) * g1
                g1_flat = g1.view(-1)
                g2_flat = g2.view(-1)
                dot = g1_flat.dot(g2_flat)
                g1_norm_sq = g1_flat.dot(g1_flat)

                if g1_norm_sq > 1e-12:
                    proj = (dot / g1_norm_sq) * g1
                else:
                    proj = torch.zeros_like(g1)

                # Perpendicular component: g2_perp = g2 - proj(g2, g1)
                g2_perp = g2 - proj

                # GSAM gradient: g2 - alpha * g2_perp
                # = (1 - alpha) * g2 + alpha * proj(g2, g1)
                p.grad.copy_(g2 - self.alpha * g2_perp)

                # Restore original weights
                p.sub_(self.state[p]['e_w'])

        # Step with the GSAM-modified gradient
        self.base_optimizer.step()
        self.state.clear()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = None
        norm_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if shared_device is None:
                        shared_device = p.grad.device
                    norm_list.append(p.grad.norm(p=2).to(shared_device))
        if len(norm_list) == 0:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norm_list), p=2)

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
