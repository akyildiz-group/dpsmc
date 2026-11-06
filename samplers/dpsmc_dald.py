import math
from samplers import register_sampler
from samplers.sampler import Sampler, SamplerConfig
from targets.target import Target
import torch
from tqdm import tqdm
from typing import Dict
from utils.distributions import stratified_resample


class DPSMCDALDConfig(SamplerConfig):
    n_aux_samples: int
    n_mala_steps: int
    n_steps: int
    T: float
    ess_threshold: float


@register_sampler("dpsmc_dald", DPSMCDALDConfig)
class DPSMCDALDSampler(Sampler):
    def __init__(
        self,
        n_aux_samples: int,
        n_mala_steps: int,
        n_steps: int,
        T: float,
        ess_threshold: float,
        score_identity: str,
        lambda_schedule: str,
        device: str,
    ) -> None:
        super().__init__(score_identity, lambda_schedule, device)

        self.n_aux_samples = n_aux_samples
        self.n_mala_steps = n_mala_steps
        self.n_steps = n_steps  # written as $K$
        self.step_size = T / self.n_steps
        self.ess_threshold = ess_threshold
        self.use_R_and_tau_init = True

    def sample(
        self,
        n_samples: int,
        target: Target,
    ) -> Dict[str, any]:
        device = self.device
        n_x = n_samples
        n_y = self.n_aux_samples
        K = self.n_steps
        h = self.step_size

        d = target.dimension()

        var_base = target.exp_squared_norm() / d
        sigma = var_base**0.5

        std_nu_log_prob = lambda x: -0.5 * (x * x).sum(dim=-1)
        base_log_prob = lambda x: std_nu_log_prob(x / sigma)
        base_grad_log_prob = lambda x: -x / var_base

        tol = torch.tensor(0.01, device=device)
        geom_scale = torch.tensor(1.1, device=device)
        accept_target = torch.tensor(0.75, device=device)
        h_k_scale = torch.tensor(1.0, device=device)

        min_acceptance_reached = False
        accept_threshold = 0.1

        sigma_tilde = (
            ((target.compact_R() ** 2) * d + target.gaussian_tau() ** 2) ** 0.5
            if self.use_R_and_tau_init
            else sigma
        )

        lambda_0 = self.lambda_schedule(0)

        ####### Initialise samples, weights, and score #######
        X_0 = sigma * torch.randn((n_x, 1, d), device=device)
        Y_0 = sigma_tilde * torch.randn((n_x, n_y, d), device=device)
        log_pi_Y_0, grad_log_pi_Y_0 = target.log_prob_and_grad(Y_0)

        A_0 = (
            base_log_prob((X_0 - (lambda_0**0.5) * Y_0) / ((1 - lambda_0) ** 0.5))
            + log_pi_Y_0
        ) - std_nu_log_prob(Y_0 / sigma_tilde)
        S_0 = base_grad_log_prob(X_0)

        X_k = X_0
        Y_k = Y_0
        A_k = A_0
        S_k = S_0
        log_pi_Y_k, grad_log_pi_Y_k = log_pi_Y_0, grad_log_pi_Y_0

        ess_traj = []

        for k in tqdm(range(1, K + 1), desc="Running DPSMC (DALD)", total=K):
            t = k / K
            lambda_k = self.lambda_schedule(t)
            c1_k = (1 - lambda_k) ** 0.5
            c2_k = lambda_k**0.5

            ####### Update X with ALD #######
            noise_scale = (2 * h) ** 0.5
            Z_k = torch.randn_like(X_k)

            ## Euler-Maruyama
            X_k_plus_one = X_k + h * S_k + noise_scale * Z_k

            ####### Resample Ys #######
            in_group_w = torch.exp(A_k - torch.logsumexp(A_k, dim=1, keepdim=True))
            within_group_ess = 1 / (in_group_w**2.0).sum(dim=1)
            should_resample = within_group_ess <= n_y * self.ess_threshold

            ess_traj.append(within_group_ess.mean().item())

            i_resample = stratified_resample(in_group_w)
            if (~should_resample).sum() > 0:
                i_resample[~should_resample] = torch.arange(n_y, device=device)
            if should_resample.sum() > 0:
                i_resample_flat = (
                    i_resample + n_y * torch.arange(n_x, device=device).unsqueeze(-1)
                ).flatten()
                Y_k = Y_k.view(-1, d)[i_resample_flat].view(n_x, n_y, d)
                log_pi_Y_k = log_pi_Y_k.view(-1)[i_resample_flat].view(n_x, n_y)
                grad_log_pi_Y_k = grad_log_pi_Y_k.view(-1, d)[i_resample_flat].view(
                    n_x, n_y, d
                )
                A_k[should_resample] = torch.logsumexp(
                    A_k[should_resample], dim=1, keepdim=True
                )

            ####### Update Ys with MALA #######
            t_plus_dt = (k + 1) / K
            lambda_k_plus_one = self.lambda_schedule(t_plus_dt)
            c1_k_plus_one = (1 - lambda_k_plus_one) ** 0.5
            c2_k_plus_one = lambda_k_plus_one**0.5

            curr_Y_k = Y_k
            log_pi_curr_Y_k = log_pi_Y_k
            grad_log_pi_curr_Y_k = grad_log_pi_Y_k
            log_pi_Y_k_plus_one_bar = None
            grad_log_pi_Y_k_plus_one_bar = None

            for _ in range(self.n_mala_steps):
                if min_acceptance_reached:
                    break
                h_k = h_k_scale * h
                noise_scale_k = (2 * h_k) ** 0.5

                grad_log_pi_k_plus_one = grad_log_pi_curr_Y_k
                grad_log_nu_k_plus_one = (c2_k_plus_one * curr_Y_k - X_k_plus_one) / (
                    c1_k_plus_one * var_base
                )
                grad_log_rho_k_plus_one = (
                    -(c2_k_plus_one / c1_k_plus_one) * grad_log_nu_k_plus_one
                    + grad_log_pi_k_plus_one
                )
                E_k = torch.randn_like(curr_Y_k)
                Y_k_plus_one_bar = (
                    curr_Y_k + h_k * grad_log_rho_k_plus_one + noise_scale_k * E_k
                )

                log_pi_Y_k_plus_one_bar, grad_log_pi_Y_k_plus_one_bar = (
                    target.log_prob_and_grad(Y_k_plus_one_bar)
                )
                grad_log_pi_k_plus_one = grad_log_pi_Y_k_plus_one_bar
                grad_log_nu_k_plus_one = (
                    c2_k_plus_one * Y_k_plus_one_bar - X_k_plus_one
                ) / (c1_k_plus_one * var_base)
                grad_log_rho_k_plus_one = (
                    -(c2_k_plus_one / c1_k_plus_one) * grad_log_nu_k_plus_one
                    + grad_log_pi_k_plus_one
                )
                B_k_bar = std_nu_log_prob(
                    (curr_Y_k - (Y_k_plus_one_bar + h_k * grad_log_rho_k_plus_one))
                    / noise_scale_k
                )

                accept_prob = (
                    (
                        base_log_prob(
                            (X_k_plus_one - c2_k_plus_one * Y_k_plus_one_bar)
                            / c1_k_plus_one
                        )
                        + log_pi_Y_k_plus_one_bar
                        - base_log_prob(
                            (X_k_plus_one - c2_k_plus_one * curr_Y_k) / c1_k_plus_one
                        )
                        - log_pi_curr_Y_k
                    )
                    + B_k_bar
                    - std_nu_log_prob(E_k)
                ).clamp(max=0)

                will_move = torch.rand_like(accept_prob) <= torch.exp(accept_prob)
                next_Y_k = torch.where(
                    will_move.unsqueeze(-1), Y_k_plus_one_bar, curr_Y_k
                )
                next_log_Y_k = torch.where(
                    will_move, log_pi_Y_k_plus_one_bar, log_pi_curr_Y_k
                )
                next_grad_log_Y_k = torch.where(
                    will_move.unsqueeze(-1),
                    grad_log_pi_Y_k_plus_one_bar,
                    grad_log_pi_curr_Y_k,
                )

                accept_mean = will_move.sum() / will_move.numel()
                decrease = accept_mean + tol < accept_target
                increase = accept_mean - tol > accept_target

                h_k_scale = torch.where(
                    decrease,
                    h_k_scale / geom_scale,
                    torch.where(increase, h_k_scale * geom_scale, h_k_scale),
                )

                if lambda_k > 0.5 and accept_mean < accept_threshold:
                    min_acceptance_reached = True
                    break

                curr_Y_k = next_Y_k
                log_pi_curr_Y_k = next_log_Y_k
                grad_log_pi_curr_Y_k = next_grad_log_Y_k

            if min_acceptance_reached:
                Y_k_plus_one[:] = X_k_plus_one
                A_k_plus_one = A_k * 0
                S_k_plus_one = target.grad_log_prob(X_k_plus_one)

                log_pi_Y_k_plus_one = log_pi_curr_Y_k
                grad_log_pi_Y_k_plus_one = grad_log_pi_curr_Y_k
            else:
                Y_k_plus_one = curr_Y_k
                log_pi_Y_k_plus_one = log_pi_curr_Y_k
                grad_log_pi_Y_k_plus_one = grad_log_pi_curr_Y_k

                ####### Weight udpate #######
                rho_k = base_log_prob((X_k - c2_k * Y_k) / c1_k) + log_pi_Y_k
                rho_k_plus_one = (
                    base_log_prob((X_k_plus_one - c2_k_plus_one * Y_k) / c1_k_plus_one)
                    + log_pi_Y_k
                )
                A_k_plus_one_bar = A_k + rho_k_plus_one - rho_k
                A_k_plus_one = A_k_plus_one_bar - torch.logsumexp(
                    A_k_plus_one_bar, dim=(0, 1)
                )

                ####### Score estimation #######
                in_group_w_bar = torch.exp(
                    A_k_plus_one_bar
                    - torch.logsumexp(A_k_plus_one_bar, dim=1, keepdim=True)
                )
                grad_log_nu_k_plus_one = (
                    c2_k_plus_one * Y_k_plus_one - X_k_plus_one
                ) / (c1_k_plus_one * var_base)
                grad_log_pi_k_plus_one = grad_log_pi_Y_k_plus_one
                grad_log_rho_k_plus_one = (
                    -(c2_k_plus_one / c1_k_plus_one) * grad_log_nu_k_plus_one
                    + grad_log_pi_k_plus_one
                )

                S_k_plus_one = self.score_identity.compute_score_estimator(
                    in_group_w_bar,
                    lambda_k_plus_one,
                    grad_log_nu_k_plus_one,
                    grad_log_pi_k_plus_one,
                    grad_log_rho_k_plus_one,
                    var_base,
                )

            X_k = X_k_plus_one
            Y_k = Y_k_plus_one
            A_k = A_k_plus_one
            S_k = S_k_plus_one

            log_pi_Y_k = log_pi_Y_k_plus_one
            grad_log_pi_Y_k = grad_log_pi_Y_k_plus_one

        self.ess_traj = torch.tensor(ess_traj).cpu()

        return X_k.squeeze(1).cpu()
