import math
from samplers import register_sampler
from samplers.sampler import Sampler, SamplerConfig
from targets.target import Target
import torch
from tqdm import tqdm
from typing import Dict
from utils.distributions import stratified_resample


class DPSMCOURDSMCConfig(SamplerConfig):
    n_aux_samples: int
    n_mala_steps: int
    n_steps: int
    n_ais_steps: int
    n_ll_steps: int
    lambda_0: float
    ess_threshold: float


@register_sampler("dpsmc_ou_rdsmc", DPSMCOURDSMCConfig)
class DPSMCOURDSMCSampler(Sampler):
    def __init__(
        self,
        n_aux_samples: int,
        n_mala_steps: int,
        n_steps: int,
        n_ais_steps: int,
        n_ll_steps: int,
        lambda_0: float,
        ess_threshold: float,
        score_identity: str,
        lambda_schedule: str,
        device: str,
    ) -> None:
        super().__init__(score_identity, lambda_schedule, device)

        self.n_aux_samples = n_aux_samples
        self.n_mala_steps = n_mala_steps
        self.n_steps = n_steps  # written as $K$
        self.n_ais_steps = n_ais_steps
        self.n_ll_steps = n_ll_steps

        self.T = -0.5 * math.log(lambda_0)
        self.step_size = self.T / self.n_steps
        self.ess_threshold = ess_threshold
        self.save_trajectory = False
        self.use_R_and_tau_init = True

        eps = 1e-4
        self.lambda_schedule = lambda t: min(
            1 - eps, math.exp(-2 * self.T * (1 - t)) + eps
        )

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
        n_ll = self.n_ll_steps
        T = self.T

        var_base = (target.compact_R() ** 2) * d + (target.gaussian_tau() ** 2.0)
        sigma = var_base**0.5

        std_nu_log_prob = lambda x: -0.5 * (x * x).sum(dim=-1) - (d / 2) * math.log(
            2 * torch.pi
        )
        nu_log_prob_with_sigma = lambda x, s: std_nu_log_prob(x / s) - d * math.log(s)
        base_log_prob = lambda x: std_nu_log_prob(x / sigma) - d * math.log(sigma)

        tol = torch.tensor(0.01, device=device)
        geom_scale = torch.tensor(1.1, device=device)
        accept_target = torch.tensor(0.75, device=device)
        h_k_scale = torch.tensor(1.0, device=device)

        min_acceptance_reached = False
        accept_threshold = 0.1

        lambda_0 = self.lambda_schedule(0)

        ####### LL initialisation settings #######
        ## Here, ts is scaled to [0, 1]. Need to multiply by T to get actual time.
        ts = torch.empty(K + 1)
        ts[: n_ll + 1] = 0
        ts[n_ll + 1 :] = torch.linspace(0, 1.0, K - n_ll)
        h_ll = (sigma**2.0) * (1 - lambda_0) / 4

        q_sigma = (var_base * min(((1 - lambda_0) / lambda_0), 1)) ** 0.5

        ####### Initialise samples, weights, and score #######
        X_0 = sigma * torch.randn((n_x, 1, d), device=device)

        ########## AIS initialisation ###########
        n_ais_steps = self.n_ais_steps

        Z_0 = q_sigma * torch.randn((n_x, n_y, d), device=device)
        AZ_0 = torch.zeros((n_x, n_y), device=device)
        Z_k = Z_0
        AZ_k = AZ_0
        AAZ_k = AZ_0.clone()

        log_pi_Z_k, grad_log_pi_Z_k = target.log_prob_and_grad(Z_k)

        last_marginal = torch.zeros((n_x,), device=device)

        c1_0 = (1 - lambda_0) ** 0.5
        c2_0 = lambda_0**0.5
        for i in range(n_ais_steps):
            beta_k = i / n_ais_steps
            beta_k_plus_one = (i + 1) / n_ais_steps

            ###### Take one MALA step ########
            h_k = h_k_scale * h
            noise_scale_k = (2 * h_k) ** 0.5

            grad_log_pi = grad_log_pi_Z_k
            grad_log_nu = (c2_0 * Z_k - X_0) / (c1_0 * var_base)
            grad_log_rho = -(c2_0 / c1_0) * grad_log_nu + grad_log_pi
            grad_log_q = -Z_k / (q_sigma**2.0)
            grad_log_geom = (
                1 - beta_k_plus_one
            ) * grad_log_q + beta_k_plus_one * grad_log_rho

            E_k = torch.randn_like(Z_k)
            Z_k_plus_one_bar = Z_k + h_k * grad_log_geom + noise_scale_k * E_k

            log_pi_Z_k_plus_one_bar, grad_log_pi_Z_k_plus_one_bar = (
                target.log_prob_and_grad(Z_k_plus_one_bar)
            )
            grad_log_pi = grad_log_pi_Z_k_plus_one_bar
            grad_log_nu = (c2_0 * Z_k_plus_one_bar - X_0) / (c1_0 * var_base)
            grad_log_rho = -(c2_0 / c1_0) * grad_log_nu + grad_log_pi
            grad_log_q = -Z_k_plus_one_bar / (q_sigma**2.0)
            grad_log_geom = (
                1 - beta_k_plus_one
            ) * grad_log_q + beta_k_plus_one * grad_log_rho

            B_k_bar = std_nu_log_prob(
                (Z_k - (Z_k_plus_one_bar + h_k * grad_log_geom)) / noise_scale_k
            )

            accept_prob = (
                (
                    (
                        (1 - beta_k_plus_one)
                        * nu_log_prob_with_sigma(Z_k_plus_one_bar, q_sigma)
                        + beta_k_plus_one
                        * (
                            base_log_prob((X_0 - c2_0 * Z_k_plus_one_bar) / c1_0)
                            + log_pi_Z_k_plus_one_bar
                        )
                        - (
                            (1 - beta_k_plus_one) * nu_log_prob_with_sigma(Z_k, q_sigma)
                            + beta_k_plus_one
                            * (base_log_prob((X_0 - c2_0 * Z_k) / c1_0) + log_pi_Z_k)
                        )
                    )
                )
                + B_k_bar
                - std_nu_log_prob(E_k)
            ).clamp(max=0)

            will_move = torch.rand_like(accept_prob) <= torch.exp(accept_prob)
            next_Z_k = torch.where(will_move.unsqueeze(-1), Z_k_plus_one_bar, Z_k)
            next_log_Z_k = torch.where(will_move, log_pi_Z_k_plus_one_bar, log_pi_Z_k)
            next_grad_log_Z_k = torch.where(
                will_move.unsqueeze(-1),
                grad_log_pi_Z_k_plus_one_bar,
                grad_log_pi_Z_k,
            )

            accept_mean = will_move.sum() / will_move.numel()
            decrease = accept_mean + tol < accept_target
            increase = accept_mean - tol > accept_target

            ## Increase scaling during LL init in case initialisation was poor
            _geom_scale = 2.0

            h_k_scale = torch.where(
                decrease,
                h_k_scale / _geom_scale,
                torch.where(increase, h_k_scale * _geom_scale, h_k_scale),
            )

            ######## Update weights ########
            AZ_update = (beta_k - beta_k_plus_one) * (
                nu_log_prob_with_sigma(Z_k, q_sigma)
                - (
                    base_log_prob((X_0 - c2_0 * Z_k) / c1_0)
                    + log_pi_Z_k
                    - d * math.log(c1_0)
                )
            )
            AZ_k = AZ_k + AZ_update
            AAZ_k = AAZ_k + AZ_update

            Z_k = next_Z_k
            log_pi_Z_k = next_log_Z_k
            grad_log_pi_Z_k = next_grad_log_Z_k

            in_group_w = torch.exp(AZ_k - torch.logsumexp(AZ_k, dim=1, keepdim=True))
            within_group_ess = 1 / (in_group_w**2.0).sum(dim=1)
            should_resample = within_group_ess <= n_y * self.ess_threshold

            i_resample = stratified_resample(in_group_w)
            if (~should_resample).sum() > 0:
                i_resample[~should_resample] = torch.arange(n_y, device=device)
            if should_resample.sum() > 0:
                i_resample_flat = (
                    i_resample + n_y * torch.arange(n_x, device=device).unsqueeze(-1)
                ).flatten()
                Z_k = Z_k.view(-1, d)[i_resample_flat].view(n_x, n_y, d)
                log_pi_Z_k = log_pi_Z_k.view(-1)[i_resample_flat].view(n_x, n_y)
                grad_log_pi_Z_k = grad_log_pi_Z_k.view(-1, d)[i_resample_flat].view(
                    n_x, n_y, d
                )

                last_marginal[should_resample] = (
                    last_marginal[should_resample]
                    + torch.logsumexp(AZ_k[should_resample], dim=1)
                    - math.log(n_y)
                )
                AZ_k[should_resample] = 0

        Y_0 = Z_k
        A_0 = AZ_k
        log_pi_Y_0, grad_log_pi_Y_0 = target.log_prob_and_grad(Y_0)
        AA_0 = AAZ_k
        ##################################

        S_0 = (
            (lambda_0**0.5)
            * (
                torch.exp(A_0 - torch.logsumexp(A_0, dim=1, keepdim=True)).unsqueeze(2)
                * Y_0
            )
            .sum(dim=1)
            .unsqueeze(1)
            - X_0
        ) / (var_base * (1 - lambda_0))

        AX_0 = (torch.logsumexp(AA_0, dim=1) - math.log(n_y)) - base_log_prob(
            X_0
        ).squeeze(1)
        AXX_0 = AX_0.clone()

        X_k = X_0
        Y_k = Y_0
        A_k = A_0
        S_k = S_0
        AA_k = AA_0
        AX_k = AX_0
        AXX_k = AXX_0
        log_pi_Y_k, grad_log_pi_Y_k = log_pi_Y_0, grad_log_pi_Y_0

        ess_traj = []

        for k in tqdm(range(1, K), desc="Running DPSMC (OU w/ RDSMC)", total=K - 1):
            t = ts[k]
            t_plus_dt = ts[min(k + 1, len(ts) - 1)]
            lambda_k = self.lambda_schedule(t)
            c1_k = (1 - lambda_k) ** 0.5
            c2_k = lambda_k**0.5

            Z_k = torch.randn_like(X_k)

            ####### Update X #######
            if k <= n_ll:
                ## ULA step for LL initialisation
                noise_scale = (2 * h_ll) ** 0.5
                X_k_plus_one = X_k + h_ll * S_k + noise_scale * Z_k
            else:
                ## Euler-Maruyama
                h = T * (t_plus_dt - t)
                noise_scale = (2 * h * var_base) ** 0.5
                X_k_plus_one = X_k + h * (X_k + 2 * var_base * S_k) + noise_scale * Z_k

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
                last_marginal[should_resample] = (
                    last_marginal[should_resample]
                    + torch.logsumexp(A_k[should_resample], dim=1)
                    - math.log(n_y)
                )
                A_k[should_resample] = 0

            ####### Update Ys with MALA #######
            t_plus_dt = ts[min(k + 1, len(ts) - 1)]
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

                ## Increase scaling during LL init in case initialisation was poor
                _geom_scale = 2.0 if k < n_ll else geom_scale

                h_k_scale = torch.where(
                    decrease,
                    h_k_scale / _geom_scale,
                    torch.where(increase, h_k_scale * _geom_scale, h_k_scale),
                )
                if k / ts.shape[0] > 0.5 and accept_mean < accept_threshold:
                    min_acceptance_reached = True
                    break

                curr_Y_k = next_Y_k
                log_pi_curr_Y_k = next_log_Y_k
                grad_log_pi_curr_Y_k = next_grad_log_Y_k

            if min_acceptance_reached:
                E_k = torch.randn_like(Y_k)
                Y_k_plus_one = (
                    X_k_plus_one / c2_k_plus_one
                    + sigma * c1_k_plus_one * E_k / c2_k_plus_one
                )

                log_pi_Y_k_plus_one, grad_log_pi_Y_k_plus_one = (
                    target.log_prob_and_grad(Y_k_plus_one)
                )
                A_k_plus_one = log_pi_Y_k_plus_one
                AA_k_plus_one = log_pi_Y_k_plus_one

                _h = T * (t_plus_dt - t)
                M_k_plus_one = (
                    last_marginal + torch.logsumexp(A_k_plus_one, dim=1) - math.log(n_y)
                )
                M_k = last_marginal + torch.logsumexp(A_k, dim=1) - math.log(n_y)

                if k == K - 1:
                    _h = T * (1 - t)
                    AX_update = (
                        target.log_prob(X_k_plus_one).squeeze(dim=-1)
                        + std_nu_log_prob(
                            (X_k - ((1 - _h) * X_k_plus_one))
                            / (sigma * ((2.0 * _h) ** 0.5))
                        ).squeeze(dim=-1)
                        - M_k
                        - std_nu_log_prob(Z_k).squeeze(dim=-1)
                    )
                else:
                    AX_update = (
                        M_k_plus_one
                        + std_nu_log_prob(
                            (X_k - ((1 - _h) * X_k_plus_one))
                            / (sigma * ((2.0 * _h) ** 0.5))
                        ).squeeze(dim=-1)
                        - M_k
                        - std_nu_log_prob(Z_k).squeeze(dim=-1)
                    )
                AX_k_plus_one = AX_k + AX_update
                AX_k_plus_one = AX_k_plus_one - torch.logsumexp(AX_k_plus_one, dim=0)
                AXX_k_plus_one = AXX_k + AX_update

                importance_w = torch.exp(
                    A_k_plus_one - torch.logsumexp(A_k_plus_one, dim=1, keepdim=True)
                ).unsqueeze(2)
                S_k_plus_one = (importance_w * grad_log_pi_Y_k_plus_one).sum(
                    dim=1
                ).unsqueeze(1) / c2_k_plus_one
            else:
                Y_k_plus_one = curr_Y_k
                log_pi_Y_k_plus_one = log_pi_curr_Y_k
                grad_log_pi_Y_k_plus_one = grad_log_pi_curr_Y_k

                ####### Weight udpate #######
                rho_k = (
                    base_log_prob((X_k - c2_k * Y_k) / c1_k)
                    + log_pi_Y_k
                    - (d / 2) * math.log(1 - lambda_k)
                )
                rho_k_plus_one = (
                    base_log_prob((X_k_plus_one - c2_k_plus_one * Y_k) / c1_k_plus_one)
                    + log_pi_Y_k
                    - (d / 2) * math.log(1 - lambda_k_plus_one)
                )
                A_k_plus_one_bar = A_k + rho_k_plus_one - rho_k
                A_k_plus_one = A_k_plus_one_bar
                AA_k_plus_one = AA_k + rho_k_plus_one - rho_k

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

                ######### Update X weights ############
                M_k_plus_one = (
                    last_marginal + torch.logsumexp(A_k_plus_one, dim=1) - math.log(n_y)
                )
                M_k = last_marginal + torch.logsumexp(A_k, dim=1) - math.log(n_y)
                if k <= n_ll:
                    AX_update = (
                        M_k_plus_one
                        + std_nu_log_prob(
                            (X_k - (X_k_plus_one + h_ll * S_k_plus_one))
                            / ((2.0 * h_ll) ** 0.5)
                        ).squeeze(dim=-1)
                        - M_k
                        - std_nu_log_prob(Z_k).squeeze(dim=-1)
                    )
                    AX_k_plus_one = AX_k + AX_update
                    AX_k_plus_one = AX_k_plus_one - torch.logsumexp(
                        AX_k_plus_one, dim=0
                    )
                    AXX_k_plus_one = AXX_k + AX_update
                else:
                    _h = T * (t_plus_dt - t)
                    if k == K - 1:
                        AX_update = (
                            target.log_prob(X_k_plus_one).squeeze(dim=-1)
                            + std_nu_log_prob(
                                (X_k - ((1 - _h) * X_k_plus_one))
                                / (sigma * ((2.0 * _h) ** 0.5))
                            ).squeeze(dim=-1)
                            - M_k
                            - std_nu_log_prob(Z_k).squeeze(dim=-1)
                        )
                    else:
                        AX_update = (
                            M_k_plus_one
                            + std_nu_log_prob(
                                (X_k - ((1 - _h) * X_k_plus_one))
                                / (sigma * ((2.0 * _h) ** 0.5))
                            ).squeeze(dim=-1)
                            - M_k
                            - std_nu_log_prob(Z_k).squeeze(dim=-1)
                        )

                    AX_k_plus_one = AX_k + AX_update
                    AX_k_plus_one = AX_k_plus_one - torch.logsumexp(
                        AX_k_plus_one, dim=0
                    )
                    AXX_k_plus_one = AXX_k + AX_update

            X_k = X_k_plus_one
            Y_k = Y_k_plus_one
            A_k = A_k_plus_one
            S_k = S_k_plus_one
            AA_k = AA_k_plus_one
            AX_k = AX_k_plus_one
            AXX_k = AXX_k_plus_one

            log_pi_Y_k = log_pi_Y_k_plus_one
            grad_log_pi_Y_k = grad_log_pi_Y_k_plus_one

        log_Z_hat = torch.logsumexp(AXX_k, dim=0) - math.log(n_x)
        self.log_Z_hat = log_Z_hat
        self.ess_traj = torch.tensor(ess_traj).cpu()

        return X_k.squeeze(1).cpu()
