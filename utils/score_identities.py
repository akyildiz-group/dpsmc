from abc import abstractmethod, ABC
import torch
from typing import Callable, Dict


class ScoreIdentity(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def compute_score_estimator(
        self,
        w: torch.Tensor,
        lambda_k: float,
        grad_log_nu: torch.Tensor,
        grad_log_pi: torch.Tensor,
        grad_log_rho: torch.Tensor,
        var_base: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError()


SCORE_IDENTITY_REGISTRY: Dict[str, ScoreIdentity] = {}


def register_score_identity(name: str) -> Callable[[ScoreIdentity], ScoreIdentity]:

    def register(identity: ScoreIdentity) -> ScoreIdentity:
        if name in SCORE_IDENTITY_REGISTRY:
            raise Exception(f"Score identity '{name}' is already in the registry")
        SCORE_IDENTITY_REGISTRY[name] = identity
        return identity

    return register


@register_score_identity("dsi")
class DSI(ScoreIdentity):
    def compute_score_estimator(
        self,
        w: torch.Tensor,
        lambda_k: float,
        grad_log_nu: torch.Tensor,
        grad_log_pi: torch.Tensor,
        grad_log_rho: torch.Tensor,
        var_base: torch.Tensor,
    ) -> torch.Tensor:
        mc_term = (1 / ((1 - lambda_k) ** 0.5)) * grad_log_nu
        return ((w.unsqueeze(2) * mc_term).sum(dim=1)).unsqueeze(dim=1)


@register_score_identity("tsi")
class TSI(ScoreIdentity):
    def compute_score_estimator(
        self,
        w: torch.Tensor,
        lambda_k: float,
        grad_log_nu: torch.Tensor,
        grad_log_pi: torch.Tensor,
        grad_log_rho: torch.Tensor,
        var_base: torch.Tensor,
    ) -> torch.Tensor:
        mc_term = (1 / (lambda_k**0.5)) * grad_log_pi
        return ((w.unsqueeze(2) * mc_term).sum(dim=1)).unsqueeze(dim=1)


@register_score_identity("msi")
class MSI(ScoreIdentity):
    def compute_score_estimator(
        self,
        w: torch.Tensor,
        lambda_k: float,
        grad_log_nu: torch.Tensor,
        grad_log_pi: torch.Tensor,
        grad_log_rho: torch.Tensor,
        var_base: torch.Tensor,
    ) -> torch.Tensor:
        alpha_k = 1 - lambda_k
        mc_term = (alpha_k / ((1 - lambda_k) ** 0.5)) * grad_log_nu + (
            (1 - alpha_k) / (lambda_k**0.5)
        ) * grad_log_pi

        return ((w.unsqueeze(2) * mc_term).sum(dim=1)).unsqueeze(dim=1)


@register_score_identity("cvsi")
class CVSI(ScoreIdentity):
    def __init__(self) -> None:
        self.alpha_k = 1.0

    def compute_score_estimator(
        self,
        w: torch.Tensor,
        lambda_k: float,
        grad_log_nu: torch.Tensor,
        grad_log_pi: torch.Tensor,
        grad_log_rho: torch.Tensor,
        var_base: torch.Tensor,
    ) -> torch.Tensor:
        alpha_k = (
            (
                (w * (grad_log_pi * grad_log_rho).sum(dim=2)).sum(dim=1)
                / (w * (grad_log_rho**2.0).sum(dim=2)).sum(dim=1)
            )
            .unsqueeze(1)
            .unsqueeze(2)
        )
        mc_term = (alpha_k / ((1 - lambda_k) ** 0.5)) * grad_log_nu + (
            (1 - alpha_k) / (lambda_k**0.5)
        ) * grad_log_pi

        return ((w.unsqueeze(2) * mc_term).sum(dim=1)).unsqueeze(dim=1)


@register_score_identity("mcvsi")
class MCVSI(ScoreIdentity):
    def __init__(self) -> None:
        self.recurring_est = 1.0
        self.est_counter = 0
        self.alpha_k = 1.0

    def compute_score_estimator(
        self,
        w: torch.Tensor,
        lambda_k: float,
        grad_log_nu: torch.Tensor,
        grad_log_pi: torch.Tensor,
        grad_log_rho: torch.Tensor,
        var_base: torch.Tensor,
    ) -> torch.Tensor:
        d = grad_log_nu.shape[-1]

        score_var_est = (
            (w * (grad_log_pi * grad_log_rho).sum(dim=2)).sum(dim=1)
        ).mean()

        self.recurring_est = self.recurring_est * (
            self.est_counter / (self.est_counter + 1)
        ) + score_var_est / (self.est_counter + 1)
        self.est_counter += 1

        self.alpha_k = 1 - (
            (lambda_k * (d / var_base))
            / (lambda_k * d / var_base + (1 - lambda_k) * self.recurring_est)
        )

        mc_term = (self.alpha_k / ((1 - lambda_k) ** 0.5)) * grad_log_nu + (
            (1 - self.alpha_k) / (lambda_k**0.5)
        ) * grad_log_pi

        return ((w.unsqueeze(2) * mc_term).sum(dim=1)).unsqueeze(dim=1)


@register_score_identity("diagonal_mcvsi")
class DiagonalMCVSI(ScoreIdentity):
    def __init__(self) -> None:
        self.recurring_est = 1.0
        self.est_counter = 0
        self.a_k = None

    def compute_score_estimator(
        self,
        w: torch.Tensor,
        lambda_k: float,
        grad_log_nu: torch.Tensor,
        grad_log_pi: torch.Tensor,
        grad_log_rho: torch.Tensor,
        var_base: torch.Tensor,
    ) -> torch.Tensor:
        score_cov_diag_est = (
            (w.unsqueeze(2) * (grad_log_pi * grad_log_rho)).sum(dim=1)
        ).mean(dim=0)

        self.recurring_est = self.recurring_est * (
            self.est_counter / (self.est_counter + 1)
        ) + score_cov_diag_est / (self.est_counter + 1)
        self.est_counter += 1

        self.alpha_k = 1 - (
            (lambda_k / var_base)
            / (lambda_k / var_base + (1 - lambda_k) * self.recurring_est)
        )

        mc_term = (self.alpha_k / ((1 - lambda_k) ** 0.5)) * grad_log_nu + (
            (1 - self.alpha_k) / (lambda_k**0.5)
        ) * grad_log_pi

        return ((w.unsqueeze(2) * mc_term).sum(dim=1)).unsqueeze(dim=1)


@register_score_identity("matrix_mcvsi")
class MatrixMCVSI(ScoreIdentity):
    def __init__(self) -> None:
        self.recurring_est = 1.0
        self.est_counter = 0
        self.A_k = None

    def compute_score_estimator(
        self,
        w: torch.Tensor,
        lambda_k: float,
        grad_log_nu: torch.Tensor,
        grad_log_pi: torch.Tensor,
        grad_log_rho: torch.Tensor,
        var_base: torch.Tensor,
    ) -> torch.Tensor:
        n_x, _, d = grad_log_nu.shape
        device = grad_log_nu.device

        w_P = (w.unsqueeze(2) * grad_log_pi).reshape(-1, d)
        R = grad_log_rho.reshape(-1, d)
        score_cov_est = (w_P.transpose(0, 1) @ R) / n_x

        self.recurring_est = self.recurring_est * (
            self.est_counter / (self.est_counter + 1)
        ) + score_cov_est / (self.est_counter + 1)
        self.est_counter += 1
        
        cv_matrix_k = (
            (1 - lambda_k)
            * torch.inverse(
                (lambda_k / var_base) * torch.eye(d, device=device)
                + (1 - lambda_k) * self.recurring_est
            )
            @ self.recurring_est
        )
        self.A_k = cv_matrix_k

        mc_term = (1 / ((1 - lambda_k) ** 0.5)) * grad_log_nu @ cv_matrix_k + (
            1 / (lambda_k**0.5)
        ) * (grad_log_pi - grad_log_pi @ cv_matrix_k)

        return ((w.unsqueeze(2) * mc_term).sum(dim=1)).unsqueeze(dim=1)
