from abc import ABC, abstractmethod
import torch
from utils.distributions import log_and_grad_log


class Target(ABC):

    @abstractmethod
    def dimension(self) -> int:
        """get dimension of target distribution"""
        raise NotImplementedError()

    # As per Grenioux, et al. (2024), if we assume the target can be
    # expressed as a convolution between $\mathcal{N}(0, \tau^2\mathbf{I})$
    # and a probability measure $\mu$ compactly supported on a ball with
    # radius $R\sqrt{d}$, then we can inform samplers of this information.
    @abstractmethod
    def compact_R(self) -> float:
        """get radius of compact set in convolution decomposition"""
        raise NotImplementedError()

    @abstractmethod
    def gaussian_tau(self) -> float:
        """get standard deviation of gaussian in convolution decomposition"""
        raise NotImplementedError()

    @abstractmethod
    def exp_squared_norm(self) -> float:
        """get second moment $\mathbb{E}_\pi[\lVert X \rVert^2]$"""
        raise NotImplementedError()

    @abstractmethod
    def to(self, device: str) -> "Target":
        """set device of parameters"""
        raise NotImplementedError()

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """log of the target density up to a normalising constant"""
        raise NotImplementedError()

    @abstractmethod
    def grad_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """score of the target density"""
        raise NotImplementedError()

    @abstractmethod
    def as_torch_distribution(self) -> torch.distributions.Distribution:
        """torch distribution we can query for true target density"""
        raise NotImplementedError()

    def log_prob_and_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return log_and_grad_log(self, x)
