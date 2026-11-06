from targets import Target, register_target
from utils.distributions import grad_log
import torch
import math

###### Adapted from Grenioux et. al, (2024)
sqrt_two_pi = math.sqrt(2 * math.pi)


def log_prob_normal(t, loc, scale):
    return -0.5 * torch.square((t - loc) / scale) - torch.log(scale * sqrt_two_pi)


class Funnel(torch.distributions.Distribution):

    arg_constraints = {
        "dim": torch.distributions.constraints.nonnegative_integer,
        "a": torch.distributions.constraints.positive,
        "b": torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real_vector
    has_rsample = True

    def __init__(self, dim, device, a=3.0, b=1.0, validate_args=False):
        self.a = a
        self.sqrt_a = math.sqrt(self.a)
        self.torch_sqrt_a = torch.tensor(self.sqrt_a)
        self.b = b
        self.dim = dim
        self.device = device
        super(Funnel, self).__init__(validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        z = self.sqrt_a * torch.randn((*sample_shape, 1))
        x = torch.exp(self.b * z / 2) * torch.randn((*sample_shape, self.dim - 1))
        return torch.concat((z, x), dim=1).to(self.device)

    def log_prob(self, value):
        return log_prob_normal(value[..., 0], 0, self.torch_sqrt_a) + torch.sum(
            log_prob_normal(
                value[..., 1:],
                0,
                torch.unsqueeze(torch.exp(self.b * value[..., 0] / 2), dim=-1),
            ),
            dim=-1,
        )


class FunnelTarget(Target):
    def __init__(self):
        self.funnel = Funnel(10, "cpu")

    def dimension(self) -> int:
        return 10

    def compact_R(self) -> float:
        return 2.12 / (10**0.5)

    def gaussian_tau(self) -> float:
        return 0

    def exp_squared_norm(self) -> float:
        return 43.34

    def to(self, device: str) -> "FunnelTarget":
        self.funnel = Funnel(10, device)
        return self

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.funnel.log_prob(x)

    def grad_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return grad_log(self.funnel, x)

    def as_torch_distribution(self) -> torch.distributions.Distribution:
        return self.funnel


register_target("funnel", FunnelTarget())
