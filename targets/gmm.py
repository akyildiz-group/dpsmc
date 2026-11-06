import math
import torch
import torch.nn as nn
from targets import register_target
from targets.target import Target
from typing import Any, Callable


class GMM(nn.Module):
    def __init__(
        self, n_modes: int, var_scaling: float, loc_scaling: float, d: int = 2
    ) -> None:
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(0)
        self.n_modes = n_modes
        self.d = d
        self.var_scaling = var_scaling
        self.register_buffer(
            "mus",
            (torch.rand((n_modes, d), generator=generator) - 0.5) * loc_scaling,
        )
        self.register_buffer(
            "scale_tril",
            torch.diag_embed(torch.ones((n_modes, d)) * (var_scaling**0.5)),
        )
        self.var_scaling = var_scaling
        self.loc_scaling = loc_scaling
        self.register_buffer("mix", torch.ones((self.n_modes,)))

    def as_target(self) -> torch.distributions.Distribution:
        return GMMTarget(self)

    def _as_torch_distribution(self) -> torch.distributions.Distribution:
        mix = torch.distributions.Categorical(probs=self.mix)
        com = torch.distributions.Independent(
            torch.distributions.Normal(self.mus, self.var_scaling**0.5), 1
        )
        com.log_prob = torch.compile(com.log_prob)
        dist = torch.distributions.MixtureSameFamily(mix, com, validate_args=False)
        return dist

    def convolve(self, var_t: float, var_base: float) -> "GMM":
        device = self.mus.device
        gmm = GMM(self.n_modes, self.var_scaling, self.loc_scaling, d=self.d).to(device)
        gmm.mus = self.mus
        gmm.mix = self.mix
        identity = torch.ones((self.n_modes, self.d), device=device)
        gmm.mus = ((1 - var_t) ** 0.5) * gmm.mus
        gmm.scale_tril = (
            (1 - var_t) * (gmm.scale_tril**2)
            + var_t * var_base * torch.diag_embed(identity)
        ) ** 0.5
        gmm.var_scaling = (1 - var_t) * self.var_scaling + var_t * var_base
        return gmm.to(device)


class GMMTarget(Target):
    def __init__(self, gmm: GMM) -> None:
        self.gmm = gmm
        self.dist = gmm._as_torch_distribution()
        self.cached_log_prob = None

    def dimension(self) -> int:
        return self.gmm.d

    def compact_R(self) -> float:
        return (
            torch.linalg.norm(
                self.gmm.mus - self.gmm.mus.mean(dim=-1, keepdim=True), dim=-1
            ).max()
        ) / (self.dimension() ** 0.5)

    def gaussian_tau(self) -> float:
        return 1.0

    def exp_squared_norm(self) -> float:
        return self.dimension() + (self.gmm.mus**2.0).sum(dim=-1).mean().item()

    @torch.compile
    def chunk_calls(self, x: torch.Tensor, fn: Callable[[torch.Tensor], Any]) -> Any:
        ## Use this if your GPU can handle large batch sizes. In our case, this
        ## would be around the ballpark of (4096 * 128, d)-shaped vectors. It's
        ## perhaps larger than expected because we're backpropagating to get scores.

        # return fn(x)

        ## If above doesn't apply, you can manually adjust largest batch size here.
        _x = x.view(-1, x.shape[-1])
        batch_size = 64 * 4096
        # batch_size = 128 * 4096
        out = []
        for chunk in _x.chunk(math.ceil(_x.shape[0] / batch_size), dim=0):
            out.append(fn(chunk))
        return torch.cat(out, dim=0).view(x.shape[:-1])

    def to(self, device: str) -> "GMMTarget":
        self.gmm.to(device)
        self.dist = self.gmm._as_torch_distribution()
        return self

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.chunk_calls(x, self.dist.log_prob)

    def grad_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob_and_grad(x)[1]

    def as_torch_distribution(self) -> torch.distributions.Distribution:
        return self.dist


register_gmm_target = lambda name, oracle: register_target(name, GMMTarget(oracle))
register_gmm_target("gmm40_dim_2", GMM(40, 1.0, 40.0, d=2))
register_gmm_target("gmm40_dim_50", GMM(40, 1.0, 40.0, d=50))
