from abc import ABC, abstractmethod
import math
from targets.target import Target
import torch
from typing import Callable, Dict
from utils.score_identities import SCORE_IDENTITY_REGISTRY, ScoreIdentity
from utils.registry import ConfigOutline, auto_property


class SamplerConfig(ConfigOutline):
    score_identity: str
    lambda_schedule: str
    device: str


class Sampler(ABC):
    def __init__(self, score_identity: str, lambda_schedule: str, device: str) -> None:
        self.resolve_schedules(score_identity, lambda_schedule)
        self.device = device

    @abstractmethod
    def sample(
        self,
        n_samples: int,
        target: Target,
    ) -> Dict[str, any]:
        raise NotImplementedError()

    @auto_property
    def score_identity(self) -> ScoreIdentity:
        pass

    @auto_property
    def lambda_schedule(self) -> Callable[[float | torch.Tensor], float | torch.Tensor]:
        pass

    @auto_property
    def lambda_schedule_ratio(self) -> Callable[[float | torch.Tensor], float | torch.Tensor]:
        pass

    @auto_property
    @abstractmethod
    def device(self) -> str:
        pass

    def resolve_schedules(self, score_identity: str, lambda_schedule: str) -> None:
        eps = 1e-5

        if lambda_schedule == "linear":
            self.lambda_schedule = lambda t: min(1 - eps, t + eps)
            self.lambda_schedule_ratio = lambda t: (1 / (t ** 0.5 + eps))
        elif lambda_schedule == "cosine":
            self.lambda_schedule = lambda t: min(
                1 - eps, math.sin(t * 0.5 * math.pi) ** 2.0 + eps
            )
            self.lambda_schedule_ratio = lambda t: math.pi * math.cos(t * 0.5 * math.pi)
        else:
            raise Exception(f"Lambda schedule '{lambda_schedule}' not found.")

        if score_identity not in SCORE_IDENTITY_REGISTRY:
            raise Exception(f"Score identity '{score_identity}' not found.")
        self.score_identity = SCORE_IDENTITY_REGISTRY.get(score_identity)()

    def compile_statistics(self, stats: Dict[str, any]) -> None:
        for k in stats:
            v = stats[k]
            if isinstance(v, (torch.Tensor, list)):
                try:
                    if isinstance(stats[k][0], torch.Tensor):
                        stats[k] = torch.stack(v, dim=0)
                    else:
                        stats[k] = torch.tensor(v)
                except Exception as e:
                    print(f"Error processing {k}")
                    print(e)
