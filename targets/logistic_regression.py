import os
import pickle
from targets import Target, register_target
import torch
from torch.distributions.utils import probs_to_logits
from torch.nn.functional import binary_cross_entropy_with_logits


###### Adapted from Grenioux et. al, (2024)
class LogisticRegression:

    def __init__(
        self, X, y, device, use_intercept=True, intercept_scale=2.5, threshold=1e-8
    ):
        # Dataset
        self.X = X.float().to(device)
        self.y = y.float().to(device).flatten()
        self.dim = self.X.shape[-1]
        self.threshold = threshold
        # Priors
        self.weights_prior = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(
                loc=torch.zeros((self.dim,), device=device),
                scale=torch.ones((self.dim,), device=device),
            ),
            reinterpreted_batch_ndims=1,
        )
        self.use_intercept = use_intercept
        if use_intercept:
            self.intercept_prior = torch.distributions.Normal(
                loc=torch.tensor(0.0).to(device),
                scale=torch.tensor(intercept_scale).to(device),
            )

    @torch.compile
    def log_prob(self, params):
        # Ensure the shape of the params
        params = params.reshape((-1, params.shape[-1]))
        # Unpack the parameters
        if self.use_intercept:
            weights, intercept = params[..., :-1], params[..., -1]
        else:
            weights = params
        # Compute the prior
        prior_log_prob = self.weights_prior.log_prob(weights)
        if self.use_intercept:
            prior_log_prob += self.intercept_prior.log_prob(intercept)
        # Compute the likelihood
        probs = torch.special.expit(
            torch.matmul(self.X, weights.T).T + intercept.unsqueeze(-1)
        )
        probs = torch.clip(probs, self.threshold, 1.0 - self.threshold)
        logits = probs_to_logits(probs, is_binary=True)
        log_prob = -binary_cross_entropy_with_logits(
            logits, self.y.unsqueeze(0).expand((logits.shape[0], -1)), reduction="none"
        ).sum(dim=-1)
        return log_prob + prior_log_prob


class LogisticRegressionTarget(Target):
    def __init__(self, path_to_dataset: str, device: str = "cpu") -> None:
        self.path_to_dataset = path_to_dataset
        with open(path_to_dataset, "rb") as f:
            d = pickle.load(f)

            _X, _y = d["X_train"], d["y_train"]

            n_val_samples = int(_X.shape[0] * (1 / 8))
            X_val, y_val = _X[:n_val_samples], _y[:n_val_samples]
            X, y = _X[n_val_samples:], _y[n_val_samples:]
            X_test, y_test = d["X_test"], d["y_test"]

        self.target = LogisticRegression(X, y, device)
        self.val = LogisticRegression(X_val, y_val, device)
        self.test = LogisticRegression(X_test, y_test, device)

    def dimension(self) -> int:
        return self.target.dim + 1

    def compact_R(self) -> float:
        return 2.5 / ((self.dimension() + 1) ** 0.5)

    def gaussian_tau(self) -> float:
        return 0.0

    def exp_squared_norm(self) -> float:
        return self.dimension() + 6.25

    def to(self, device: str) -> "LogisticRegressionTarget":
        return LogisticRegressionTarget(self.path_to_dataset, device)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.target.log_prob(x.view(-1, x.shape[-1])).view(x.shape[:-1])

    def grad_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob_and_grad(x)[-1]

    def as_torch_distribution(self) -> torch.distributions.Distribution:
        raise NotImplementedError()


PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


class Ionosphere(LogisticRegressionTarget):
    def __init__(self, path_to_dataset=os.path.join(PATH, "ionosphere.pkl")):
        super().__init__(path_to_dataset)


class Sonar(LogisticRegressionTarget):
    def __init__(self, path_to_dataset=os.path.join(PATH, "sonar.pkl")):
        super().__init__(path_to_dataset)


register_target("ionosphere", Ionosphere())
register_target("sonar", Sonar())
