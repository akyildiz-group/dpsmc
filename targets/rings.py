import torch
from targets import register_target
from targets.target import Target
from utils.distributions import grad_log


###### Adapted from Grenioux et. al, (2024)
class PolarTransform(torch.distributions.transforms.Transform):
    """Polar transformation"""

    domain = torch.distributions.constraints.real_vector
    codomain = torch.distributions.constraints.real_vector
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, PolarTransform)

    def _call(self, x):
        return torch.stack(
            [x[..., 0] * torch.cos(x[..., 1]), x[..., 0] * torch.sin(x[..., 1])], dim=-1
        )

    def _inverse(self, y):
        x = torch.stack(
            [torch.norm(y, p=2, dim=-1), torch.atan2(y[..., 1], y[..., 0])], dim=-1
        )
        x[..., 1] = x[..., 1] + (x[..., 1] < 0).type_as(y) * (2 * torch.pi)
        return x

    def log_abs_det_jacobian(self, x, y):
        return torch.log(x[..., 0])


class Rings(torch.nn.Module):
    """Rings distribution"""

    def __init__(
        self, device, num_modes=4, radius=1.0, sigma=0.15, validate_args=False
    ):
        """Constructor

        The distribution is centered at 0

        Args:
            device (torch.device): Device used for computations
            num_modes (int): Number of circles (default is 4)
            radius (float): Radius of the smallest circle (default is 1.0)
            sigma (float): Width of the circles (default is 0.15)
        """

        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0, device=device))
        # Make the radius distribution
        self.radius_dist = torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(
                torch.ones((num_modes,), device=device)
            ),
            component_distribution=torch.distributions.Normal(
                loc=radius * (torch.arange(num_modes).to(device) + 1), scale=sigma
            ),
        )
        # Make the angle distribution
        self.angle_dist = torch.distributions.Uniform(
            low=torch.zeros((1,), device=device).squeeze(),
            high=2 * torch.pi * torch.ones((1,), device=device).squeeze(),
        )
        # Make the polar transform
        self.transform = PolarTransform()
        # Set the extreme values
        self.x_min = -radius * num_modes - sigma
        self.x_max = radius * num_modes + sigma
        self.y_min = -radius * num_modes - sigma
        self.y_max = radius * num_modes + sigma

    def sample(self, sample_shape=torch.Size()):
        """Sample the distribution

        Args:
            sample_shape (tuple of int): Shape of the samples

        Returns
            samples (torch.Tensor of shape (*sample_shape, 2)): Samples
        """

        r = self.radius_dist.sample(sample_shape)
        theta = self.angle_dist.sample(sample_shape)
        if len(sample_shape) == 0:
            x = torch.FloatTensor([r, theta])
        else:
            x = torch.stack([r, theta], dim=1)
        return self.transform(x)

    def log_prob(self, value):
        """Evaluate the log-likelihood of the distribution

        Args:
            value (torch.Tensor of shape (batch_size, 2)): Sample

        Returns
            log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood of the samples
        """

        x = self.transform.inv(value)
        return (
            self.radius_dist.log_prob(x[..., 0])
            + self.angle_dist.log_prob(x[..., 1])
            - self.transform.log_abs_det_jacobian(x, value)
        )

    def _apply(self, fn):
        """Apply the fn function on the distribution

        Args:
            fn (function): Function to apply on tensors
        """

        new_self = super(Rings, self)._apply(fn)
        # Radius distribution
        new_self.radius_dist.mixture_distribution.probs = fn(
            new_self.radius_dist.mixture_distribution.probs
        )
        new_self.radius_dist.component_distribution.loc = fn(
            new_self.radius_dist.component_distribution.loc
        )
        new_self.radius_dist.component_distribution.scale = fn(
            new_self.radius_dist.component_distribution.scale
        )
        # Angle distribution
        new_self.angle_dist.low = fn(new_self.angle_dist.low)
        new_self.angle_dist.high = fn(new_self.angle_dist.high)
        return new_self


class RingsTarget(Target):
    def __init__(self) -> None:
        self.rings = Rings("cpu")

    def dimension(self) -> int:
        return 2

    def compact_R(self) -> float:
        return 4 / (2**0.5)

    def gaussian_tau(self) -> float:
        return 0.15

    def exp_squared_norm(self) -> float:
        return 7.52

    def to(self, device: str) -> "RingsTarget":
        self.rings = Rings(device)
        return self

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.rings.log_prob(x)

    def grad_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        grad_log_prob_x = grad_log(self.rings, x)
        return grad_log_prob_x

    def as_torch_distribution(self) -> torch.distributions.Distribution:
        return self.rings


register_target("rings", RingsTarget())
