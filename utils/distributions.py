from functools import partial
import logging
import math
import ot
import torch
import numpy as np

logger = logging.getLogger()


def grad_log(p: torch.distributions.Distribution, x: torch.Tensor) -> torch.Tensor:
    r"""Helper to evaluate $\nabla_x \log p(x)$"""
    y = x.clone()
    already_requires_grad = y.requires_grad
    if not already_requires_grad:
        y.requires_grad = True
    (grad_log_prob_y,) = torch.autograd.grad(p.log_prob(y).sum(), y)
    if not already_requires_grad:
        y.requires_grad = False
    return grad_log_prob_y


def log_and_grad_log(
    p: torch.distributions.Distribution, x: torch.Tensor
) -> torch.Tensor:
    r"""Helper to evaluate $\nabla_x \log p(x)$"""
    y = x.clone()
    already_requires_grad = y.requires_grad
    if not already_requires_grad:
        y.requires_grad = True
    log_prob_y = p.log_prob(y)
    (grad_log_prob_y,) = torch.autograd.grad(log_prob_y.sum(), y)
    if not already_requires_grad:
        y.requires_grad = False
    return log_prob_y.detach(), grad_log_prob_y


def compute_tvds(samples1, dist):
    samples0 = dist.sample((len(samples1),)).to(dtype=samples1.dtype)
    radius1, radius0 = samples1.norm(dim=1), samples0.norm(dim=1)

    angle1 = torch.atan2(samples1[:, 1], samples1[:, 0])
    angle0 = torch.atan2(samples0[:, 1], samples0[:, 0])

    radius_tvd = _compute_tvd_and_ksd(radius1, radius0, min_=0, max_=8)[0]
    angle_tvd = _compute_tvd_and_ksd(angle1, angle0, min_=-math.pi, max_=math.pi)[0]

    return radius_tvd, angle_tvd


# Taken from (Wu et al., 2025)
def _compute_tvd_and_ksd(
    samples, true_samples, bins=256, min_=None, max_=None, weights=None
):
    """
    Compute the total variation distance between model samples and validation samples
    """
    assert (
        len(samples.shape) == 1
    ), f"Samples should be 1D tensors, but got {samples.shape}"
    assert (
        len(true_samples.shape) == 1
    ), f"True samples should be 1D tensors, but got {true_samples.shape}"

    # Ensure tensors are on the CPU and convert to NumPy
    samples = samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples
    true_samples = (
        true_samples.cpu().numpy()
        if isinstance(true_samples, torch.Tensor)
        else true_samples
    )

    if min_ is None:
        min_ = min(samples.min(), true_samples.min())
    if max_ is None:
        max_ = max(samples.max(), true_samples.max())

    hist1, bins = np.histogram(
        samples,
        bins=bins,
        range=(min_, max_),
        weights=weights.cpu().numpy() if weights is not None else None,
    )
    hist2, _ = np.histogram(true_samples, bins=bins, range=(min_, max_))

    # Normalize histograms to get probability distributions
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)

    # Calculate the Total Variation distance
    tv_distance = 0.5 * np.sum(np.abs(hist1 - hist2))

    cdf1 = np.cumsum(hist1)
    cdf2 = np.cumsum(hist2)
    ks_distance = np.max(np.abs(cdf1 - cdf2))

    return tv_distance, ks_distance


# Divergence metrics
def compute_wasserstein(samples1, dist, method=None, reg=0.05, power=2, weights=None):
    """Compute the Wasserstein (1 or 2) distance (wrt Euclidean cost) between a source and a target
    distributions.

    Adapted from https://github.com/h2o64/slips/blob/master/slips/metrics/wasserstein.py#L71

    Args:
        samples0 (torch.Tensor of shape (bs, *dim): Represents the source minibatch
        samples1 (torch.Tensor of shape (bs, *dim): Represents the source minibatch
        method (str): Use exact Wasserstein or an entropic regularization (default is None)
        reg (float): Entropic regularization coefficients (default is 0.05)
        power (int): Power of the Wasserstein distance (1 or 2) (default is 2)
        weights (torch.Tensor of shape (batch_size,)): Optional reweighting of samples1

    Returns
        ret (float): Wasserstein distance
    """
    samples0 = dist.sample((len(samples1),)).to(dtype=samples1.dtype)

    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = ot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(ot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a = ot.unif(samples0.shape[0])
    if weights is not None:
        b = weights.cpu().numpy()
    else:
        b = ot.unif(samples1.shape[0])
    if samples0.dim() > 2:
        samples0 = samples0.reshape(samples0.shape[0], -1)
    if samples1.dim() > 2:
        samples1 = samples1.reshape(samples1.shape[0], -1)
    M = torch.cdist(samples0, samples1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))
    if power == 2:
        ret = math.sqrt(ret)
    return ret


def compute_random_proj_cdf(
    samples,
    random_projs,
    n_bins,
    min_x=None,
    max_x=None,
    weights=None,
    return_min_max=False,
):
    """Compute the CDF of randomly 1D-projected samples

    Args:
        samples (torch.Tensor of shape (batch_size, dim)): Samples to project
        random_projs (torch.Tensor of shape (n_random_projections, dim)): Random 1D projections
        n_bins (int): Number of bins for histogram computations
        min_x, max_x (float): Ranges for histogram computation (default is None)
        weights (torch.Tensor of shape (batch_size,)): Weights for histogram computation (default is None)
        return_min_max (bool): Whether to return the ranges of histogram computations

    Returns
        random_proj_cdf (torch.Tensor of shape (n_random_projections, n_bins)): CDF of the projected samples
    """

    samples_random_proj = torch.matmul(samples, random_projs[..., None])[..., 0]
    if min_x is None and max_x is None:
        min_x, max_x = (
            samples_random_proj.min(dim=-1).values,
            samples_random_proj.max(dim=-1).values,
        )
    random_proj_hist = torch.zeros((random_projs.shape[0], n_bins))
    for i in range(random_projs.shape[0]):
        random_proj_hist[i] = torch.histogram(
            samples_random_proj[i],
            bins=n_bins,
            range=(float(min_x[i]), float(max_x[i])),
            weight=weights,
        ).hist
    random_proj_hist /= random_proj_hist.sum(dim=-1)[..., None]
    random_proj_cdf = random_proj_hist.cumsum(dim=-1)
    if return_min_max:
        return random_proj_cdf, min_x, max_x
    else:
        return random_proj_cdf


def _ks(samples1, samples2, weights=None, n_random_projections=128, n_bins=256):
    """Compute the sliced Kolmogorov-Smirnov distance

    Args:
        samples1 (torch.Tensor (batch_size, dim)): Samples from the first distribution
        samples2 (torch.Tensor (batch_size, dim)): Samples from the second distribution
        weights (torch.Tensor of shape (batch_size,)): Reweithing of the samples from the second distribution
                (default is None)
        n_random_projections (int): Number of random projections (default is 128)
        n_bins (int): Number of bins in the histogram computation (default is 256)

    Returns:
        ks (float): Approximation of the sliced KS distance
    """

    # Move to CPU
    samples1, samples2 = samples1.cpu(), samples2.cpu()
    if weights is not None:
        weights = weights.cpu()
    # Define the random projections
    random_projs = torch.randn((n_random_projections, samples1.shape[-1]))
    random_projs /= torch.linalg.norm(random_projs, axis=-1)[..., None]
    # Compute the CDFs
    samples1_random_proj_cdf, min_x, max_x = compute_random_proj_cdf(
        samples1, random_projs, n_bins=n_bins, return_min_max=True
    )
    samples2_random_proj_cdf = compute_random_proj_cdf(
        samples2, random_projs, min_x=min_x, max_x=max_x, weights=weights, n_bins=n_bins
    )
    # Approximate the KS distance
    return torch.max(
        torch.abs(samples1_random_proj_cdf - samples2_random_proj_cdf), dim=-1
    ).values.mean()


def ks(x: torch.Tensor, dist: torch.distributions.Distribution) -> float:
    y = dist.sample((len(x),))
    return _ks(y, x).item()


def soft_mode_coverage(
    samples: torch.Tensor,
    means: torch.Tensor,
    var: float,
    weights: torch.Tensor = None,
    chunk_size: int = 1 << 20,
):
    r"""Compute the soft mode TVD and mode weight estimates of samples against an isotropic Gaussian mixture

    For a target $\pi(x) = \sum_i w_i N(x; m_i, \sigma^2 I)$, the recovered weight of mode i is
    the mean posterior responsibility $\hat{p}_i = \frac{1}{n} \sum_j p(i | x_j)$, and the TVD is
    $\frac{1}{2} \|\hat{p} - w\|_1$. Unlike a hard assignment, this is unbiased under mode overlap.

    Args:
        samples (torch.Tensor of shape (n, dim)): Samples to evaluate
        means (torch.Tensor of shape (K, dim)): Means of the mixture components
        var (float): Shared isotropic variance of the mixture components
        weights (torch.Tensor of shape (K,)): Mixture weights, need not be normalized
                (default is None, i.e. uniform)
        chunk_size (int): Number of entries of the (n, K) responsibility matrix per chunk
                (default is 2**20)

    Returns:
        tvd (float): Soft mode-coverage TVD
        p_hat (torch.Tensor of shape (K,)): Recovered mode weights
    """
    samples = samples.reshape(-1, samples.shape[-1])
    device, dtype = samples.device, samples.dtype
    means = means.to(device=device, dtype=dtype)
    K, d = means.shape
    if weights is None:
        weights = torch.ones(K, device=device, dtype=dtype)
    weights = (weights.to(device=device, dtype=dtype)).clone()
    weights = weights / weights.sum()
    log_w = torch.log(weights)

    n = samples.shape[0]
    resp_sum = torch.zeros(K, device=device, dtype=dtype)
    per_row = max(1, chunk_size // max(K, 1))
    for start in range(0, n, per_row):
        x = samples[start : start + per_row]
        sq = torch.cdist(x, means) ** 2.0
        logits = -0.5 * sq / var + log_w
        resp_sum += torch.softmax(logits, dim=-1).sum(dim=0)
    p_hat = resp_sum / n
    tvd = 0.5 * (p_hat - weights).abs().sum()
    return float(tvd), p_hat


def soft_mode_coverage_gmm(samples: torch.Tensor, gmm) -> float:
    """Soft mode TVD for GMM class"""
    tvd, _ = soft_mode_coverage(samples, gmm.mus, gmm.var_scaling, gmm.mix)
    return tvd


# Low-variance resampling methods
def residual_resample(w: torch.Tensor) -> torch.Tensor:
    squeeze = False
    if w.ndim == 1:
        w = w.view(1, -1)
        squeeze = True

    assert w.ndim == 2

    N, K = w.shape
    c_k = torch.floor(K * w).int()
    r_k = K * w - c_k
    R = K - torch.sum(c_k, axis=1)

    indices = []
    for i in range(N):
        i_C = torch.repeat_interleave(torch.arange(K).to(w.device), c_k[i])
        if R[i] == 0:
            indices.append(i_C)
            continue
        i_R = torch.multinomial(r_k[i], R[i], replacement=True)
        indices.append(torch.cat((i_R, i_C)))
    indices = torch.stack(indices)

    if squeeze:
        return indices.squeeze(0)
    return indices


def stratified_resample(w: torch.Tensor) -> torch.Tensor:
    squeeze = False
    if w.ndim == 1:
        w = w.view(1, -1)
        squeeze = True

    assert w.ndim == 2

    N, K = w.shape
    w_cumsum = torch.cumsum(w, 1)
    samples = (torch.rand((N, K)) + torch.arange(K).float().view(1, K)) / K
    indices = torch.searchsorted(w_cumsum, samples.to(w.device))
    indices[indices == w.shape[-1]] = w.shape[-1] - 1

    if squeeze:
        return indices.squeeze(0)
    return indices


def systematic_resample(w: torch.Tensor) -> torch.Tensor:
    squeeze = False
    if w.ndim == 1:
        w = w.view(1, -1)
        squeeze = True

    assert w.ndim == 2

    N, K = w.shape
    w_cumsum = torch.cumsum(w, 1)
    samples = (
        torch.rand(N).repeat_interleave(K).view(N, K)
        + torch.arange(K).float().view(1, K)
    ) / K
    indices = torch.searchsorted(w_cumsum, samples.to(w.device))
    indices[indices == w.shape[-1]] = w.shape[-1] - 1

    if squeeze:
        return indices.squeeze(0)
    return indices


def multinomial_resample(w: torch.Tensor) -> torch.Tensor:
    squeeze = False
    if w.ndim == 1:
        w = w.view(1, -1)
        squeeze = True

    assert w.ndim == 2

    K = w.shape[1]
    indices = torch.multinomial(w, K, replacement=True)

    if squeeze:
        return indices.squeeze(0)
    return indices
