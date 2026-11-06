from experiments import register_experiment
import json
from omegaconf import DictConfig, open_dict
import os
from samplers import SAMPLER_REGISTRY
from targets import TARGET_REGISTRY
import torch
from tqdm import tqdm
from utils.distributions import (
    ks,
    compute_wasserstein,
    compute_tvds,
    soft_mode_coverage_gmm,
)
from utils.path import out_dir


@register_experiment("sample")
def sample(cfg: DictConfig) -> None:
    cfg.sampler.device = cfg.device

    target_cfg = cfg.target
    sampler_cfg = cfg.sampler

    if target_cfg.name not in TARGET_REGISTRY:
        raise Exception(
            f"Target '{target_cfg.name}' not found. Consider: {', '.join(TARGET_REGISTRY.keys())}"
        )

    target = TARGET_REGISTRY.get(target_cfg.name)
    if "dald" in sampler_cfg.name:
        with open_dict(sampler_cfg):
            sampler_cfg["T"] = target_cfg["xi"] * (
                (
                    sampler_cfg["n_steps"]
                    * target.exp_squared_norm()
                    / target.dimension()
                )
                ** (1 / 3)
            )

    sampler_cls, config_resolver = SAMPLER_REGISTRY.get(sampler_cfg.name)
    sampler_params = config_resolver(sampler_cfg)
    sampler = sampler_cls(**sampler_params)

    n_samples = cfg.sampler.n_samples

    torch.manual_seed(cfg.seed)
    samples = sampler.sample(n_samples, target.to(cfg.device)).cpu()

    out = out_dir()
    samples_dir = os.path.join(out, "samples")
    os.mkdir(samples_dir)
    torch.save(samples, os.path.join(samples_dir, "samples.pt"))

    stats_dir = os.path.join(out, "stats")
    os.mkdir(stats_dir)

    res_dict = {}
    samples_device = samples.to(cfg.device)

    bar = tqdm(
        range(1),
        "Computing divergence metrics",
        miniters=1,
        mininterval=0,
    )

    metrics = {}

    if "rdsmc" in sampler_cfg.name:
        metrics["log_Z_bias"] = abs(sampler.log_Z_hat.item())

    if target_cfg.name in {"gmm40_dim_2", "gmm40_dim_50", "rings"}:
        target_dist = target.to(cfg.device).as_torch_distribution()
        metrics["ws2"] = compute_wasserstein(samples_device, target_dist)
    if target_cfg.name in {"gmm40_dim_2", "gmm40_dim_50"}:
        metrics["mode_tvd_soft"] = soft_mode_coverage_gmm(
            samples_device, target.to(cfg.device).gmm
        )
    if target_cfg.name == "rings":
        target_dist = target.to(cfg.device).as_torch_distribution()
        metrics["radius_tvd"], metrics["angle_tvd"] = compute_tvds(
            samples_device, target_dist
        )
    elif target_cfg.name == "funnel":
        target_dist = target.to(cfg.device).as_torch_distribution()
        metrics["ks"] = ks(samples_device, target_dist)
    elif target_cfg.name in {"ionosphere", "sonar"}:
        metrics["pred_val_log_prob"] = float(target.val.log_prob(samples).mean())
        metrics["pred_target_log_prob"] = float(target.test.log_prob(samples).mean())

    res_dict = {**res_dict, **metrics}
    bar.update(1)
    bar.close()

    print(
        f'\n{"  ".join(f"{k}: {v:.5f}" for k, v in res_dict.items() if isinstance(v, (int, float)))}\n'
    )
    with open(os.path.join(stats_dir, "stats.json"), "w") as f:
        json.dump(res_dict, f, indent=2)

    dict_export = [
        {
            "algorithm_name": "dpsmc",
            "target_type": target_cfg.name,
            "n_samples": sampler_cfg.n_samples,
            "K": sampler_cfg.n_steps,
            "n_mcmc_steps": 1,
            "params": None,
            "metric": {**metrics},
            "X_traj": getattr(sampler, "X_traj", None),
            "ess": sampler.ess_traj.cpu(),
        }
    ]

    import pickle

    with open(
        os.path.join(stats_dir, f"dpsmc_{target_cfg.name}_{cfg.seed}.pkl"), "wb"
    ) as f:
        pickle.dump(dict_export, f)
