import gc
import logging
from omegaconf import DictConfig
import pathlib
import time
import torch
from typing import Callable, Dict
from utils.path import import_all_files_in_directory, out_dir

EXPERIMENT_REGISTRY: Dict[str, Callable[[DictConfig], any]] = {}

logger = logging.getLogger(__name__)

def register_experiment(name: str) -> Callable[[DictConfig], any]:

    def experiment_job(fn: Callable[[DictConfig], any]) -> Callable[[DictConfig], any]:

        def wrapper(cfg: DictConfig) -> None:
            experiment_name = cfg.experiment.name
            t_start = time.time()
            logger.info(f"Started {experiment_name} experiment.")

            fn(cfg)

            t_elapsed_str = time.strftime(
                f"%H:%M:%S", time.gmtime(time.time() - t_start)
            )
            logger.info(f"Finished {experiment_name} experiment in {t_elapsed_str}.")
            logger.info(f"Output folder can be found at: {out_dir()}")
            gc.collect()

        if name in EXPERIMENT_REGISTRY:
            raise Exception(f"Experiment '{name}' is already in the registry")
        EXPERIMENT_REGISTRY[name] = wrapper

        return wrapper

    return experiment_job


import_all_files_in_directory(pathlib.Path(__file__).parent.resolve(), "experiments")
