import pathlib
from samplers.sampler import Sampler, SamplerConfig
from typing import Callable, Dict, Tuple
from utils.path import import_all_files_in_directory

SAMPLER_REGISTRY: Dict[
    str, Tuple[Sampler, Callable[[Dict[str, any]], Dict[str, any]]]
] = {}


def register_sampler(name: str, config: SamplerConfig) -> Callable[[Sampler], Sampler]:

    def register(sampler: Sampler) -> Sampler:
        if name in SAMPLER_REGISTRY:
            raise Exception(f"Sampler '{name}' is already in the registry")
        config_resolver = config.get_resolver(name)
        SAMPLER_REGISTRY[name] = (sampler, config_resolver)
        return sampler

    return register


import_all_files_in_directory(pathlib.Path(__file__).parent.resolve(), "samplers")
