import pathlib
from targets.target import Target
from typing import Dict
from utils.path import import_all_files_in_directory

TARGET_REGISTRY: Dict[str, Target] = {}


def register_target(name: str, target: Target) -> None:
    if name in TARGET_REGISTRY:
        raise Exception(f"Target '{name}' is already in the registry")
    TARGET_REGISTRY[name] = target
    return target


import_all_files_in_directory(pathlib.Path(__file__).parent.resolve(), "targets")
