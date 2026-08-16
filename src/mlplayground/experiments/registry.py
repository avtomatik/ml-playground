from collections.abc import Callable
from importlib import import_module

Experiment = Callable[[], object]
EXPERIMENTS: dict[str, Experiment] = {}


def register(name: str):
    def decorator(func: Experiment) -> Experiment:
        if name in EXPERIMENTS:
            raise ValueError(f"Experiment already registered: {name}")
        EXPERIMENTS[name] = func
        return func

    return decorator


def load_experiments() -> None:
    modules = ("mlplayground.experiments.cobb_douglas.linear",)
    for module in modules:
        import_module(module)
