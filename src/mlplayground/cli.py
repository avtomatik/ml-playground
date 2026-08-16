from mlplayground.experiments.registry import EXPERIMENTS, load_experiments


def main() -> None:
    load_experiments()
    for name in EXPERIMENTS:
        print(name)
