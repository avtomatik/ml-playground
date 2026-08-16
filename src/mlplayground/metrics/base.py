from dataclasses import dataclass


@dataclass(frozen=True)
class RegressionResult:
    model_name: str
    r2: float
    mse: float
    coefficients: tuple[float, ...] | None
