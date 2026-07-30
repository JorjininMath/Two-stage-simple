"""Scenario-level M/M/1 DGP with finite input samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mm1 import simulate_mm1_output
from .mm1 import simulate_mm1_outputs


INPUT_SIZE_CHOICES = (20, 50, 200, 1000)


@dataclass
class Scenario:
    """One exchangeable queueing scenario.

    The algorithm observes finite input samples and one output. True parameters
    are retained only for oracle CDF diagnostics.
    """

    n_a: int
    n_s: int
    inter_arrivals: np.ndarray
    services: np.ndarray
    y: float
    lambda_rate: float
    mu_rate: float
    rho: float


def draw_rho(rng: np.random.Generator) -> float:
    """Draw scenario traffic intensity from a light/heavy mixture."""

    if rng.random() < 0.7:
        return float(rng.uniform(0.35, 0.75))
    return float(rng.uniform(0.75, 0.95))


def draw_input_sizes(rng: np.random.Generator) -> tuple[int, int]:
    """Draw finite input sample sizes."""

    n_a = int(rng.choice(INPUT_SIZE_CHOICES))
    return n_a, 2 * n_a


def generate_mm1_scenario(rng: np.random.Generator) -> Scenario:
    """Generate one M/M/1 scenario with finite input samples."""

    n_a, n_s = draw_input_sizes(rng)
    rho = draw_rho(rng)
    mu_rate = float(rng.uniform(0.8, 1.2))
    lambda_rate = rho * mu_rate
    inter_arrivals = rng.exponential(scale=1.0 / lambda_rate, size=n_a)
    services = rng.exponential(scale=1.0 / mu_rate, size=n_s)
    y = simulate_mm1_output(lambda_rate, mu_rate, rng)
    return Scenario(
        n_a=n_a,
        n_s=n_s,
        inter_arrivals=inter_arrivals,
        services=services,
        y=y,
        lambda_rate=lambda_rate,
        mu_rate=mu_rate,
        rho=rho,
    )


def generate_training_scenarios(n: int, rng: np.random.Generator) -> list[Scenario]:
    """Generate exchangeable training scenarios for the CKME-CDF estimator."""

    return [generate_mm1_scenario(rng) for _ in range(n)]


def generate_scenarios(n: int, rng: np.random.Generator, dgp: str = "exp_mm1") -> list[Scenario]:
    """Generate scenario-level observations for the named DGP."""

    if dgp != "exp_mm1":
        raise ValueError(f"unsupported DGP: {dgp}")
    return [generate_mm1_scenario(rng) for _ in range(n)]


def scenario_outputs(scenario: Scenario, rng: np.random.Generator, n_outputs: int) -> np.ndarray:
    """Generate fresh queue outputs from one scenario's hidden distributions."""

    return simulate_mm1_outputs(scenario.lambda_rate, scenario.mu_rate, rng, n_outputs)


def training_output_reps(
    scenarios: list[Scenario],
    rng: np.random.Generator,
    r_train: int,
) -> np.ndarray:
    """Return output replications with shape ``(n_scenarios, r_train)``."""

    if r_train <= 0:
        raise ValueError("r_train must be positive")
    outputs = np.empty((len(scenarios), r_train), dtype=float)
    for idx, scenario in enumerate(scenarios):
        outputs[idx, 0] = scenario.y
        if r_train > 1:
            outputs[idx, 1:] = scenario_outputs(scenario, rng, r_train - 1)
    return outputs


def find_representative_scenario(
    rng: np.random.Generator,
    target: str = "highU_heavy",
    max_attempts: int = 10_000,
) -> Scenario:
    """Find a scenario for visual diagnostics.

    Defaults to a hard case: small finite input sample and high estimated true
    traffic. This makes input uncertainty visible in the CDF envelope.
    """

    for _ in range(max_attempts):
        scenario = generate_mm1_scenario(rng)
        high_u = scenario.n_a in {20, 50}
        heavy = scenario.rho > 0.75
        label = ("highU" if high_u else "lowU") + "_" + ("heavy" if heavy else "light")
        if label == target:
            return scenario
    raise RuntimeError(f"could not find scenario with label {target}")
