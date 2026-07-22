"""M/M/1 queue simulation helpers for CDF diagnostics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def compute_sojourn_times(inter_arrivals: np.ndarray, services: np.ndarray) -> np.ndarray:
    """Compute FIFO single-server sojourn times from arrival and service draws."""

    inter_arrivals = np.asarray(inter_arrivals, dtype=float).ravel()
    services = np.asarray(services, dtype=float).ravel()
    if inter_arrivals.shape != services.shape:
        raise ValueError("inter_arrivals and services must have the same length")
    if inter_arrivals.size == 0:
        raise ValueError("at least one customer is required")
    if np.any(inter_arrivals <= 0) or np.any(services <= 0):
        raise ValueError("queue times must be positive")

    arrivals = np.cumsum(inter_arrivals)
    departure_prev = 0.0
    sojourn = np.empty_like(arrivals)
    for idx, arrival_time in enumerate(arrivals):
        begin_service = max(arrival_time, departure_prev)
        departure_prev = begin_service + services[idx]
        sojourn[idx] = departure_prev - arrival_time
    return sojourn


def average_sojourn(inter_arrivals: np.ndarray, services: np.ndarray) -> float:
    """Return the average sojourn time for supplied draws."""

    return float(np.mean(compute_sojourn_times(inter_arrivals, services)))


def simulate_queue_output(
    rng: np.random.Generator,
    interarrival_sampler: Callable[[np.random.Generator, int], np.ndarray],
    service_sampler: Callable[[np.random.Generator, int], np.ndarray],
    n_customers: int = 10,
) -> float:
    """Generate one average-sojourn output from the true scenario distributions."""

    arrivals = interarrival_sampler(rng, n_customers)
    services = service_sampler(rng, n_customers)
    return average_sojourn(arrivals, services)


def simulate_mm1_output(
    lambda_rate: float,
    mu_rate: float,
    rng: np.random.Generator,
    n_customers: int = 10,
) -> float:
    """Generate one M/M/1 average-sojourn output."""

    if lambda_rate <= 0 or mu_rate <= 0:
        raise ValueError("rates must be positive")

    def sample_arrivals(local_rng: np.random.Generator, n: int) -> np.ndarray:
        return local_rng.exponential(scale=1.0 / lambda_rate, size=n)

    def sample_services(local_rng: np.random.Generator, n: int) -> np.ndarray:
        return local_rng.exponential(scale=1.0 / mu_rate, size=n)

    return simulate_queue_output(rng, sample_arrivals, sample_services, n_customers)


def simulate_mm1_outputs(
    lambda_rate: float,
    mu_rate: float,
    rng: np.random.Generator,
    n_outputs: int,
    n_customers: int = 10,
) -> np.ndarray:
    """Vectorized M/M/1 average-sojourn outputs for one scenario."""

    if lambda_rate <= 0 or mu_rate <= 0:
        raise ValueError("rates must be positive")
    if n_outputs <= 0:
        raise ValueError("n_outputs must be positive")
    inter_arrivals = rng.exponential(scale=1.0 / lambda_rate, size=(n_outputs, n_customers))
    services = rng.exponential(scale=1.0 / mu_rate, size=(n_outputs, n_customers))
    arrivals = np.cumsum(inter_arrivals, axis=1)
    departure_prev = np.zeros(n_outputs, dtype=float)
    sojourn = np.empty_like(arrivals)
    for idx in range(n_customers):
        begin_service = np.maximum(arrivals[:, idx], departure_prev)
        departure_prev = begin_service + services[:, idx]
        sojourn[:, idx] = departure_prev - arrivals[:, idx]
    return np.mean(sojourn, axis=1)


def oracle_mm1_outputs(
    lambda_rate: float,
    mu_rate: float,
    rng: np.random.Generator,
    n_outputs: int,
    n_customers: int = 10,
) -> np.ndarray:
    """Monte Carlo sample from the true output law for one M/M/1 scenario."""

    return simulate_mm1_outputs(lambda_rate, mu_rate, rng, n_outputs, n_customers)
