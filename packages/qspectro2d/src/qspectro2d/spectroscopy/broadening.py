"""Inhomogeneous broadening helpers."""

from __future__ import annotations

from typing import Union

import numpy as np

__all__ = ["normalized_gauss", "sample_from_gaussian"]


def _fwhm_to_sigma(fwhm: float) -> float:
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def normalized_gauss(x_vals: np.ndarray, fwhm: float, mu: float = 0.0) -> np.ndarray:
    """Return the normalized Gaussian density with the given FWHM and center."""
    sigma_value = _fwhm_to_sigma(float(fwhm))
    if sigma_value == 0.0:
        return np.where(np.isclose(x_vals, mu), np.inf, 0.0)
    norm = 1.0 / (sigma_value * np.sqrt(2.0 * np.pi))
    exponent = -0.5 * ((x_vals - mu) / sigma_value) ** 2
    return norm * np.exp(exponent)


def sample_from_gaussian(
    n_samples: int,
    fwhm: Union[float, np.ndarray],
    mu: Union[float, np.ndarray],
    max_detuning: float = 10.0,
) -> np.ndarray:
    """Draw samples from one or more truncated Gaussian distributions."""
    if n_samples <= 0:
        return np.empty((0,), dtype=float)

    mu_array = np.asarray(mu, dtype=float)
    mu_scalar = mu_array.ndim == 0
    if mu_scalar:
        mu_array = mu_array.reshape(1)

    fwhm_array = np.asarray(fwhm, dtype=float)
    if fwhm_array.ndim == 0:
        fwhm_array = np.full_like(mu_array, float(fwhm_array))
    elif fwhm_array.shape != mu_array.shape:
        try:
            fwhm_array = np.broadcast_to(fwhm_array, mu_array.shape)
        except ValueError as exc:
            raise ValueError(
                f"fwhm with shape {np.shape(fwhm_array)} is not broadcastable to mu shape {mu_array.shape}"
            ) from exc

    if np.allclose(fwhm_array, 0.0):
        output = np.tile(mu_array, (n_samples, 1))
        return output.squeeze() if mu_scalar else output

    sigma_array = _fwhm_to_sigma(1.0) * fwhm_array
    width = max_detuning * fwhm_array
    lower = mu_array - width
    upper = mu_array + width

    sample_shape = (n_samples, mu_array.size)
    samples = np.random.normal(loc=mu_array, scale=sigma_array, size=sample_shape)
    mask = (samples < lower) | (samples > upper)

    iterations = 0
    max_iterations = 1000
    while mask.any():
        samples[mask] = np.random.normal(
            loc=mu_array[np.newaxis, :].repeat(n_samples, axis=0)[mask],
            scale=sigma_array[np.newaxis, :].repeat(n_samples, axis=0)[mask],
        )
        mask = (samples < lower) | (samples > upper)
        iterations += 1
        if iterations >= max_iterations:
            samples = np.clip(samples, lower, upper)
            break

    return samples.squeeze() if mu_scalar else samples