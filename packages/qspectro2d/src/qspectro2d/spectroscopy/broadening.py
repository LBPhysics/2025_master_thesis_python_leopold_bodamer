"""Inhomogeneous broadening helpers."""

from __future__ import annotations

from typing import Union

import numpy as np

__all__ = [
    "normalized_gauss",
    "sample_static_disorder",
    "sample_from_gaussian",
    "sample_from_correlated_gaussian",
]


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


def _normalize_mean_and_width(
    fwhm: Union[float, np.ndarray],
    mu: Union[float, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, bool]:
    mu_array = np.asarray(mu, dtype=float)
    mu_scalar = mu_array.ndim == 0
    if mu_scalar:
        mu_array = mu_array.reshape(1)

    dim = mu_array.size

    fwhm_array = np.asarray(fwhm, dtype=float)
    if fwhm_array.ndim == 0:
        fwhm_array = np.full(dim, float(fwhm_array))
    elif fwhm_array.shape != (dim,):
        try:
            fwhm_array = np.broadcast_to(fwhm_array, (dim,))
        except ValueError as exc:
            raise ValueError(
                f"fwhm with shape {np.shape(fwhm_array)} is not broadcastable to shape {(dim,)}"
            ) from exc

    return mu_array, fwhm_array, mu_scalar


def _normalize_correlation(
    corr: Union[float, np.ndarray, None],
    *,
    dim: int,
) -> np.ndarray:
    if corr is None:
        return np.eye(dim, dtype=float)

    corr_array = np.asarray(corr, dtype=float)
    if corr_array.ndim == 0:
        if dim != 2:
            raise ValueError("Scalar corr is only valid for dim=2.")
        rho = float(corr_array)
        corr_array = np.array([[1.0, rho], [rho, 1.0]], dtype=float)

    if corr_array.shape != (dim, dim):
        raise ValueError(f"corr must have shape {(dim, dim)}, got {corr_array.shape}")

    if not np.allclose(corr_array, corr_array.T):
        raise ValueError("corr must be symmetric.")

    if not np.allclose(np.diag(corr_array), 1.0):
        raise ValueError("corr must have ones on the diagonal.")

    return corr_array


def sample_static_disorder(
    n_samples: int,
    fwhm: Union[float, np.ndarray],
    mu: Union[float, np.ndarray],
    *,
    corr: Union[float, np.ndarray, None] = None,
    max_detuning: float = 10.0,
) -> np.ndarray:
    """Draw truncated static-disorder samples for one or more site energies.

    ``corr=None`` means independent site-energy disorder. Otherwise ``corr`` is
    interpreted as a correlation matrix, with scalar shorthand supported for
    dimers.
    """
    mu_array, fwhm_array, _ = _normalize_mean_and_width(fwhm, mu)
    dim = mu_array.size

    if n_samples <= 0:
        return np.empty((0, dim), dtype=float)

    sigma_array = _fwhm_to_sigma(1.0) * fwhm_array
    if np.allclose(sigma_array, 0.0):
        return np.tile(mu_array, (n_samples, 1))

    corr_array = _normalize_correlation(corr, dim=dim)
    cov = np.outer(sigma_array, sigma_array) * corr_array

    eigvals = np.linalg.eigvalsh(cov)
    if np.min(eigvals) < -1e-12:
        raise ValueError("Covariance matrix is not positive semidefinite.")

    width = max_detuning * fwhm_array
    lower = mu_array - width
    upper = mu_array + width

    samples = np.random.multivariate_normal(mean=mu_array, cov=cov, size=n_samples)
    mask = ((samples < lower) | (samples > upper)).any(axis=1)

    iterations = 0
    max_iterations = 1000
    while mask.any():
        samples[mask] = np.random.multivariate_normal(
            mean=mu_array,
            cov=cov,
            size=int(mask.sum()),
        )
        mask = ((samples < lower) | (samples > upper)).any(axis=1)
        iterations += 1
        if iterations >= max_iterations:
            samples = np.clip(samples, lower, upper)
            break

    return samples


def sample_from_gaussian(
    n_samples: int,
    fwhm: Union[float, np.ndarray],
    mu: Union[float, np.ndarray],
    max_detuning: float = 10.0,
) -> np.ndarray:
    """Draw samples from one or more truncated independent Gaussian distributions."""
    _, _, mu_scalar = _normalize_mean_and_width(fwhm, mu)
    samples = sample_static_disorder(
        n_samples=n_samples,
        fwhm=fwhm,
        mu=mu,
        corr=None,
        max_detuning=max_detuning,
    )
    return samples[:, 0] if mu_scalar else samples


def sample_from_correlated_gaussian(
    n_samples: int,
    fwhm: Union[float, np.ndarray],
    mu: Union[float, np.ndarray],
    corr: Union[float, np.ndarray],
    max_detuning: float = 10.0,
) -> np.ndarray:
    """Draw samples from a truncated multivariate Gaussian with correlations.

    Parameters
    ----------
    n_samples
        Number of realizations.
    fwhm
        FWHM per site or a scalar applied to all sites.
    mu
        Mean frequencies per site.
    corr
        Correlation coefficient matrix. For a dimer a scalar is also accepted:
        corr = +1 -> fully correlated
        corr =  0 -> uncorrelated
        corr = -1 -> fully anticorrelated
    """
    return sample_static_disorder(
        n_samples=n_samples,
        fwhm=fwhm,
        mu=mu,
        corr=corr,
        max_detuning=max_detuning,
    )
