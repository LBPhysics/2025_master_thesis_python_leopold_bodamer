# this module provides functions to handle inhomogeneous broadening
# via Gaussian distributions characterized by FWHM and center frequencies.
# NOTE AT the moment not used
import numpy as np
from typing import Union


def _fwhm_to_sigma(fwhm: float) -> float:
    """Convert full width at half maximum (FWHM) to standard deviation σ.

    Uses the relation σ = fwhm / (2*sqrt(2*ln 2)).
    """
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def normalized_gauss(x_vals: np.ndarray, fwhm: float, mu: float = 0.0) -> np.ndarray:
    """
    Compute the normalized Gaussian density σ(x_vals - mu) with given FWHM.

    Parameters
    ----------
    x_vals : np.ndarray
        Energy value(s) at which to evaluate σ(x_vals - mu).
    fwhm : float
        Full width at half maximum of the Gaussian (same units as x and mu).
    mu : float, optional
        Center energy (default: 0.0).

    Returns
    -------
    np.ndarray
        The value(s) of σ(x_vals - mu) at x_vals.

    Notes
    -----
    The function is normalized such that ∫ σ(x - mu) dx = 1 for all fwhm.
    """
    sigma_val = _fwhm_to_sigma(float(fwhm))
    if sigma_val == 0.0:
        # Delta distribution limit; approximate by a tall, narrow spike.
        # For numerical stability, return zeros except exact matches to mu.
        return np.where(np.isclose(x_vals, mu), np.inf, 0.0)
    norm = 1.0 / (sigma_val * np.sqrt(2.0 * np.pi))
    exponent = -0.5 * ((x_vals - mu) / sigma_val) ** 2
    return norm * np.exp(exponent)


def sample_from_gaussian(
    n_samples: int,
    fwhm: Union[float, np.ndarray],
    mu: Union[float, np.ndarray],
    max_detuning: float = 10.0,
) -> np.ndarray:
    """
    Draw samples from Gaussian distributions defined by centers ``mu`` and common FWHM.

    This function supports both scalar and vector inputs:
    - If ``mu`` is a scalar, returns a 1D array of shape (n_samples,).
    - If ``mu`` is 1D array-like of length ``M``, returns a 2D array of shape (n_samples, M),
      where each column corresponds to samples around the respective center in ``mu``.

    Sampling is truncated to the interval [mu - max_detuning*fwhm, mu + max_detuning*fwhm]
    via vectorized rejection sampling. If fwhm == 0, the distribution collapses to a delta
    at ``mu`` and the function returns copies of ``mu``.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate per center.
    fwhm : float | np.ndarray
        Full width at half maximum. Can be a scalar or broadcastable to ``mu``.
    mu : float | np.ndarray
        Center(s) of the Gaussian(s). Scalar or 1D array-like.
    max_detuning : float, default=10.0
        Truncation half-width in units of fwhm.

    Returns
    -------
    np.ndarray
        Samples with shape (n_samples,) if ``mu`` is scalar, else (n_samples, len(mu)).
    """
    if n_samples <= 0:
        return np.empty((0,), dtype=float)

    mu_arr = np.asarray(mu, dtype=float)
    mu_scalar = mu_arr.ndim == 0
    if mu_scalar:
        mu_arr = mu_arr.reshape(1)

    # Broadcast fwhm to match centers
    fwhm_arr = np.asarray(fwhm, dtype=float)
    if fwhm_arr.ndim == 0:
        fwhm_arr = np.full_like(mu_arr, float(fwhm_arr))
    elif fwhm_arr.shape != mu_arr.shape:
        try:
            fwhm_arr = np.broadcast_to(fwhm_arr, mu_arr.shape)
        except ValueError as exc:
            raise ValueError("fwhm is not broadcastable to mu shape") from exc

    # Handle zero-FWHM (delta) case early
    if np.allclose(fwhm_arr, 0.0):
        out = np.tile(mu_arr, (n_samples, 1))
        return out.squeeze() if mu_scalar else out

    sigma_arr = _fwhm_to_sigma(1.0) * fwhm_arr  # vectorized via scalar factor

    # Truncation bounds per center
    width = max_detuning * fwhm_arr
    lower = mu_arr - width
    upper = mu_arr + width

    # Initial draw
    rng_shape = (n_samples, mu_arr.size)
    samples = np.random.normal(loc=mu_arr, scale=sigma_arr, size=rng_shape)

    # Rejection loop for truncation
    mask = (samples < lower) | (samples > upper)
    # Iterate until all within bounds; in practice converges very fast
    max_iters = 1000
    it = 0
    while mask.any():
        n_to_fix = mask.sum()
        # Resample only the rejected positions
        resampled = np.random.normal(
            loc=mu_arr[np.newaxis, :].repeat(n_samples, axis=0)[mask],
            scale=sigma_arr[np.newaxis, :].repeat(n_samples, axis=0)[mask],
        )
        samples[mask] = resampled
        mask = (samples < lower) | (samples > upper)
        it += 1
        if it >= max_iters:
            # Give up further tightening to avoid infinite loops; clip to bounds
            samples = np.clip(samples, lower, upper)
            break

    return samples.squeeze() if mu_scalar else samples
