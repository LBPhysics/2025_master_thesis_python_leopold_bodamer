from numpy.typing import ArrayLike
import numpy as np
from qutip.utilities import n_thermal
from qutip import BosonicEnvironment

"""
This module contains functions for calculating spectral density and power spectrum
for various types of bosonic baths, including Drude-Lorentz and ohmic baths.
It mirrors the structure of the QuTip library's bath functions
"""


def spectral_density_func_drude_lorentz(w: float | ArrayLike, **args) -> float | ArrayLike:
    """
    Spectral density function for a Drude-Lorentz bath.
    Compatible with scalar and array inputs.
    """
    alpha = args["alpha"]
    cutoff = args["cutoff"]
    lambda_ = alpha * cutoff / 2  # Reorganization energy (coupling strength)
    gamma = cutoff  # Drude decay rate (cutoff frequency)
    w_input = w  # Store original input
    w = np.asarray(w, dtype=float)
    result = (2 * lambda_ * gamma * w) / (w**2 + gamma**2)

    # Return scalar if input was scalar
    if np.isscalar(w_input):
        return float(result)
    return result


def power_spectrum_func_drude_lorentz(w: float | ArrayLike, **args) -> float | ArrayLike:
    """
    power spectrum function in the frequency domain for an drude lorentzian bath.
    Handles both positive and negative frequencies, compatible with arrays.
    """
    temp = args["temp"]

    Boltzmann = args["Boltzmann"] if "Boltzmann" in args else 1.0
    hbar = args["hbar"] if "hbar" in args else 1.0

    w_input = w  # Store original input
    w = np.asarray(w, dtype=float)

    # Avoid division by zero in tanh
    w_safe = np.where(w == 0, 1e-10, w)
    w_th = Boltzmann * temp / hbar  # Thermal energy in frequency units
    coth_term = 1 / np.tanh(w_safe / (2 * w_th))

    result = np.sign(w) * spectral_density_func_drude_lorentz(np.abs(w), **args) * (coth_term + 1)

    # Return scalar if input was scalar
    if np.isscalar(w_input):
        return float(result)
    return result


def spectral_density_func_ohmic(w: float | ArrayLike, **args) -> float | ArrayLike:
    """
    Spectral density function for an ohmic bath.
    Compatible with scalar and array inputs.
    """
    wc = args["cutoff"]
    alpha = args["alpha"]
    s = args["s"] if "s" in args else 1.0  # Default to ohmic (s=1)

    w = np.asarray(w, dtype=float)
    result = np.zeros_like(w)

    positive_mask = w > 0
    w_mask = w[positive_mask]
    result[positive_mask] = alpha * w_mask**s / (wc ** (s - 1)) * np.exp(-np.abs(w_mask) / wc)

    return result.item() if w.ndim == 0 else result


def power_spectrum_func_ohmic(w: float | ArrayLike, **args) -> float | ArrayLike:
    """
    power spectrum function in the frequency domain for an ohmic bath.
    Handles both positive and negative frequencies, compatible with arrays.
    """
    s = args["s"] if "s" in args else 1.0  # Default to ohmic (s=1)
    if s > 1:
        sd_derivative = 0
    elif s == 1:
        sd_derivative = args["alpha"]
    else:
        sd_derivative = None  # I changed from np.inf to None
    temp = args["temp"]

    # Boltzmann = args["Boltzmann"] if "Boltzmann" in args else 1.0
    # hbar = args["hbar"] if "hbar" in args else 1.0

    # derivative: value of J'(0)
    if temp is None:
        raise ValueError("The temperature must be specified for this operation.")

    w = np.asarray(w, dtype=float)
    if temp == 0:
        return 2 * np.heaviside(w, 0) * spectral_density_func_ohmic(w, **args)

    # at zero frequency, we do numerical differentiation
    # S(0) = 2 J'(0) / beta
    zero_mask = w == 0
    nonzero_mask = np.invert(zero_mask)

    S = np.zeros_like(w)
    if sd_derivative is None:
        eps = 1e-10  # Small value to avoid division by zero
        S[zero_mask] = 2 * temp * spectral_density_func_ohmic(eps, **args) / eps
    else:
        S[zero_mask] = 2 * temp * sd_derivative
    S[nonzero_mask] = (
        2
        * np.sign(w[nonzero_mask])
        * spectral_density_func_ohmic(np.abs(w[nonzero_mask]), **args)
        * (n_thermal(w[nonzero_mask], temp) + 1)
    )
    return S.item() if w.ndim == 0 else S


# BATH FUNCTIONS as defined in the paper


def spectral_density_func_paper(w: float | ArrayLike, **args) -> float | ArrayLike:
    """
    Spectral density function for a bath as given in the paper.
    Compatible with scalar and array inputs.

    Parameters:
    -----------
    w : array_like
        Frequency array
    **args : dict
        Parameters dictionary containing 'alpha' and 'cutoff'

    Returns:
    --------
    array_like
        Spectral density values
    """
    alpha = args["alpha"]
    cutoff = args["cutoff"]

    w_input = w  # Store original input
    w = np.asarray(w, dtype=float)

    result = alpha * (w / cutoff) * np.exp(-w / cutoff) * (w > 0)

    # Return scalar if input was scalar
    if np.isscalar(w_input):
        return float(result)
    return result


def power_spectrum_func_paper(w: float | ArrayLike, **args) -> float | ArrayLike:
    """
    power spectrum function in the frequency domain as given in the paper.
    Compatible with scalar and array inputs.

    Parameters:
    -----------
    w : array_like
        Frequency array
    **args : dict
        Parameters dictionary containing 'temp', 'alpha', 'cutoff',
        and optionally 'Boltzmann' and 'hbar'

    Returns:
    --------
    array_like
        Power spectrum values
    """
    # Extract constants from args
    Boltzmann = args.get("Boltzmann", 1.0)
    hbar = args.get("hbar", 1.0)

    temp = args["temp"]
    alpha = args["alpha"]
    cutoff = args["cutoff"]

    w_th = Boltzmann * temp / hbar  # Thermal energy in frequency units

    w_input = w  # Store original input
    w = np.asarray(w, dtype=float)
    result = np.zeros_like(w)

    # Positive frequency
    pos_mask = w > 0
    neg_mask = w < 0
    result[pos_mask] = (1 + n_thermal(w[pos_mask], w_th)) * spectral_density_func_paper(
        w[pos_mask], **args
    )

    # Negative frequency
    result[neg_mask] = n_thermal(-w[neg_mask], w_th) * spectral_density_func_paper(
        -w[neg_mask], **args
    )

    # Zero frequency C(0)
    zero_mask = w == 0
    result[zero_mask] = alpha * w_th / cutoff
    # Return scalar if input was scalar
    if np.isscalar(w_input):
        return 2 * float(result)
    return 2 * result


# Convert bath coupling constant alpha to ME rates and vice versa
def bath_to_rates(
    env: BosonicEnvironment, w: float = None, mode: str = "decay"
) -> tuple[float, float] | float:
    """
    Wrapper to convert bath coupling constant alpha to ME rates.
    Args:
        alpha: Coupling constant of the bath.
        env: BosonicEnvironment instance with the bath parameters.
        w: System transition frequency (difference of two energy levels) (required for decay mode).
        mode: 'decay' for decay rates, 'deph' for dephasing rate.
    Returns:
        Decay rates (emission_rate, gamma_absorption) or dephasing rate (deph_rate).
    """
    if mode == "decay":
        if w is None:
            raise ValueError("System frequency w must be provided for decay mode.")
        return bath_to_decay_rates(env, w)
    elif mode == "deph":
        return bath_to_dephasing_rate(env)
    else:
        raise ValueError("Invalid mode. Use 'decay' or 'deph'.")


def rates_to_alpha(
    rate: float | tuple[float, float],
    env: BosonicEnvironment,
    w: float = None,
    wc=None,
    mode: str = "decay",
) -> float:
    """
    Wrapper to convert ME rates to bath coupling constant alpha.
    Args:
        rate: Decay rates (emission_rate, gamma_absorption) or dephasing rate (deph_rate).
        env: BosonicEnvironment instance with the bath parameters.
        w: System frequency (required for decay mode).
        mode: 'decay' for decay rates, 'deph' for dephasing rate.
    Returns:
        alpha: Coupling constant of the bath.
    """
    if mode == "decay":
        if wc is None:
            raise ValueError("Cutoff frequency wc must be provided for decay mode.")
        return decay_rates_to_alpha(rate, env, w, wc)
    elif mode == "deph":
        if not isinstance(rate, float):
            raise ValueError("Rate must be a float (deph_rate) for dephasing mode.")
        return dephasing_rate_to_alpha(rate, env)
    else:
        raise ValueError("Invalid mode. Use 'decay' or 'deph'.")


def bath_to_decay_rates(env: BosonicEnvironment, w: float) -> tuple[float, float]:
    """
    Convert bath coupling constant alpha to ME decay channel rates.
    Args:
        alpha: Coupling constant of the bath.
        env: BosonicEnvironment instance with the bath parameters.
        w: System frequency.
    Returns:
        emission_rate: Spontaneous emission rate.
        gamma_absorption: Thermal absorption rate.
    """
    P_plus = env.power_spectrum(w)  # S(+ω) - emission rate
    P_minus = env.power_spectrum(-w)  # S(-ω) - absorption rate
    return P_plus, P_minus


def decay_rates_to_alpha(
    emission_rate: float, env: BosonicEnvironment, w: float, wc: float
) -> float:
    """
    Convert ME decay channel rates to bath coupling constant alpha.
    Args:
        emission_rate: Spontaneous emission rate.
        env: BosonicEnvironment instance with the bath parameters.
        w: System frequency.
    Returns:
        alpha: Coupling constant of the bath.
    """
    # Avoid division by zero if w is very small
    if w < 1e-12:
        raise ValueError("w is too small; cannot determine alpha reliably.")

    temp = env.T

    P_plus = emission_rate
    # Use constants directly to avoid circular import
    BOLTZMANN_LOCAL = 1.0  # Using normalized units
    HBAR_LOCAL = 1.0  # Using normalized units

    w_th = BOLTZMANN_LOCAL * temp / HBAR_LOCAL
    alpha = P_plus / (2 * w * np.exp(-w / wc) * (n_thermal(w, w_th) + 1))
    return alpha


def bath_to_dephasing_rate(env: BosonicEnvironment) -> float:
    """
    Convert bath coupling constant alpha to ME dephasing rate.
    Args:
        alpha: Coupling constant of the bath.
        env: BosonicEnvironment instance with the bath parameters.
    Returns:
        deph_rate: Pure dephasing rate.
    """
    P_zero = env.power_spectrum(0)  # S(0) - dephasing rate
    deph_rate = P_zero
    return deph_rate


def dephasing_rate_to_alpha(deph_rate: float, env: BosonicEnvironment) -> float:
    """
    Convert ME dephasing rate to bath coupling constant alpha.
    Args:
        deph_rate: Pure dephasing rate.
        env: BosonicEnvironment instance with the bath parameters.
    Returns:
        alpha: Coupling constant of the bath.
    """
    # Use constant directly to avoid circular import
    BOLTZMANN_LOCAL = 1.0  # Using normalized units

    temp = env.T
    P_zero = deph_rate
    alpha = P_zero / (2 * BOLTZMANN_LOCAL * temp)

    return alpha


def extract_bath_parameters(bath, w0=None) -> dict:
    """
    Extract parameters from a QuTip Environment instance for serialization/plotting.

    Args:
        bath: QuTip Environment instance (BosonicEnvironment or subclass)

    Returns:
        dict: Dictionary containing the extractable bath parameters
    """
    import qutip

    # Base parameters available in all BosonicEnvironment instances
    params = {
        "T": getattr(bath, "T", None),  # Temperature
        "temp": getattr(bath, "T", None),
        "tag": getattr(bath, "tag", None),  # Tag/identifier
        "S(0)": getattr(bath, "power_spectrum", lambda x: None)(0),
    }
    if w0 is not None:
        params["S(w0)"] = getattr(bath, "power_spectrum", lambda x: None)(w0)
    # Extract specific parameters based on environment type
    if isinstance(bath, qutip.DrudeLorentzEnvironment):
        params.update(
            {
                "lam": getattr(bath, "lam", None),  # Coupling strength
                "gamma": getattr(bath, "gamma", None),  # Cutoff frequency
                "cutoff": getattr(bath, "gamma", None),
                "Nk": getattr(bath, "Nk", None),  # Number of Pade exponents
            }
        )
    elif isinstance(bath, qutip.OhmicEnvironment):
        params.update(
            {
                "alpha": getattr(bath, "alpha", None),  # Coupling strength
                "wc": getattr(bath, "wc", None),  # Cutoff parameter
                "cutoff": getattr(bath, "wc", None),
                "s": getattr(bath, "s", None),  # Power of omega
            }
        )
    elif isinstance(bath, qutip.UnderDampedEnvironment):
        params.update(
            {
                "lam": getattr(bath, "lam", None),  # Coupling strength
                "gamma": getattr(bath, "gamma", None),  # Damping rate
                "w0": getattr(bath, "w0", None),  # Resonance frequency
            }
        )
    elif isinstance(bath, qutip.ExponentialBosonicEnvironment):
        params.update({"exponents": getattr(bath, "exponents", None)})  # CF exponents

    # Remove None values to keep dict clean
    return {k: v for k, v in params.items() if v is not None}
