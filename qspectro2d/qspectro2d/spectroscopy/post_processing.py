"""Concise post-processing utilities for extending time-domain data."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


__all__ = [
    "extend_time_domain_data",
    "compute_spectra",
]


def _extend_axis(axis: np.ndarray, new_len: int) -> np.ndarray:
    """Extend a strictly increasing 1D axis to ``new_len`` in + direction."""
    if axis.ndim != 1:
        raise ValueError("axis must be 1D")
    n_old = int(axis.size)
    n_new = int(new_len)
    if n_new <= n_old:
        return axis.copy()
    if n_old == 0:
        raise ValueError("axis must have at least one element")

    if n_old == 1:
        last = float(axis[-1])
        extra = np.full(n_new - n_old, last, dtype=float)
        return np.concatenate([axis.astype(float, copy=True), extra])

    diffs = np.diff(axis)
    if not np.all(diffs > 0):
        raise ValueError("axis must be strictly increasing")
    dt = float(np.median(diffs))
    last = float(axis[-1])
    extra_n = n_new - n_old
    cont = last + dt * np.arange(1, extra_n + 1, dtype=float)
    return np.concatenate([axis.astype(float, copy=True), cont])


def extend_time_domain_data(
    datas: List[np.ndarray],
    t_det: np.ndarray,
    t_coh: Optional[np.ndarray] = None,
    *,
    pad: float = 1.0,
) -> Tuple[List[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """Extend axes by ceil(pad * N) and zero-pad data on the positive side.

    - 1D (t_coh is None): data shape (N_det,), extend det-axis only.
    - 2D (t_coh provided): data shape (N_coh, N_det), extend both axes.
    """
    if pad < 1.0:
        raise ValueError("pad must be >= 1.0 for extension-only padding")
    if not datas:
        raise ValueError("datas must be a non-empty list of arrays")
    if t_det.ndim != 1:
        raise ValueError("t_det must be a 1D array")

    n_det = int(t_det.size)
    n_det_ext = int(np.ceil(pad * n_det))

    if t_coh is None:
        extended_t_det = _extend_axis(t_det, n_det_ext)
        extended_datas = []
        for idx, arr in enumerate(datas):
            if arr.ndim != 1:
                raise ValueError(f"datas[{idx}] must be 1D when t_coh is None")
            if arr.shape[0] != n_det:
                raise ValueError(
                    f"datas[{idx}] length {arr.shape[0]} does not match t_det length {n_det}"
                )
            extra = n_det_ext - n_det
            if extra <= 0:
                extended_datas.append(arr.copy())
                continue
            pad_width = (0, extra)
            padded = np.pad(arr, pad_width, mode="constant", constant_values=0)
            extended_datas.append(padded)

        return extended_datas, extended_t_det, None

    if t_coh.ndim != 1:
        raise ValueError("t_coh must be a 1D array when provided")

    n_coh = int(t_coh.size)
    n_coh_ext = int(np.ceil(pad * n_coh))

    extended_t_det = _extend_axis(t_det, n_det_ext)
    extended_t_coh = _extend_axis(t_coh, n_coh_ext)

    extended_datas: List[np.ndarray] = []
    for idx, arr in enumerate(datas):
        if arr.ndim != 2:
            raise ValueError(f"datas[{idx}] must be 2D when t_coh is provided")
        if arr.shape != (n_coh, n_det):
            raise ValueError(
                f"datas[{idx}] shape {arr.shape} does not match (len(t_coh), len(t_det)) = {(n_coh, n_det)}"
            )
        extra_c = n_coh_ext - n_coh
        extra_d = n_det_ext - n_det
        if extra_c <= 0 and extra_d <= 0:
            extended_datas.append(arr.copy())
            continue
        pad_width = ((0, max(0, extra_c)), (0, max(0, extra_d)))
        padded = np.pad(arr, pad_width, mode="constant", constant_values=0)
        extended_datas.append(padded)

    return extended_datas, extended_t_det, extended_t_coh


def compute_spectra(
    datas: List[np.ndarray],
    signal_types: List[str] = ["rephasing"],
    t_det: np.ndarray = None,
    t_coh: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray, List[np.ndarray], List[str]]:
    """Compute spectra along detection (and optional coherence) axes.
    Based on the paper: https://doi.org/10.1063/5.0214023

    For each input data array:
    - Along detection time, always use +i convention: S(w_det) = ∫ E(t) e^{+i w t} dt
      (implemented via IFFT, no normalization scaling applied).
    - If coherence axis is present:
        - nonrephasing:  S_NR(w_coh, *) = ∫ E(t_coh, *) e^{-i w t} dt (FFT)
        - rephasing/else: S_R(w_coh, *) = ∫ E(t_coh, *) e^{+i w t} dt (IFFT)

    Returns:
        (nu_cohs, nu_dets, datas_nu, signal_types_out)
        - nu_cohs: frequency axis (cm^-1) for coherence (or None if t_coh is None)
        - nu_dets: frequency axis (cm^-1) for detection
        - datas_nu: list of spectra arrays in frequency domain
        - signal_types_out: list of signal labels, aligned with datas_nu

    Notes:
        Frequency-to-wavenumber conversion follows the user's formula:
            nu = np.fft.fftfreq(N, d=dt) / 2.998 * 10
    """
    if not datas:
        raise ValueError("datas must be a non-empty list of arrays")
    if t_det is None:
        raise ValueError("t_det must be provided")
    if t_det.ndim != 1:
        raise ValueError("t_det must be a 1D array")

    # Normalize signal types to length of datas.
    if len(signal_types) == 1 and len(datas) > 1:
        sig_types = [signal_types[0]] * len(datas)
    else:
        if len(signal_types) != len(datas):
            raise ValueError(
                "signal_types must have same length as datas or be a single entry"
            )
        sig_types = list(signal_types)

    # Detection axis frequency and wavenumber
    n_det = int(t_det.size)
    dt_det = float(np.median(np.diff(t_det))) if n_det > 1 else 1.0
    freq_dets = np.fft.fftfreq(n_det, d=dt_det)
    nu_dets = freq_dets / 2.998 * 10

    # Coherence axis frequency and wavenumber (optional)
    if t_coh is None:
        nu_cohs = None
    else:
        if t_coh.ndim != 1:
            raise ValueError("t_coh must be 1D when provided")
        n_coh = int(t_coh.size)
        dt_coh = float(np.median(np.diff(t_coh))) if n_coh > 1 else 1.0
        freq_cohs = np.fft.fftfreq(n_coh, d=dt_coh)
        nu_cohs = freq_cohs / 2.998 * 10

    # Build spectra
    datas_nu: List[np.ndarray] = []
    out_types: List[str] = []
    for idx, (arr, stype) in enumerate(zip(datas, sig_types)):
        st_norm = str(stype).strip().lower()
        if t_coh is None:
            if arr.ndim != 1 or arr.shape[0] != n_det:
                raise ValueError(
                    f"datas[{idx}] must be 1D with length len(t_det) when t_coh is None"
                )
            spec_det = np.fft.ifft(arr, axis=0)
            datas_nu.append(spec_det)
            out_types.append(stype)
            continue

        # 2D case: (N_coh, N_det)
        n_coh = int(t_coh.size)
        if arr.ndim != 2 or arr.shape != (n_coh, n_det):
            raise ValueError(
                f"datas[{idx}] must be 2D with shape (len(t_coh), len(t_det))"
            )

        # Detection axis (+i) via IFFT along last axis
        spec_2d = np.fft.ifft(arr, axis=1)

        # Coherence axis sign depends on signal type
        if st_norm == "nonrephasing":
            spec_2d = np.fft.fft(spec_2d, axis=0)
            out_types.append("nonrephasing")
        else:
            spec_2d = np.fft.ifft(spec_2d, axis=0)
            out_types.append("rephasing")

        datas_nu.append(spec_2d)

    # NOTE Optional absorptive combination (commented per request)
    # if t_coh is not None:
    #     have_r  = any(t.lower().startswith("rephasing") for t in out_types)
    #     have_nr = any(t.lower().startswith("nonrephasing") for t in out_types)
    #     if have_r and have_nr:
    #         # Example: match by index or by separate provided lists;
    #         # here we simply combine the first found pair.
    #         try:
    #             r_idx = next(i for i, t in enumerate(out_types) if t.startswith("rephasing"))
    #             nr_idx = next(i for i, t in enumerate(out_types) if t.startswith("nonrephasing"))
    #             absorptive = np.real(datas_nu[r_idx] + datas_nu[nr_idx])
    #             datas_nu.append(absorptive)
    #             out_types.append("absorptive")
    #         except StopIteration:
    #             pass

    return nu_cohs, nu_dets, datas_nu, out_types
