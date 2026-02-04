"""Concise post-processing utilities for extending time-domain data."""

from __future__ import annotations
from typing import List, Optional, Tuple, Union
import scipy.sparse as sp
import numpy as np

__all__ = [
    "compute_spectra",
]


ArrayOrSparse = Union[np.ndarray, sp.spmatrix]


def compute_spectra(
    datas: List[np.ndarray],
    signal_types: List[str] = ["rephasing"],
    t_det: np.ndarray = None,
    t_coh: Optional[np.ndarray] = None,
    pad: float = 1.0,
    *,
    # ---- rectangular band settings in 10^4 cm^-1 ----
    nu_win_det: float = 2.0,
    nu_win_coh: Optional[float] = 2.0,  # ignored if t_coh is None
    # ---- outputs / memory tuning ----
    return_sparse: bool = True,
    force_dense: bool = False,
) -> Tuple[Optional[np.ndarray], np.ndarray, List[ArrayOrSparse], List[str]]:
    """Compute spectra along detection (and optional coherence) axes.
    Based on the paper: https://doi.org/10.1063/5.0214023
    For each input data array:
    - Along detection time, always use +i convention: S(w_det) = ∫ E(t) e^{+i w t} dt
      (implemented via IFFT with virtual padding, no normalization scaling applied).
    - If coherence axis is present:
        - rephasing/else: S_R(w_coh, *) = ∫ E(t_coh, *) e^{-i w t} dt (FFT)
        (- nonrephasing:  S_NR(w_coh, *) = ∫ E(t_coh, *) e^{+i w t} dt (IFFT))

    Args:
        datas: List of time-domain data arrays (original size, not extended).
        signal_types: List of signal type strings.
        t_det: Detection time axis (1D array).
        t_coh: Coherence time axis (1D array, optional).
        pad: Padding factor (>=1.0) for virtual zero-padding in FFTs.

    Returns:
        (nu_cohs, nu_dets, datas_nu, signal_types_out)
        - nu_cohs: Coherence frequency axis (10^4 cm^-1), extended if t_coh provided.
        - nu_dets: Detection frequency axis (10^4 cm^-1), extended.
        - datas_nu: List of spectra arrays in frequency domain (virtually padded).
        - signal_types_out: List of signal labels, aligned with datas_nu.
    """
    if pad < 1.0:
        raise ValueError("pad must be >= 1.0")
    if t_det is None or t_det.ndim != 1:
        raise ValueError("t_det must be 1D")
    if t_coh is not None and t_coh.ndim != 1:
        raise ValueError("t_coh must be 1D when provided")
    if not datas:
        raise ValueError("datas must be non-empty")

    # Normalize signal types
    if len(signal_types) == 1 and len(datas) > 1:
        sig_types = [signal_types[0]] * len(datas)
    else:
        if len(signal_types) != len(datas):
            raise ValueError("signal_types length must match datas or be a single entry")
        sig_types = list(signal_types)

    # Sizes and extended sizes
    n_det = int(t_det.size)
    if t_coh is not None:
        min_det = min(arr.shape[1] for arr in datas if arr.ndim == 2)
        if min_det != n_det:
            t_det = t_det[:min_det]
            n_det = int(t_det.size)
    n_det_ext = int(np.ceil(pad * n_det))
    dt_det = float(t_det[1] - t_det[0])

    # Detection frequency axes
    freq_dets = np.fft.fftfreq(n_det_ext, d=dt_det)
    nu_det_unshifted = freq_dets / 2.998 * 10  # for masking (no shift)
    nu_dets = np.fft.fftshift(nu_det_unshifted)  # for user ergonomics

    # Coherence axes
    if t_coh is None:
        n_coh = None
        n_coh_ext = None
        nu_cohs = None
        nu_coh_unshifted = None
    else:
        n_coh = int(t_coh.size)
        n_coh_ext = int(np.ceil(pad * n_coh))
        dt_coh = float(t_coh[1] - t_coh[0])
        freq_cohs = np.fft.fftfreq(n_coh_ext, d=dt_coh)
        nu_coh_unshifted = freq_cohs / 2.998 * 10  # for masking (no shift)
        nu_cohs = np.fft.fftshift(nu_coh_unshifted)  # for convenience

    datas_nu: List[ArrayOrSparse] = []
    out_types: List[str] = []

    # Precompute 2D rectangular mask (unshifted) if 2D & sparse/dense-ROI requested
    rect_mask_2d = None
    if (t_coh is not None) and not force_dense and return_sparse:
        if nu_win_coh is None:
            raise ValueError("nu_win_coh must be set for 2D rectangular ROI")
        # Build outer-product mask in shifted coordinates
        det_mask = (np.abs(nu_dets) < float(nu_win_det))[None, :]  # shape (1, n_det_ext)
        coh_mask = (np.abs(nu_cohs) < float(nu_win_coh))[:, None]  # shape (n_coh_ext, 1)
        rect_mask_2d = coh_mask & det_mask  # shape (n_coh_ext, n_det_ext)

    for idx, (arr, stype) in enumerate(zip(datas, sig_types)):
        st_norm = str(stype).strip().lower()

        # 1D case
        if t_coh is None:
            if arr.ndim != 1 or arr.shape[0] != n_det:
                raise ValueError(f"datas[{idx}] must be 1D with length len(t_det)")

            # Detection axis (+i): IFFT with virtual padding
            spec_det = np.fft.ifft(arr, n=n_det_ext, axis=0)

            if force_dense or not return_sparse:
                # Keep dense; shift for user-facing axis
                datas_nu.append(np.fft.fftshift(spec_det))
            else:
                # Sparse ROI: keep |nu_det| < nu_win_det (shifted)
                spec_det_shifted = np.fft.fftshift(spec_det)
                keep = np.abs(nu_dets) < float(nu_win_det)
                idxs = np.nonzero(keep)[0]
                vals = spec_det_shifted[idxs]
                rows = idxs
                cols = np.zeros_like(rows)  # store as column vector
                spvec = sp.coo_matrix((vals, (rows, cols)), shape=(n_det_ext, 1))
                datas_nu.append(spvec)

            out_types.append(stype)
            continue

        # 2D case
        if arr.ndim != 2 or arr.shape[0] != n_coh:
            raise ValueError(f"datas[{idx}] must be 2D with shape (len(t_coh), len(t_det))")
        if arr.shape[1] != n_det:
            arr = arr[:, :n_det]

        # Detection axis (+i): IFFT along det axis (virtual padding)
        spec_2d = np.fft.ifft(arr, n=n_det_ext, axis=1)

        # Coherence axis depends on signal type (virtual padding)
        if st_norm == "nonrephasing":
            spec_2d = np.fft.ifft(spec_2d, n=n_coh_ext, axis=0)
            out_types.append("nonrephasing")
        else:
            spec_2d = np.fft.fft(spec_2d, n=n_coh_ext, axis=0)
            out_types.append("rephasing")

        if force_dense or not return_sparse:
            # Dense output; shift for convenient axes as before
            datas_nu.append(np.fft.fftshift(spec_2d, axes=(0, 1)))
        else:
            # Sparse rectangular ROI in *shifted* coordinates
            spec_2d_shifted = np.fft.fftshift(spec_2d, axes=(0, 1))
            if rect_mask_2d is None:
                # Fallback: keep everything but in sparse form
                rows, cols = np.nonzero(np.ones((n_coh_ext, n_det_ext), dtype=bool))
            else:
                rows, cols = np.nonzero(rect_mask_2d)
            vals = spec_2d_shifted[rows, cols]
            spmat = sp.coo_matrix((vals, (rows, cols)), shape=(n_coh_ext, n_det_ext))
            datas_nu.append(spmat)
            
    if t_coh is not None:
        have_r  = any(t.lower().startswith("rephasing") for t in out_types)
        have_nr = any(t.lower().startswith("nonrephasing") for t in out_types)
        if have_r and have_nr:
            # Example: match by index or by separate provided lists;
            # here we simply combine the first found pair.
            try:
                r_idx = next(i for i, t in enumerate(out_types) if t.startswith("rephasing"))
                nr_idx = next(i for i, t in enumerate(out_types) if t.startswith("nonrephasing"))
                absorptive = np.real(datas_nu[r_idx] + datas_nu[nr_idx])
                datas_nu.append(absorptive)
                out_types.append("absorptive")
            except StopIteration:
                pass
    return nu_cohs, nu_dets, datas_nu, out_types
