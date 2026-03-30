from __future__ import annotations

from typing import Union

import numpy as np
import scipy.sparse as sp

__all__ = [
    "compute_spectra",
]


ArrayOrSparse = Union[np.ndarray, sp.spmatrix]


def _normalize_range(section: tuple[float, float]) -> tuple[float, float]:
    left, right = float(section[0]), float(section[1])
    return (left, right) if left <= right else (right, left)


def _shifted_roi_indices(
    axis_shifted: np.ndarray,
    section: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return shifted ROI indices and the corresponding unshifted FFT indices."""
    section = _normalize_range(section)
    mask_shifted = (axis_shifted >= section[0]) & (axis_shifted <= section[1])
    idx_shifted = np.flatnonzero(mask_shifted)
    idx_unshifted = np.fft.ifftshift(np.arange(axis_shifted.size))[idx_shifted]
    return idx_shifted, idx_unshifted


def compute_spectra(
    datas: list[np.ndarray],
    signal_types: list[str] | None = None,
    t_det: np.ndarray | None = None,
    t_coh: np.ndarray | None = None,
    pad: float = 1.0,
    *,
    section: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[np.ndarray | None, np.ndarray, list[ArrayOrSparse], list[str]]:
    """Compute spectra along detection and optional coherence axes.

    Convention
    ----------
    The spectrum pipeline uses the emitted-field / ``P^+(t)`` convention.

    Detection axis:
        S(omega_det) = ∫ dt_det E(t_det) exp(-i omega_det t_det)

    Coherence axis:
        rephasing:    ∫ dt_coh E(t_coh, *) exp(+i omega_coh t_coh)  -> IFFT
        nonrephasing: ∫ dt_coh E(t_coh, *) exp(-i omega_coh t_coh)  -> FFT

    Notes
    -----
    - The returned frequency axes are in units of ``10^4 cm^-1`` when the time
      axes are given in fs.
    - ``section`` is applied on the raw FFT axes returned by this function.
      If later code shifts axes from a rotating frame to a lab frame
      (for example by adding a carrier frequency), then cropping in lab-frame
      units should be done later, not here.

    Parameters
    ----------
    datas
        Time-domain signals. Each entry must be:
        - 1D, shape ``(len(t_det),)`` when ``t_coh is None``
        - 2D, shape ``(len(t_coh), len(t_det))`` otherwise
    signal_types
        Signal labels corresponding to ``datas``. If ``None``, defaults to
        ``["rephasing"]``. If one label is provided for multiple datasets, it is
        broadcast to all.
    t_det
        Detection-time axis in fs.
    t_coh
        Coherence-time axis in fs, or ``None`` for 1D spectra.
    pad
        Virtual zero-padding factor for FFT lengths. Must be >= 1.
    section
        Optional ROI in the returned FFT-axis units (``10^4 cm^-1``).

        Always specified as

            ((x_min, x_max), (y_min, y_max))

        Interpretation:
        - 2D: ``((coh_min, coh_max), (det_min, det_max))``
        - 1D: only the first tuple is used and interpreted as ``(det_min, det_max)``

        If provided, the function always returns sparse ROI objects.

    Returns
    -------
    (nu_cohs, nu_dets, datas_nu, signal_types_out)
        nu_cohs
            Coherence-frequency axis in ``10^4 cm^-1``, or ``None`` for 1D.
        nu_dets
            Detection-frequency axis in ``10^4 cm^-1``.
        datas_nu
            Frequency-domain spectra, either dense full arrays or sparse ROI matrices.
        signal_types_out
            Output labels aligned with ``datas_nu``. If both rephasing and
            nonrephasing are present, an additional ``"absorptive"`` entry is appended.
    """
    if pad < 1.0:
        raise ValueError("pad must be >= 1.0")

    if t_det is None:
        raise ValueError("t_det must be provided")
    t_det = np.asarray(t_det, dtype=float)
    if t_det.ndim != 1:
        raise ValueError("t_det must be 1D")
    if t_det.size < 2:
        raise ValueError("t_det must contain at least 2 points")

    if t_coh is not None:
        t_coh = np.asarray(t_coh, dtype=float)
        if t_coh.ndim != 1:
            raise ValueError("t_coh must be 1D when provided")
        if t_coh.size < 2:
            raise ValueError("t_coh must contain at least 2 points when provided")

    if not datas:
        raise ValueError("datas must be non-empty")

    if signal_types is None:
        signal_types = ["rephasing"]

    # Normalize signal types
    if len(signal_types) == 1 and len(datas) > 1:
        sig_types = [signal_types[0]] * len(datas)
    else:
        if len(signal_types) != len(datas):
            raise ValueError("signal_types length must match datas or be a single entry")
        sig_types = list(signal_types)

    # Sizes
    n_det = int(t_det.size)
    if t_coh is not None:
        min_det = min(arr.shape[1] for arr in datas if np.asarray(arr).ndim == 2)
        if min_det != n_det:
            t_det = t_det[:min_det]
            n_det = int(t_det.size)

    n_det_ext = int(np.ceil(pad * n_det))
    dt_det = float(t_det[1] - t_det[0])

    # Detection axis in 10^4 cm^-1
    freq_dets = np.fft.fftfreq(n_det_ext, d=dt_det)
    nu_dets = np.fft.fftshift(freq_dets / 2.998 * 10.0)

    # Coherence axis
    if t_coh is None:
        n_coh = None
        n_coh_ext = None
        nu_cohs = None
    else:
        n_coh = int(t_coh.size)
        n_coh_ext = int(np.ceil(pad * n_coh))
        dt_coh = float(t_coh[1] - t_coh[0])

        freq_cohs = np.fft.fftfreq(n_coh_ext, d=dt_coh)
        nu_cohs = np.fft.fftshift(freq_cohs / 2.998 * 10.0)

    datas_nu: list[ArrayOrSparse] = []
    out_types: list[str] = []

    # Parse optional sparse ROI section
    use_sparse_roi = section is not None
    det_idx_shifted: np.ndarray | None = None
    det_idx_unshifted: np.ndarray | None = None
    coh_idx_shifted: np.ndarray | None = None
    coh_idx_unshifted: np.ndarray | None = None

    if section is not None:
        if len(section) != 2:
            raise ValueError("section must be of the form ((x_min, x_max), (y_min, y_max))")

        first_range = _normalize_range(section[0])
        second_range = _normalize_range(section[1])

        if t_coh is None:
            # 1D: use only the first tuple as detection range
            det_idx_shifted, det_idx_unshifted = _shifted_roi_indices(
                nu_dets,
                first_range,
            )
        else:
            # 2D: first tuple is coherence range, second tuple is detection range
            assert nu_cohs is not None
            coh_idx_shifted, coh_idx_unshifted = _shifted_roi_indices(
                nu_cohs,
                first_range,
            )
            det_idx_shifted, det_idx_unshifted = _shifted_roi_indices(
                nu_dets,
                second_range,
            )

    for idx, (arr, stype) in enumerate(zip(datas, sig_types)):
        arr = np.asarray(arr)
        st_norm = str(stype).strip().lower()

        # 1D case
        if t_coh is None:
            if arr.ndim != 1 or arr.shape[0] != n_det:
                raise ValueError(f"datas[{idx}] must be 1D with length len(t_det)")

            spec_det = np.fft.fft(arr, n=n_det_ext, axis=0)

            if not use_sparse_roi:
                datas_nu.append(np.fft.fftshift(spec_det))
            else:
                assert det_idx_shifted is not None and det_idx_unshifted is not None
                vals = spec_det[det_idx_unshifted]
                rows = det_idx_shifted
                cols = np.zeros_like(rows)
                spvec = sp.coo_matrix((vals, (rows, cols)), shape=(n_det_ext, 1))
                datas_nu.append(spvec)

            out_types.append(str(stype))
            continue

        # 2D case
        if arr.ndim != 2 or arr.shape[0] != n_coh:
            raise ValueError(f"datas[{idx}] must be 2D with shape (len(t_coh), len(t_det))")
        if arr.shape[1] != n_det:
            arr = arr[:, :n_det]

        # Detection axis
        spec_2d = np.fft.fft(arr, n=n_det_ext, axis=1)

        # Coherence axis
        if st_norm == "nonrephasing":
            spec_2d = np.fft.fft(spec_2d, n=n_coh_ext, axis=0)
            out_types.append("nonrephasing")
        else:
            spec_2d = np.fft.ifft(spec_2d, n=n_coh_ext, axis=0)
            out_types.append("rephasing")

        if not use_sparse_roi:
            datas_nu.append(np.fft.fftshift(spec_2d, axes=(0, 1)))
        else:
            assert coh_idx_shifted is not None and coh_idx_unshifted is not None
            assert det_idx_shifted is not None and det_idx_unshifted is not None

            roi_block = spec_2d[np.ix_(coh_idx_unshifted, det_idx_unshifted)]
            rr, cc = np.meshgrid(coh_idx_shifted, det_idx_shifted, indexing="ij")

            spmat = sp.coo_matrix(
                (roi_block.ravel(), (rr.ravel(), cc.ravel())),
                shape=(n_coh_ext, n_det_ext),
            )
            datas_nu.append(spmat)

    # Add absorptive signal if both are present
    try:
        r_idx = next(i for i, t in enumerate(out_types) if t.lower().startswith("rephasing"))
        nr_idx = next(i for i, t in enumerate(out_types) if t.lower().startswith("nonrephasing"))

        r_data = datas_nu[r_idx]
        nr_data = datas_nu[nr_idx]

        if sp.issparse(r_data) or sp.issparse(nr_data):
            absorptive = (r_data + nr_data).real
        else:
            absorptive = np.real(r_data + nr_data)

        datas_nu.append(absorptive)
        out_types.append("absorptive")
    except StopIteration:
        pass
    return nu_cohs, nu_dets, datas_nu, out_types
