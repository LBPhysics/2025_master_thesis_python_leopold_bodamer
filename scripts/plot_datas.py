"""
Unified 1D/2D Electronic Spectroscopy Data Plotting Script

Clean, flexible CLI to load results and produce time/frequency-domain plots.

Examples (Windows PowerShell):
    # Base path (no suffix) or an .npz file
    python plot_datas.py --abs_path "C:/path/to/data/run_001"
    python plot_datas.py --abs_path "C:/path/to/data/run_042.npz"
    # Control zero-padding factor for FFT (1 disables, default 10)
    python plot_datas.py --abs_path "C:/path/to/data/run_001" --extend 5
    # Plot only time or frequency domain
    python plot_datas.py --abs_path "C:/path/to/data/run_001" --no-freq OR --no-time
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Mapping
from qutip import BosonicEnvironment
import sys
import argparse
import numpy as np
import warnings

### Project-specific imports
from qspectro2d.visualization.plotting import (
    plot_1d_el_field,
    plot_2d_el_field,
)
from qspectro2d.spectroscopy.post_processing import (
    extend_time_domain_data,
    compute_spectra,
)

from qspectro2d import generate_unique_plot_filename
from qspectro2d.core.bath_system.bath_fcts import extract_bath_parameters
from qspectro2d import load_simulation_data
from qspectro2d.utils.constants import convert_cm_to_fs
from qspectro2d.utils.file_naming import _generate_unique_filename

from plotstyle import init_style, save_fig

init_style()

SCRIPTS_DIR = Path(__file__).parent.resolve()
for _parent in SCRIPTS_DIR.parents:
    if (_parent / ".git").is_dir():
        PROJECT_ROOT = _parent
        break
DATA_DIR = (PROJECT_ROOT / "data").resolve()
FIGURES_DIR = (PROJECT_ROOT / "figures").resolve()
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# Suppress noisy but harmless warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")


# Lightweight proxies for dict-based payloads (new artifact format)
class _DictProxy:
    def __init__(self, payload: Dict[str, Any] | None, *, name: str) -> None:
        self._payload = dict(payload or {})
        self._name = name

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._payload)

    def __getattr__(self, attr: str) -> Any:
        if attr in self._payload:
            return self._payload[attr]
        raise AttributeError(f"{self._name} has no attribute '{attr}'")


class _SimulationConfigProxy(_DictProxy):
    def __init__(self, payload: Dict[str, Any] | None) -> None:
        super().__init__(payload, name="SimulationConfig")
        if "signal_types" not in self._payload or self._payload["signal_types"] is None:
            self._payload["signal_types"] = []

    @property
    def signal_types(self) -> List[str]:
        raw = self._payload.get("signal_types", [])
        return list(raw) if isinstance(raw, (list, tuple)) else [raw]


class _SystemProxy(_DictProxy):
    def __init__(self, payload: Dict[str, Any] | None) -> None:
        super().__init__(payload, name="AtomicSystem")

    def __getattr__(self, attr: str) -> Any:
        if attr == "frequencies_fs":
            freqs = self._payload.get("frequencies_fs")
            if freqs is None and "frequencies_cm" in self._payload:
                cm_vals = np.asarray(self._payload["frequencies_cm"], dtype=float)
                freqs = list(convert_cm_to_fs(cm_vals))
                self._payload["frequencies_fs"] = freqs
            if freqs is not None:
                return freqs
        return super().__getattr__(attr)


class _LaserProxy(_DictProxy):
    def __init__(self, payload: Dict[str, Any] | None) -> None:
        super().__init__(payload, name="LaserPulseSequence")


def _resolve_input_path(raw: str | None) -> Path:
    if raw is None:
        raise ValueError("--abs_path must be provided")

    candidate = Path(raw).expanduser()
    candidate = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
    candidate = candidate.resolve()

    if candidate.is_dir():
        run_candidates = sorted(candidate.glob("*_run_*.npz"))
        if not run_candidates:
            run_candidates = sorted(candidate.glob("*.npz"))
        if not run_candidates:
            raise FileNotFoundError(f"No .npz artifacts found in directory: {candidate}")
        return Path(run_candidates[-1]).resolve()

    if candidate.suffix == "":
        implicit_npz = candidate.with_suffix(".npz")
        if implicit_npz.exists():
            return implicit_npz.resolve()
        raise ValueError("Provide a path to a .npz run artifact produced by save_run_artifact")

    if candidate.suffix.lower() != ".npz":
        raise ValueError("plot_datas expects a .npz run artifact")

    if not candidate.exists():
        raise FileNotFoundError(f"Artifact not found: {candidate}")

    return candidate


# Helper to collect paths from save_fig (which may return a single path or a list of paths)
def _collect_saved_paths(accumulator: List[str], saved: Any) -> None:
    """Append saved file path(s) to ``accumulator`` (handles list / scalar)."""
    if isinstance(saved, (list, tuple, set)):
        accumulator.extend([str(s) for s in saved])
    else:  # single path-like
        accumulator.append(str(saved))


def _extend_and_fft(
    loaded_data_and_info: Dict[str, Any],
    pad_factor: float,
    dimension: str,
):
    """Zero-pad in time (positive direction) and compute spectra.

    Returns (freq_axes, data_freq, freq_labels).
    """
    # Extract inputs from loaded dict
    sim_cfg_payload = loaded_data_and_info.get("sim_config") or loaded_data_and_info.get(
        "simulation_config"
    )
    signal_types: Sequence[str]
    if sim_cfg_payload is not None:
        if hasattr(sim_cfg_payload, "signal_types"):
            signal_types = getattr(sim_cfg_payload, "signal_types")
        elif isinstance(sim_cfg_payload, Mapping):
            signal_types = sim_cfg_payload.get("signal_types", [])
        else:
            signal_types = []
    else:
        signal_types = []

    if not signal_types:
        signal_types = loaded_data_and_info.get("signal_types", [])

    if not signal_types:
        raise KeyError("Signal types missing from run artifact; cannot compute spectra")

    signal_types = list(signal_types)
    t_det = loaded_data_and_info.get("t_det")
    t_coh = loaded_data_and_info.get("t_coh") if dimension == "2d" else None

    datas: Sequence[Any] = [loaded_data_and_info.get(st) for st in signal_types]
    if any(d is None for d in datas):
        missing_signals = [st for st, d in zip(signal_types, datas) if d is None]
        raise KeyError(f"Missing signal arrays for: {missing_signals}")

    # Positive-direction multiplicative pad (1 disables)
    pad_factor = max(1.0, float(pad_factor))

    # Extend all signals at once
    ext_datas, ext_t_det, ext_t_coh = extend_time_domain_data(
        list(datas),
        np.asarray(t_det),
        np.asarray(t_coh) if t_coh is not None else None,
        pad=pad_factor,
    )

    # Compute spectra with requested sign conventions
    nu_cohs, nu_dets, datas_nu, out_types = compute_spectra(
        ext_datas, list(signal_types), ext_t_det, ext_t_coh
    )

    # Keep detection axis first in tuple
    freq_axes = (nu_dets, nu_cohs) if nu_cohs is not None else nu_dets
    return freq_axes, datas_nu, out_types


def _plot_components(
    *,
    datas: Sequence[Any],
    signal_types: Sequence[str],
    axis_det,
    axis_coh,
    domain: str,
    components: Sequence[str],
    plot_func,
    base_metadata: Dict[str, Any],
    system,
    sim_config,
    dimension: str,
    **kwargs,
) -> List[str]:
    """Loop over components (and signals for time domain) producing figures.

    For both time and frequency domain we make one figure per (signal/label, component).
    Frequency labels come from the FFT step (e.g., 'rephasing','nonrephasing','absorptive').
    """
    saved_paths: List[str] = []
    for comp in components:
        try:
            if domain == "time":
                for data, st in zip(datas, signal_types):
                    md = {**base_metadata, "signal_type": st}
                    fig = (
                        plot_func(
                            axis_det=axis_det,
                            data=data,
                            domain=domain,
                            component=comp,
                            function_symbol=r"$E_{k_s}$",
                            **md,
                        )
                        if dimension == "1d"
                        else plot_func(
                            axis_det=axis_det,
                            axis_coh=axis_coh,
                            data=data,
                            domain=domain,
                            use_custom_colormap=True,
                            component=comp,
                            **md,
                        )
                    )
                    base_name = Path(
                        generate_unique_plot_filename(
                            system=system,
                            sim_config=sim_config,
                            domain=domain,
                            component=comp,
                            figures_root=FIGURES_DIR,
                        )
                    )
                    safe_label = str(st).replace(" ", "_")
                    unique_path = _generate_unique_filename(
                        base_name.parent, f"{base_name.name}_{safe_label}"
                    )
                    filename = unique_path
                    saved = save_fig(
                        fig,
                        filename=filename,
                    )
                    _collect_saved_paths(saved_paths, saved)
            else:  # frequency domain: iterate and save one per spectrum label
                for data, st in zip(datas, signal_types):
                    md = {**base_metadata, "signal_type": st}
                    fig = (
                        plot_func(
                            axis_det=axis_det,
                            data=data,
                            domain=domain,
                            use_custom_colormap=True,
                            component=comp,
                            **kwargs,
                            **md,
                        )
                        if dimension == "1d"
                        else plot_func(
                            axis_det=axis_det,
                            axis_coh=axis_coh,
                            data=data,
                            domain=domain,
                            use_custom_colormap=True,
                            component=comp,
                            **kwargs,
                            **md,
                        )
                    )
                    base_name = Path(
                        generate_unique_plot_filename(
                            system=system,
                            sim_config=sim_config,
                            domain=domain,
                            component=comp,
                            figures_root=FIGURES_DIR,
                        )
                    )
                    safe_label = str(st).replace(" ", "_")
                    unique_path = _generate_unique_filename(
                        base_name.parent, f"{base_name.name}_{safe_label}"
                    )
                    filename = unique_path
                    saved = save_fig(
                        fig,
                        filename=filename,
                    )
                    _collect_saved_paths(saved_paths, saved)
        except Exception as e:
            # Defensive: continue other components even if one fails
            print(f"❌ Error plotting {dimension.upper()} {domain} {comp}: {e}")
    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Plot 1D or 2D electronic spectroscopy data (time/frequency domains)."
    )

    # Required input
    parser.add_argument(
        "--abs_path",
        type=str,
        default=None,
        help=(
            "Absolute path to the saved results file (ends with '_data.npz' or '_info.pkl'). "
            "Alternatively, pass the base path without suffix to auto-resolve."
        ),
    )

    parser.add_argument(
        "--extend",
        type=float,
        default=10.0,
        help="Zero-padding factor for (>t_max) before FFT (1 disables)",
    )
    parser.add_argument(
        "--section",
        type=float,
        nargs="+",
        default=(1.5, 1.7),
        help="Frequency window: 1D -> two floats (min max), 2D -> four floats (min max min max)",
    )

    # Plot selection (both enabled by default)
    parser.add_argument(
        "--no_time",
        dest="plot_time",
        action="store_false",
        default=True,
        help="Disable time-domain plotting",
    )
    parser.add_argument(
        "--no_freq",
        dest="plot_freq",
        action="store_false",
        default=True,
        help="Disable frequency-domain plotting",
    )

    args = parser.parse_args()
    resolved_path = _resolve_input_path(args.abs_path)

    try:
        # Parse section from command line arguments
        section: Optional[Any] = None
        if len(args.section) == 2:
            section = (args.section[0], args.section[1])
        elif len(args.section) == 4:
            section = (
                args.section[0],
                args.section[1],
                args.section[2],
                args.section[3],
            )
        else:
            raise ValueError("--section expects 2 (1D/2D) or 4 (2D) floats")

        # Plot selections (CLI flags)
        plot_time = bool(args.plot_time)
        plot_freq = bool(args.plot_freq)

        if args.abs_path and Path(args.abs_path).expanduser().resolve() != resolved_path:
            print(f"🔄 Loading: {args.abs_path} → {resolved_path}")
        else:
            print(f"🔄 Loading: {resolved_path}")

        loaded_data_and_info = load_simulation_data(abs_path=resolved_path)
        # Quick probe to decide dimension and basic info
        from qspectro2d.core.atomic_system.system_class import AtomicSystem
        from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
        from qspectro2d.core.simulation import SimulationConfig

        sim_cfg_payload = loaded_data_and_info.get("sim_config") or loaded_data_and_info.get(
            "simulation_config"
        )
        if sim_cfg_payload is None:
            raise KeyError("Simulation configuration missing from loaded data")
        if isinstance(sim_cfg_payload, SimulationConfig):
            sim_config = sim_cfg_payload
        elif isinstance(sim_cfg_payload, dict):
            sim_config = _SimulationConfigProxy(sim_cfg_payload)
        else:
            to_dict = getattr(sim_cfg_payload, "to_dict", None)
            if callable(to_dict):
                sim_config = _SimulationConfigProxy(to_dict())
            else:
                raise TypeError("Unsupported simulation configuration payload type")

        system_payload = loaded_data_and_info.get("system")
        if isinstance(system_payload, AtomicSystem):
            system = system_payload
        elif isinstance(system_payload, dict):
            system = _SystemProxy(system_payload)
        else:
            to_dict = getattr(system_payload, "to_dict", None)
            system = _SystemProxy(to_dict()) if callable(to_dict) else _SystemProxy({})

        laser_payload = loaded_data_and_info.get("laser")
        if isinstance(laser_payload, LaserPulseSequence):
            laser = laser_payload
        elif isinstance(laser_payload, dict):
            laser = _LaserProxy(laser_payload)
        elif laser_payload is None:
            laser = _LaserProxy({})
        else:
            to_dict = getattr(laser_payload, "to_dict", None)
            laser = _LaserProxy(to_dict()) if callable(to_dict) else _LaserProxy({})

        bath_payload = loaded_data_and_info.get("bath")
        bath_env = bath_payload if isinstance(bath_payload, BosonicEnvironment) else None

        is_2d = getattr(sim_config, "sim_type", "1d") == "2d"
        dimension = "2d" if is_2d else "1d"

        t_det = loaded_data_and_info.get("t_det")
        if t_det is not None:
            t_det = np.asarray(t_det, dtype=float)

        t_coh = loaded_data_and_info.get("t_coh") if dimension == "2d" else None
        if t_coh is not None:
            t_coh = np.asarray(t_coh, dtype=float)

        # Print axes info
        n_t_det = len(t_det) if t_det is not None else 0
        det_rng = f"[{float(t_det[0]):.2f},{float(t_det[-1]):.2f}] fs" if n_t_det > 0 else "[—]"
        if is_2d:
            n_t_coh = len(t_coh) if t_coh is not None else 0
            coh_rng = f"[{float(t_coh[0]):.2f},{float(t_coh[-1]):.2f}] fs" if n_t_coh > 0 else "[—]"
        else:
            n_t_coh, coh_rng = 0, "—"
        print(
            f"   Axes: t_det n={n_t_det} {det_rng}; t_coh n={n_t_coh if is_2d else '—'} {coh_rng if is_2d else ''}"
        )

        metadata_block = dict(loaded_data_and_info.get("metadata") or {})
        signal_types: Sequence[str] = getattr(sim_config, "signal_types", [])
        if not signal_types:
            signal_types = metadata_block.get("signal_types", [])
        if not signal_types:
            signal_types = loaded_data_and_info.get("signal_types", [])
        if not signal_types:
            signal_types = [
                key for key, val in loaded_data_and_info.items() if isinstance(val, np.ndarray)
            ]
        signal_types = list(signal_types)

        # Collect raw data arrays
        datas = [loaded_data_and_info.get(st) for st in signal_types]
        if any(d is None for d in datas):
            missing_signals = [st for st, d in zip(signal_types, datas) if d is None]
            raise KeyError(f"Missing signal arrays for: {missing_signals}")

        # Detect if time-domain signals are all-zero (informative warning only)
        try:
            if datas and all(
                isinstance(a, np.ndarray) and a.size > 0 and np.allclose(a, 0) for a in datas
            ):
                print("⚠️  All-zero time-domain signals detected.")
        except Exception:
            pass

        # Prepare metadata for plotting
        ode_solver = getattr(sim_config, "ode_solver", "")
        if ode_solver == "BR" or ode_solver == "Paper_eqs":
            try:
                w0 = float(np.asarray(system.frequencies_fs)[0])
            except Exception:
                w0 = None
            if bath_env is not None and w0 is not None:
                bath_dict = extract_bath_parameters(bath_env, w0)
            else:
                bath_dict = {}
        else:  # NOTE have to adjust if making ME also time-dep.
            bath_dict = {
                "deph_rate": 1 / 100,
                "down_rate": 1 / 300,
            }

        sim_dict = sim_config.to_dict()
        sim_dict.pop("signal_types", None)  # Remove to avoid duplication
        laser_dict = {k: v for k, v in laser.to_dict().items() if k != "pulses"}
        meta = {**system.to_dict(), **bath_dict, **laser_dict, **sim_dict}
        if metadata_block:
            meta.update(metadata_block)

        # Announce plotting context
        print(f"➡️  Plotting: dimension={dimension}, signals={list(signal_types)}")
        print(f"   t_det: n={n_t_det} range={det_rng}")
        if dimension == "2d" and t_coh is not None:
            print(f"   t_coh: n={n_t_coh} range={coh_rng}")

        # Parse and normalize section formats
        pad_factor = float(args.extend)
        components = ["real", "abs", "img"]

        # Normalize section format based on dimension
        plot_section = None
        if section is not None and isinstance(section, (list, tuple)):
            try:
                if len(section) == 2:
                    if dimension == "1d":
                        plot_section = (float(section[0]), float(section[1]))
                    else:  # 2D: use same range for both axes
                        plot_section = (
                            (float(section[0]), float(section[1])),
                            (float(section[0]), float(section[1])),
                        )
                elif len(section) == 4:
                    if dimension == "1d":
                        plot_section = (float(section[0]), float(section[1]))
                    else:  # 2D: separate ranges for each axis
                        plot_section = (
                            (float(section[0]), float(section[1])),
                            (float(section[2]), float(section[3])),
                        )
            except Exception:
                plot_section = None

        # Choose plotting function
        plot_func = plot_1d_el_field if dimension == "1d" else plot_2d_el_field

        # ---- Time domain plotting ----
        all_saved: List[str] = []
        if plot_time:
            print("📊 Time domain ...")
            saved = _plot_components(
                datas=datas,
                signal_types=signal_types,
                axis_det=t_det,
                axis_coh=t_coh,
                domain="time",
                components=components,
                plot_func=plot_func,
                base_metadata=meta,
                system=system,
                sim_config=sim_config,
                dimension=dimension,
            )
            all_saved.extend(saved)
            for p in saved:
                print(f"   💾 {p}")

        # ---- Frequency domain plotting ----
        if plot_freq:
            print(f"📊 Frequency domain ... (extend={pad_factor})")
            try:
                freq_axes, freq_datas, kept_types = _extend_and_fft(
                    loaded_data_and_info=loaded_data_and_info,
                    pad_factor=pad_factor,
                    dimension=dimension,
                )
                axis_det_f, axis_coh_f = (freq_axes, None) if dimension == "1d" else freq_axes
                saved = _plot_components(
                    datas=freq_datas,
                    signal_types=kept_types,
                    axis_det=axis_det_f,
                    axis_coh=axis_coh_f,
                    domain="freq",
                    components=components,
                    plot_func=plot_func,
                    base_metadata=meta,
                    system=system,
                    sim_config=sim_config,
                    dimension=dimension,
                    section=plot_section,
                )
                all_saved.extend(saved)
                for p in saved:
                    print(f"   💾 {p}")
            except Exception as e:
                print(f"❌ Frequency domain skipped: {e}")

        plt.close("all")

        # Final summary
        print(f"✅ Done. Saved {len(all_saved)} file(s).")
        print(
            f"To see them go to the directory:\n{Path(all_saved[0]).parent if all_saved else '—'}"
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
