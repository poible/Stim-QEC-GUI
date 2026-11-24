"""
Stim Studio: an interactive GUI around Stim (+ PyMatching) for QEC simulation.

Run:
    pip install -r requirements.txt
    streamlit run stim_studio.py

Tabs
    Overview & export   circuit stats, noise model summary, .stim/.dem/QASM export, Crumble/Quirk links
    Diagrams            every stim diagram type (timeline, detslice, timeslice, matchgraph, 3D, Crumble)
    Sampling            measurement sampling, detection events, per-shot syndrome inspection + matching
    Error model         detector error model, graphlike distance, undetectable-error search,
                        error explanation, detecting regions
    Decoding            PyMatching logical error rate with confidence intervals
    Sweeps              threshold / Lambda sweeps over any noise parameter and distance
    Tableau lab         step-through TableauSimulator, Clifford tableau / unitary / Pauli propagation
    Gate reference      every gate Stim knows, with tableau, unitary and flows
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import stim

try:
    import pymatching

    HAS_PM = True
except ImportError:  # app still works without a decoder
    HAS_PM = False

st.set_page_config(page_title="Stim Studio", page_icon="🧊", layout="wide")

# =====================================================================================
# Static configuration
# =====================================================================================

CODE_TASKS = {
    "Repetition code memory": "repetition_code:memory",
    "Rotated surface code memory (Z basis)": "surface_code:rotated_memory_z",
    "Rotated surface code memory (X basis)": "surface_code:rotated_memory_x",
    "Unrotated surface code memory (Z basis)": "surface_code:unrotated_memory_z",
    "Unrotated surface code memory (X basis)": "surface_code:unrotated_memory_x",
    "Color code memory (XYZ)": "color_code:memory_xyz",
}

# (key, label, default, min, max, step, format, help)
P = (0.0, 0.5, 1e-4, "%.5f")
NOISE_PARAMS: dict[str, list[tuple]] = {
    "None (noiseless)": [],
    "Keep embedded noise": [],
    "Stim generator built-in": [
        ("after_clifford_depolarization", "After-Clifford depolarization", 1e-3, *P,
         "DEPOLARIZE1/2 after every Clifford gate."),
        ("before_round_data_depolarization", "Data depolarization per round", 1e-3, *P,
         "DEPOLARIZE1 on data qubits at the start of each round."),
        ("before_measure_flip_probability", "Measurement flip", 1e-3, *P,
         "X_ERROR (or Z_ERROR) before each measurement."),
        ("after_reset_flip_probability", "Reset flip", 1e-3, *P,
         "X_ERROR (or Z_ERROR) after each reset."),
    ],
    "Uniform circuit-level depolarizing": [
        ("p", "p", 1e-3, *P, "Same p for 1q gates, 2q gates, measurement, reset and idle."),
    ],
    "SI1000 (superconducting-inspired)": [
        ("p", "p", 1e-3, *P,
         "Gidney et al. SI1000: 2q DEP2(p), 1q DEP1(p/10), reset flip 2p, "
         "measure flip 5p, idle DEP1(p/10), idle during measure/reset DEP1(2p)."),
    ],
    "Transmon hardware model": [
        ("T1_us", "T1 (µs)", 100.0, 0.1, 10000.0, 1.0, "%.1f", "Energy relaxation time."),
        ("T2_us", "T2 echo (µs)", 80.0, 0.1, 20000.0, 1.0, "%.1f",
         "Total dephasing time (includes T1 contribution). Must satisfy T2 <= 2*T1."),
        ("t1q_ns", "1q gate duration (ns)", 25.0, 0.0, 10000.0, 1.0, "%.1f", ""),
        ("t2q_ns", "2q gate duration (ns)", 40.0, 0.0, 10000.0, 1.0, "%.1f", ""),
        ("tmeas_ns", "Readout + ringdown (ns)", 500.0, 0.0, 100000.0, 10.0, "%.1f",
         "Duration of a measurement layer; idle qubits decohere during it."),
        ("treset_ns", "Reset duration (ns)", 200.0, 0.0, 100000.0, 10.0, "%.1f", ""),
        ("e1q", "1q average gate infidelity", 5e-4, *P, "From RB. Mapped to DEPOLARIZE1(1.5 r)."),
        ("e2q", "2q average gate infidelity", 5e-3, *P,
         "From interleaved RB / XEB. Mapped to DEPOLARIZE2(1.25 r)."),
        ("e_ro", "Readout assignment error", 1e-2, *P,
         "Symmetric flip probability; use (P(0|1)+P(1|0))/2."),
        ("e_reset", "Reset / init error", 2e-3, *P, "Residual excited population after reset."),
    ],
}
TRANSMON_ERROR_KEYS = ["e1q", "e2q", "e_ro", "e_reset"]

DIAGRAM_TYPES = {
    "Timeline (text)": "timeline-text",
    "Timeline (SVG)": "timeline-svg",
    "Timeline (3D)": "timeline-3d-html",
    "Time slices (SVG)": "timeslice-svg",
    "Detector slices (SVG)": "detslice-svg",
    "Detector slices with ops (SVG)": "detslice-with-ops-svg",
    "Match graph (SVG)": "matchgraph-svg",
    "Match graph (3D)": "matchgraph-3d-html",
    "Crumble interactive editor": "interactive-html",
}
TICK_DIAGRAMS = {"timeslice-svg", "detslice-svg", "detslice-with-ops-svg", "timeline-svg", "timeline-text"}

ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK", "MPAD"}

DEFAULT_CUSTOM = """# Distance-3 bit-flip code, 2 rounds (edit freely)
R 0 1 2 3 4
TICK
CX 0 1 2 3
TICK
CX 2 1 4 3
TICK
MR 1 3
DETECTOR(1, 0) rec[-2]
DETECTOR(3, 0) rec[-1]
TICK
CX 0 1 2 3
TICK
CX 2 1 4 3
TICK
MR 1 3
SHIFT_COORDS(0, 1)
DETECTOR(1, 0) rec[-2] rec[-4]
DETECTOR(3, 0) rec[-1] rec[-3]
M 0 2 4
DETECTOR(1, 1) rec[-2] rec[-3] rec[-5]
DETECTOR(3, 1) rec[-1] rec[-2] rec[-4]
OBSERVABLE_INCLUDE(0) rec[-1]
"""

# =====================================================================================
# Noise models (rewriter that inserts hardware-motivated Pauli noise into any circuit)
# =====================================================================================


@dataclass
class NoiseModel:
    p1: float = 0.0  # DEPOLARIZE1 after 1q unitaries
    p2: float = 0.0  # DEPOLARIZE2 after 2q unitaries
    p_meas: float = 0.0  # measurement result flip
    p_reset: float = 0.0  # flip after reset
    idle: Callable[[set], list] = field(default=lambda kinds: [])  # noise on idle qubits
    active: Callable[[set], list] = field(default=lambda kinds: [])  # extra noise on busy qubits
    notes: list = field(default_factory=list)


def pauli_twirl(t_ns: float, T1_us: float, T2_us: float) -> tuple[float, float, float]:
    """Pauli-twirled amplitude + phase damping for an idle of duration t."""
    t_us = t_ns * 1e-3
    a = 1.0 - math.exp(-t_us / T1_us)
    px = py = a / 4.0
    pz = (1.0 - math.exp(-t_us / T2_us)) / 2.0 - a / 4.0
    return px, py, max(pz, 0.0)


def make_noise_model(mode: str, prm: dict, active_decoherence: bool = False) -> NoiseModel | None:
    if mode == "Uniform circuit-level depolarizing":
        p = prm["p"]
        return NoiseModel(p, p, p, p, idle=lambda k: [("DEPOLARIZE1", [p])])
    if mode == "SI1000 (superconducting-inspired)":
        p = prm["p"]

        def idle(kinds):
            if kinds & {"meas", "reset"}:
                return [("DEPOLARIZE1", [2 * p])]
            return [("DEPOLARIZE1", [p / 10])]

        return NoiseModel(p / 10, p, 5 * p, 2 * p, idle=idle)
    if mode == "Transmon hardware model":
        T1, T2 = prm["T1_us"], prm["T2_us"]
        notes = []
        if T2 > 2 * T1:
            notes.append("T2 > 2*T1 is unphysical; the Z part of the idle channel is clipped to 0.")
        dur = {"1q": prm["t1q_ns"], "2q": prm["t2q_ns"], "meas": prm["tmeas_ns"], "reset": prm["treset_ns"]}

        def channel(kinds):
            if not kinds:
                return []
            t = max(dur[k] for k in kinds)
            px, py, pz = pauli_twirl(t, T1, T2)
            return [("PAULI_CHANNEL_1", [px, py, pz])]

        return NoiseModel(
            p1=min(1.5 * prm["e1q"], 0.75),
            p2=min(1.25 * prm["e2q"], 15 / 16),
            p_meas=prm["e_ro"],
            p_reset=prm["e_reset"],
            idle=channel,
            active=channel if active_decoherence else (lambda k: []),
            notes=notes,
        )
    return None


def gate_kind(name: str) -> str | None:
    if name in ANNOTATIONS:
        return None
    g = stim.gate_data(name)
    if g.produces_measurements:
        return "meas"
    if g.is_reset:
        return "reset"
    if g.is_unitary:
        return "2q" if g.is_two_qubit_gate else "1q"
    return None


def is_noise_channel(name: str) -> bool:
    if name in ANNOTATIONS:
        return False
    g = stim.gate_data(name)
    return g.is_noisy_gate and not g.is_unitary and not g.produces_measurements and not g.is_reset


def reset_error_gate(name: str) -> str:
    return "Z_ERROR" if name.endswith("X") else "X_ERROR"


def collect_qubits(circuit: stim.Circuit) -> set[int]:
    qs: set[int] = set()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            qs |= collect_qubits(inst.body_copy())
        elif gate_kind(inst.name):
            qs.update(t.qubit_value for t in inst.targets_copy() if t.qubit_value is not None)
    return qs


def _emit_moment(ops, nm: NoiseModel, all_qubits: set[int], out: stim.Circuit):
    kinds, touched, post = set(), set(), []
    for op in ops:
        k = gate_kind(op.name)
        targets = op.targets_copy()
        qs = [t.qubit_value for t in targets if t.qubit_value is not None]
        if k:
            kinds.add(k)
            touched.update(qs)
        if k == "meas":
            if nm.p_meas > 0 and not op.name.startswith("HERALDED"):
                out.append(stim.CircuitInstruction(op.name, targets, [nm.p_meas]))
            else:
                out.append(op)
            if stim.gate_data(op.name).is_reset and nm.p_reset > 0:
                post.append((reset_error_gate(op.name), qs, [nm.p_reset]))
        elif k == "reset":
            out.append(op)
            if nm.p_reset > 0:
                post.append((reset_error_gate(op.name), qs, [nm.p_reset]))
        elif k == "1q":
            out.append(op)
            if nm.p1 > 0 and qs:
                post.append(("DEPOLARIZE1", qs, [nm.p1]))
        elif k == "2q":
            out.append(op)
            pairs = []
            for a, b in zip(targets[::2], targets[1::2]):
                if a.qubit_value is not None and b.qubit_value is not None:
                    pairs += [a.qubit_value, b.qubit_value]
            if nm.p2 > 0 and pairs:
                post.append(("DEPOLARIZE2", pairs, [nm.p2]))
        else:
            out.append(op)
    for name, qs, args in post:
        out.append(name, qs, args)
    if kinds:
        for name, args in nm.active(kinds):
            if touched and any(a > 0 for a in args):
                out.append(name, sorted(touched), args)
        idle = sorted(all_qubits - touched)
        for name, args in nm.idle(kinds):
            if idle and any(a > 0 for a in args):
                out.append(name, idle, args)


def apply_noise(circuit: stim.Circuit, nm: NoiseModel, all_qubits: set[int]) -> stim.Circuit:
    """Strip existing noise channels and insert the noise model moment by moment (TICK delimited)."""
    out = stim.Circuit()
    moment: list = []
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            _emit_moment(moment, nm, all_qubits, out)
            moment = []
            out.append(stim.CircuitRepeatBlock(inst.repeat_count, apply_noise(inst.body_copy(), nm, all_qubits)))
        elif inst.name == "TICK":
            _emit_moment(moment, nm, all_qubits, out)
            moment = []
            out.append("TICK")
        elif is_noise_channel(inst.name):
            continue
        else:
            moment.append(inst)
    _emit_moment(moment, nm, all_qubits, out)
    return out


def describe_noise(mode: str, prm: dict, active: bool) -> pd.DataFrame | None:
    rows = []
    if mode == "Transmon hardware model":
        for kind, key in [("1q layer", "t1q_ns"), ("2q layer", "t2q_ns"), ("Measure layer", "tmeas_ns"), ("Reset layer", "treset_ns")]:
            px, py, pz = pauli_twirl(prm[key], prm["T1_us"], prm["T2_us"])
            rows.append({"Moment": kind, "Duration (ns)": prm[key], "Idle pX": px, "Idle pY": py, "Idle pZ": pz,
                         "Idle total": px + py + pz})
        df = pd.DataFrame(rows)
        df.attrs["gates"] = {
            "DEPOLARIZE1 after 1q gates": 1.5 * prm["e1q"],
            "DEPOLARIZE2 after 2q gates": 1.25 * prm["e2q"],
            "Measurement flip": prm["e_ro"],
            "Reset flip": prm["e_reset"],
            "Decoherence also on active qubits": active,
        }
        return df
    if mode == "SI1000 (superconducting-inspired)":
        p = prm["p"]
        return pd.DataFrame([
            {"Operation": "CZ / 2q gate", "Channel": "DEPOLARIZE2", "Probability": p},
            {"Operation": "1q Clifford", "Channel": "DEPOLARIZE1", "Probability": p / 10},
            {"Operation": "Reset", "Channel": "X_ERROR", "Probability": 2 * p},
            {"Operation": "Measurement", "Channel": "flip", "Probability": 5 * p},
            {"Operation": "Idle", "Channel": "DEPOLARIZE1", "Probability": p / 10},
            {"Operation": "Idle during measure/reset", "Channel": "DEPOLARIZE1", "Probability": 2 * p},
        ])
    if mode == "Uniform circuit-level depolarizing":
        p = prm["p"]
        return pd.DataFrame([{"Operation": o, "Probability": p} for o in
                             ["1q gate (DEP1)", "2q gate (DEP2)", "Measurement flip", "Reset flip", "Idle (DEP1)"]])
    if mode == "Stim generator built-in":
        return pd.DataFrame([{"Parameter": k, "Probability": v} for k, v in prm.items()])
    return None


# =====================================================================================
# Cached builders
# =====================================================================================


@st.cache_data(show_spinner=False, max_entries=256)
def build_circuit_text(cfg_json: str) -> tuple[str, list[str]]:
    cfg = json.loads(cfg_json)
    mode, prm = cfg["noise"], cfg["params"]
    if cfg["source"] == "generated":
        task, d, r = CODE_TASKS[cfg["task"]], cfg["distance"], cfg["rounds"]
        if mode == "Stim generator built-in":
            return str(stim.Circuit.generated(task, distance=d, rounds=r, **prm)), []
        base = stim.Circuit.generated(task, distance=d, rounds=r)
    else:
        base = stim.Circuit(cfg["text"])
    if mode == "Keep embedded noise":
        return str(base), []
    if mode == "None (noiseless)":
        return str(base.without_noise()), []
    nm = make_noise_model(mode, prm, cfg.get("active_decoherence", False))
    return str(apply_noise(base, nm, collect_qubits(base))), nm.notes


@st.cache_data(show_spinner=False, max_entries=64)
def compute_dem(circuit_text: str, decompose: bool = True) -> tuple[str | None, str | None]:
    c = stim.Circuit(circuit_text)
    try:
        return str(c.detector_error_model(decompose_errors=decompose, approximate_disjoint_errors=True)), None
    except Exception as e:  # noqa: BLE001
        if decompose:
            try:
                dem = c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
                return str(dem), f"Could not decompose into graphlike errors, using the undecomposed model. ({e})"
            except Exception as e2:  # noqa: BLE001
                return None, str(e2)
        return None, str(e)


@st.cache_resource(show_spinner=False, max_entries=32)
def get_matching(dem_text: str):
    return pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel(dem_text))


def text_hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


# =====================================================================================
# Statistics helpers
# =====================================================================================


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 1.0
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return ph, max(0.0, c - h), min(1.0, c + h)


def per_round(p_l: float, rounds: int) -> float:
    if rounds <= 1:
        return p_l
    if p_l >= 0.5:
        return 0.5
    return 0.5 * (1 - (1 - 2 * p_l) ** (1 / rounds))


def estimate_ler(circuit_text: str, shots: int, max_errors: int, batch: int = 20_000, progress=None):
    dem_text, err = compute_dem(circuit_text, True)
    if dem_text is None:
        raise RuntimeError(err)
    m = get_matching(dem_text)
    sampler = stim.Circuit(circuit_text).compile_detector_sampler()
    errors = done = 0
    while done < shots and errors < max_errors:
        n = min(batch, shots - done)
        det, obs = sampler.sample(n, separate_observables=True)
        pred = m.decode_batch(det)
        errors += int(np.any(pred != obs, axis=1).sum())
        done += n
        if progress:
            progress(done / shots)
    return errors, done


def detector_xyz(circuit: stim.Circuit) -> np.ndarray:
    coords = circuit.get_detector_coordinates()
    xyz = np.zeros((circuit.num_detectors, 3))
    for k in range(circuit.num_detectors):
        c = coords.get(k, [])
        if len(c) >= 3:
            xyz[k] = c[:3]
        elif len(c) == 2:
            xyz[k] = [c[0], 0, c[1]]
        elif len(c) == 1:
            xyz[k] = [c[0], 0, 0]
        else:
            xyz[k] = [k, 0, 0]
    return xyz


def detector_figure(xyz, color=None, fired=None, edges=None, title="", cbar="", height=560):
    fig = go.Figure()
    hover = [f"D{k}<br>coords {tuple(np.round(xyz[k], 2))}" for k in range(len(xyz))]
    if color is not None:
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", name="detectors",
            marker=dict(size=5, color=color, colorscale="Viridis", colorbar=dict(title=cbar)),
            text=[f"{h}<br>{cbar}: {v:.4g}" for h, v in zip(hover, color)], hoverinfo="text"))
    else:
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", name="detectors",
            marker=dict(size=3, color="rgba(140,140,160,0.45)"), text=hover, hoverinfo="text"))
    if fired is not None:
        idx = np.flatnonzero(fired)
        fig.add_trace(go.Scatter3d(
            x=xyz[idx, 0], y=xyz[idx, 1], z=xyz[idx, 2], mode="markers", name="detection events",
            marker=dict(size=7, color="#e4572e", line=dict(width=1, color="black")),
            text=[hover[i] for i in idx], hoverinfo="text"))
    if edges is not None and len(edges):
        lx, ly, lz, bx, by, bz = [], [], [], [], [], []
        for a, b in edges:
            if a >= 0 and b >= 0:
                for arr, i in ((lx, 0), (ly, 1), (lz, 2)):
                    arr += [xyz[a, i], xyz[b, i], None]
            else:
                q = a if a >= 0 else b
                bx.append(xyz[q, 0]); by.append(xyz[q, 1]); bz.append(xyz[q, 2])
        if lx:
            fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", name="matched edges",
                                       line=dict(width=6, color="#2e86ab")))
        if bx:
            fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode="markers", name="matched to boundary",
                                       marker=dict(size=9, symbol="diamond-open", color="#2e86ab")))
    fig.update_layout(title=title, height=height, margin=dict(l=0, r=0, t=40, b=0),
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="t (round)"),
                      legend=dict(orientation="h", y=-0.02))
    return fig


def bits_to_01(arr: np.ndarray) -> str:
    return "\n".join("".join("1" if b else "0" for b in row) for row in arr)


def show_html(content: str, height: int = 620):
    # Render raw HTML inside a sandboxed iframe (scripts allowed, needed for Stim's 3D viewers).
    components.html(content, height=height, scrolling=True)


def show_svg(svg: str, height: int = 620):
    show_html(f"<div style='background:white;overflow:auto'>{svg}</div>", height)


# =====================================================================================
# Sidebar: circuit + noise configuration
# =====================================================================================

st.sidebar.title("🧊 Stim Studio")
st.sidebar.caption(f"stim {stim.__version__}  |  pymatching {pymatching.__version__ if HAS_PM else 'not installed'}")

source = st.sidebar.radio("Circuit source", ["Generated QEC circuit", "Custom circuit"], horizontal=True)
cfg: dict = {"source": "generated" if source.startswith("Generated") else "custom"}

if cfg["source"] == "generated":
    cfg["task"] = st.sidebar.selectbox("Code / experiment", list(CODE_TASKS))
    c1, c2 = st.sidebar.columns(2)
    cfg["distance"] = int(c1.number_input("Distance", 2, 25, 3, 1))
    cfg["rounds"] = int(c2.number_input("Rounds", 1, 100, 3, 1))
    modes = [m for m in NOISE_PARAMS if m != "Keep embedded noise"]
    default_mode = "Transmon hardware model"
else:
    up = st.sidebar.file_uploader("Load a .stim file", type=["stim", "txt"])
    initial = up.read().decode() if up else DEFAULT_CUSTOM
    cfg["text"] = st.sidebar.text_area("Stim circuit", initial, height=260, key=f"custom_{text_hash(initial)}")
    modes = [m for m in NOISE_PARAMS if m != "Stim generator built-in"]
    default_mode = "Keep embedded noise"

cfg["noise"] = st.sidebar.selectbox("Noise model", modes, index=modes.index(default_mode))
cfg["params"] = {}
spec = NOISE_PARAMS[cfg["noise"]]
if spec:
    with st.sidebar.expander("Noise parameters", expanded=True):
        for key, label, default, lo, hi, step, fmt, hlp in spec:
            cfg["params"][key] = float(st.number_input(label, lo, hi, float(default), step, fmt,
                                                       help=hlp or None, key=f"{cfg['noise']}_{key}"))
        if cfg["noise"] == "Transmon hardware model":
            cfg["active_decoherence"] = st.checkbox(
                "Also apply T1/T2 to qubits during their own gates",
                False, help="Leave off if RB/XEB infidelities already include coherence-limited error.")

with st.sidebar.expander("Model limits"):
    st.markdown(
        "Stim simulates Clifford circuits with Pauli noise only. In the transmon model, amplitude and phase "
        "damping are Pauli-twirled, readout error is symmetrized, and leakage, ZZ crosstalk and non-Markovian "
        "noise are not represented. Treat results as a Pauli-channel approximation of your device."
    )

try:
    circuit_text, noise_notes = build_circuit_text(json.dumps(cfg, sort_keys=True))
    circuit = stim.Circuit(circuit_text)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not build the circuit: {e}")
    st.stop()

CH = text_hash(circuit_text)
for n in noise_notes:
    st.warning(n)

# =====================================================================================
# Header
# =====================================================================================

title = cfg["task"] if cfg["source"] == "generated" else "Custom circuit"
sub = f"d = {cfg['distance']}, {cfg['rounds']} rounds, " if cfg["source"] == "generated" else ""
st.markdown(f"### {title}")
st.caption(f"{sub}noise: {cfg['noise']}")

m = st.columns(6)
m[0].metric("Qubits", circuit.num_qubits)
m[1].metric("Measurements", circuit.num_measurements)
m[2].metric("Detectors", circuit.num_detectors)
m[3].metric("Observables", circuit.num_observables)
m[4].metric("Ticks", circuit.num_ticks)
m[5].metric("Sweep bits", circuit.num_sweep_bits)

tabs = st.tabs(["Overview & export", "Diagrams", "Sampling", "Error model", "Decoding",
                "Sweeps", "Tableau lab", "Gate reference"])

# =====================================================================================
# Overview & export
# =====================================================================================
with tabs[0]:
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Circuit")
        lines = circuit_text.splitlines()
        shown = "\n".join(lines[:400]) + (f"\n# ... {len(lines) - 400} more lines" if len(lines) > 400 else "")
        st.code(shown, language="text")
    with right:
        st.markdown("#### Noise model")
        df = describe_noise(cfg["noise"], cfg["params"], cfg.get("active_decoherence", False))
        if df is not None:
            st.dataframe(df, hide_index=True, width="stretch")
            if "gates" in df.attrs:
                st.dataframe(pd.DataFrame(df.attrs["gates"].items(), columns=["Gate channel", "Value"]).astype(str),
                             hide_index=True, width="stretch")
        else:
            st.caption("No noise model inserted.")

        st.markdown("#### Export")
        st.download_button("Download circuit (.stim)", circuit_text, "circuit.stim", width="stretch")
        dem_text, dem_err = compute_dem(circuit_text, True)
        if dem_text:
            st.download_button("Download detector error model (.dem)", dem_text, "model.dem", width="stretch")
        try:
            st.download_button("Download OpenQASM 3", circuit.to_qasm(open_qasm_version=3, skip_dets_and_obs=False),
                               "circuit.qasm", width="stretch")
        except Exception as e:  # noqa: BLE001
            st.caption(f"QASM export unavailable: {e}")
        if circuit.num_qubits <= 200 and len(lines) < 3000:
            try:
                st.link_button("Open in Crumble", circuit.to_crumble_url(), width="stretch")
            except Exception:  # noqa: BLE001
                pass
            try:
                st.link_button("Open in Quirk (noiseless)", circuit.without_noise().to_quirk_url(),
                               width="stretch")
            except Exception:  # noqa: BLE001
                st.caption("Quirk link unavailable for this circuit.")

        st.markdown("#### Determinism checks")
        try:
            st.write(f"Determined measurements: {circuit.count_determined_measurements()}")
        except Exception:  # noqa: BLE001
            pass
        try:
            missing = circuit.without_noise().missing_detectors()
            st.write(f"Missing detectors (deterministic parities not annotated): {missing.num_detectors}")
        except Exception:  # noqa: BLE001
            pass

# =====================================================================================
# Diagrams
# =====================================================================================
with tabs[1]:
    c1, c2 = st.columns([2, 3])
    dlabel = c1.selectbox("Diagram type", list(DIAGRAM_TYPES))
    dtype = DIAGRAM_TYPES[dlabel]
    kwargs = {}
    if dtype in TICK_DIAGRAMS and circuit.num_ticks > 0:
        lo, hi = c2.slider("Tick range", 0, circuit.num_ticks, (0, min(circuit.num_ticks, 8 if "slice" in dtype else circuit.num_ticks)))
        if "slice" in dtype or (lo, hi) != (0, circuit.num_ticks):
            kwargs["tick"] = range(lo, hi + 1) if hi > lo else lo
    if len(circuit_text) > 400_000 and dtype != "interactive-html":
        st.warning("Large circuit: rendering may be slow. Reduce distance or rounds for faster diagrams.")
    height = st.slider("Panel height", 300, 1600, 650, 50)
    try:
        dgm = circuit.diagram(dtype, **kwargs)
        if dtype == "timeline-text":
            st.code(str(dgm), language="text")
        elif dtype.endswith("svg"):
            show_svg(str(dgm), height)
        else:
            show_html(str(dgm), height)
    except Exception as e:  # noqa: BLE001
        st.error(f"Diagram failed: {e}")

# =====================================================================================
# Sampling
# =====================================================================================
with tabs[2]:
    st.markdown("#### Measurement sampling")
    c1, c2, c3 = st.columns(3)
    ms = int(c1.number_input("Shots", 1, 1_000_000, 1000, 100, key="m_shots"))
    mseed = int(c2.number_input("Seed", 0, 2**31 - 1, 1, key="m_seed"))
    if c3.button("Sample measurements", width="stretch"):
        with st.spinner("Sampling"):
            samp = circuit.compile_sampler(seed=mseed).sample(ms)
            st.session_state["meas"] = (CH, samp, circuit.reference_sample())
    if st.session_state.get("meas", (None,))[0] == CH:
        _, samp, ref = st.session_state["meas"]
        st.caption("Reference sample (noiseless, deterministic choice for random results)")
        st.code("".join("1" if b else "0" for b in ref), language="text")
        k = min(200, samp.shape[0])
        fig = go.Figure(go.Heatmap(z=samp[:k].astype(int), colorscale=[[0, "#f2f2f2"], [1, "#1f3b73"]], showscale=False))
        fig.update_layout(title=f"First {k} shots (rows) x measurements (columns)", height=380,
                          xaxis_title="measurement index", yaxis_title="shot", margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        fig = go.Figure(go.Bar(y=samp.mean(axis=0), marker_color="#1f3b73"))
        fig.update_layout(title="P(result = 1) per measurement", height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        st.download_button("Download measurements (.01)", bits_to_01(samp), "measurements.01")

    st.divider()
    st.markdown("#### Detection events and syndrome inspection")
    if circuit.num_detectors == 0:
        st.info("This circuit has no DETECTOR annotations.")
    else:
        c1, c2, c3 = st.columns(3)
        ds = int(c1.number_input("Shots", 1, 1_000_000, 5000, 500, key="d_shots"))
        dseed = int(c2.number_input("Seed", 0, 2**31 - 1, 7, key="d_seed"))
        if c3.button("Sample detection events", width="stretch"):
            with st.spinner("Sampling"):
                det, obs = circuit.compile_detector_sampler(seed=dseed).sample(ds, separate_observables=True)
                st.session_state["det"] = (CH, det, obs)
        if st.session_state.get("det", (None,))[0] == CH:
            _, det, obs = st.session_state["det"]
            xyz = detector_xyz(circuit)
            rates = det.mean(axis=0)
            s = st.columns(4)
            s[0].metric("Mean detection fraction", f"{rates.mean():.4f}")
            s[1].metric("Shots with any event", f"{np.any(det, axis=1).mean():.3f}")
            s[2].metric("Mean events / shot", f"{det.sum(axis=1).mean():.2f}")
            if obs.shape[1]:
                s[3].metric("Raw observable flip rate", f"{np.any(obs, axis=1).mean():.4f}",
                            help="Without decoding.")
            st.plotly_chart(detector_figure(xyz, color=rates, title="Detector firing rate", cbar="rate"),
                            width="stretch")

            # firing rate vs round (useful for spotting leakage-like growth or boundary effects)
            tvals = np.round(xyz[:, 2], 6)
            ut = np.unique(tvals)
            fig = go.Figure(go.Scatter(x=ut, y=[rates[tvals == t].mean() for t in ut], mode="lines+markers",
                                       line=dict(color="#1f3b73")))
            fig.update_layout(title="Mean detection fraction per time coordinate", height=280,
                              xaxis_title="t", yaxis_title="fraction", margin=dict(t=40, b=10))
            st.plotly_chart(fig, width="stretch")

            st.markdown("##### Single shot")
            shot = st.slider("Shot index", 0, det.shape[0] - 1, 0)
            only_nontrivial = st.checkbox("Jump to next shot with a logical error", False)
            edges, pred = None, None
            dem_text, dem_err = compute_dem(circuit_text, True)
            if HAS_PM and dem_text and obs.shape[1]:
                try:
                    mt = get_matching(dem_text)
                    if only_nontrivial:
                        preds = mt.decode_batch(det)
                        bad = np.flatnonzero(np.any(preds != obs, axis=1))
                        nxt = bad[bad >= shot]
                        if len(nxt):
                            shot = int(nxt[0])
                            st.caption(f"Showing shot {shot} ({len(bad)} logical failures in this batch).")
                        else:
                            st.caption("No logical failures at or after this index.")
                    pred = mt.decode(det[shot])
                    edges = mt.decode_to_edges_array(det[shot])
                except Exception as e:  # noqa: BLE001
                    st.caption(f"Matching unavailable: {e}")
            fired = det[shot]
            info = f"Shot {shot}: {int(fired.sum())} detection events."
            if obs.shape[1]:
                info += f" Actual observables {obs[shot].astype(int).tolist()}"
                if pred is not None:
                    ok = np.array_equal(pred.astype(bool), obs[shot])
                    info += f", decoder predicts {pred.astype(int).tolist()} ({'correct' if ok else 'LOGICAL ERROR'})."
            st.write(info)
            st.plotly_chart(detector_figure(xyz, fired=fired, edges=edges, title=f"Syndrome of shot {shot}"),
                            width="stretch")
            with st.expander("Fired detectors"):
                st.write([f"D{k}" for k in np.flatnonzero(fired)])
            st.download_button("Download detection events (.01)", bits_to_01(det), "detection_events.01")
            if obs.shape[1]:
                st.download_button("Download observable flips (.01)", bits_to_01(obs), "observables.01")

# =====================================================================================
# Detector error model
# =====================================================================================
with tabs[3]:
    decompose = st.checkbox("Decompose hyperedges into graphlike errors", True)
    dem_text, dem_err = compute_dem(circuit_text, decompose)
    if dem_err:
        st.warning(dem_err)
    if dem_text:
        dem = stim.DetectorErrorModel(dem_text)
        errs = []
        for inst in dem.flattened():
            if inst.type == "error":
                tg = inst.targets_copy()
                errs.append({
                    "p": inst.args_copy()[0],
                    "detectors": " ".join(f"D{t.val}" for t in tg if t.is_relative_detector_id()),
                    "observables": " ".join(f"L{t.val}" for t in tg if t.is_logical_observable_id()),
                    "components": 1 + sum(1 for t in tg if t.is_separator()),
                })
        edf = pd.DataFrame(errs)
        c = st.columns(4)
        c[0].metric("Error mechanisms", len(edf))
        c[1].metric("Detectors", dem.num_detectors)
        c[2].metric("Observables", dem.num_observables)
        if len(edf):
            c[3].metric("Max symptom size", int(edf["detectors"].str.split().str.len().max() or 0))
            fig = go.Figure(go.Histogram(x=np.log10(edf["p"].clip(lower=1e-15)), nbinsx=40, marker_color="#1f3b73"))
            fig.update_layout(title="Error probabilities", xaxis_title="log10(p)", height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(edf.head(2000), width="stretch", height=280)
        with st.expander("DEM text"):
            dl = dem_text.splitlines()
            st.code("\n".join(dl[:500]) + (f"\n# ... {len(dl) - 500} more lines" if len(dl) > 500 else ""), "text")
        with st.expander("Match graph of the DEM"):
            try:
                show_svg(str(dem.diagram("matchgraph-svg")), 600)
            except Exception as e:  # noqa: BLE001
                st.caption(f"Unavailable: {e}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Circuit-level distance")
        st.caption("Shortest graphlike error that flips an observable without triggering detectors.")
        if st.button("Find shortest graphlike error"):
            try:
                with st.spinner("Searching"):
                    sh = circuit.shortest_graphlike_error(ignore_ungraphlike_errors=True,
                                                          canonicalize_circuit_errors=True)
                st.success(f"Graphlike distance = {len(sh)}")
                for i, e in enumerate(sh):
                    with st.expander(f"Error {i + 1}"):
                        st.code(str(e), "text")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    with c2:
        st.markdown("#### Undetectable logical errors (heuristic)")
        a = st.number_input("Max detection-event set size", 1, 20, 4)
        b = st.number_input("Max edge degree to explore", 1, 20, 4)
        inc = st.checkbox("Explore edges that increase symptom degree", False)
        if st.button("Search"):
            try:
                with st.spinner("Searching"):
                    res = circuit.search_for_undetectable_logical_errors(
                        dont_explore_detection_event_sets_with_size_above=int(a),
                        dont_explore_edges_with_degree_above=int(b),
                        dont_explore_edges_increasing_symptom_degree=not inc,
                        canonicalize_circuit_errors=True)
                st.success(f"Found an undetectable logical error of weight {len(res)}")
                st.code("\n\n".join(str(e) for e in res), "text")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Explain a symptom")
        sym = st.text_input("Detectors / observables (e.g. D0 D4 L0)", "D0")
        if st.button("Explain"):
            try:
                filt = stim.DetectorErrorModel(f"error(1) {sym}")
                ex = circuit.explain_detector_error_model_errors(dem_filter=filt,
                                                                 reduce_to_one_representative_error=False)
                if not ex:
                    st.info("No single circuit error produces exactly this symptom.")
                for e in ex:
                    st.code(str(e), "text")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    with c2:
        st.markdown("#### Detecting regions")
        st.caption("Pauli sensitivity of a detector or observable at each tick.")
        tgt = st.text_input("Target (e.g. D5 or L0)", "D0")
        if st.button("Compute regions"):
            try:
                reg = circuit.detecting_regions(targets=[tgt])
                for dt, ticks in reg.items():
                    st.dataframe(pd.DataFrame([{"tick": t, "sensitivity": str(p)} for t, p in sorted(ticks.items())]),
                                 hide_index=True, width="stretch")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

# =====================================================================================
# Decoding
# =====================================================================================
with tabs[4]:
    if not HAS_PM:
        st.error("Install pymatching to decode: pip install pymatching")
    elif circuit.num_observables == 0:
        st.info("The circuit has no observables to protect.")
    else:
        c1, c2, c3 = st.columns(3)
        shots = int(c1.number_input("Max shots", 1000, 100_000_000, 100_000, 10_000))
        maxe = int(c2.number_input("Stop after this many logical errors", 1, 1_000_000, 1000, 100))
        rounds = cfg.get("rounds", 1)
        if "color_code" in CODE_TASKS.get(cfg.get("task", ""), ""):
            st.caption("Note: matching is a suboptimal decoder for color codes, so these rates are pessimistic.")
        if c3.button("Estimate logical error rate", width="stretch"):
            bar = st.progress(0.0)
            try:
                k, n = estimate_ler(circuit_text, shots, maxe, progress=bar.progress)
                st.session_state["ler"] = (CH, k, n)
            except Exception as e:  # noqa: BLE001
                st.error(f"Decoding failed: {e}")
        if st.session_state.get("ler", (None,))[0] == CH:
            _, k, n = st.session_state["ler"]
            p, lo, hi = wilson(k, n)
            c = st.columns(4)
            c[0].metric("Logical errors / shots", f"{k} / {n}")
            c[1].metric("P_L (whole experiment)", f"{p:.3e}", help=f"95% Wilson CI [{lo:.2e}, {hi:.2e}]")
            c[2].metric("Per-round ε_L", f"{per_round(p, rounds):.3e}",
                        help=f"95% CI [{per_round(lo, rounds):.2e}, {per_round(hi, rounds):.2e}]; "
                             "ε = (1 - (1 - 2P)^(1/r)) / 2")
            c[3].metric("Relative uncertainty", f"{(hi - lo) / (2 * p) if p else float('inf'):.1%}")
            dem_text, _ = compute_dem(circuit_text, True)
            mt = get_matching(dem_text)
            st.caption(f"Matching graph: {mt.num_nodes} nodes, {mt.num_edges} edges, "
                       f"{mt.num_fault_ids} fault ids.")

# =====================================================================================
# Sweeps
# =====================================================================================
with tabs[5]:
    if not HAS_PM:
        st.error("Install pymatching to run sweeps.")
    else:
        numeric = [(k, lbl) for k, lbl, *_ in NOISE_PARAMS[cfg["noise"]]]
        options = {lbl: k for k, lbl in numeric}
        if cfg["noise"] == "Stim generator built-in":
            options = {"All four probabilities together": "__all__", **options}
        if cfg["noise"] == "Transmon hardware model":
            options = {"Error scale λ (multiplies all errors and 1/T1, 1/T2)": "__scale__", **options}
        if not options:
            st.info("Pick a noise model with parameters in the sidebar to sweep it.")
        else:
            c1, c2 = st.columns(2)
            plabel = c1.selectbox("Swept parameter", list(options))
            pkey = options[plabel]
            vals_txt = c2.text_input("Values (comma separated)",
                                     "0.25, 0.5, 1, 2" if pkey == "__scale__" else
                                     ("50, 100, 200, 400" if pkey.startswith("T") else "0.001, 0.002, 0.004, 0.008"))
            if cfg["source"] == "generated":
                c1, c2 = st.columns(2)
                ds_txt = c1.text_input("Distances", "3, 5, 7")
                r_eq_d = c2.checkbox("Rounds = distance", True)
            else:
                ds_txt, r_eq_d = "", False
            c1, c2 = st.columns(2)
            s_shots = int(c1.number_input("Max shots per point", 1000, 100_000_000, 100_000, 10_000, key="sw_s"))
            s_err = int(c2.number_input("Max logical errors per point", 10, 1_000_000, 500, 50, key="sw_e"))

            if st.button("Run sweep", type="primary"):
                try:
                    vals = [float(v) for v in vals_txt.split(",") if v.strip()]
                    dists = [int(v) for v in ds_txt.split(",") if v.strip()] or [None]
                except ValueError:
                    st.error("Values and distances must be comma-separated numbers.")
                    st.stop()
                rows, bar, total, done = [], st.progress(0.0), len(vals) * len(dists), 0
                status = st.empty()
                for d in dists:
                    for v in vals:
                        c = json.loads(json.dumps(cfg))
                        if d is not None:
                            c["distance"] = d
                            if r_eq_d:
                                c["rounds"] = d
                        if pkey == "__all__":
                            for kk in c["params"]:
                                c["params"][kk] = v
                        elif pkey == "__scale__":
                            for kk in TRANSMON_ERROR_KEYS:
                                c["params"][kk] = cfg["params"][kk] * v
                            c["params"]["T1_us"] = cfg["params"]["T1_us"] / v
                            c["params"]["T2_us"] = cfg["params"]["T2_us"] / v
                        else:
                            c["params"][pkey] = v
                        status.caption(f"d = {d}, {plabel} = {v}")
                        try:
                            txt, _ = build_circuit_text(json.dumps(c, sort_keys=True))
                            k, n = estimate_ler(txt, s_shots, s_err)
                            r = c.get("rounds", 1)
                            p, lo, hi = wilson(k, n)
                            rows.append(dict(distance=d, value=v, rounds=r, errors=k, shots=n, P_L=p, P_lo=lo, P_hi=hi,
                                             eps=per_round(p, r), eps_lo=per_round(lo, r), eps_hi=per_round(hi, r)))
                        except Exception as e:  # noqa: BLE001
                            st.warning(f"d={d}, value={v}: {e}")
                        done += 1
                        bar.progress(done / total)
                status.empty()
                st.session_state["sweep"] = (pd.DataFrame(rows), plabel)

            if "sweep" in st.session_state:
                sdf, slabel = st.session_state["sweep"]
                if len(sdf):
                    logx = st.checkbox("Log x axis", True)
                    fig = go.Figure()
                    for d, g in sdf.groupby(sdf["distance"].fillna(-1)):
                        g = g.sort_values("value")
                        fig.add_trace(go.Scatter(
                            x=g["value"], y=g["eps"], mode="lines+markers",
                            name=f"d = {int(d)}" if d >= 0 else "circuit",
                            error_y=dict(type="data", symmetric=False, array=g["eps_hi"] - g["eps"],
                                         arrayminus=g["eps"] - g["eps_lo"])))
                    fig.update_layout(title="Logical error per round", height=480,
                                      xaxis=dict(title=slabel, type="log" if logx else "linear"),
                                      yaxis=dict(title="ε_L per round", type="log"))
                    st.plotly_chart(fig, width="stretch")

                    # Lambda = eps_d / eps_{d+2}
                    if sdf["distance"].notna().any() and sdf["distance"].nunique() > 1:
                        piv = sdf.pivot_table(index="value", columns="distance", values="eps")
                        cols = sorted(piv.columns)
                        lam = pd.DataFrame(index=piv.index)
                        for a, b in zip(cols, cols[1:]):
                            lam[f"Λ {a}/{b}"] = piv[a] / piv[b]
                        st.markdown("##### Suppression factor Λ = ε(d) / ε(d+Δ)")
                        st.caption("Λ > 1 means operating below threshold.")
                        st.dataframe(lam, width="stretch")
                    st.dataframe(sdf, width="stretch", hide_index=True)
                    st.download_button("Download sweep (.csv)", sdf.to_csv(index=False), "sweep.csv")

# =====================================================================================
# Tableau lab
# =====================================================================================
with tabs[6]:
    st.markdown("#### Step-through stabilizer simulation")
    st.caption("Runs the noiseless circuit with stim.TableauSimulator, one operation at a time.")
    flat = [op for op in circuit.without_noise().flattened() if op.name not in ANNOTATIONS]
    used = sorted(collect_qubits(circuit))
    if not flat:
        st.info("No operations.")
    elif len(used) > 64:
        st.info(f"{len(used)} qubits is too many to display stabilizers. Use a smaller circuit (64 qubits max).")
    else:
        flat = flat[:5000]
        c1, c2 = st.columns([3, 1])
        k = c1.slider("Operations applied", 0, len(flat), min(len(flat), 1))
        tseed = int(c2.number_input("Seed", 0, 2**31 - 1, 0, key="t_seed"))
        sim = stim.TableauSimulator(seed=tseed)
        sim.set_num_qubits(max(used) + 1)
        sub = stim.Circuit()
        for op in flat[:k]:
            sub.append(op)
        sim.do_circuit(sub)
        st.write(f"Last applied: `{flat[k - 1]}`" if k else "Initial state |0...0>")
        if k < len(flat):
            st.write(f"Next: `{flat[k]}`")
        c1, c2 = st.columns([3, 2])
        with c1:
            rows = []
            for ps in sim.canonical_stabilizers():
                s = "".join("_XYZ"[ps[q]] for q in used)
                if s.strip("_"):
                    rows.append({"sign": "+" if ps.sign == 1 else "-", "stabilizer (qubits " + ",".join(map(str, used)) + ")": s})
            st.markdown("Canonical stabilizers")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=360)
        with c2:
            st.markdown("Single-qubit Bloch vectors")
            st.dataframe(pd.DataFrame([{"qubit": q, "bloch": str(sim.peek_bloch(q))} for q in used]),
                         hide_index=True, width="stretch", height=360)
        rec = sim.current_measurement_record()
        st.caption(f"Measurement record ({len(rec)} results): " + "".join("1" if b else "0" for b in rec[-200:]))

    st.divider()
    st.markdown("#### Clifford analyzer")
    c1, c2 = st.columns([2, 3])
    ctext = c1.text_area("Unitary Clifford circuit", "H 0\nCX 0 1\nS 1", height=160)
    pstr = c1.text_input("Pauli string to propagate", "XZ")
    with c2:
        try:
            cc = stim.Circuit(ctext)
            tab = stim.Tableau.from_circuit(cc)
            st.markdown("Tableau (images of X and Z generators)")
            st.code(str(tab), "text")
            try:
                ps = stim.PauliString(pstr)
                if len(ps) < len(tab):
                    ps = ps * stim.PauliString(len(tab))
                st.write(f"Heisenberg propagation: `{ps}` → `{tab(ps)}`")
            except Exception as e:  # noqa: BLE001
                st.caption(f"Pauli propagation: {e}")
            if len(tab) <= 4:
                u = tab.to_unitary_matrix(endian="big")
                st.markdown("Unitary (big endian)")
                st.dataframe(pd.DataFrame(np.round(u, 3)).astype(str), width="stretch")
            if len(tab) <= 8:
                s2 = stim.TableauSimulator()
                s2.do_circuit(cc)
                sv = s2.state_vector(endian="big")
                nz = [(format(i, f"0{len(tab)}b"), np.round(a, 4)) for i, a in enumerate(sv) if abs(a) > 1e-9]
                st.markdown("Output state from |0...0>")
                st.write(" + ".join(f"({a})|{b}>" for b, a in nz))
            try:
                st.markdown("Flow generators")
                st.code("\n".join(str(f) for f in cc.flow_generators()), "text")
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            st.error(f"Not a valid unitary Clifford circuit: {e}")

# =====================================================================================
# Gate reference
# =====================================================================================
with tabs[7]:
    gates = stim.gate_data()
    names = sorted({g.name for g in gates.values()})
    table = []
    for n in names:
        g = stim.gate_data(n)
        table.append({"gate": n, "aliases": ", ".join(a for a in g.aliases if a != n), "unitary": g.is_unitary,
                      "1q": g.is_single_qubit_gate, "2q": g.is_two_qubit_gate, "measures": g.produces_measurements,
                      "resets": g.is_reset, "noisy": g.is_noisy_gate})
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch", height=560)
    with c2:
        sel = st.selectbox("Gate", names, index=names.index("CX") if "CX" in names else 0)
        g = stim.gate_data(sel)
        st.write(f"Aliases: {', '.join(g.aliases)}")
        st.write(f"Parens arguments: {list(g.num_parens_arguments_range)}  |  "
                 f"Pauli targets: {g.takes_pauli_targets}  |  record targets: {g.takes_measurement_record_targets}")
        try:
            if g.inverse is not None:
                st.write(f"Inverse: {g.inverse.name}")
        except Exception:  # noqa: BLE001
            pass
        if g.tableau is not None:
            st.markdown("Tableau")
            st.code(str(g.tableau), "text")
            u = g.unitary_matrix
            if u is not None:
                st.markdown("Unitary")
                st.dataframe(pd.DataFrame(np.round(u, 3)).astype(str), width="stretch")
        try:
            fl = g.flows
            if fl:
                st.markdown("Stabilizer flows")
                st.code("\n".join(str(f) for f in fl), "text")
        except Exception:  # noqa: BLE001
            pass
