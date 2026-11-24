# Stim Studio

An interactive browser GUI for [Stim](https://github.com/quantumlib/Stim) and
[PyMatching](https://github.com/oscarhiggott/PyMatching), built with Streamlit. Generate or paste a
quantum error correction circuit, add a noise model, and then inspect it, sample it, decode it and
sweep its parameters, all without writing any code.

The app is a single file, [`stim_studio.py`](stim_studio.py).

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/<your-username>/stim-studio.git
cd stim-studio
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run stim_studio.py
```

Streamlit opens the app at <http://localhost:8501>.

PyMatching is optional. Without it the app still runs, but the decoding, sweep and per-shot matching
features are turned off.

## Circuits

Use the sidebar to choose where the circuit comes from:

- **Generated QEC circuit**: Stim's built-in generators for repetition code memory, rotated and
  unrotated surface code memory (Z or X basis), and XYZ color code memory. You set the code distance
  and the number of rounds.
- **Custom circuit**: paste Stim circuit text or upload a `.stim` file. The app opens with an example
  distance-3 bit-flip code to edit.

## Noise models

| Model | What it does |
|---|---|
| None | Noiseless circuit. |
| Keep embedded noise | Uses the noise channels already written in a custom circuit. |
| Stim generator built-in | Stim's own generator parameters: after-Clifford depolarization, data depolarization at the start of each round, measurement flips and reset flips. |
| Uniform circuit-level depolarizing | One probability `p` for 1q gates, 2q gates, measurement, reset and idling. |
| SI1000 | The superconducting-inspired model of Gidney et al.: 2q `DEP2(p)`, 1q `DEP1(p/10)`, reset flip `2p`, measurement flip `5p`, idle `DEP1(p/10)`, and idle during measurement or reset `DEP1(2p)`. |
| Transmon hardware model | Built from device numbers: T1, T2, gate, readout and reset durations, RB/XEB infidelities, and readout and reset errors. |

Every model except the generator's built-in one is applied by rewriting the circuit one moment at a
time, so it works on custom circuits as well as generated ones.

In the transmon model:

- Idle decoherence is turned into a Pauli-twirled `PAULI_CHANNEL_1`, with the idle time set by the
  slowest operation in the moment.
- Gate infidelities `r` become depolarizing noise: `DEPOLARIZE1(1.5 r)` for 1q gates and
  `DEPOLARIZE2(1.25 r)` for 2q gates.
- An option also applies T1/T2 decay to qubits while their own gates run.

## Tabs

| Tab | Contents |
|---|---|
| Overview & export | Circuit statistics, a summary of the noise model, export to `.stim`, `.dem` and OpenQASM, and links that open the circuit in Crumble and Quirk. |
| Diagrams | Every Stim diagram type: timeline (text, SVG, 3D), time slices, detector slices, match graph (SVG, 3D) and the Crumble interactive editor. |
| Sampling | Measurement samples, detection events, and a per-shot view of the syndrome with its matching. |
| Error model | The detector error model, the graphlike code distance, a search for undetectable logical errors, error explanations and detecting regions. |
| Decoding | Logical error rate from PyMatching, with Wilson confidence intervals and a per-round rate. |
| Sweeps | Threshold sweeps over any noise parameter and code distance, plus the suppression factor Λ = ε(d) / ε(d+Δ). |
| Tableau lab | Runs the noiseless circuit step by step with `stim.TableauSimulator`, and shows Clifford tableaus, unitaries and how Paulis propagate. |
| Gate reference | Every gate Stim supports, with its tableau, unitary and flows. |

## Limitations

- Stim only simulates Clifford circuits with Pauli noise. The transmon model is therefore a
  Pauli-channel approximation: amplitude and phase damping are Pauli-twirled, readout error is
  treated as symmetric, and leakage, ZZ crosstalk and non-Markovian noise are not modeled.
- PyMatching is a minimum-weight perfect matching decoder, which does not suit color codes well, so
  the logical error rates it gives for color codes are pessimistic.
- Large distances or many rounds increase both the diagram rendering time and the sampling time.

## Acknowledgements

The simulation is done by [Stim](https://github.com/quantumlib/Stim) (Craig Gidney) and the decoding
by [PyMatching](https://github.com/oscarhiggott/PyMatching) (Oscar Higgott and Craig Gidney). The
SI1000 noise model comes from Gidney, Newman, Fowler and Broughton, *A Fault-Tolerant Honeycomb
Memory* (2021). This project is not affiliated with either library.

## License

[MIT](LICENSE)
