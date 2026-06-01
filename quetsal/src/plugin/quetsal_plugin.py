"""
quetsal/src/plugin/quetsal_plugin.py

Integrates the trained Quetsal RL agent into Qiskit's transpiler pipeline as
a PassManagerStagePlugin that replaces the optimization stage.

After ``pip install quetsal`` registers the entry point declared in
``pyproject.toml`` under ``[project.entry-points."qiskit.transpiler.optimization"]``,
a user calls:

    pm = generate_preset_pass_manager(
        optimization_level=1,           # L1 layout + routing + translation
        optimization_plugin="quetsal",  # ← Quetsal replaces the optimization stage
        basis_gates=HERON_R2_BASIS,
        coupling_map=cm,
    )
    optimized_circuit = pm.run(raw_circuit)

The trained agent ships inside the package, so no model path is required.
To use a different checkpoint, instantiate directly with an explicit path:

    plugin = QuetsalPlugin(model_path="path/to/model.zip")   # or QuetsalPlugin() for bundled
    pm.optimization = plugin.pass_manager(pass_manager_config=None)
"""

from __future__ import annotations

__all__ = ["QuetsalOptimizationPass", "QuetsalPlugin"]

from importlib.resources import files
from pathlib import Path
from typing import Optional

from qiskit.converters import dag_to_circuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

from quetsal.src.constants import MAX_STEPS_PER_EPISODE, SKIP_GATES
from quetsal.src.environment.pass_env import PassManagerEnv
from quetsal.src.agent.ppo_agent import load_agent


def _bundled_model_path() -> Path:
    """Filesystem path to the trained agent shipped inside the installed package.

    Declared as package data in pyproject.toml ([tool.setuptools.package-data]
    quetsal = ["models/*.zip"]).  Lets ``QuetsalPlugin()`` run with zero config
    after ``pip install quetsal``.
    """
    return Path(str(files("quetsal").joinpath("models", "best_model.zip")))


class QuetsalOptimizationPass(TransformationPass):
    """Qiskit TransformationPass that runs the Quetsal RL agent on a DAGCircuit.

    Receives a DAGCircuit that has already been through layout + routing +
    translation, runs the PPO agent episodically (deterministic policy), and
    returns the optimized DAGCircuit back to the Qiskit pipeline.

    After ``run()`` completes, two attributes are populated for inspection:

    ``last_pass_sequence`` : list[str]
        Pass names chosen by the agent, in order.
    ``last_trace_2q`` : list[int]
        2-qubit gate count at each step: ``[initial, after_p1, after_p2, ...]``.
        Length == ``len(last_pass_sequence) + 1``.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        model_path : Path to the trained PPO model (.zip).  Defaults to the
                     agent bundled with the package when None.
        device     : PyTorch device for inference (``"cpu"`` or ``"cuda"``).
        verbose    : If True, prints the chosen pass sequence to stdout.
        """
        super().__init__()
        self.model_path = Path(model_path) if model_path else _bundled_model_path()
        self.device = device
        self.verbose = verbose
        self._model = None  # lazy-loaded on first call

        # Populated after each run() call — available for notebooks / eval scripts
        self.last_pass_sequence: list[str] = []
        self.last_trace_2q: list[int] = []

    def run(self, dag):
        """Apply the Quetsal agent to optimize *dag*.

        Parameters
        ----------
        dag : DAGCircuit
            Circuit at the optimization stage (basis-translated, layout applied).

        Returns
        -------
        DAGCircuit
            Optimized circuit in the same basis.
        """
        qc = dag_to_circuit(dag)

        # Guard: nothing to optimize if there are no 2-qubit gates
        two_q = sum(
            1 for inst in qc.data
            if len(inst.qubits) >= 2 and inst.operation.name not in SKIP_GATES
        )
        if two_q == 0:
            self.last_pass_sequence = []
            self.last_trace_2q = [0]
            return dag

        # Fresh env for this circuit
        env = PassManagerEnv(circuits=[qc], max_steps=MAX_STEPS_PER_EPISODE)

        # Load model on first call; rebind env on subsequent calls
        if self._model is None:
            self._model = load_agent(self.model_path, env=env, device=self.device)
        else:
            self._model.set_env(env)

        obs, info = env.reset()
        self.last_pass_sequence = []
        self.last_trace_2q = [info["initial_2q"]]

        while True:
            action, _ = self._model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            self.last_pass_sequence.append(info["action_name"])
            self.last_trace_2q.append(info["current_2q"])
            if terminated or truncated:
                break

        if self.verbose:
            print(
                f"  [QuetsalPass] {self.last_trace_2q[0]} → {self.last_trace_2q[-1]} "
                f"2q gates | {' → '.join(self.last_pass_sequence)}"
            )

        return env._dag


class QuetsalPlugin(PassManagerStagePlugin):
    """Qiskit PassManagerStagePlugin that replaces the optimization stage with Quetsal.

    Registered in ``pyproject.toml`` as:

    .. code-block:: toml

        [project.entry-points."qiskit.transpiler.optimization"]
        quetsal = "quetsal.src.plugin.quetsal_plugin:QuetsalPlugin"

    After registration, invoke via:

    .. code-block:: python

        pm = generate_preset_pass_manager(
            optimization_level=1,
            optimization_plugin="quetsal",
            basis_gates=HERON_R2_BASIS,
            coupling_map=cm,
        )

    Or directly (no entry-point needed):

    .. code-block:: python

        plugin = QuetsalPlugin(model_path=MODEL_PATH)
        pm.optimization = plugin.pass_manager(pass_manager_config=None)

    After ``pm.run()`` completes, the agent's trace is available on
    ``plugin._quetsal_pass``.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        model_path : Path to the trained model ``.zip``.  Defaults to the agent
                     bundled with the package when None (zero-config usage).
        device     : ``"cpu"`` or ``"cuda"``.
        verbose    : Passed through to ``QuetsalOptimizationPass``.
        """
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.verbose = verbose
        self._quetsal_pass: Optional[QuetsalOptimizationPass] = None

    def pass_manager(
        self,
        pass_manager_config,
        optimization_level: Optional[int] = None,
    ) -> PassManager:
        """Build and return the Quetsal optimization ``PassManager``.

        Called automatically by Qiskit when ``optimization_plugin="quetsal"``
        is set, or manually from notebooks / scripts.

        Parameters
        ----------
        pass_manager_config : ``PassManagerConfig`` supplied by Qiskit's
            preset-pass-manager machinery, or ``None`` for direct calls.
            Not used by this implementation (model path comes from ``__init__``).
        optimization_level  : Supplied by Qiskit; ignored.

        Returns
        -------
        PassManager
            Single-pass manager wrapping ``QuetsalOptimizationPass``.

        After ``pm.run()`` completes, inspect:
            ``plugin._quetsal_pass.last_pass_sequence``
            ``plugin._quetsal_pass.last_trace_2q``
        """
        # model_path may be None here — QuetsalOptimizationPass falls back to the
        # agent bundled with the package, so the plugin works with zero config.
        self._quetsal_pass = QuetsalOptimizationPass(
            model_path=self.model_path,
            device=self.device,
            verbose=self.verbose,
        )
        return PassManager([self._quetsal_pass])
