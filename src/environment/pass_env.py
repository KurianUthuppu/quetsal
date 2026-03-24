"""
Gymnasium environment for Qiskit PassManager optimization.
"""


class PassManagerEnv:
    """
    Gymnasium environment for wrapping Qiskit PassManager.
    """

    def __init__(self):
        """
        Initialize the environment.
        """
        pass

    def reset(self):
        """
        Reset the environment to an initial state.

        Returns:
            The initial state.
        """
        pass

    def step(self, action):
        """
        Take an action in the environment.

        Args:
            action: The action to take.

        Returns:
            A tuple of (next_state, reward, done, info).
        """
        pass
