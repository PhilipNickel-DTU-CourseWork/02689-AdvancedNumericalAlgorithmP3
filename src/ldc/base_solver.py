from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import time
from datastructures import RuntimeConfig, Results

# -----------------------------
# Abstract base solver
# -----------------------------
class LidDrivenCavitySolver(ABC):
    """
    Abstract contract for any lid-driven cavity solver.
    Subclasses must implement the numerical step and
    provide access to the velocity field.
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config

    # ---- Required methods for a concrete solver ----

    @abstractmethod
    def step(self, dt: float) -> float:
        """
        Advance the solution by one pseudo-timestep of size dt.
        """
        pass

    @abstractmethod
    def get_velocity_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return physical-space velocity fields (u, v).
        Shapes and layouts are solver-dependent, but must be real arrays.
        """
        pass

    @abstractmethod
    def get_pressure_field(self) -> np.ndarray:
        """
        Return physical-space pressure field (p). 
        """
        pass



    # ---- Provided method: common solve loop ----

    def solve(self):
        """
        Generic pseudo-timestep-loop.
        """

        start = time.time()

        for _ in range(max_steps):
            residual = self.step(dt)

            self.time += dt
            self.step_count += 1
            self.residual_history.append(residual)

            if residual < self.config.tolerance:
                break

            if self.time >= T:
                break


        # Save results: 
        self.config.

        return None
