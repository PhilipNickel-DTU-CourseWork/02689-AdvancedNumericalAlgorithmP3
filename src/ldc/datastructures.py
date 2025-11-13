"""Data structures for solver configuration and results.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

@dataclass
class SolverConfig: 
    # Problem setup
    
    Re: Optional[int] = 100
    lid_velocity: Optional[float] = 1
    Lx: float = 1.0
    Ly: float = 1.0

    # Solver configuration
    discretization_method: Optional[str] = None
 

@dataclass
class RuntimeConfig:
    
    tolerance: Optional[float] = None

    # Pseudo-timestepping options
    max_iter: Optional[int] = None

    # Spectral solver configuration
    N: Optional[int] = None


@dataclass
class Results:

    # Solution 
    u: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None
    p: Optional[np.ndarray] = None

    # Residual history 
    res_his: list = None 

    
   
    # Runtime results
    iterations: Optional[int] = None
    converged: Optional[bool] = None
    final_alg_residual: Optional[float] = None

    # performance metrics
    wall_time: Optional[float] = None


