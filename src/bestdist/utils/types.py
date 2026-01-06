"""Type definitions for the pdist package."""

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd

# Type aliases
ArrayLike = Union[np.ndarray, pd.Series, List[float]]
FitResult = Dict[str, Union[float, str, Dict[str, float]]]
TestResult = Tuple[float, float]  # (statistic, p_value)
Parameters = Dict[str, float]

