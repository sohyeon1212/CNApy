import cobra
from cobra.sampling import sample
from typing import Dict, Tuple
import pandas as pd

def perform_sampling(model: cobra.Model, n: int, thinning: int, processes: int) -> pd.DataFrame:
    with model as model:
        s = sample(model, n, thinning=thinning, processes=processes)
    return s
