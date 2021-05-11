from .gaussian_process import GaussianProcess, trend
from .random_forest import RandomForest, SurrogateAggregation

__all__ = ["GaussianProcess", "RandomForest", "SurrogateAggregation", "trend"]
