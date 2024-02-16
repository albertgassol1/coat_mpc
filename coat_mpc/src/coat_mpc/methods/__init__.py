from .base_optimizer import BaseOptimizer
from .wml import WeightedMaximumLikelihood
from .metropolis_hastings import MetropolisHastings
from .bayesopt import BayesianOptimizer
from .crbo.crbo import ConfidenceRegionBayesianOptimization

optimizer_dict = {
    "WML": WeightedMaximumLikelihood,
    "MH": MetropolisHastings,
    "BO": BayesianOptimizer,
    "WML": ConfidenceRegionBayesianOptimization,
}
