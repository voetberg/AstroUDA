import logging
import math
from typing import Any

import numpy

from astrouda.optimization.search_space import ParameterSpecification, SearchSpace

logger: logging.Logger = logging.getLogger(__name__)


def sample_value(specification: ParameterSpecification, generator: numpy.random.Generator) -> Any:
    if specification.distribution == "uniform":
        return float(generator.uniform(specification.low, specification.high))

    if specification.distribution == "loguniform":
        return float(math.exp(generator.uniform(math.log(specification.low), math.log(specification.high))))

    if specification.distribution == "int":
        return int(generator.integers(specification.low, specification.high, endpoint=True))

    return specification.choices[int(generator.integers(len(specification.choices)))]


def sample_trial_parameters(search_space: SearchSpace, seed: int, trial_index: int) -> dict[str, Any]:
    "Depends only on (seed, trial_index, search space); parameters are drawn in sorted name order."
    generator: numpy.random.Generator = numpy.random.default_rng([seed, trial_index])
    sampled_parameters: dict[str, Any] = {
        specification.name: sample_value(specification, generator) for specification in search_space.parameter_specifications
    }
    logger.debug(f"Trial {trial_index} (seed {seed}) sampled {sampled_parameters}")

    return sampled_parameters
