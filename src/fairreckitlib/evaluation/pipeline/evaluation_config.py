"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Any

KEY_EVALUATION = 'evaluation'


@dataclass
class MetricConfig:
    """Metric Configuration."""

    name: str
    params: {str: Any}
    prefilters: []
