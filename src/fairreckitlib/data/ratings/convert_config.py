"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ConvertConfig:
    """Dataset rating conversion Configuration."""

    name: str
    params: Dict[str, Any]
