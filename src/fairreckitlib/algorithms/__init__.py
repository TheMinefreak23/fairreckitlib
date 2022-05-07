"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""
from .elliot_alg.factory import ELLIOT_API
from .lenskit_alg.factory import LENSKIT_API
from .implicit_alg.factory import IMPLICIT_API

# Export (approach) API names TODO refactor
ELLIOT_API = ELLIOT_API
LENSKIT_API = LENSKIT_API
IMPLICIT_API = IMPLICIT_API
