"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from to_explicit import create_to_explicit_converter

def create_converter(rating_modifier : str):
    """Creates a converter with the specified modifier."""
    if rating_modifier == 'to_explicit':
        return create_to_explicit_converter()
    else: raise KeyError('Converter does not exist: ' + rating_modifier)
