# FairRecKit Lib
[![Pylint](https://github.com/TheMinefreak23/fairreckitlib/actions/workflows/pylint.yml/badge.svg)](https://github.com/TheMinefreak23/fairreckitlib/actions/workflows/pylint.yml)
[![PEP257](https://github.com/TheMinefreak23/fairreckitlib/actions/workflows/pydocstyle.yml/badge.svg)](https://github.com/TheMinefreak23/fairreckitlib/actions/workflows/pydocstyle.yml)
[![Pytest with Coverage](https://github.com/TheMinefreak23/fairreckitlib/actions/workflows/pytest-coverage.yml/badge.svg)](https://github.com/TheMinefreak23/fairreckitlib/actions/workflows/pytest-coverage.yml)
[![Upload to PyPI](https://github.com/TheMinefreak23/fairreckitlib/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/TheMinefreak23/fairreckitlib/actions/workflows/pypi-publish.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/TheMinefreak23/fairreckitlib?label=Release)

FairRecKitLib is a library that functions as a combinatory interface between a set of existing recommender libraries, such as [Lenskit](https://pypi.org/project/lenskit/), [Implicit](https://pypi.org/project/implicit/), and [Surprise](https://pypi.org/project/scikit-surprise/). It was made to accompany the [FairRecKit application (FairRecKitApp)](https://github.com/FairRecKit/FairRecKitApp).

This program has been developed by students from the bachelor Computer Science program at Utrecht University within the Software Project course, commissioned by Christine Bauer.

Lennard Chung, Aleksej Cornelissen, Isabelle van Driessel, Diede van der Hoorn, Yme de Jong, Lan Le, Sanaz Najiyan Tabriz, Roderick Spaans, Casper Thijsen, Robert Verbeeten, Vos Wesseling, Fern Wieland

© Copyright Utrecht University (Department of Information and Computing Sciences)

# Installation Requirements
FairRecKitLib utilises the scikit-surprise package, which relies on having a suitable C/C++ compiler present on the system to be able to install itself. For this purpose, make sure you have [Cython](https://pypi.org/project/Cython/) installed before attempting to install FairRecKitLib. If your system lacks a compiler, install the 'Desktop development with C++' build tools through the [Visual Studio installer](https://aka.ms/vs/17/release/vs_buildtools.exe).

Meeting these requirements, you can install FairRecKitLib like any PyPI package, using e.g. pip or conda.

**pip:**  
`pip install fairreckitlib`

**conda**  
`conda install fairreckitlib`

# Documentation
Please check out the [FairRecKitLib Wiki](https://github.com/FairRecKit/FairRecKitLib/wiki) and [FairRecKitLib API](https://FairRecKit.github.io/FairRecKitLib/src/fairreckitlib) for instructions and guides on how to utilise the library or add new functionality.
