# FairRecKitLib

FairRecKitLib is a library that functions as a combinatory interface between a set of existing recommender libraries, such as [Lenskit](https://pypi.org/project/lenskit/), [Implicit](https://pypi.org/project/implicit/), and [Elliot](https://github.com/sisinflab/elliot). It was made to accompany the [FairRecKit application](https://github.com/TheMinefreak23/fair-rec-kit-app).

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.  
© Copyright Utrecht University (Department of Information and Computing Sciences)

# Installation Requirements
This package requires the Elliot package, which is not available on PyPI and can instead be found on its [GitHub page](https://github.com/sisinflab/elliot).

FairRecKitLib also utilises the scikit-surprise package, which relies on having a suitable C/C++ compiler present on the system to be able to install itself. For this purpose, make sure you have [Cython](https://pypi.org/project/Cython/) installed before attempting to install FairRecKitLib. If your system lacks a compiler, install the 'Desktop development with C++' build tools through the [Visual Studio installer](https://aka.ms/vs/17/release/vs_buildtools.exe).

Meeting these requirements, you can install FairRecKitLib like any PyPI package, using e.g. pip or conda.

**pip:**  
`pip install fairreckitlib`

**conda**  
`conda install fairreckitlib`