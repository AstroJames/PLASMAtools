from setuptools import setup, find_packages
import os

setup(
    name="PLASMAtools",
    version="0.1.0",
    author="James R. Beattie and Collaborators",
    author_email="james.beattie@princeton.edu",
    description="A toolkit (JIT compiled) for doing fluid plasma physics analysis",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/AstroJames/PLASMAtools",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core scientific computing dependencies
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        # JIT compilation and parallelization
        "numba>=0.56.0", 
        "joblib>=1.0.0",
        # File I/O and data handling
        "h5py>=3.0.0",
    ],
    extras_require={
        # Fast FFT library (recommended for best performance)
        "fftw": [
            "pyfftw>=0.12.0",
        ],
        # MPI support for large memory parallel jobs
        "mpi": [
            "mpi4py>=3.1.0",
        ],
        # Alternative FFT library fallback
        "fftw3": [
            "fftw3>=0.2.2",
        ],
        # Documentation tools
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "numpydoc>=1.1",
        ],
        # Complete installation with all optional features
        "all": [
            "numpy>=1.20.0",
            "scipy>=1.7.0",
            "h5py>=3.0.0",
            "pyfftw>=0.12.0",
            "numba>=0.56.0", 
            "joblib>=1.0.0",
            "mpi4py>=3.1.0",
            "fftw3>=0.2.2",
            "black>=21.0",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "numpydoc>=1.1",
        ],
        # Recommended for most users (core + fast FFT + MPI)
        "recommended": [
            "pyfftw>=0.12.0",
            "mpi4py>=3.1.0",
        ],
    },
    # Include package data
    include_package_data=True,
    package_data={
        "PLASMAtools": ["*.txt", "*.md"],
    },
    zip_safe=False,
)