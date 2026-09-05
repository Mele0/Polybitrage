"""Build Cython hot-path extensions.

Usage:
    # Compile in-place (editable install):
    python setup_cython.py build_ext --inplace

    # Or via pip (if declared in pyproject.toml build-system):
    pip install -e ".[speed]" --no-build-isolation

The compiled modules are placed in src/fast/ alongside the .pyx sources.
If compilation fails, the simulator falls back to scan_numpy.py silently.
"""
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Cython and numpy are required to build the C extensions.\n"
        "Install them with: pip install cython numpy"
    ) from exc

_CFLAGS = ["-O3", "-march=native", "-ffast-math", "-funroll-loops"]

ext_modules = cythonize(
    [
        Extension(
            "src.fast._scan_ext",
            sources=["src/fast/scan.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=_CFLAGS,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
    ],
    compiler_directives={
        "language_level": 3,
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "nonecheck": False,
        "initializedcheck": False,
        "embedsignature": True,
    },
    annotate=False,
)

setup(
    name="polybitrage-cython",
    ext_modules=ext_modules,
)
