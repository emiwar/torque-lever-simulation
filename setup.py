from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='torque_lever_cython',
    ext_modules=cythonize(
        'torque_lever_cython.pyx',
        language_level='3',
        compiler_directives={
            'boundscheck': True,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': True,
        }
    ),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
