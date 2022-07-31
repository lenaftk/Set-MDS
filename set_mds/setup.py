from numpy.distutils.misc_util import Configuration

import numpy
import os
from Cython.Build import cythonize

def configuration(parent_package="", top_path=None):
    os.environ['CFLAGS'] = '-Wno-cpp -fno-strict-aliasing  -ffast-math -O3 -Wall -fPIC'
    config = Configuration('set_mds', parent_package, top_path)
    libraries = []
    if os.name == 'posix':
        libraries.append('m')
    config.add_extension('setmds_fast',
                         sources=["setmds_fast.pyx", "setmds_pertubations.c"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries)
    config.ext_modules[-1] = cythonize(config.ext_modules[-1], language='c')[0]
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration().todict())