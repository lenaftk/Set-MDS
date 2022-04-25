import errno
import functools
import os
import time

import numpy as np


def load_embeddings(data_file):
    with open(data_file, 'r') as fd:
        lines = fd.readlines()
        words = [l.strip().split()[0] for l in lines]
        d = [map(float, l.strip().split()[1:]) for l in lines]
        xs = np.array(d)
    return words, xs


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def order_of_magnitude(x):
    return int(np.floor(np.log10(x)))


def timefunc(func):
    """
    Decorator that measure the time it takes for a function to complete
    Usage:
      @timefunc
      def time_consuming_function(...):
    """
    @functools.wraps(func)
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        elapsed = '{0}'.format(te - ts)
        print('{f} took: {t} sec'.format(f=func.__name__, t=elapsed))
        return result
    return timed


def timemethod(func):
    """
    Decorator that measure the time it takes for a function to complete
    Usage:
      @timemethod
      def time_consuming_method(...):
    """
    @functools.wraps(func)
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        elapsed = '{0}'.format(te - ts)
        print('{c}.{f} took: {t} sec'.format(
            c=args[0].__class__.__name__, f=func.__name__, t=elapsed))
        return result
    return timed
