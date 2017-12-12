from sklearn.neural_network import MLPClassifier
import pandas as pd
from functools import wraps
from time import time


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func: {0} args: [{1}, {2}] took: {3} sec'.format(f.__name__, args, kw, te-ts))
        return result
    return wrap

@timeit
def transform():
    """Aggregate the data to prepare the required features for analysis"""

    pass


if __name__ == "__main__":
    pass