from sklearn.neural_network import MLPClassifier
import pandas as pd
import glob
from functools import wraps
from time import time

DATA_DIR = 'data'


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func: {0}  took: {1} sec'.format(f.__name__, te-ts))
        return result
    return wrap


@timeit
def fetch_data(dir):
    """ Go to the data directory and filter it to return only the files we will need.
        I.e. Aviva Premiership games for 2017/18 season
        Returns a dataframe. 
    """
    l = [pd.read_csv(filename, encoding='latin-1') for filename in glob.glob(DATA_DIR + "/*.csv")]
    df = pd.concat(l, axis=0)
    df = df[df["competition"].str.contains("Aviva Premiership 2018") == True]
    return df

@timeit
def get_single_row(df):
    """Aggregate the data into a single row per team per game with the columns we will need to
    impute features"""

    ids = set(list(df['game_id']))
    result = []

    for game_id in ids:
        single_game = df[df['game_id'] == game_id]
        date = single_game['date'][0]
        single_game["result"] = 0
        if single_game['score'].iloc[0] > single_game['score'].iloc[-1]:
            single_game["result"][0:23] = 1
            single_game["result"][23:] = 0
        else:
            single_game["result"][0:23] = 0
            single_game["result"][23:] = 1
        for i in [1, -1]:
            r = []
            r.append(single_game['score'].iloc[i])
            r.append(single_game['score'].iloc[i*-1])
            r.append(single_game['team'].iloc[i])
            r.append(i)
            r.append(date)
            r.append(single_game['result'].iloc[i])
            result.append(r)
    return result


if __name__ == "__main__":
    data = fetch_data(DATA_DIR)
    grouped  = get_single_row(data)
