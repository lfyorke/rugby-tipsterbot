from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import glob
from functools import wraps
from time import time

DATA_DIR = 'data'

FEATURES = ['Ranking', 'avg_conceded_past_5', 'avg_scored_past_5', 'away_perf_cumulative',
            'home_perf_cumulative', 'conceded_cumulative', 'location', 'perf_past_1', 'perf_past_5',
            'performance_cumulative', 'scored_cumulative']


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
def get_rows(df):
    """Aggregate the data into a single row per team per game with the columns we will need to
    impute features"""

    #ids = set(list(df['game_id']))
    ids = [291584, 291591, 291600]
    result = []
    columns = ['scored', 'conceded', 'team', 'location', 'date', 'result', 'home_res', 'away_res']

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
            r.append(single_game['result'].iloc[i]) # overall
            if i == 1:
                r.append(single_game['result'].iloc[i])
                r.append(0)
            else:
                r.append(0)
                r.append(single_game['result'].iloc[i])
            result.append(r)
    return pd.DataFrame(result, columns=columns)


@timeit
def produce_easy_features(df):
    """ Produce the features from the aggregated dataset, except the ranking
        Return a dataframe 
    """
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['week_number'] = [date.isocalendar()[1] for date in list(pd.Series(df['date']))]
    teams = set(list(df['team']))
    df.sort_values(by=['date'], axis=0, inplace=True)
    dfs = []
    for team in teams:
        temp_df = df[df['team'] == team]
        temp_df['scored_cumulative'] = temp_df.scored.cumsum()
        temp_df['conceded_cumulative'] = temp_df.conceded.cumsum()
        temp_df['performance_cumulative'] = temp_df.result.cumsum()*2
        temp_df['away_perf_cumulative'] = temp_df.away_res.cumsum()*2
        temp_df['home_perf_cumulative'] = temp_df.home_res.cumsum()*2
        temp_df['avg_scored_past_5'] = temp_df['scored'].rolling(min_periods=0, window=5).mean()
        temp_df['avg_conceded_past_5'] = temp_df['conceded'].rolling(min_periods=0, window=5).mean()
        temp_df['perf_past_5'] = temp_df['result'].rolling(min_periods=0, window=5).sum()*2
        temp_df['perf_past_1'] = temp_df['result'].rolling(min_periods=0, window=1).sum()*2
        dfs.append(temp_df)
    return pd.concat(dfs, axis=0)


@timeit
def produce_ranking_feature(df):
    """Produce the rankings feature. i.e a rolling league table
    """
    weeks = set(list(df['week_number']))
    ranking_dfs = []

    for week in weeks:
        temp_df = df[df['week_number'] == week]
        temp_df.sort_values(by=['performance_cumulative'], axis=0, inplace=True, ascending=False)
        temp_df = temp_df.reset_index()
        temp_df['Ranking'] = temp_df.index + 1
        ranking_dfs.append(temp_df)
    return pd.concat(ranking_dfs, axis=0)


@timeit
def get_future_games(df, dir=''):
    """Add the next game to the dataframe to be predicted.
       Checks the date and then picks the correct game, forward fill the data to do the prediction.
    """
    upcoming_games = pd.read_csv('upcoming-games.csv')
    upcoming_games['date'] = pd.to_datetime(upcoming_games['date'], format='%d-%b-%y')
    teams = set(list(df['team']))
    max_date = max(df['date'])
    possible_games = upcoming_games[upcoming_games['date'] > max_date]
    min_date = min(possible_games['date'])
    next_round = possible_games[possible_games['date'] == min_date]
    result = pd.concat([df, next_round], axis=0)
    cols_to_fill = list(result.columns)
    cols_to_fill.remove('result')

    dfs = []
    for team in teams:
        temp_df = result[result['team'] == team]
        temp_df[cols_to_fill] = temp_df[cols_to_fill].fillna(method='ffill')
        dfs.append(temp_df)

    return pd.concat(dfs, axis=0)


@timeit
def mlp_modelling(dataset):
    train = dataset.dropna()
    test = dataset[dataset.isnull().any(axis=1)]
    test.set_index('team', inplace=True)
    X = train[FEATURES]
    y = train['result']
    X_test = test[FEATURES]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(11, 11, 11), random_state=1)
    clf.fit(X, y)
    predictions = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    X_test['Predicted'] = predictions
    probs_df = pd.DataFrame(probs, columns=['Prob_Loss', 'Prob_Win'])
    X_test['Prob_Loss'] = list(pd.Series(probs_df['Prob_Loss']))
    X_test['Prob_Win'] = list(pd.Series(probs_df['Prob_Win']))
    return X_test



if __name__ == "__main__":
    data = fetch_data(DATA_DIR)
    grouped  = get_rows(data)
    f_grouped = produce_easy_features(grouped)
    f_complete = produce_ranking_feature(f_grouped)
    combined = get_future_games(f_complete)
    preds = mlp_modelling(combined)
    print(preds)
