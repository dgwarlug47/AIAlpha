import numpy as np
import pandas as pd
import copy


def labelling(next_day_returns, label_type):
    # next_day_returns should have dimensions (num_tickers, num_days)
    if isinstance(label_type, int):
        df = pd.DataFrame(copy.deepcopy(next_day_returns)).T
        df = df.shift(-(label_type - 1)).rolling(label_type).mean()
        return np.array(df.fillna(0).T)
    else:
        raise "this option of label_type is not provided yet"


def leverage(next_day_returns, info):
    if info['leverage_type'] == 'volatility_scaling':
        df = pd.DataFrame(next_day_returns)
        vol_tgt = info['vol_tgt']
        leverage_factor = (vol_tgt/(np.sqrt(252)*df.ewm(span=60).std())).shift(1)
        leverage_factor = leverage_factor.replace(np.inf, 0)
        leverage_factor = leverage_factor.replace(-np.inf, 0)
        leverage_factor = leverage_factor.fillna(0)
        return np.array(leverage_factor)
    elif info['leverage_type'] == 'variance_scaling':
        df = pd.DataFrame(next_day_returns)
        vol_tgt = info['vol_tgt']
        leverage_factor = (vol_tgt/(np.sqrt(252)*df.ewm(span=60).var())).shift(1)
        leverage_factor = leverage_factor.replace(np.inf, 0)
        leverage_factor = leverage_factor.replace(-np.inf, 0)
        leverage_factor = leverage_factor.fillna(0)
        return np.array(leverage_factor)
    elif info['leverage_type'] == 'None':
        return np.ones(next_day_returns.shape)
    else:
        raise "this option is not currenlty available"