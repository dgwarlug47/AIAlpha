import numpy as np
import pandas as pd


def spectral(df, width):
    """
    Compute rolling Fourier decomposition on the given window size.
    df: price values for each asset
    width: number of frequencies to compute
    *returns* tensor of shape (days, assets, frequencies)
    """
    df = df.pct_change() # returns
    spectrum = {}
    for col in df.columns:
        x = df[col].values
        ft_mat = np.zeros((len(x), 2*width))
        for i in range(2*width, len(x)):
            ft_mat[i] = np.abs(fft(x[i - (2*width - 1):i + 1]))
        # throw away redundant half
        spectrum[col] = pd.DataFrame(ft_mat[:, :width]).apply(normalize, args=[width])
    spectrum = np.stack(list(spectrum.values())).transpose(1, 0, 2)
    return spectrum


def simple_momentum(returns, days):
    # returns should have dimensions (sim)
  return returns.rolling(window = int(days)).sum()/(returns.ewm(span=60).std()*np.sqrt(days))


def macd_momentum(P, s, l):
  EWMAS = P.ewm(halflife=((np.log(0.5))/(np.log(1 -(1/s) )))).mean()
  EWMAL = P.ewm(halflife=((np.log(0.5))/(np.log(1 -(1/l) )))).mean()
  x = EWMAS - EWMAL
  STD = P.rolling(63).std()
  y = (x/STD)
  z = (y/(y.rolling(252).std().fillna(0)))
  return z


def positive_part(x):
    if(x > 0.0):
        y = x
    else:
        y = 0
    return y


def negative_part(x):
    if(x < 0.0):
        y = x
    else:
        y = 0
    return y


def raw_returns_lags(returns, max_lag):
  df = pd.DataFrame()
  for lag in range(max_lag):
    df['return_lag_' + str(lag)] = returns.shift(lag)
  return df


def prices_normalized(prices, rolling_window, max_lag):
  df = pd.DataFrame()
  prices_mean = prices.rolling(rolling_window).mean()
  prices_std = prices.rolling(rolling_window).std()
  z_score = (prices - prices_mean) / prices_std
  for lag in range(max_lag):
    df['price_normalized_rolling_window_' + str(rolling_window) + '_' + str(lag)] = z_score.shift(lag)
  return df


def chap_features(df1):
    # To be Modified to replicate the features in the paper Enhancing Momentum Strategies
    # from Kim et al. There are 2 main sets of features: 
    # 1- Momentum based features
    # 2 - MACD features based on the AHL paper dissecting time series momentum etc ... 
  vol_tgt = 0.010
  # df = pd.read_csv(file_name, parse_dates=['Date'], index_col='Date')
  # df = pd.read_csv(file_name, index_col='Date')
  # mom macd features CTA style paper AHL
  # append the risk indexes to the data
  # df = pd.concat([df1, df_risk_indexes],axis=1)
  # dont append    
  df = df1.copy()

  price_lambda1 =  8
  df['prices_mv1'] =  df['Price'].ewm(com = price_lambda1, adjust = False).mean()
  price_lambda2 = 16
  df['prices_mv2'] =  df['Price'].ewm(com = price_lambda2, adjust = False).mean()
  price_lambda3 = 32
  df['prices_mv3'] =  df['Price'].ewm(com = price_lambda3, adjust = False).mean()
  price_lambda4 = 24
  df['prices_mv4'] =  df['Price'].ewm(com = price_lambda4, adjust = False).mean()
  price_lambda5 = 48
  df['prices_mv5'] =  df['Price'].ewm(com = price_lambda5, adjust = False).mean()
  price_lambda6 = 96
  df['prices_mv6'] =  df['Price'].ewm(com = price_lambda6, adjust = False).mean() 
  df['prices_x1'] = df['prices_mv1'] - df['prices_mv4'] 
  df['prices_x2'] = df['prices_mv2'] - df['prices_mv5']
  df['prices_x3'] = df['prices_mv3'] - df['prices_mv6']
  df['vol_prices'] = df['Price'].rolling(63).std()
  df['prices_y1'] = df['prices_x1']/(df['vol_prices']+0.00001)
  df['prices_y2'] = df['prices_x2']/(df['vol_prices']+0.00001)
  df['prices_y3'] = df['prices_x3']/(df['vol_prices']+0.00001)
  df['vol_y1'] = df['prices_y1'].rolling(252).std()
  df['vol_y2'] = df['prices_y2'].rolling(252).std()
  df['vol_y3'] = df['prices_y3'].rolling(252).std()
  df['prices_z1'] = df['prices_y1']/(df['vol_y1']+0.00001)
  df['prices_z2'] = df['prices_y2']/(df['vol_y2']+0.00001)
  df['prices_z3'] = df['prices_y3']/(df['vol_y3']+0.00001)

  # momentum features
  df['rets'] = (df['Price']/df['Price'].shift(1) - 1.0)
  #df = df[df['rets'] != 0.0]
  df['sigma_daily1'] =  df['rets'].rolling(21).std()
  df['sigma_daily2'] =  df['rets'].rolling(42).std()
  df['sigma_daily3'] =  df['rets'].rolling(84).std()
  
  df['sigma_daily'] = (df['sigma_daily1'] + df['sigma_daily2']  +  df['sigma_daily3'])/3.0
  
    
  #df['mom_2d'] =   df['rets'].shift(1)/(np.sqrt(1)*(df['sigma_daily']+0.00001))
  #df['mom_3d'] =   df['rets'].shift(2)/(np.sqrt(1)*(df['sigma_daily']+0.00001))
  #df['mom_4d'] =   df['rets'].shift(3)/(np.sqrt(1)*(df['sigma_daily']+0.00001))
  #df['mom_5d'] =   df['rets'].shift(4)/(np.sqrt(1)*(df['sigma_daily']+0.00001))
  #df['mom_6d'] =   df['rets'].shift(5)/(np.sqrt(1)*(df['sigma_daily']+0.00001))
  #df['mom_7d'] =   df['rets'].shift(6)/(np.sqrt(1)*(df['sigma_daily']+0.00001))
  #df['mom_8d'] =   df['rets'].shift(7)/(np.sqrt(1)*(df['sigma_daily']+0.00001))
      
  
  df['mom_1d'] =   df['rets'].rolling(5).sum()/(np.sqrt(5)*(df['sigma_daily']+0.001))
  df['mom_20d'] =  df['rets'].rolling(21).sum()/(np.sqrt(21)*(df['sigma_daily']+0.0001)) #0.00001 original
  df['mom_60d'] =  df['rets'].rolling(63).sum()/(np.sqrt(63)*(df['sigma_daily']+0.0001))
  df['mom_120d'] = df['rets'].rolling(126).sum()/(np.sqrt(126)*(df['sigma_daily']+0.00001))
  df['mom_200d'] = df['rets'].rolling(252).sum()/(np.sqrt(252)*(df['sigma_daily']+0.00001)) 
  
  # new features
  df['skew_250'] = df['rets'].rolling(252).skew() #3.0*(df['mom_60d'].rolling(63).corr(df['sigma_daily'])) #df['rets'].rolling(252).skew()
  df['vol_250'] = 3.0*(df['sigma_daily'] -df['sigma_daily'].rolling(250).mean())/(df['sigma_daily'].rolling(250).max() - df['sigma_daily'].rolling(250).min())
#3.0*(df['mom_60d'].rolling(63).corr(df['sigma_daily'])) #
#    df['mom_1d_pos'] =   df['mom_1d'].apply(positive_part)
#    df['mom_1d_neg'] =   df['mom_1d'].apply(negative_part)
#    df['mom_2d_pos'] =   df['mom_2d'].apply(positive_part)
#    df['mom_2d_neg'] =   df['mom_2d'].apply(negative_part)
#    df['mom_3d_pos'] =   df['mom_3d'].apply(positive_part)
#    df['mom_3d_neg'] =   df['mom_3d'].apply(negative_part)
#    df['mom_4d_pos'] =   df['mom_4d'].apply(positive_part)
#    df['mom_4d_neg'] =   df['mom_4d'].apply(negative_part)
#    df['mom_5d_pos'] =   df['mom_5d'].apply(positive_part)
#    df['mom_5d_neg'] =   df['mom_5d'].apply(negative_part)
#    df['mom_6d_pos'] =   df['mom_6d'].apply(positive_part)
#    df['mom_6d_neg'] =   df['mom_6d'].apply(negative_part)
#    df['mom_7d_pos'] =   df['mom_7d'].apply(positive_part)
#    df['mom_7d_neg'] =   df['mom_7d'].apply(negative_part)
#    df['mom_8d_pos'] =   df['mom_8d'].apply(positive_part)
#    df['mom_8d_neg'] =   df['mom_8d'].apply(negative_part)
  
  df['prices_z1_pos'] = df['prices_z1'].apply(positive_part)
  df['prices_z1_neg'] = df['prices_z1'].apply(negative_part)
      
  df['prices_z2_pos'] = df['prices_z2'].apply(positive_part)
  df['prices_z2_neg'] = df['prices_z2'].apply(negative_part)
  
  df['prices_z3_pos'] = df['prices_z3'].apply(positive_part)
  df['prices_z3_neg'] = df['prices_z3'].apply(negative_part)
  
  
  df['mom_1d_pos'] =  df['mom_1d'].apply(positive_part)
  df['mom_1d_neg'] =  df['mom_1d'].apply(negative_part)     
  df['mom_20d_pos'] =  df['mom_20d'].apply(positive_part)
  df['mom_20d_neg'] =  df['mom_20d'].apply(negative_part)
  df['mom_60d_pos'] =  df['mom_60d'].apply(positive_part)
  df['mom_60d_neg'] =  df['mom_60d'].apply(negative_part)
  df['mom_120d_pos'] = df['mom_120d'].apply(positive_part)
  df['mom_120d_neg'] = df['mom_120d'].apply(negative_part)    
  df['mom_200d_pos'] =  df['mom_200d'].apply(positive_part)
  df['mom_200d_neg'] = df['mom_200d'].apply(negative_part)
  df['skew250_pos'] = df['skew_250'].apply(positive_part)
  df['skew250_neg'] = df['skew_250'].apply(negative_part)
  df['vol_250_pos'] = df['vol_250'].apply(positive_part)
  df['vol_250_neg'] = df['vol_250'].apply(negative_part)
  
  df['rets_d1'] = df['rets'].shift(-1)
  #df = df.dropna()
  
  df.replace(np.nan,0.0, inplace = True)
  df_out = df.copy()
  #print('mom_3d',np.corrcoef(df['mom_3d'],df['rets_d1']))
  #print('mom_20d',np.corrcoef(df['mom_20d'],df['rets_d1']))
  #print('mom_60d',np.corrcoef(df['mom_60d'],df['rets_d1']))
  #print('mom_120d',np.corrcoef(df['mom_120d'],df['rets_d1']))
  #print('mom_200d',np.corrcoef(df['mom_200d'],df['rets_d1']))
  #print('prices_z1',np.corrcoef(df['prices_z1'],df['rets_d1']))
  #print('prices_z2',np.corrcoef(df['prices_z2'],df['rets_d1']))
  #print('prices_z3',np.corrcoef(df['prices_z2'],df['rets_d1']))
  
  #return df_out[['rets_norm','mom_1d_pos','mom_1d_neg','mom_2d_pos','mom_2d_neg','mom_3d_pos','mom_3d_neg',
  #              'mom_4d_pos','mom_4d_neg','mom_5d_pos','mom_5d_neg','mom_20d_pos','mom_20d_neg',
  #              'mom_60d_pos','mom_60d_neg','mom_120d_pos','mom_120d_neg','mom_200d_pos','mom_200d_neg']]
  return df_out[['mom_1d_pos','mom_1d_neg','mom_20d_pos','mom_20d_neg','mom_60d_pos','mom_60d_neg', 
                'mom_120d_pos','mom_120d_neg','mom_200d_pos','mom_200d_neg','skew250_pos','skew250_neg','vol_250_pos','vol_250_neg']]


def davi_features2(df1):
    # To be Modified to replicate the features in the paper Enhancing Momentum Strategies
    # from Kim et al. There are 2 main sets of features: 
    # 1- Momentum based features
    # 2 - MACD features based on the AHL paper dissecting time series momentum etc ... 
  df = pd.DataFrame()
  df2 = pd.DataFrame()
  df2['rets'] = (df1['Price']/df1['Price'].shift(1) - 1.0)
  rollings = [10, 21, 63, 126, 252]
  skew_rollings = [63, 126, 252]
  for rolling in rollings:
    df['mom_davi_' + str(rolling) + 'd'] = df2['rets'].rolling(rolling).mean()
  for skew_rolling in skew_rollings:
    df['skew_' + str(skew_rolling)] = df2['rets'].rolling(skew_rolling).skew()

  features = list(df.columns)

  final_df = pd.DataFrame()

  for feature in features:
    final_df[feature + '_pos'] = df[feature].apply(positive_part)
    final_df[feature + '_neg'] = df[feature].apply(negative_part)

  df_out = final_df.copy()

  return df_out

def davi_features(df1):
    # To be Modified to replicate the features in the paper Enhancing Momentum Strategies
    # from Kim et al. There are 2 main sets of features: 
    # 1- Momentum based features
    # 2 - MACD features based on the AHL paper dissecting time series momentum etc ... 
  df = pd.DataFrame()
  df2 = pd.DataFrame()
  df2['rets'] = (df1['Price']/df1['Price'].shift(1) - 1.0)
  mean_rollings = [5, 21, 63, 126, 252]
  std_rollings = [10, 21, 63, 126, 252]
  skew_rollings = [63, 126, 252]
  for mean_rolling in mean_rollings:
    df['ret_mean_' + str(mean_rolling) + 'd'] = df2['rets'].rolling(mean_rolling).mean()
  for std_rolling in std_rollings:
    df['ret_std_' + str(std_rolling)] = (df2['rets'].rolling(std_rolling).std())
  for skew_rolling in skew_rollings:
    df['skewnesss_' + str(skew_rolling)] = df2['rets'].rolling(skew_rolling).skew()

  features = list(df.columns)

  final_df = pd.DataFrame()

  for feature in features:
    final_df[feature + '_pos'] = df[feature].apply(positive_part)
    final_df[feature + '_neg'] = df[feature].apply(negative_part)

  df_out = final_df.copy()

  return df_out