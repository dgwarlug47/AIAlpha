

# basic script to plot the correlation between different features, the distribution per say

base_path = './Futures Deep Momentum Features Bloomberg/'
name='US10Y'

path_for_real_returns = base_path + name + '.csv'
import pandas as pd
Data = pd.read_csv(path_for_real_returns)
Data = Data.rename(columns = {'Unnamed: 0': 'date'})
Data = Data.query("date >="+ "'" + '2000-01-01' + "'")
Data = Data.query("date <"+ "'" + '2021-01-01' + "'")

import seaborn as sns
import matplotlib.pyplot as plt
sns.relplot(data=Data, x='returns next day', y = 'MACD_short_32_long_96')
plt.show()