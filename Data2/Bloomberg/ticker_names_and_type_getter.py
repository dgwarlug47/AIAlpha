import pandas as pd
all_assets = pd.read_excel('assets.xlsx', engine = 'openpyxl')
num_assests = all_assets.shape[0]
tickers_name_type = {}
for asset_index in range(num_assests):
    tickers_name_type[all_assets.loc[asset_index]['Name']] = all_assets.loc[asset_index]['Class']
import pickle
with open('tickers_names_and_type.pkl', 'wb') as f:
    pickle.dump(tickers_name_type, f)
for typ in list(tickers_name_type.values()):
    with open(typ + "_tickers", 'wt') as f:
        for ticker_name in tickers_name_type.keys():
            if tickers_name_type[ticker_name] == typ:
                f.write(ticker_name + '\n')