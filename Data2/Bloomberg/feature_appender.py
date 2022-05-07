import pandas as pd
import os
tickers_file_path = 'tickers_Equity_but_SGX_and_something_else'
new_directory = 'Equity_directory3'
old_directory1 = 'All Futures Momentum Features Bloomberg14'
old_directory2 = 'Specific Equities Features'

def loadTickers(fileName):
    # receives as input the address where the names of the tickers are stored.
    # outputs the list of the tickers names.
    lineList = list()
    with open(fileName) as f:
        for line in f:
            lineList.append(line)
    lineList = [line.rstrip('\n') for line in open(fileName)]
    return lineList

tickers = loadTickers(tickers_file_path)

path = os.path.join(os.getcwd(), new_directory)

try:
    os.rmdir(path)
except:
    print('been trying to meet you')

os.mkdir(path)

for ticker in tickers:
    path1 = old_directory1 + '/' + ticker + '.csv'
    path2 = old_directory2 + '/' + ticker + '.csv'
    df1 = pd.read_csv(path1)
    df1.index = df1.iloc[:,0]
    df1 = df1 = df1.iloc[:,1:]
    minimum = min(df1.index)
    maximum = max(df1.index)
    df2 = pd.read_csv(path2)
    df2.index = df2.iloc[:,0]
    df2 = df2.iloc[:,1:]
    df2 = df2[minimum:maximum]
    df = pd.concat([df1, df2], axis=1)
    
    os.chdir('./' + new_directory)
    df.to_csv(ticker+'.csv')
    os.chdir('./..')