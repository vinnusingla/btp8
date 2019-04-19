import pandas as pd

files = ['SBIN.NS.csv','INFY.NS.csv','LT.NS.csv','RELIANCE.NS.csv','MARUTI.NS.csv', 'USDINR.csv']

name = 'MARUTI.NS.csv'
data = pd.read_csv('./'+name)[['Close','Volume','Date']]


for name2 in files: 
	if name2 == name:
		continue
	data2 = pd.read_csv('./'+name2)
	data2 = data2[['Close','Date']]
	data2.rename(columns={'Close': name2[:-7]}, inplace=True)
	# data2.columns[0] = name2[:-6] 
	data = data.merge(data2,how='left', on='Date')

data = data.drop(['Date'],axis=1)	
data.to_csv('ALL.'+name, encoding='utf-8', index=False) 
