import pandas as pd

usd = pd.read_csv('./USD_INR.csv')
name = 'SBIN.NS.csv'
data = pd.read_csv('./'+name)

y = data.merge(usd[['USD','Date']],how='left', on='Date')
	
y.to_csv(name, encoding='utf-8', index=False) 
