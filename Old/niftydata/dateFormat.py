import pandas as pd

usd = pd.read_csv('./USD_INR.csv')

key = {}
key['Jan'] = '01'
key['Feb'] = '02'
key['Mar'] = '03'
key['Apr'] = '04'
key['May'] = '05'
key['Jun'] = '06'
key['Jul'] = '07'
key['Aug'] = '08'
key['Sep'] = '09'
key['Oct'] = '10'
key['Nov'] = '11'
key['Dec'] = '12'

for i in range (0,len(usd['Date'])):
	x = usd['Date'][i]
	print x,
	x = x[8:] + '-' + key[x[:3]] + '-' + x[4:6]
	usd['Date'][i] = x
	print x
print 'hola'
	
usd.to_csv('USD_INR.csv', encoding='utf-8', index=False) 
