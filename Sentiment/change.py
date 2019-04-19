import pandas as pd

fileName = 'sbi_news.xls'
sheetName = fileName[:-4]

data = pd.read_excel('./'+fileName,sheetName)

data = data[['Date','Close','Volume','TB_para','TB_sent','USD_INR','Nifty']]

vals = data['Date'].values

for i in range(0,len(data)):
	date  = vals[i]
	dateSplit = vals[i].split('-')
	for j in range(0,len(dateSplit)):
		if(len(dateSplit[j]) == 1):
			dateSplit[j] = '0' + dateSplit[j]
	newDate = dateSplit[2]+'-'+dateSplit[1]+'-'+dateSplit[0]
	data['Date'][i]=newDate
	# print date, dateSplit , newDate , data['Date'][i]

data.to_csv(fileName[:-4]+'.csv', encoding='utf-8', index=False) 
print data.head()