# import libraries
import urllib.request
from bs4 import BeautifulSoup
#import xlwt 
#from xlwt import Workbook   
# Workbook is created 
from xlutils.copy import copy
import xlrd
#from xlwt import workbook  
wb = copy(xlrd.open_workbook('Nifty_2015_till_date_news.xls'))

# for loop

maruti_suzuki=[] 
j=-1
for yr in range(2015,2020):
    for i in range(1,35):
        if i%10==1:
            j=j+1
            x=str('https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=MU01&scat=&pageno=')+str(i)+str('&next=')+str(j)+str('&durationType=Y&Year=')+str(yr)+str('&duration=1&news_type=')
        else:
            x=str('https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=MU01&scat=&pageno=')+str(i)+str('&next=')+str(j)+str('&durationType=Y&Year=')+str(yr)+str('&duration=1&news_type=')
        maruti_suzuki.append(x)
#print (maruti_suzuki)
#query the website and return the html to the variable ‘page’
# for loop

data = []
for i in maruti_suzuki:
 # query the website and return the html to the variable ‘page’
 page = urllib.request.urlopen(i)
 # parse the html using beautiful soap and store in variable `soup`
 soup = BeautifulSoup(page, 'html.parser')
 for m in soup:
    news_box = soup.find('a', attrs={'class': 'g_14bl'})
    if news_box!=None:
        #print ('This is the news:',news_box)
        news = news_box.text.strip() # strip() is used to remove starting and trailing
        #print ('This is the news:',news)
        # get the index price
        soup.find('a', attrs={'class': 'g_14bl'}).decompose()
        date_box = soup.find('p', attrs={'class':'PT3 a_10dgry'})
        #print(date_box)
        date = date_box.text
        #print ('Time and date:',date)
        # save the data in tuple
        soup.find('p', attrs={'class': 'PT3 a_10dgry'}).decompose()
        data.append((news, date))
        #print ("This is data:",data)
    elif news_box==None:
        break
# open a csv file with append, so old data will not be erased
# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Maruti_Suzuki_news')
sheet1.write(0,0,'News')
sheet1.write(0,1,'Time | Date | Source') 
for i in range(0,len(data)):
    sheet1.write(i+1,0,data[i][0])
    sheet1.write(i+1,1,data[i][1])
print ("Done")
'''
sbi=[] 
j=-1
for yr in range(2015,2020):
    for i in range(1,35):
        if i%10==1:
            j=j+1
            x=str('https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=SBI&scat=&pageno=')+str(i)+str('&next=')+str(j)+str('&durationType=Y&Year=')+str(yr)+str('&duration=1&news_type=')
        else:
            x=str('https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=SBI&scat=&pageno=')+str(i)+str('&next=')+str(j)+str('&durationType=Y&Year=')+str(yr)+str('&duration=1&news_type=')
        sbi.append(x)
data = []
for i in sbi:
 # query the website and return the html to the variable ‘page’
 page = urllib.request.urlopen(i)
 # parse the html using beautiful soap and store in variable `soup`
 soup = BeautifulSoup(page, 'html.parser')
 for m in soup:
    news_box = soup.find('a', attrs={'class': 'g_14bl'})
    if news_box!=None:
        #print ('This is the news:',news_box)
        news = news_box.text.strip() # strip() is used to remove starting and trailing
        #print ('This is the news:',news)
        # get the index price
        soup.find('a', attrs={'class': 'g_14bl'}).decompose()
        date_box = soup.find('p', attrs={'class':'PT3 a_10dgry'})
        #print(date_box)
        date = date_box.text
        #print ('Time and date:',date)
        # save the data in tuple
        soup.find('p', attrs={'class': 'PT3 a_10dgry'}).decompose()
        data.append((news, date))
        #print ("This is data:",data)
    elif news_box==None:
        break
# open a csv file with append, so old data will not be erased
sheet1 = wb.add_sheet('sbi_news')
sheet1.write(0,0,'News')
sheet1.write(0,1,'Time | Date | Source') 
for i in range(0,len(data)):
    sheet1.write(i+1,0,data[i][0])
    sheet1.write(i+1,1,data[i][1])
print ("Done")

infosys=[] 
j=-1
for yr in range(2015,2020):
    for i in range(1,35):
        if i%10==1:
            j=j+1
            x=str('https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=IT&scat=&pageno=')+str(i)+str('&next=')+str(j)+str('&durationType=Y&Year=')+str(yr)+str('&duration=1&news_type=')
        else:
            x=str('https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=IT&scat=&pageno=')+str(i)+str('&next=')+str(j)+str('&durationType=Y&Year=')+str(yr)+str('&duration=1&news_type=')
        infosys.append(x)
data = []
for i in infosys:
 # query the website and return the html to the variable ‘page’
 page = urllib.request.urlopen(i)
 # parse the html using beautiful soap and store in variable `soup`
 soup = BeautifulSoup(page, 'html.parser')
 for m in soup:
    news_box = soup.find('a', attrs={'class': 'g_14bl'})
    if news_box!=None:
        #print ('This is the news:',news_box)
        news = news_box.text.strip() # strip() is used to remove starting and trailing
        #print ('This is the news:',news)
        # get the index price
        soup.find('a', attrs={'class': 'g_14bl'}).decompose()
        date_box = soup.find('p', attrs={'class':'PT3 a_10dgry'})
        #print(date_box)
        date = date_box.text
        #print ('Time and date:',date)
        # save the data in tuple
        soup.find('p', attrs={'class': 'PT3 a_10dgry'}).decompose()
        data.append((news, date))
        #print ("This is data:",data)
    elif news_box==None:
        break
# open a csv file with append, so old data will not be erased

sheet1 = wb.add_sheet('Infosys_news')
sheet1.write(0,0,'News')
sheet1.write(0,1,'Time | Date | Source') 
for i in range(0,len(data)):
    sheet1.write(i+1,0,data[i][0])
    sheet1.write(i+1,1,data[i][1])
print ("Done")
'''
reliance_industries=[] 
j=-1
for yr in range(2015,2020):
    for i in range(1,35):
        if i%10==1:
            j=j+1
            x=str('https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=RI&scat=&pageno=')+str(i)+str('&next=')+str(j)+str('&durationType=Y&Year=')+str(yr)+str('&duration=1&news_type=')
        else:
            x=str('https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=RI&scat=&pageno=')+str(i)+str('&next=')+str(j)+str('&durationType=Y&Year=')+str(yr)+str('&duration=1&news_type=')
        reliance_industries.append(x)
data = []
for i in reliance_industries:
 # query the website and return the html to the variable ‘page’
 page = urllib.request.urlopen(i)
 # parse the html using beautiful soap and store in variable `soup`
 soup = BeautifulSoup(page, 'html.parser')
 for m in soup:
    news_box = soup.find('a', attrs={'class': 'g_14bl'})
    if news_box!=None:
        #print ('This is the news:',news_box)
        news = news_box.text.strip() # strip() is used to remove starting and trailing
        #print ('This is the news:',news)
        # get the index price
        soup.find('a', attrs={'class': 'g_14bl'}).decompose()
        date_box = soup.find('p', attrs={'class':'PT3 a_10dgry'})
        #print(date_box)
        date = date_box.text
        #print ('Time and date:',date)
        # save the data in tuple
        soup.find('p', attrs={'class': 'PT3 a_10dgry'}).decompose()
        data.append((news, date))
        #print ("This is data:",data)
    elif news_box==None:
        break
# open a csv file with append, so old data will not be erased
print ("Done")
sheet1 = wb.add_sheet('Reliance_industries_news')
sheet1.write(0,0,'News')
sheet1.write(0,1,'Time | Date | Source') 
for i in range(0,len(data)):
    sheet1.write(i+1,0,data[i][0])
    sheet1.write(i+1,1,data[i][1])
        
                     
larsen_and_toubro=[] 
j=-1
for yr in range(2015,2020):
    for i in range(1,35):
        if i%10==1:
            j=j+1
            x=str('https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=LT&scat=&pageno=')+str(i)+str('&next=')+str(j)+str('&durationType=Y&Year=')+str(yr)+str('&duration=1&news_type=')
        else:
            x=str('https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=LT&scat=&pageno=')+str(i)+str('&next=')+str(j)+str('&durationType=Y&Year=')+str(yr)+str('&duration=1&news_type=')
        larsen_and_toubro.append(x)
data = []
for i in larsen_and_toubro:
 # query the website and return the html to the variable ‘page’
 page = urllib.request.urlopen(i)
 # parse the html using beautiful soap and store in variable `soup`
 soup = BeautifulSoup(page, 'html.parser')
 for m in soup:
    news_box = soup.find('a', attrs={'class': 'g_14bl'})
    if news_box!=None:
        #print ('This is the news:',news_box)
        news = news_box.text.strip() # strip() is used to remove starting and trailing
        #print ('This is the news:',news)
        # get the index price
        soup.find('a', attrs={'class': 'g_14bl'}).decompose()
        date_box = soup.find('p', attrs={'class':'PT3 a_10dgry'})
        #print(date_box)
        date = date_box.text
        #print ('Time and date:',date)
        # save the data in tuple
        soup.find('p', attrs={'class': 'PT3 a_10dgry'}).decompose()
        data.append((news, date))
        #print ("This is data:",data)
    elif news_box==None:
        break
# open a csv file with append, so old data will not be erased
sheet1 = wb.add_sheet('larsen_and_toubro_news')
sheet1.write(0,0,'News')
sheet1.write(0,1,'Time | Date | Source') 
for i in range(0,len(data)):
    sheet1.write(i+1,0,data[i][0])
    sheet1.write(i+1,1,data[i][1])
print ("Done")
print ("Excel Sheet:'Nifty_2015_till_date_news.xls' updated")
wb.save('Nifty_2015_till_date_news.xls')    
