import xlrd 
#import pandas as pd
loc = ("Nifty_2015_till_date_news.xls")   
wb = xlrd.open_workbook(loc) 

from xlwt import Workbook 
  
# Workbook is created 
wb1 = Workbook() 
#print(sheet.cell_value(1,0) )
def sort_news(sheet,stock):
    news=[]
    date=[]
    time=[] 
    date_time=[]
    for i in range(1,sheet.nrows):
        date_time.append(sheet.cell_value(i, 1))
        #print(date_time[i])
        x=(date_time[i-1].split('|'))
        #print(x[0])
        time.append(x[0])
        date.append(x[1])
    for i in range(1,sheet.nrows): 
        news.append(sheet.cell_value(i, 0))
    #print (news)
    #print (time)
    #print (date)
    shape=[sheet.nrows-1,sheet.ncols]

    for i in range(0,shape[0]):
        date[i]=(date[i].replace(' Jan ','-1-'))
        date[i]=(date[i].replace(' Feb ','-2-'))
        date[i]=(date[i].replace(' Mar ','-3-'))
        date[i]=(date[i].replace(' Apr ','-4-'))
        date[i]=(date[i].replace(' May ','-5-'))
        date[i]=(date[i].replace(' Jun ','-6-'))
        date[i]=(date[i].replace(' Jul ','-7-'))
        date[i]=(date[i].replace(' Aug ','-8-'))
        date[i]=(date[i].replace(' Sep ','-9-'))
        date[i]=(date[i].replace(' Oct ','-10-'))
        date[i]=(date[i].replace(' Nov ','-11-'))
        date[i]=(date[i].replace(' Dec ','-12-'))
        time[i]=(time[i].replace(' pm ','-pm'))
        time[i]=(time[i].replace(' am ','-am'))
    for i in range(0,shape[0]):
        x=time[i].split('-')
        if x[1]=='pm':
            time[i]=round((float(x[0])+12.00),2)
        else:
            time[i]=x[0]
    mod_date=[]
    for i in range(0,shape[0]):
        x=date[i].split('-')
        if float(time[i])>=17.00:
            x[0]=int(x[0])+1
            y=str(x[0])+str('-')+str(x[1])+str('-')+str(x[2])
            #print (y)
            y=y.replace(u'\xa0',u'')
            mod_date.append(y)
        else:
            x[0]=int(x[0])
            y=str(x[0])+str('-')+str(x[1])+str('-')+str(x[2])
            #print (y)
            y=y.replace(u'\xa0',u'')
            mod_date.append(y)
        
    #print(mod_date)
    new_news_data=[]
    for i in range(0,len(mod_date)):
        news_append=[]
        #new_dates_data.append(mod_date[i])
        news_append.append(news[i])
        news_append.append(str(" | "))
        for j in range(i+1,len(mod_date)):
            if mod_date[i]==mod_date[j]:
                news_append.append(news[j])
                news_append.append(str(" | "))
                mod_date[j]=str('')
                news[j]=str('')
            else:
                pass    
        new_news_data.append(news_append)
    #print(new_news_data)

    sheet1 = wb1.add_sheet(str(stock))
    k=0
    sheet1.write(0,0,'News')
    sheet1.write(0,1,'Date')
    for i in range(0,len(news)):
        # for j in range(0,len(new_news_data[i])-1,2):
        if new_news_data[i][0]!=str(''):
            k=k+1
            sheet1.write(k,0,new_news_data[i])
            sheet1.write(k,1,mod_date[i])
        else:
            pass

nifty=['sbi_news','Infosys_news','Maruti_Suzuki_news','Reliance_industries_news','larsen_and_toubro_news']

for i in range(0,len(nifty)):
    sheet = wb.sheet_by_index(i)
    sort_news(sheet,nifty[i])
print("Excel Sheet:'Nifty_sorted.xls' updated")
wb1.save('Nifty_sorted.xls')