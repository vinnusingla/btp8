import pandas
from xlwt import Workbook 
import xlrd  
import datetime
from xlrd import open_workbook
# Workbook is created 
from xlutils.copy import copy
from textblob import TextBlob
import numpy as np
from xlrd import open_workbook
#import matplotlib.pyplot as plt
#import sklearn.svm as svm
import statistics

wb1 = Workbook() 

def data(prices1,news1,stock,num,usd_inr,nifty_data):
    shape_prices=prices1.shape
    shape_news=news1.shape
    shape_nifty=nifty_data.shape
    shape_usd_inr=usd_inr.shape
    #usd_inr['Date'] = pandas.to_datetime(usd_inr['Date'])
    usd_inr['Date'] = pandas.to_datetime(usd_inr.Date)
    #print (usd_inr)
    usd_inr['Date'] = usd_inr['Date'].dt.strftime('%#d-%#m-%Y')
    nifty_data['Date'] = pandas.to_datetime(nifty_data.Date)
    nifty_data['Date'] = nifty_data['Date'].dt.strftime('%#d-%#m-%Y')       
    #print (usd_inr)
    news=[]
    date_news=[]
    date_prices=[]
    date_usdinr=[]
    date_nifty=[]
    usdinr=[]
    nifty88=[]
    price=[]
    volume=[]
    for i in range(0,shape_prices[0]):
        price.append(prices1.iloc[i]['Close'])
        x=(prices1.iloc[i]['Date']).split('/')
        x[0]=int(x[0])
        x[1]=int(x[1])
        x[2]=int(x[2])
        y=str(x[1])+str('-')+str(x[0])+str('-')+str(x[2])
        date_prices.append(y)
        volume.append(int(prices1.iloc[i]['Volume']))
        #print(date_time[i])
            #print(date_time[i])
    for i in range(0,shape_news[0]):
        news.append(news1.iloc[i]['News'])
        date_news.append(news1.iloc[i]['Date'])

    for i in range(0,shape_nifty[0]):
        nifty88.append(nifty_data.iloc[i]['Close'])
        #x=(nifty_data.iloc[i]['Date']).split('/')
        #print (x)
        #x[0]=int(x[0])
        #x[1]=int(x[1])
        #x[2]=int(x[2])
        #y=str(x[1])+str('-')+str(x[2])+str('-')+str(x[0])
        date_nifty.append(nifty_data.iloc[i]['Date'])
        #volume.append(int(prices1.iloc[i]['Volume']))    
        #print(date_prices)
    #print(date_news[2],date_prices[2])
    #test=0
    for i in range(0,shape_usd_inr[0]):
        usdinr.append(usd_inr.iloc[i]['Price'])
        #x=(usd_inr.iloc[i]['Date']).split('/')
        #print(x)
        #x[0]=int(x[0])
        #x[1]=int(x[1])
        #x[2]=int(x[2])
        #y=str(x[1])+str('-')+str(x[0])+str('-')+str(x[2])
        #print(y)
        #test=test+1
        #print(test,shape_usd_inr)
        date_usdinr.append(usd_inr.iloc[i]['Date'])
        #print(date_usdinr[0])

    sheet1 = wb1.add_sheet(str(stock))
    k=0
    sheet1.write(0,0,'Date')    
    sheet1.write(0,1,'Close')
    sheet1.write(0,2,'News')
    sheet1.write(0,3,'Volume')
    sheet1.write(0,7,'USD_INR')
    sheet1.write(0,8,'Nifty')
    for i in range(0,len(date_prices)):
        sheet1.write(i+1,2,'|x')
        sheet1.write(i+1,0,date_prices[i])    
        sheet1.write(i+1,1,price[i])
        sheet1.write(i+1,3,volume[i])
    wb1.save('Nifty_intermediate.xls')
    rb = open_workbook('Nifty_intermediate.xls')
    sheet1 = copy(rb)
    x=sheet1.get_sheet(num)
    #x.write(0,0,'|x')
    for i in range(0,len(date_prices)):
        for j in range(0,len(date_news)):
            if str(date_news[j])==str(date_prices[i]):
                x.write(i+1,2,news[j])
                k=k+1
            else:
                pass
    for i in range(0,len(date_prices)):
        for j in range(0,len(date_usdinr)):
            if str(date_usdinr[j])==str(date_prices[i]):
                x.write(i+1,7,usdinr[j])
            else:
                pass
    for i in range(0,len(date_prices)):
        for j in range(0,len(date_nifty)):
            if str(date_nifty[j])==str(date_prices[i]):
                x.write(i+1,8,nifty88[j])
            else:
                pass
            
    sht=str(stock)+'.xls'
    sheet1.save(sht)
nifty22=['sbi','infosys','maruti','reliance','l&t']
nifty11=['SBI','Infosys','Maruti Suzuki','Reliance Industries','L&T']
nifty=['sbi_news','Infosys_news','Maruti_Suzuki_news','Reliance_industries_news','larsen_and_toubro_news']
sheets=['SBIN.NS.csv','INFY.NS.csv','MARUTI.NS.csv','RELIANCE.NS.csv','LT.NS.csv']
for m in range(0,len(nifty)):
    #sheet = wb1.sheet_by_index(i)
    print('Updating data for stock:',nifty11[m])
    prices1=pandas.read_csv(sheets[m])
    usd_inr=pandas.read_csv('USD_INR Historical Data.csv')
    nifty_data=pandas.read_csv('^NSEI.csv')
    #print(usd_inr)
    news1=pandas.read_excel('Nifty_sorted.xls',nifty[m])
    data(prices1,news1,nifty[m],m,usd_inr,nifty_data)
    
print("Excel sheets updated")

positive_txt = open('positive1.txt') #importing positive words corpus  
negative_txt = open('negative1.txt') #importing negative words corpus 

pos_words = []
neg_words = []
for line in positive_txt:
    pos_words.append(line.split())

for line in negative_txt:
    neg_words.append(line.split())

posit_words=[]
for i in range(0,len(pos_words)):
        feats = {}
        feats["({0})".format(pos_words[i])] = "Positive sentiment word"
        posit_words.append(feats)


negat_words=[]
for i in range(0,len(neg_words)):
        feats = {}

        feats["({0})".format(neg_words[i])] = "Negative sentiment word"
        negat_words.append(feats)


####################################################

def sentiment(text2,stock):  
  comments1=str(text2) 
  commentsfilter=[]
  x=comments1.split('|')
  for i in range(1,len(x)):
    commentsfilter.append(x[i])
  #print (commentsfilter)
  text2 = '. '.join(commentsfilter)
  #print (text2)
 
  name=str('maruti')  #main subject
  s = list(text2)
  
  for i in range(0,len(s)):
    if s[i]=="'":
       s[i]=" "
    else:
       pass
 
  text2="".join(s)


  #print text2
  text1=(text2)
  blob=TextBlob(text2)
  sentences_break=blob.sentences
  polarity=blob.sentiment.polarity
  sum1=0
  for i in range(1,len(x)):
    blob=TextBlob(x[i])
    polarity1=blob.sentiment.polarity
    sum1=sum1+polarity1
  polarity2=sum1/len(x)


  #filterednouns=raw_input ("Please enter the name of the company to be analysed")
  text1=text1.lower()
  blob1=TextBlob(text1)
  sentences_break1=blob1.sentences

  #print 'Total Number of Sentences in given text is: '+str(len(sentences_break1))

#  words_in_extracted_sentences=[]
  sentences_filtered=[]
  t=[]

  for i in range(0,len(sentences_break1)):
      x=sentences_break1[i].words.count(name)    #checking for the number of times the subject have occured
    
      #if (x==0):
        
      #      pass
      #else:
            #print 'The word '+name+' occured in '+str(i+1)+'th sentence '+str(x)+' times'
      t.append(i+1)
      words=sentences_break1[i].words
      st=""
      for k in range(0,len(words)):
            st=st+words[k]+" "
      sentences_filtered.append(st)       #filtering out those sentences which contains the subject

  ascii_coded_sentences=[]
  #print 'Filtered Sentences from Text given containing word '+name+':'
  for i in range(0,len(sentences_filtered)):
     a=sentences_filtered[i]
     type(a)
     a.encode('utf-8')
     #print ('filtered sentences: ',a)
     ascii_coded_sentences.append(a)

  k = []
  position=[]
  for i in range(0,len(sentences_filtered)):   #finding out the position of subject in the filtered sentences
      k=sentences_filtered[i].split()
      for j in range(0,len(k)):
          if str(k[j])==str(name): 
              position.append(j)
          else:
              pass
  #print ('The position at which the word '+name+' occurs = '+str(position))

  k1=[]          #k1=matrix in which range(-5,+5) of words(in numbers) is mentioned
  for z in range(0,len(position)):
          #x2=position[z]
          k2=[]
          m=len(sentences_filtered[z].split())  #min(x2+6,len(sentences_filtered[i].split()))
          #if(x2>=5):
          #      for i1 in range(x2-5,m):
          #              k2.append(i1)
          #      
          #if(x2<5):
          for i1 in range(0,m):
              k2.append(i1)      
          k1.append(k2)
  #print ('Indexes of words used in sentences filtered: ',str(k1))                

  sentences=[]
  for i in range(0,len(position)):
        q=sentences_filtered[i].split()
        st=""
        for i1 in k1[i]:
                        st=st+q[i1]+" "
        #print (st)
        sentences.append(str(st))
        
  #print ('Filtered sentences words range for sentiment analysis: ',str(sentences))

  
  z=[]
  no_of_posii_words=[]
  no_of_negaa_words=[]
  posii_words=[]
  negaa_words=[]
  pos_and_neg_words_in_filtered_sentences=[]
  def text(document,index):               #filtering out the frequency of positive and negative words in a tweet
                                        #as well as storing the filtered words in an array
    tokens=document.split()
    y=[]
#    k=[]
    pos=0
    neg=0
    
    for j in range(0,len(tokens)):
        for i in range(0,len(pos_words)):
           if((tokens[j])==(pos_words[i][0])):
               x=[]
               q=tokens[j]
               posii_words.append(q)
               #print (posit_words[i])
               x.append(posit_words[i])
               y.append(x)
               pos=pos+1
           else:
               pass
           
           
        for i in range(0,len(neg_words)):
           if((tokens[j])==(neg_words[i][0])):
               q=tokens[j]
               negaa_words.append(q)
               x=[]
               neg=neg+1
               #print (negat_words[i])
               x.append(negat_words[i])
               y.append(x)
           else:
               pass
    #print ('Positive and negative words in '+str(t[index])+'th sentence is: '+str(y))        
    no_of_posii_words.append(pos)
    #print 'Number of positive words in '+str(t[index])+'th sentence is: '+str(no_of_posii_words[index])
    no_of_negaa_words.append(neg)
    #print 'Number of Negative words in '+str(t[index])+'th sentence is: '+str(no_of_negaa_words[index])
    pos_and_neg_words_in_filtered_sentences.append(no_of_posii_words)
    pos_and_neg_words_in_filtered_sentences.append(no_of_negaa_words)   
    z.append(y)
  for i in range(0,len(sentences)):
    senti_words=text(sentences[i],i)
  #print 'Positive words in filtered sentences: '+str(posii_words)
  #print 'Negative words in filtered sentences: '+str(negaa_words)

  
  neutralcount=0
  sentiment=[]
  for i in range(0,len(sentences)):     #temporarily assigning the sentence to be either positive, negative or neutral sentences
      if(no_of_posii_words[i]==no_of_negaa_words[i]):
         sentiment.append("Can't be classified approximately")
         neutralcount=neutralcount+1
      elif(no_of_posii_words[i]>no_of_negaa_words[i]):
         sentiment.append("Positive")
      elif(no_of_posii_words[i]<no_of_negaa_words[i]):
         sentiment.append("Negative")


  for i in range(0,len(sentences)):   #updating the sentiment of the sentence if the text contains words like 'not','but' or 'but not',etc.
          x=sentences[i].split()
          v=len(k1[i])
          for j in range(0,len(x)):    
            if(x[j]=='not'):
                for k in range(0,len(posii_words)):
                    if(x[j+1]==posii_words[k]):
                        sentiment[i]=("Negative")
                    elif((j+2)<=k1[i][v-1]):
                        if(x[j+2]==posii_words[k]):
                            sentiment[i]=("Negative")        
                    else:
                        pass
                for k in range(0,len(negaa_words)):        
                    if(x[j+1]==negaa_words[k]):
                        sentiment[i]=("Positive")
                    elif((j+2)<=k1[i][v-1]):
                        if(x[j+2]==negaa_words[k]):    
                            sentiment[i]=("Positive")     
                    else:
                        pass
            if(x[j]=='but'):
                for k in range(0,len(posii_words)):
                    if(x[j+1]==posii_words[k]):
                        sentiment[i]=("Positive")
                    elif((j+2)<=k1[i][v-1]):
                        if(x[j+2]==posii_words[k]):
                            sentiment[i]=("Positive")    
                    else:
                        pass
                                  
                for k in range(0,len(negaa_words)):
                    if(x[j+1]==negaa_words[k]):
                        sentiment[i]=("Negative")
                    elif((j+2)<=k1[i][v-1]):
                        if(x[j+2]==negaa_words[k]):    
                            sentiment[i]=("Negative")    
                    else:
                        pass
            if(x[j]=='but' and x[j+1]=='not'):
                for k in range(0,len(posii_words)):
                    if(x[j+2]==posii_words[k]):
                        sentiment[i]=("Negative")
                    elif((j+3)<=k1[i][v-1]):
                        if(x[j+3]==posii_words[k]):    
                            sentiment[i]=("Negative")    
                    else:
                        pass
                for k in range(0,len(negaa_words)):        
                    if(x[j+2]==negaa_words[k]):
                        sentiment[i]=("Positive")
                    elif((j+3)<=k1[i][v-1]):
                        if(x[j+3]==negaa_words[k]):    
                            sentiment[i]=("Positive")        
                    else:
                        pass
  #for i in range(0,len(sentences)):
      #print ('The sentiment of '+str(name)+' in-\" '+str(sentences_break[t[i]-1])+'\"  is: '+sentiment[i])
  #print ('\n')


###################################################################
  for i in range(0,len(sentiment)):
    sentiment[i]=sentiment[i].lower()
  finalpos=0
  finalneg=0
  for i in range(0,len(sentiment)):
     if sentiment[i]=='positive':
          finalpos=finalpos+1
     if sentiment[i]=='negative':
          finalneg=finalneg+1
  if len(sentences_filtered)==0 or len(sentences_filtered)==neutralcount:
    finalpos=0
    finalneg=0
  else:
    finalpos=(finalpos/((len(sentences_filtered)-neutralcount)*1.0))
    finalneg=(finalneg/((len(sentences_filtered)-neutralcount)*1.0))
  maximum=0.0
  #finalsent=str('none')
  if finalpos==0:   
    maximum=0-finalneg
  elif finalneg==0:
    maximum=0+finalpos
  elif finalpos>finalneg:
    maximum=float(1.0)
  elif finalpos<finalneg:
    maximum=float(-1.0)
  #print 'Sentiment polarity: '+str(maximum)+'\n'
  return maximum,polarity,polarity2
  
for k in range(0,len(nifty)):
    polarity_bin=[]
    sent_bin=[]
    polarity_par=[]
    polarity_sent=[]
    #textblob 
    print ('Stock under processing:',nifty11[k])
    stock_name=str(nifty[k])+'.xls'
    book = open_workbook(stock_name)
    book = open_workbook(stock_name)
    #sheet = book.sheet_by_name("Sheet2") #If your data is on sheet 1
    sheet1 = book.sheet_by_name(nifty[k]) #If your data is on sheet 1
    closing_price = []
    news = []
    for row in range(1,sheet1.nrows): #start from 1, to leave out row 0
        closing_price.append(sheet1.cell(row, 1))#extract from second col
        
    for row in range(1,sheet1.nrows): #start from 1, to leave out row 0
        news.append(sheet1.cell(row, 2))
  
    #print (closing_price)
    #print (news)
    for i in range(0,len(news)):
        a=sentiment(news[i],nifty22[k])
        polarity_bin.append(a[0])
        #sent_bin.append(a[1])
        polarity_par.append(a[1])
        polarity_sent.append(a[2])
    #print (polarity_bin)
    #print (sent_bin)
    pos_bin=0
    neg_bin=0
    neut_bin=0
    for i in range(0,len(polarity_bin)):
        if polarity_bin[i]==1.0:
            pos_bin=pos_bin+1
        elif polarity_bin[i]==-1.0:
            neg_bin=neg_bin+1
        else:
            neut_bin=neut_bin+1
    #print('Binary classification:',pos_bin,neg_bin,neut_bin,len(news))


    #filterednouns=str('maruti')
    #print '\nTopic to be analyzed: '+filterednouns
    
    #print (polarity)
    poss=0
    negg=0
    neutt=0
    for i in range(0,len(polarity_par)):
        if polarity_par[i]>0:
            poss=poss+1
        elif polarity_par[i]<0:
            negg=negg+1
        else:
            neutt=neutt+1
    #print('textblob_para',poss,negg,neutt,len(news))

    poss=0
    negg=0
    neutt=0
    for i in range(0,len(polarity_sent)):
        if polarity_sent[i]>0:
            poss=poss+1
        elif polarity_sent[i]<0:
            negg=negg+1
        else:
            neutt=neutt+1
    #print('textblob_sentences_avg',poss,negg,neutt,len(news))

    sheet2 = copy(book)
    xx=sheet2.get_sheet(k) 
        
    xx.write(0,4,'Bin_sent')    
    xx.write(0,5,'TB_para')
    xx.write(0,6,'TB_sent')
    #xx.write(0,5,'TB_sent')
    #xx.write(0,5,'TB_sent')
    for i in range(0,len(polarity_bin)):
        xx.write(i+1,4,polarity_bin[i])
        xx.write(i+1,5,polarity_par[i])
        xx.write(i+1,6,polarity_sent[i])
    sheet2.save(stock_name)
    print ("The sheet: ",str(stock_name)," is updated...")
print ("All sheets updated...")