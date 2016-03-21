
# coding: utf-8

# In[1]:

import pandas as pd
import datetime
import numpy as np
from pandas.io.data import DataReader
import urllib2
import matplotlib.pyplot as plt



# In[17]:

def get_google_data(symbol, period, window):
    url_root = 'http://www.google.com/finance/getprices?i='
    url_root += str(period) + '&p=' + str(window)
    url_root += 'd&f=d,o,h,l,c,v&df=cpct&q=' + symbol
    response = urllib2.urlopen(url_root,timeout=5.0)
    print('Get response...')
    data0 = response.read()
    data = data0.split('\n')
    print('done with data fetching')
    #actual data starts at index = 7
    #first line contains full timestamp,
    #every other line is offset of period from timestamp
    parsed_data = []
    anchor_stamp = ''
    end = len(data)
    for i in range(7, end):
        cdata = data[i].split(',')
        if 'a' in cdata[0]:
            #first one record anchor timestamp
            anchor_stamp = cdata[0].replace('a', '')
            cts = int(anchor_stamp)
            parsed_data.append((datetime.datetime.fromtimestamp(float(cts)), float(cdata[1]), float(cdata[2]), float(cdata[3]), float(cdata[4]), float(cdata[5])))            
        else:
            try:
                coffset = int(cdata[0])
                cts = int(anchor_stamp) + (coffset * period)
                parsed_data.append((datetime.datetime.fromtimestamp(float(cts)), float(cdata[1]), float(cdata[2]), float(cdata[3]), float(cdata[4]), float(cdata[5])))
            except:
                pass # for time zone offsets thrown into data
    df = pd.DataFrame(parsed_data)
    df.columns = ['ts', 'o', 'h', 'l', 'c', 'v']
    df.index = df.ts
    del df['ts']
    return df


# In[18]:

spy=get_google_data('SPY',60,15)
mf = (spy['o']-spy['c'])*spy['v']


# In[ ]:

sp500stocks = pd.DataFrame.from_csv('data/constituents.csv', sep=',')


# In[ ]:

count=0
for index, row in sp500stocks.iterrows():
    count=count+1
    print(count)
    indext=index.replace('-','.')
    print(indext)
    data=get_google_data(indext,60,15)
    print('down with fetch data from'+indext)
    mf = (data['o']-data['c'])*data['v']
    if count==1:
        mfall=pd.DataFrame(mf, columns=[indext])
    else:
        mfall[indext] = mf
        
get_ipython().magic(u'matplotlib inline')

mfsum = mfall.sum(axis=1)
mfsum.plot()
    

