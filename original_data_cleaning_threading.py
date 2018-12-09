from __future__ import print_function
import sys
import urllib
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import json
import seaborn as sns 
import math
import threading
try:
	# For Python 3.0 and later
	from urllib.error import HTTPError
	from urllib.parse import quote
	from urllib.parse import urlencode
except ImportError:
	# Fall back to Python 2's urllib2 and urllib
	from urllib2 import HTTPError
	from urllib import quote
	from urllib import urlencode

def buildPrevInfo(df,start,end):
	start = int(start)
	end = int(end)
	local_df = df.iloc[start:end,:]
	prev = None
	try:
		if start!=0:
			prev = df.iloc[start-1,:]
		for i in range (start,end):
			if(i%100 == 0):
				print("Thread-{}: Querying {}th entry\n".format(start/5000, i))
			curr = df.iloc[i,:]
			if prev is not None and prev.CAMIS2==curr.CAMIS2:
			# add prev info to curr row
				local_df.loc[i,'last_crit_flag'] = str(prev['CRITICAL FLAG'])
				local_df.loc[i,'last_violation_code'] = str(prev['VIOLATION CODE'])
				local_df.loc[i,'last_score'] = str(prev['SCORE'])
				local_df.loc[i,'last_inspection_date'] = str(prev['DATE2'])
			prev = curr  
		local_df.to_csv('with_prev_inspection_{}.csv'.format(int(start/5000)))
	except Exception as e:
		print("Thread-{}: Run into error!!! Position-{}".format(start/5000, i))
		print(e)
	return local_df


df = pd.read_csv('original.csv')
df['DATE'] = pd.to_datetime(df['INSPECTION DATE'])
a = df['DATE'].min()
df[df['DATE'] == a] = np.nan
df.sort_values(['DATE','CAMIS'],ascending = [0,0],inplace = True)
df['CRITICAL FLAG'] = df['CRITICAL FLAG'].map({'Critical': 1, 'Not Critical': 0,'Not Applicable':0})
df.drop(['BUILDING','STREET','BORO','ZIPCODE','PHONE','RECORD DATE','INSPECTION DATE','GRADE DATE','INSPECTION TYPE'],axis=1,inplace=True)
df['CAMIS2'] = df['CAMIS']
df['DATE2'] = df['DATE']
def sericoncat(seri):
	return seri.str.cat()
def getfirst(seri):
	return seri.iloc[0]
df = df.groupby(['CAMIS','DATE']).agg({'CRITICAL FLAG': np.sum,'VIOLATION CODE':sericoncat, 'ACTION':sericoncat, 'VIOLATION DESCRIPTION':sericoncat, 'SCORE': np.mean,'CUISINE DESCRIPTION': getfirst, 'CAMIS2':getfirst, 'DATE2':getfirst})
df['last_crit_flag'] = np.nan
df['last_violation_code'] = np.nan
df['last_score'] = np.nan
df['last_inspection_date'] = np.nan
count = int(len(df)/5000)

for i in range(count):
	threading.Thread(target=buildPrevInfo, args=(df, 5000*i, np.minimum(len(df),5000*(i+1)))).start()



