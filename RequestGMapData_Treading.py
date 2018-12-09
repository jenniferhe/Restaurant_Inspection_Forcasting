from __future__ import print_function

import argparse
import json
import pprint
import requests
import sys
import urllib
import numpy as np
import pandas as pd
import json
import threading
import googlemaps
from datetime import datetime
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


def batch_process_gmap_reviews(start, end):
    local_df = df.iloc[start:end,:]
    for i in range(start,end):
        if i % 50 == 0:
            print("Thread-{}: Querying {}th entry".format(start, i))
        try:
            business = gmaps.find_place(df['DBA'][i],'textquery',
                 location_bias='point:{},{}'.format(df['yelp_latitude'][i],df['yelp_longitude'][i]))
            if business['status'] =='OK':
                gmap_id = business['candidates'][0]['place_id']
                response = gmaps.place(gmap_id)
                if response['status'] == 'OK':
                    gmap_reviews = []
                    for r in response['result']['reviews']:
                        tmp = dict()
                        tmp['time'] = r['time']
                        tmp['text'] = r['text']
                        tmp['rating'] = r['rating']
                        gmap_reviews.append(tmp)
                    local_df.loc[i,'gmap_reviews'] = str(gmap_reviews)
        except json.decoder.JSONDecodeError:
            print("Thread-{}: Json Decoder Error in running {}. Position-{}".format(start, local_df['DBA'][i], i))
        except HTTPError as error:
            sys.exit(
                'Encountered HTTP error {0} on {1}:\n {2}\nAbort program.'.format(
                    error.code,error.url,error.read(),
                )
            )
        except Exception as e:
            print("Thread-{}: Run into error!!! Position-{}".format(start, i))
            print(e)
    local_df.to_csv('1001_Project_data_8000_{}.csv'.format(start))
    return local_df



api_key = 'AIzaSyCW2wWWgjoNEqN2Nao9ITLZEHE2TV7t4AU'
gmaps = googlemaps.Client(key=api_key)
df = pd.read_csv('dataset_v6.csv')
for i in range(5):
    threading.Thread(target=batch_process_gmap_reviews, args=(200*i+8000, np.minimum(len(df),200*(i+1)+8000))).start()

