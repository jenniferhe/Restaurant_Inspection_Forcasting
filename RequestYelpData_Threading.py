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

# This client code can run on Python 2.x or 3.x.  Your imports can be
# simpler if you only need one of those.
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


# Yelp Fusion no longer uses OAuth as of December 7, 2017.
# You no longer need to provide Client ID to fetch Data
# It now uses private keys to authenticate requests (API Key)
# You can find it on
# https://www.yelp.com/developers/v3/manage_app
API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
MATCH_PATH = '/v3/businesses/matches'
BUSINESS_PATH = '/v3/businesses/'  # Business ID will come after slash.
# Defaults for our simple example.
#DEFAULT_TERM = 'dinner'
DEFAULT_LOCATION = 'New York City, NY'
SEARCH_LIMIT = 10


def request(host, path, api_key, url_params=None):
    """Given your API_KEY, send a GET request to the API.
    Args:
        host (str): The domain host of the API.
        path (str): The path of the API after the domain.
        API_KEY (str): Your API Key.
        url_params (dict): An optional set of query parameters in the request.
    Returns:
        dict: The JSON response from the request.
    Raises:
        HTTPError: An error occurs from the HTTP request.
    """
    url_params = url_params or {}
    url = '{0}{1}'.format(host, quote(path.encode('utf8')))
    headers = {'Authorization': 'Bearer %s' % api_key,}

    # print(u'Querying {0} ...'.format(url))
    response = requests.request('GET', url, headers=headers, params=url_params)
    return response.json()

def query_match(term, location, api_key):
    """Query the MATCH API by a term and location.
    Args:
        term (str): The search term passed to the API.
        location (str): The search location passed to the API.
        Note city and state parameters are set to default
    Returns:
        dict: The JSON response from the request.
    """
    name = '?name='+term.replace(' ','+')
    address = 'address1='+location.replace(' ','+')
    headers = {'Authorization': 'Bearer %s' % api_key,}
    url = API_HOST + MATCH_PATH + name +'&city=New+York+City&state=NY&country=US&'+ address
    return requests.request('GET', url, headers = headers, params = {'match_threshold': 'default'}).json()

def search(term, location, api_key):
    """Query the Search API by a search term and location.
    Args:
        term (str): The search term passed to the API.
        location (str): The search location passed to the API.
    Returns:
        dict: The JSON response from the request.
    """
    url_params = {
        'term': term.replace(' ', '+'),
        'location': location.replace(' ', '+'),
        'limit': SEARCH_LIMIT
    }
    return request(API_HOST, SEARCH_PATH, api_key, url_params=url_params)

def get_reviews(business_id, api_key):
    business_path = BUSINESS_PATH + business_id + '/reviews'
    return request(API_HOST, business_path, api_key)

def get_business(business_id, api_key):
    """Query the Business API by a business ID.
    Args:
        business_id (str): The ID of the business to query.
    Returns:
        dict: The JSON response from the request.
    """
    business_path = BUSINESS_PATH + business_id
    return request(API_HOST, business_path, api_key)

def query_api(term, api_key):
    """Queries the API by the input values from the user.
    Args:
        term (str): The search term to query.
        location (str): The location of the business to query.
    """
    response = search(term, DEFAULT_LOCATION, api_key)
    businesses = response.get('businesses')
    if not businesses:
        print(u'No businesses for {0} in {1} found.'.format(term, DEFAULT_LOCATION))
        return
    business_id = businesses[0]['id']
#     print(u'{0} businesses found, querying business info for the top result "{1}" ...'.format(len(businesses), business_id))
    response = get_business(business_id, api_key)
#     print(u'Result for business "{0}" found:'.format(business_id))
#     pprint.pprint(response, indent=2)
    return response


def batch_process(start, end, api_key):
    local_df = df.iloc[start:end,:]
    for i in range(start,end):
        if i % 50 == 0:
            print("Thread-{}: Querying {}th entry".format(start, i))
        try:
            data = get_reviews(local_df['yelp_id'][i], api_key)
            reviews = []
            for item in data['reviews']:
                r = dict()
                r['text'] = item['text']
                r['time_created'] = item['time_created']
                r['rating'] = item['rating']
                reviews.append(r)
            local_df.loc[i,'yelp_reviews'] = str(reviews)
                # for key in data.keys():
                #     if key not in keep:
                #         continue
                #     elif key in keep1:
                #         local_df.loc[i,'yelp_'+key] = data[key]
                #     elif key == 'location':
                #         for j in location_keys:
                #             local_df.loc[i,'yelp_'+j] = data[key][j]
                #     elif key == 'coordinates':
                #         local_df.loc[i,'yelp_latitude'] = data[key]['latitude']
                #         local_df.loc[i,'yelp_longitude']  = data[key]['longitude']
                #     elif key == 'transactions':
                #         local_df.loc[i,'yelp_'+key] = (', '.join(data['transactions']))
                #     elif key == 'categories':
                #         temp1 = json.dumps([item['alias'] for item in data[key]])
                #         temp2 = json.dumps([item['title'] for item in data[key]])
                #         local_df.loc[i,'yelp_'+key+'_a'] = temp1[1:len(temp1)-1]
                #         local_df.loc[i,'yelp_'+key+'_t'] = temp2[1:len(temp2)-1]
                #     elif key == 'hours':
                #         # dic = {item['day']:item['start']+','+item['end'] for item in data[key][0]['open']} 
                #         dic = {item['day']:item['start']+item['end'] for item in data[key][0]['open']}
                #         for day in range(7):
                #             if day in dic.keys():
                #                 local_df.loc[i,'yelp_day'+str(day)] = dic[day]
                #             # local_df.loc[i,'yelp_day'+str(day)] = dic[day] if day in dic.keys() else np.nan
                #     elif key == 'price':
                #         local_df.loc[i,'yelp_'+key] = len(data[key])
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

    local_df.to_csv('1001_Project_data_{}.csv'.format(start))
    return local_df


df = pd.read_csv('dataset_v5.csv')
# keep = ['name','id','is_closed','url','review_count','rating','phone','location','coordinates','categories','price','hours','transactions']
# keep1 = ['name','id','is_closed','review_count','rating','phone','url']
# location_keys = ['address1','city','state','zip_code']

API_KEYS = ['DnKa6kFX558t8UIkXIITOk5NGiUW1H4ypRVTJ6caM_bGd_ukfcs1GugjdK73JjWD_DGzN1kZuQ9ehKs9NnW6iz-zHQrlyVpTx87mro33EJ5ukYraRaw1kQ_oToPLW3Yx',
'aA1yqFtuThy9pQMwDBNXoQsIEIBhwsfagdZWeO2se7cwM_utRD8cpX2LAPkTpQV8BJhBX7-DRLQxQfJR9CSjIGOzKMPKufCF_nKSTD0SwXhVhPZ0uIxRtvcrhN7tW3Yx',
'I4EpqW3yupDMXGdxyNAUrYPStOBulGvAVSs6C7Z72T84CCV4_OdlZwfn3dLe9MHU48xSUXZeoacYy_1kxc4lZbjKuzijxb0TCAc9mIpTC8D9OSgxmIxRJG0YFBfZW3Yx',
'agUXwmM8cW5bCP2js7Iceu_YPvIxIT4tdUbPdKHL9QYxB6BbSIOJ3rPx_icxo260xzwobeOvfd4wncfZ02mhyR4mWj16hoQT1ynSEQlLH3E3brDsisnKwuhWaaj0W3Yx',
'opNQnuXzHK-dTyi5qhs9mxcs3_VbD7vZKjAnHij-n0BH8W6Y9-56SJpRXZSZ9b8KEsgWmzTUIWXCOOp5Olyh8yG1qwX-sTaJuxfFauvuYI2TUueHCGPJXDPMlar0W3Yx',
'Ey8H8bHU6_BIUVL04isi9wyox74i1YV1QSxjlMf_EbwqOycls7V8d2zJoyZtzyNZq73PIQv8qOnY1Eti_xrqCsvGyzyyFGAgNeQzuRxTRbTr_TKnpUxhdOAz5ZDrW3Yx',
'aZRnf-PXJMCCtLacE1uJ6uKRJYPyuhIbRU8Bly4NOyAMeFO7aXIKcCG2fl7Hui4CEN22BRRKDbfXhEoKrr4_JNFPXF3VSs0ajxmbOCiVqYhFXDmzZaw_v_SMm-ztW3Yx',
'tBr90NpecyROtHshn7aZtuK7GOPfRIbHnBXkX6ZCaNZ-7TFLEk0GCtqOcF0GXm6h6D17nOPay1lNiAns2oxqbKbyPuNVdLOrkQs5hRTeqRG1qsLGV58QRVJZVgrqW3Yx',
'392qu0-zBLl7823TDUDCvRSghalLly-pEPb-6-F2rROzGMk_R2MZTYyocueJQ9ZAlagGiGOOUToIStluP6Sdr9q5WIIHQ5QaM7N6CyeIcOK0FjYZG_l7EY1bCiT4W3Yx',
'XLmmmQu_kBccKnKMkzTrghXuYIdn3w13xKTMi0xrBm9ggDnNMl0Ofna64NKqJedMn34qeGZv4XoTdJzjP8piYhqsbTWYPSLyXwbxtuZaIAGkjxbLcqRbTjH-NB34W3Yx',
'vcJ9Op_XhRA4I4w2mWT2Tu98DS7wFLEbbkdbutVXwIHLZ4OuzIFz89tTt9ooZqJmuD4Jiap-yvGQe46ERb9P5UVJJBcIHi2wspqkbA0JEQj3MnV-fZc_53TJN6brW3Yx',
'6Lj-KuXW4n83sRT2G2HR22WFWDqUEQ8Y9S8K2EfygF5Dk76T8oT0oBp_tAIFRR2FY85QmE3YGA4WN4-v0sF5K9OZBWh0Ldr32a-ILd9qi9pV-BFeUqYnylK9p9j4W3Yx']

for i in range(7):
    key = API_KEYS[i]
    threading.Thread(target=batch_process, args=(2000*i, np.minimum(len(df),2000*(i+1)), key)).start()
# threading.Thread(target=batch_process_3000, args=(500, 999, API_KEYS[0])).start()
# threading.Thread(target=batch_process_3000, args=(1000, 1499, API_KEYS[0])).start()
# threading.Thread(target=batch_process_3000, args=(1500, 1999, API_KEYS[0])).start()
# threading.Thread(target=batch_process_3000, args=(2000, 2499, API_KEYS[0])).start()
# threading.Thread(target=batch_process_3000, args=(2500, 2999, API_KEYS[1])).start()
# threading.Thread(target=batch_process_3000, args=(3000, 3499, API_KEYS[1])).start()
# threading.Thread(target=batch_process_3000, args=(3500, 3999, API_KEYS[1])).start()
# threading.Thread(target=batch_process_3000, args=(4000, 4499, API_KEYS[1])).start()
# threading.Thread(target=batch_process_3000, args=(4500, 4999, API_KEYS[1])).start()

