__author__ = 'eczech'


import traceback
from . import parser
import pandas as pd
from nfl import storage as nfl_db
from nfl.fdb.const import *
from weather.parser import get_weather_url
import functools


def process_airports(airports, dates, verified_fields=[]):
    failures = []
    for airport in airports:
        print('Processing airport code "{}"'.format(airport))
        res = []
        for i, d in enumerate(dates):
            if i % 100 == 0:
                print('\tOn date {} of {}'.format(i, len(dates)))
            try:
                weather = _get_airport_weather(airport, d, verified_fields=verified_fields)
            except KeyboardInterrupt:
                raise
            except:
                print('Failed to parse airport {}, date {}'.format(airport, d))
                failures.append((airport, d))
                traceback.print_exc()
                continue
            res.append(weather)
        if len(res) > 0:
            _insert_airport_weather(functools.reduce(pd.DataFrame.append, res))
    return failures


def _insert_airport_weather(weather):
    nfl_db.insert_data(DB, COLL_WEATHER_WU, weather, ['AirportCode', 'Date'])


def _get_airport_weather(airport, date, verified_fields=[]):
    weather = parser.get_weather(airport, date.year, date.month, date.day)
    res = pd.DataFrame([weather])
    res['AirportCode'] = airport
    res['Date'] = date
    for field in verified_fields:
        if field not in res or pd.isnull(res[field].iloc[0]):
            raise ValueError('Verified field {} was not present in result for airport "{}", date "{}"'
                             .format(field, airport, date))
    return res


def process_airport(airport, date, print_res=False, verified_fields=[]):
    res = _get_airport_weather(airport, date, verified_fields=verified_fields)
    non_id_cols = [c for c in res if c not in ['AirportCode', 'Date']]
    if len(non_id_cols) == 0:
        print('No data found for URL {}'.format(get_weather_url(airport, date.year, date.month, date.day)))
    else:
        if print_res:
            print(res)
        _insert_airport_weather(res)
