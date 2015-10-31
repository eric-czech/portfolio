__author__ = 'eczech'

import pandas as pd
from bs4 import BeautifulSoup
import urllib
import re
import sports_math
import numpy as np


def get_weather_url(airport, year, month, day):
    return 'http://www.wunderground.com/history/airport/{}/{}/{}/{}/DailyHistory.html?req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo='\
        .format(airport, year, month, day)


def _get_doc(year, month, day, airport):
    url = get_weather_url(airport, year, month, day)
    try:
        #print(url)
        doc = urllib.request.urlopen(url).read()
        return url, BeautifulSoup(doc, 'html.parser')
    except:
        print('Failed to parse url "{}"'.format(url))
        raise


def _parse_doc(doc):
    record = {}
    for tbl in doc.findAll('table'):
        if 'id' not in tbl.attrs or tbl['id'] != 'historyTable':
            continue
        if 'class' not in tbl.attrs or 'airport-history-summary-table' not in tbl['class']:
            continue
        data = pd.read_html(str(tbl))[0]
        data = data.set_index(data.columns[0]).dropna(how='all', axis=0)

        # Fetch the temperature record;
        record['Temp'] = _get_value(data, 'Mean Temperature', None)

        # In some cases, usually outside of the US, the known temp isn't present
        # but only the average min and max temps are -- when this happens use the
        # average of those 2 values instead
        if record['Temp'] is None:
            max_temp = _get_value(data, 'Max Temperature', None)
            min_temp = _get_value(data, 'Min Temperature', None)
            if max_temp is not None and min_temp is not None:
                record['Temp'] = float((max_temp + min_temp) / 2)

        record['Humidity'] = _get_value(data, 'Average Humidity', None)
        record['Precip'] = _get_value(data, 'Precipitation', None, allow_average=False)
        record['Snow'] = _get_value(data, 'Snow', 0.)
        record['Wind'] = _get_value(data, 'Wind Speed', None)
        break
    return record


def _get_value(data, metric, default_value, allow_average=True):
    if _is_present(data, metric, 'Actual'):
        value = _parse_number(data.loc[metric]['Actual'])
        if value is not None:
            return value
    average_col = 'Average'
    for c in data:
        if 'Average' in c:
            average_col = c
    if allow_average and _is_present(data, metric, average_col):
        value = _parse_number(data.loc[metric][average_col])
        if value is not None:
            return value
    return default_value


def _is_present(data, metric, value):
    return metric in data.index and value in data.loc[metric]


def _parse_number(value):
    if not isinstance(value, str) and (sports_math.isnan(value) or value == np.nan):
        return None
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", value)
    if len(numbers) > 0:
        return float(numbers[0])
    if value.strip().upper().startswith('MM') or re.sub(r'\([^)]*\)', '', value).strip() == '-':
        return None
    if value.strip().upper().startswith('T'):
        return 0.
    raise ValueError('Cannot parse value "{}"'.format(value))


def get_weather(airport, year, month, day):
    """
    Returns weather record (via HTML) scrape of Weather Underground

    URL Example: http://www.wunderground.com/history/airport/KGRB/1994/10/2/DailyHistory.html?req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=
    :param airport: Airport code to get weather for
    :param year: Year for weather record (int)
    :param month: Month for weather record (int)
    :param day: Day for weather record (int)
    :return: Dict like: {'humidity': 86.0, 'precip': 0.09, 'snow': 0.0, 'temp': 71.0, 'wind': 6.0}
    """
    url, doc = _get_doc(year, month, day, airport)
    try:
        return _parse_doc(doc)
    except:
        print('Failed to get weather from URL "{}"'.format(url))
        raise