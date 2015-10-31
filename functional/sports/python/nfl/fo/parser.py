__author__ = 'eczech'

import requests

import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = 'http://www.footballoutsiders.com/premium'
COOKIE = None


def set_cookie(cookie):
    global COOKIE
    COOKIE = cookie


def _parse_url(url):
    if COOKIE is None:
        raise ValueError('Cookie must be set by calling "set_cookie" before parsing can be done.')
    try:
        doc = requests.get(url, cookies=dict(cookies_are=COOKIE)).text
        return BeautifulSoup(doc, 'html.parser')
    except:
        print('Failed to parse url "{}"'.format(url))
        raise


def get_overall_dvoa(year, week):
    url = '{}/oneWeek.php?year={}&week={}'.format(BASE_URL, year, week)
    doc = _parse_url(url)
    for tbl in doc.findAll('table'):
        if 'id' not in tbl.attrs or tbl['id'] != 'dataTable':
            continue
        data = pd.read_html(str(tbl))[0]
        data['Year'] = year
        data['Week'] = week
        return data
    return None


def get_team_dvoa(year, team):
    url = '{}/weekByTeam.php?year={}&team={}'.format(BASE_URL, year, team)
    doc = _parse_url(url)
    for tbl in doc.findAll('table'):
        if 'id' not in tbl.attrs or tbl['id'] != 'dataTable':
            continue
        data = pd.read_html(str(tbl))[0]
        data['Year'] = year
        data['Team'] = team
        return data
    return None
