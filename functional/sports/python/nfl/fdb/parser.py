__author__ = 'eczech'


from bs4 import BeautifulSoup
import urllib
import pandas as pd
import numpy as np
import functools

BASE_URL = 'http://www.footballdb.com'


def _parse_url(url):
    try:
        doc = urllib.request.urlopen(url).read()
        return BeautifulSoup(doc, 'html.parser')
    except:
        print('Failed to parse url "{}"'.format(url))
        raise


# def get_teams(year):
#     """
#     Returns
#     :param year:
#     :return:
#     """
#     url = '{}/standings/index.html?lg=NFL&yr={}'.format(BASE_URL, year)
#     doc = _parse_url(url)
#     res = []
#     for t1 in doc.findAll('div'):
#         if 'class' not in t1.attrs:
#             continue
#         if 'nfldiv' in t1.attrs['class']:
#             for t2 in t1.findAll('a'):
#                 if 'href' not in t2.attrs:
#                     continue
#                 if not t2.text or not t2.text.strip():
#                     continue
#                 link = t2['href']
#                 id = link.split('/')[-1]
#                 team = t2.text.strip()
#                 res.append((id, team, link, year))
#     assert res, 'No teams found for url "{}"'.format(url)
#     return pd.DataFrame(res, columns=['Id', 'Name', 'Link', 'Year'])


def get_teams(year):
    """
    Returns
    :param year:
    :return:
    """
    url = '{}/standings/index.html?lg=NFL&yr={}'.format(BASE_URL, year)
    doc = _parse_url(url)
    res = []
    for tbl in doc.findAll('table'):
        if 'class' not in tbl.attrs:
            continue
        if 'statistics' not in tbl['class'] or 'scrollable' not in tbl['class']:
            continue

        for tr in tbl.find('tbody').findAll('tr'):
            for span in tr.findAll('span'):
                if 'class' not in span.attrs or 'highres' not in span['class']:
                    continue
                if not span.find('a'):
                    continue
                link = span.find('a')['href']
                if not link.startswith('/teams/nfl/'):
                    continue
                id = link.split('/')[3]
                link = '/'.join(link.split('/')[:4])
                team = span.find('a').text
                res.append((id, team, link, year))
    assert res, 'No teams found for url "{}"'.format(url)
    return pd.DataFrame(res, columns=['Id', 'Name', 'Link', 'Year'])

def to_date(x):
    x = pd.to_datetime(x) if not pd.isnull(x) else np.nan
    return np.nan if pd.isnull(x) or not isinstance(x, pd.tslib.Timestamp) else x


def get_roster(team_id, year):
    url = '{}/teams/nfl/{}/roster/{}'.format(BASE_URL, team_id, year)
    doc = _parse_url(url)
    for tbl in doc.findAll('table'):
        if not tbl.find('thead') or not tbl.find('tbody'):
            continue
        cells = tbl.find('thead').findAll('th')
        if len(cells) < 2 or cells[1].text != 'Player':
            continue
        cols = [c.text for c in cells]
        cols[0] = 'Number'
        res = []
        for row in tbl.find('tbody').findAll('tr'):
            cells = row.findAll('td')
            player = cells[1]
            player_link = np.nan
            if player.find('a'):
                player_link = player.find('a')['href']

            def clean_value(x):
                if not isinstance(x, str):
                    return x
                return x.replace('&nbsp;', '').replace('--', '').strip()

            res.append([player_link] + [clean_value(c.text) for c in cells])
        res = pd.DataFrame(res, columns=['PlayerLink'] + cols)
        res['Birthdate'] = res['Birthdate'].apply(to_date)
        res['Year'] = year
        return res.convert_objects(convert_numeric=True)
    return None


def get_games(team_id, year):
    url = '{}/teams/nfl/{}/results/{}'.format(BASE_URL, team_id, year)
    doc = _parse_url(url)
    res = []
    for tbl in doc.findAll('table'):
        sib = tbl.findPreviousSibling()
        if sib.name != 'div' or \
            'class' not in sib.attrs or \
            'divider' not in sib['class']:
            continue
        game_type = sib.text.strip()
        if not game_type:
            continue
        columns = tbl.find('thead').findAll('td')
        columns = [c.text for c in columns]

        rows = []
        for tr in tbl.find('tbody').findAll('tr'):
            cells = tr.findAll('td')

            opponent = cells[1]
            opponent_link = np.nan
            if opponent.find('a'):
                opponent_link = opponent.find('a')['href']

            outcome = cells[3]
            outcome_link = np.nan
            if outcome.find('a'):
                outcome_link = outcome.find('a')['href']
            rows.append([c.text for c in cells] + [opponent_link, outcome_link, game_type, year])

        data = pd.DataFrame(rows, columns=columns+['OpponentLink', 'OutcomeLink', 'GameType', 'Year'])
        data['Date'] = data['Date'].str.split(' ').str.get(0).apply(to_date)
        res.append(data)

    return functools.reduce(pd.DataFrame.append, res)


def _parse_stat_table(tbl, year):
    cols = None
    rows = []
    for tr in tbl.findAll('tr'):
        if 'class' in tr.attrs and 'header' in tr['class']:
            cols = [c.text for c in tr.findAll('td')][1:]
            continue
        cells = tr.findAll('td')
        row_id = cells[0]
        row_link = np.nan
        if row_id.find('a'):
            row_link = row_id.find('a')['href']
        row_name = row_id.text
        rows.append([row_link, row_name] + [c.text for c in cells[1:]] + [year])
    if cols is None:
        return None
    return pd.DataFrame(rows, columns=['RowLink', 'RowName'] + cols + ['Year'])


def get_game_stats(game_link, year):
    tables = {}
    url = '{}{}'.format(BASE_URL, game_link)
    doc = _parse_url(url)
    for tbl in doc.findAll('table'):
        if 'class' not in tbl.attrs or 'sports_math' not in tbl['class']:
            continue
        sib = tbl.findPreviousSibling()
        if sib.name != 'div' or \
            'class' not in sib.attrs or \
            'divider' not in sib['class']:
            continue
        category = sib.text
        if category == 'Scoring Summary':
            continue

        tables[category] = _parse_stat_table(tbl, year)
    return tables