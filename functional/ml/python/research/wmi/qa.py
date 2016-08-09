
import pandas as pd
import itertools
import cufflinks as cf


def get_corrected_geo_data(d):
    def fix_coords(r):
        lat = r['OriginalLat']
        lon = r['OriginalLon']
        cty = r['Country']

        res = (lat, lon)
        if cty == 'Peru':
            res = (-abs(lat), -abs(lon))
        elif cty == 'Haiti':
            res = (abs(lat), -abs(lon))
        elif cty == 'Indonesia':
            res = (lat, abs(lon))
        elif cty in ['Mexico', 'Honduras']:
            res = (lat, -abs(lon))
        elif cty == 'Liberia':
            res = (abs(lat), lon)
        elif cty == 'Malawi':
            res = (-abs(lat), abs(lon))
        elif cty in ['Tanzania', 'Uganda', 'Kenya']:
            res = (lat, abs(lon))
        is_error = lat != res[0] or lon != res[1]

        return pd.Series((res[0], res[1], is_error), index=['CorrectedLat', 'CorrectedLon', 'IsError'])

    d_geo = d.copy()\
        .filter(regex='AssessmentSummaryInformation|DistributionPoints')\
        .drop('DistributionPoints.AssessmentID', axis=1)\
        .drop_duplicates()\
        .rename(columns=lambda c: c.split('.')[1])\
        .rename(columns={
            'GPSLatitude': 'OriginalLat',
            'GPSLongitude': 'OriginalLon'
        })

    d_coord = d_geo.apply(fix_coords, axis=1)
    for c in d_coord:
        d_geo[c] = d_coord[c].values

    d_geo = d_geo[(d_geo['OriginalLat'].notnull()) & (d_geo['OriginalLon'].notnull())]

    return d_geo


def plot_corrected_geo_data(d_geo, title):

    d_geo = d_geo[d_geo['IsError'] == 1].copy()

    text_cols = [
        'AssessmentName', 'AssessmentID', 'DistributionPointID'
    ]
    coord_cols = [
        'OriginalLat', 'OriginalLon', 'CorrectedLat', 'CorrectedLon'
    ]

    def get_text(row):
        r = row.copy()
        r[coord_cols] = r[coord_cols].apply(lambda x: round(x, 2))
        r['OriginalCoords'] = '({}, {})'.format(r['OriginalLat'], r['OriginalLon'])
        r['CorrectedCoords'] = '({}, {})'.format(r['CorrectedLat'], r['CorrectedLon'])
        cols = ['OriginalCoords', 'CorrectedCoords'] + text_cols
        return '<br>'.join(['{}: {}'.format(k, v) for k, v in r[cols].to_dict().items()])
    d_geo['Text'] = d_geo.apply(get_text, axis=1)

    ctys = d_geo['Country'].unique()
    colors = itertools.cycle(cf.colors.cl.scales['10']['qual']['dflt'])
    colors = dict(zip(ctys, colors))
    traces = [
        dict(
            type = 'scattergeo',
            lat = d_geo['OriginalLat'],
            lon = d_geo['OriginalLon'],
            hoverinfo = 'text',
            text = d_geo['Text'],
            mode = 'markers',
            marker = dict(
                size=4,
                color='black'
            )
        ),
        dict(
            type = 'scattergeo',
            lat = d_geo['CorrectedLat'],
            lon = d_geo['CorrectedLon'],
            hoverinfo = 'text',
            text = d_geo['Text'],
            mode = 'markers',
            marker = dict(
                size=8,
                color=d_geo['Country'].map(colors)
            )
        )
    ]
    for i, r in d_geo.query("IsError == 1").iterrows():
        traces.append(
            dict(
                type='scattergeo',
                lon=[ r['OriginalLon'], r['CorrectedLon'] ],
                lat = [ r['OriginalLat'], r['CorrectedLat'] ],
                text = 'Test',
                mode = 'lines',
                line = dict(
                    width = 1,
                    color = colors[r['Country']],
                ),
                opacity=.5
            )
        )

    layout = dict(
        geo = dict(
            projection=dict(type='natural earth'),
            showland=True,
            landcolor='white',
            showcountries=True,
            showsubunits=True
        ),
        showlegend = False,
        autosize=False,
        width=800,
        height=800,
        title=title
    )
    fig = dict(data=traces, layout=layout)
    return cf.iplot(fig)
