__author__ = 'eczech'

from pymongo import MongoClient
from pymongo import ASCENDING
import pandas as pd

_client = None


def init():
    global _client
    if _client is None:
        _client = MongoClient()


def close():
    global _client
    if _client is not None:
        _client.close()
        _client = None


def insert_data(db, collection, data, index_cols):
    """ Inserts data frame into local Mongo instance
    :param db: Mongo db name
    :param collection: Mongo collection name
    :param data: DataFrame
    :param index_cols: Columns in data frame to use as index
    :return:
    """
    init()
    coll = _client[db][collection]
    coll.create_index([(c, ASCENDING) for c in index_cols], unique=True)
    for i, r in data.iterrows():
        doc = r.to_dict()
        query = {k: v for k, v in doc.items() if k in index_cols}
        coll.update(query, doc, upsert=True)


def get_years_query(years):
    if years is None:
        return {}
    if isinstance(years, range):
        return {'Year': {'$gte': years.start, '$lte': years.stop}}
    if isinstance(years, str) or not hasattr(years, '__iter__'):
        years = [int(years)]
    return {'Year': {'$in': years}}


def run_query(db, coll, years, other=None):
    query = get_years_query(years)
    if other is not None:
        query.update(other)
    return get_data(db, coll, query)


def get_collections(db):
    init()
    return _client[db].collection_names()


def get_data(db, collection, query={}, keep_id=False, projection=None):
    init()
    cursor = _client[db][collection].find(query, projection)
    cursor.batch_size(1000)

    try:
        res = []
        i = 0
        for d in cursor:
            i += 1
            if i % 10000 == 0:
                print('[Mongo] On extraction of document #{}'.format(i))
            res.append(d)

        # res = pd.DataFrame([d for d in cursor])
        res = pd.DataFrame(res)
        if not keep_id and '_id' in res:
            res = res.drop('_id', axis=1)
        return res
    finally:
        cursor.close()


def add_index(db, collection, keys):
    init()
    return _client[db][collection].create_index(keys)