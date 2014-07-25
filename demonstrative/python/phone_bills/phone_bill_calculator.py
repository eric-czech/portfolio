#!/usr/bin/python

import sys
import cPickle
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# ### Load in raw phone bill data and remove unnecessary fields
data = pd.io.parsers.read_csv(sys.argv[1], sep=None)
data = data[[col for col in data.columns.tolist() if not col in ('Airtime', 'Charges', 'Total', 'Description', 'Application Price')]]


# ### Extract relvant features from raw fields
# Parse city and state out of call origin
data = data.merge(data.Origin.apply(lambda x: pd.Series({
    'Origin State' : x.split(' ')[-1], 
    'Origin City'  : ' '.join(x.split(' ')[:-1])
})), left_index=True, right_index=True)

# Parse city and state out of call destination
data = data.merge(data.Destination.apply(lambda x: pd.Series({
    'Destination State' : x.split(' ')[-1], 
    'Destination City'  : ' '.join(x.split(' ')[:-1])
})), left_index=True, right_index=True)

# Parse out start time as number of minutes into day of call
data['Start Time'] = data.Time.apply(lambda x: pd.to_datetime('1970-01-01 ' + x, format='%Y-%m-%d %H:%M%p').value / (60 * 1E9))

# Parse out area code from phone number
data['Area Code'] = data.Number.apply(lambda x: x.split('-')[0])



# Load classifier and label encoder
with open('phone_bill_classifier.pkl', 'rb') as f:
    clf, encoders = cPickle.load(f)

# ### Encode categorical fields as numeric
unencoded_data = data.copy()

def encode_labels(col):
    if not col.name in encoders.keys():
        return col
    encoder = encoders[col.name]
    unknown = set()
    col = col.fillna('Unknown')
    def encode(value, uknown):
        if not value in encoder.classes_:
            uknown.add(value)
            value = 'Unknown'
        return encoder.transform([value])[0]
    res = col.apply(encode, args=(unknown,))        
    if len(unknown) > 0:
        print 'Unknown values found for column ', col.name, ': ', unknown
    return res

data = data.apply(encode_labels)
data = data[[col for col in data.columns.tolist() if col not in ('Date', 'Time', 'Origin', 'Destination', 'Business')]]

print 'Classifier input data:'
print data.to_string()

unencoded_data['Business'] = clf.predict(data)

print 'Data with predicted call type:'
print unencoded_data.to_string()

print 'Summary stats by call type:'
stat_summary = unencoded_data.groupby('Business')['Minutes'].describe().unstack()
total_summary = DataFrame(unencoded_data.groupby('Business')['Minutes'].sum())
total_summary.columns = ['total_minutes']
summary = pd.merge(stat_summary, total_summary, right_index = True, left_index = True)

print summary.to_string()

print summary.to_html()

