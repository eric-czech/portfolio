
# coding: utf-8

# # Phone Call Classification
# 
# Nothing grinds my gears like sifting through a spreadsheet of calls on my phone bill to figure out which business-related ones I can expense.  I do it every month and hate it more each time so I finally thought I'd giving solving the problem with a computer a whirl.
# 
# Below, I walk through training an ensemble tree classifier to identify which calls can be expensed based on csv downloads from my Verizon account.  The statements are pretty detailed and it turns out that a classifier can use them to great effect, performing at least as well as I would (especially considering that I normally say "screw it, call it 50/50" half way through when doing it manually).
# 
# The accuracy of the classifier on test data sets is nearly perfect so any more I just sit back, press a button to do the expensing, and spend my very valuable 30 extra minutes per month doing more important things like napping or browsing Reddit.
# 

# In[ ]:

# Some standard pandas/numpy/sklearn imports I'll need
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from pandas import DataFrame, Series


# ### Load in raw (Verizon) phone bill data and remove unnecessary fields

# In[567]:

data = pd.io.parsers.read_csv('/home/ubuntu/phone_bills/phone_bill_20140221.txt', sep=None)
data = data[[col for col in data.columns.tolist() if not col in ('Airtime', 'Charges', 'Total', 'Description', 'Application Price')]]
data.head()


# Each call record above comes with details like when the call was made, how it was to, how long it was, etc. and for this particular dataset, I went through and added a __Business__ field to indicate whether the call was personal or for my job (0 = personal, 1 = business-related).
# 
# There seems to be a good amount of useful information in each field, but we'll have to munge out some of the features to make them more amenable to a classifier.

# ### Extract relvant features from raw fields

# In[568]:

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

# Restrict input data, again, to only the fields relevant after feature extraction
data = data[[col for col in data.columns.tolist() if col not in ('Date', 'Time', 'Origin', 'Destination')]]

data.head()


# Ok, now we've got a more useful data set where the most likely useful features like state of origin/destination, call time of day as seconds, and area code are extracted correctly.  Before we can start to train a classifier though, we'll have to deal with all these categorical/non-numeric values.

# ### Encode categorical fields as numeric

# In[569]:

from sklearn.preprocessing import LabelEncoder

# Group column names by data type
by_type = data.columns.to_series().groupby([t.name for t in data.dtypes])

# Apply (and save) a label encoder for each non-numeric field
encoders = {}
for col in by_type.groups['object']:
    encoders[col] = LabelEncoder()
    vals = list(set(data[col]))
    vals.append('Unknown')
    encoders[col] = encoders[col].fit(vals)
    data[col] = data[col].fillna('Unknown')
    data[col] = encoders[col].transform(data[col])

data.head()


# Now we're ready to go!  All the string fields have been encoded on some one-dimensional numeric space which might not be as good as something like [One-Hot Encoding](http://en.wikipedia.org/wiki/One-hot), but for my purposes it will do just fine.

# ### Split data into training and test sets

# In[570]:

# Use 75% train/25% test split
i = np.random.permutation(data.shape[0])
split = int(len(i)*.75)
train, test = data.iloc[i[:split]], data.iloc[i[split:]]


# We'll ignore the test set for now and instead see how well a classifier can do with cross-validation on the training data

# ### Run 5-Fold classification on training set

# In[571]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold, cross_val_score

clf = RandomForestClassifier(n_estimators=100)
cv = KFold(train.shape[0], 5, shuffle=True, random_state=0)
scores = cross_val_score(clf, train.iloc[:,1:], train.iloc[:,0], cv=cv)

# Plot the accuracy for each fold
Series(scores).plot(kind='bar', title='Accuracy per Fold')


# Not bad!  When I do the classification manually I'd say I'm only about 90% correct so the error of the classifier is similar to my own (i.e. I'll take it).  And now lets see how well it generalizes..

# ### Check generalization to test data

# In[573]:

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Initialize features and response for each set (testing and training)
x_train, y_train = train.iloc[:,1:], train.iloc[:,0]
x_test, y_test = test.iloc[:,1:], test.iloc[:,0]

# Train classifier using training set
fit = clf.fit(x_train, y_train)

# Make predictions on the test set
predicted = clf.predict(x_test)

# Show precision, recall, support, and accuracy
print 'Performance Measures:'
print classification_report(y_test.ravel(), predicted)
print 'Overall Accuracy:'
print accuracy_score(y_test.ravel(), predicted)


# Beautiful.  It would be nice if my training and test data sets were larger but I'll shoot myself if I have to go through that list and do any more manually, and this level accuracy should suffice. 
# 
# Out of curiosity, lets see which features are most important before using the classifier.

# ### Check feature influence

# In[556]:

Series(dict(zip(x_train.columns.tolist(), fit.feature_importances_))).order(ascending=False).plot(kind='bar')


# No concerns here at all.  I would have expected the area code and phone numbers themselves to be the most helpful, and I'm not surprised that destination state was right behind them given that most calls from New York should be business-related.

# ### Export classifier and field encoders

# In[557]:

import cPickle
with open('/home/ubuntu/phone_bills/phone_bill_classifier.pkl', 'wb') as f:
    cPickle.dump((clf, encoders), f)    


# With the classifier saved, I can apply it to other bills and automatically break out which were personal or not.
# 
# One thing to note here is that the field encoders also have to be saved for the sake of applying this thing.  The field encoding for new data sets has to match this one exactly and it's also very likely that new values not seen in this data set (i.e. new phone numbers, states, cities, etc.) will come up and the script applying the classifier should encode them as such.

# ## Applying the classifier

# To apply this all, I wrote a separate script to use the saved classifier on arbitrary bills.  The script goes through the same rigamarole -- extracting features, encoding fields, subsetting fields, etc. -- and the determines the classification for each record.
# 
# Given that classification, then I can figure out how I spend my time on calls which I can then use to set my expenses for the month.
# 
# For example, ouput of the script would look like this:

# ### Summary of call lengths by type
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>count</th>
#       <th>mean</th>
#       <th>std</th>
#       <th>min</th>
#       <th>25%</th>
#       <th>50%</th>
#       <th>75%</th>
#       <th>max</th>
#       <th>total_minutes</th>
#     </tr>
#     <tr>
#       <th>Business</th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td> 72</td>
#       <td>  5.277778</td>
#       <td> 10.287723</td>
#       <td> 1</td>
#       <td> 1</td>
#       <td> 2</td>
#       <td>  4.00</td>
#       <td> 59</td>
#       <td> 380</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td> 18</td>
#       <td> 14.888889</td>
#       <td> 15.940658</td>
#       <td> 1</td>
#       <td> 2</td>
#       <td> 9</td>
#       <td> 21.75</td>
#       <td> 59</td>
#       <td> 268</td>
#     </tr>
#   </tbody>
# </table>

# Not surprisingly, I make more personal calls but they're usually shorter (average of 5 mins vs 15 for business calls).
# 
# Either way, this is sweet! I can use the split between minutes spent making calls of one type or another to determine how much of my phone bill to submit for reimbursement.
