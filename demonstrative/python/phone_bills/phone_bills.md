

    import pandas as pd
    import numpy as np
    import datetime as dt
    import matplotlib.pyplot as plt
    from pandas import DataFrame, Series
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import svm
    from sklearn.cross_validation import train_test_split, KFold, cross_val_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import Pipeline

# Modeling

<br>

### Load in raw phone bill data and remove unnecessary fields


    data = pd.io.parsers.read_csv('/home/ubuntu/phone_bills/phone_bill_20140221.txt', sep=None)

    Using Python parser to sniff delimiter



    data.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Business</th>
      <th>Date</th>
      <th>Time</th>
      <th>Number</th>
      <th>Rate</th>
      <th>Usage Type</th>
      <th>Origin</th>
      <th>Destination</th>
      <th>Minutes</th>
      <th>Airtime</th>
      <th>Charges</th>
      <th>Total</th>
      <th>Description</th>
      <th>Application Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td> 0</td>
      <td> 01/07/2014</td>
      <td>  9:48AM</td>
      <td> 843-270-9399</td>
      <td> Peak</td>
      <td>                M2MAllow</td>
      <td> Charleston SC</td>
      <td>   Incoming CL</td>
      <td>  4</td>
      <td> --</td>
      <td> --</td>
      <td> --</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td> 1</td>
      <td> 01/07/2014</td>
      <td> 11:32AM</td>
      <td> 516-384-4100</td>
      <td> Peak</td>
      <td>               PlanAllow</td>
      <td> Charleston SC</td>
      <td> Gardencity NY</td>
      <td>  1</td>
      <td> --</td>
      <td> --</td>
      <td> --</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td> 1</td>
      <td> 01/07/2014</td>
      <td> 11:50AM</td>
      <td> 646-398-6444</td>
      <td> Peak</td>
      <td>               PlanAllow</td>
      <td> Charleston SC</td>
      <td>   Incoming CL</td>
      <td> 13</td>
      <td> --</td>
      <td> --</td>
      <td> --</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td> 0</td>
      <td> 01/07/2014</td>
      <td>  5:13PM</td>
      <td> 717-673-8466</td>
      <td> Peak</td>
      <td> PlanAllow,NoAnsBusyXfer</td>
      <td>     Butler PA</td>
      <td>    Lebanon PA</td>
      <td>  1</td>
      <td> --</td>
      <td> --</td>
      <td> --</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td> 1</td>
      <td> 01/07/2014</td>
      <td>  5:24PM</td>
      <td> 646-398-6444</td>
      <td> Peak</td>
      <td>               PlanAllow</td>
      <td> Charleston SC</td>
      <td>   Incoming CL</td>
      <td>  6</td>
      <td> --</td>
      <td> --</td>
      <td> --</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 14 columns</p>
</div>




    data = data[[col for col in data.columns.tolist() if not col in ('Airtime', 'Charges', 'Total', 'Description', 'Application Price')]]

### Extract relvant features from raw fields


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


    data.head()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Business</th>
      <th>Date</th>
      <th>Time</th>
      <th>Number</th>
      <th>Rate</th>
      <th>Usage Type</th>
      <th>Origin</th>
      <th>Destination</th>
      <th>Minutes</th>
      <th>Origin City</th>
      <th>Origin State</th>
      <th>Destination City</th>
      <th>Destination State</th>
      <th>Start Time</th>
      <th>Area Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td> 0</td>
      <td> 01/07/2014</td>
      <td>  9:48AM</td>
      <td> 843-270-9399</td>
      <td> Peak</td>
      <td>                M2MAllow</td>
      <td> Charleston SC</td>
      <td>   Incoming CL</td>
      <td>  4</td>
      <td> Charleston</td>
      <td> SC</td>
      <td>   Incoming</td>
      <td> CL</td>
      <td> 588</td>
      <td> 843</td>
    </tr>
    <tr>
      <th>1</th>
      <td> 1</td>
      <td> 01/07/2014</td>
      <td> 11:32AM</td>
      <td> 516-384-4100</td>
      <td> Peak</td>
      <td>               PlanAllow</td>
      <td> Charleston SC</td>
      <td> Gardencity NY</td>
      <td>  1</td>
      <td> Charleston</td>
      <td> SC</td>
      <td> Gardencity</td>
      <td> NY</td>
      <td> 692</td>
      <td> 516</td>
    </tr>
    <tr>
      <th>2</th>
      <td> 1</td>
      <td> 01/07/2014</td>
      <td> 11:50AM</td>
      <td> 646-398-6444</td>
      <td> Peak</td>
      <td>               PlanAllow</td>
      <td> Charleston SC</td>
      <td>   Incoming CL</td>
      <td> 13</td>
      <td> Charleston</td>
      <td> SC</td>
      <td>   Incoming</td>
      <td> CL</td>
      <td> 710</td>
      <td> 646</td>
    </tr>
    <tr>
      <th>3</th>
      <td> 0</td>
      <td> 01/07/2014</td>
      <td>  5:13PM</td>
      <td> 717-673-8466</td>
      <td> Peak</td>
      <td> PlanAllow,NoAnsBusyXfer</td>
      <td>     Butler PA</td>
      <td>    Lebanon PA</td>
      <td>  1</td>
      <td>     Butler</td>
      <td> PA</td>
      <td>    Lebanon</td>
      <td> PA</td>
      <td> 313</td>
      <td> 717</td>
    </tr>
    <tr>
      <th>4</th>
      <td> 1</td>
      <td> 01/07/2014</td>
      <td>  5:24PM</td>
      <td> 646-398-6444</td>
      <td> Peak</td>
      <td>               PlanAllow</td>
      <td> Charleston SC</td>
      <td>   Incoming CL</td>
      <td>  6</td>
      <td> Charleston</td>
      <td> SC</td>
      <td>   Incoming</td>
      <td> CL</td>
      <td> 324</td>
      <td> 646</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 15 columns</p>
</div>



### Encode categorical fields as numeric


    by_type = data.columns.to_series().groupby([t.name for t in data.dtypes])


    encoders = {}
    for col in by_type.groups['object']:
        encoders[col] = LabelEncoder()
        vals = list(set(data[col]))
        vals.append('Unknown')
        encoders[col] = encoders[col].fit(vals)
        data[col] = data[col].fillna('Unknown')
        data[col] = encoders[col].transform(data[col])


    data = data[[col for col in data.columns.tolist() if col not in ('Date', 'Time', 'Origin', 'Destination')]]

### Split data into training and test sets


    i = np.random.permutation(data.shape[0])
    split = int(len(i)*.75)
    train, test = data.iloc[i[:split]], data.iloc[i[split:]]

### Run 5-Fold classification on training set


    clf = RandomForestClassifier(n_estimators=100)
    cv = KFold(train.shape[0], 5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, train.iloc[:,1:], train.iloc[:,0], cv=cv)
    Series(scores).plot(kind='bar', title='Accuracy per Fold')




    <matplotlib.axes.AxesSubplot at 0x7f06205ce910>




![png](phone_bills-checkpoint_files/phone_bills-checkpoint_16_1.png)


### Check generalization to test data


    x_train, y_train = train.iloc[:,1:], train.iloc[:,0]
    x_test, y_test = test.iloc[:,1:], test.iloc[:,0]
    fit = clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    print 'Performance Measures:'
    print classification_report(y_test.ravel(), predicted)
    print 'Overall Accuracy:'
    print accuracy_score(y_test.ravel(), predicted)

    Performance Measures:
                 precision    recall  f1-score   support
    
              0       0.95      0.95      0.95        19
              1       0.83      0.83      0.83         6
    
    avg / total       0.92      0.92      0.92        25
    
    Overall Accuracy:
    0.92


### Show incorrectly predicted cases (if any)


    t = test.copy()
    t['Predicted'] = predicted
    
    # Function to invert encoded field values
    def invert_labels(col):
        return encoders[col.name].inverse_transform(col) if col.name in encoders.keys() else col
            
    t[t.Business != t.Predicted].apply(invert_labels)




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Business</th>
      <th>Number</th>
      <th>Rate</th>
      <th>Usage Type</th>
      <th>Minutes</th>
      <th>Origin City</th>
      <th>Origin State</th>
      <th>Destination City</th>
      <th>Destination State</th>
      <th>Start Time</th>
      <th>Area Code</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td> 1</td>
      <td> 816-935-6846</td>
      <td> Peak</td>
      <td> PlanAllow</td>
      <td>  6</td>
      <td> Charleston</td>
      <td> SC</td>
      <td> Incoming</td>
      <td> CL</td>
      <td> 671</td>
      <td> 816</td>
      <td> 0</td>
    </tr>
    <tr>
      <th>44</th>
      <td> 0</td>
      <td> 415-729-9643</td>
      <td> Peak</td>
      <td> PlanAllow</td>
      <td> 14</td>
      <td> Charleston</td>
      <td> SC</td>
      <td> Incoming</td>
      <td> CL</td>
      <td> 662</td>
      <td> 415</td>
      <td> 1</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 12 columns</p>
</div>



### Check feature influence


    Series(dict(zip(x_train.columns.tolist(), fit.feature_importances_))).order(ascending=False).plot(kind='bar')




    <matplotlib.axes.AxesSubplot at 0x7f0620581fd0>




![png](phone_bills-checkpoint_files/phone_bills-checkpoint_22_1.png)



    import cPickle
    with open('/home/ubuntu/phone_bills/phone_bill_classifier.pkl', 'wb') as f:
        cPickle.dump((clf, encoders), f)    

<br>
# Application
<br>

### Call time summary (by type)
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>total_minutes</th>
    </tr>
    <tr>
      <th>Business</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td> 72</td>
      <td>  5.277778</td>
      <td> 10.287723</td>
      <td> 1</td>
      <td> 1</td>
      <td> 2</td>
      <td>  4.00</td>
      <td> 59</td>
      <td> 380</td>
    </tr>
    <tr>
      <th>1</th>
      <td> 18</td>
      <td> 14.888889</td>
      <td> 15.940658</td>
      <td> 1</td>
      <td> 2</td>
      <td> 9</td>
      <td> 21.75</td>
      <td> 59</td>
      <td> 268</td>
    </tr>
  </tbody>
</table>
