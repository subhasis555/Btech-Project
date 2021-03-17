#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
from sklearn import *
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from flask import Flask,request,jsonify,render_template


# In[2]:


app=Flask(__name__)
best_model=pickle.load(open('kmeans+LR_model.pkl','rb'))
dataset = pickle.load(open('dataset.pkl', 'rb'))
headers = ["Status", "Duration", "Credit_History", "Purpose", "Credit_amount", "Savings_account",
               "Employment_duration", "Installment_rate", "Personal_status&sex", "Guarantors", "Residence_duration",
               "Property", "Age", "Installment_plan", "Housing", "Existing_credit", "Job", "People", "Telephone",
               "Foreign_worker", "Result"]
header1=headers[:20]

# In[3]:


@app.route('/')
def home():
    return render_template('credit_score.html')
@app.route('/predict',methods=['POST'])
def predict():
    features= [int(x) for x in request.form.values()]
    x1 = pd.DataFrame(features)
    x1=x1.T
    x1.columns=header1
    cp_data = dataset.iloc[:, :20]
    cp = cp_data.append(x1, ignore_index=True)
    scaler = preprocessing.MinMaxScaler()
    new_data1 = scaler.fit_transform(cp)
    x_data1 = pd.DataFrame(new_data1,columns=header1)
    data2=x_data1.copy()
    for iter in range(10):
        out = []
        ans = 0
        for x in header1:
            l = []
            y = data2[x]
            quantile1, quantile3 = np.percentile(y, [25, 75])
            iqr = quantile3 - quantile1
            lb_value = quantile1 - (1.5 * iqr)
            ub_value = quantile3 + (1.5 * iqr)
            for i in range(1000):
                if y[i] < lb_value or y[i] > ub_value:
                    l.append(y[i])
                    y[i] = np.median(y)
            out.append(l)
        for x in out:
            if len(x) > 0:
                ans += 1
        if ans == 0:
            break
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, random_state=29)
    km = km.fit(data2)
    data2['Clusters'] = km.labels_
    sample_data = data2.iloc[1000, :]
    prediction=best_model.predict([sample_data])
    if prediction[0]==0:
        ans="Yes\U0001f60a"
    else:
        ans="No\U0001f61e"
    return render_template('credit_score.html',prediction_text="Is applicant creditworthy?:{}".format(ans))
if __name__=="__main__":
    app.run(debug=True)
