#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 15:48:34 2018

@author: alexmerryman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import the dataset
full_data = pd.read_table('Medicare_Provider_Util_Payment_PUF_CY2015.txt')

#print len(full_data)

# drop "Copyright..." row
full_data = full_data.drop([0], axis=0)

headers_list = list(full_data.columns.values)

numeric_features = ['line_srvc_cnt', 'bene_unique_cnt', 'bene_day_srvc_cnt',
                   'average_Medicare_allowed_amt', 'average_submitted_chrg_amt',
                   'average_Medicare_payment_amt', 'average_Medicare_standard_amt']


# clean numeric data - remove data points where any of their features lie
# outside 2 standard deviations from the mean
for h in numeric_features:
    full_data.drop(full_data[(full_data[h] < (full_data[h].mean() - 2*full_data[h].std())) | (full_data[h] > (full_data[h].mean() + 2*full_data[h].std()))].index, inplace=True)

# drop unnecessary/redundant/too granualar features
full_data.drop(['npi', 'nppes_provider_last_org_name', 'nppes_provider_first_name',
                'nppes_provider_mi', 'nppes_credentials', 'nppes_provider_zip', 'nppes_provider_street1',
                'nppes_provider_street2', 'nppes_provider_city',
                'provider_type', 'hcpcs_code', 'hcpcs_description'], axis=1, inplace=True)


full_data.drop(['nppes_provider_gender', 'nppes_provider_state', 'nppes_provider_country',
                'nppes_entity_code'], axis=1, inplace=True)



categorical_features = ['nppes_provider_gender', 'nppes_entity_code',
                        'nppes_provider_country', 'hcpcs_code',
                        'medicare_participation_indicator', 'place_of_service',
                        'hcpcs_drug_indicator']

full_data.describe()

# pull a random sample from the overall dataset
sample_data = full_data.sample(n=1000000)

sample_data.describe()


print 'Graphing...'

#sns.pairplot(sample_data)

from sklearn import preprocessing

one_hot_variables = ['nppes_provider_gender', 'nppes_entity_code', 'nppes_provider_state',
                        'nppes_provider_country', 'medicare_participation_indicator',
                        'place_of_service', 'hcpcs_drug_indicator']

# one-hot encode place_of_service feature
one_hot_place_service = pd.get_dummies(sample_data['place_of_service'], prefix='place_service')
sample_data = sample_data.drop(['place_of_service'], axis=1)
sample_data = pd.concat([sample_data, one_hot_place_service], axis=1)

# one-hot encode medicare_participation_indicator feature
one_hot_medicare_ind = pd.get_dummies(sample_data['medicare_participation_indicator'], prefix='medicare')
sample_data = sample_data.drop(['medicare_participation_indicator'], axis=1)
sample_data = pd.concat([sample_data, one_hot_medicare_ind], axis=1)

# one-hot encode hcpcs_drug_indicator feature
one_hot_drug_ind = pd.get_dummies(sample_data['hcpcs_drug_indicator'], prefix='drug_ind')
sample_data = sample_data.drop(['hcpcs_drug_indicator'], axis=1)
sample_data = pd.concat([sample_data, one_hot_drug_ind], axis=1)

# convert Pandas dataframe to Numpy array
#sample_data_array = sample_data.as_matrix
#sample_data_array = sample_data_array[~np.isnan(sample_data_array).any(axis=1)]
sample_data_array = sample_data.values

# standardize the data
standardized_data = preprocessing.scale(sample_data_array)


from sklearn.cluster import KMeans
print "Clustering..."
kmeans = KMeans(n_clusters=8)
kmeans.fit(standardized_data)

labels = kmeans.labels_
sample_data['clusters'] = labels


print 'CLUSTER MEANS'
print sample_data.groupby(['clusters']).mean()


print 'CLUSTER PLOT'
sns.lmplot('average_Medicare_standard_amt', 'bene_day_srvc_cnt',
           data=sample_data,
           fit_reg=False,
           hue="clusters",
           scatter_kws={"marker": "D",
                        "s": 25})
plt.title('Clusters average_Medicare_standard_amt vs bene_day_srvc_cnt')
plt.xlabel('average_Medicare_standard_amt')
plt.ylabel('bene_day_srvc_cnt')


from sklearn.metrics import silhouette_score
silhouettes = silhouette_score(sample_data_array, labels)
print 'SILHOUETTE SCORES'
print silhouettes








