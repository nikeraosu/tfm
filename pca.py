#!/usr/bin/env python3

#This file creates the trained models for a given neural network configuration
# Uses pca and clustering to select the diferent categories


import pandas as pd
import numpy as np
import sys
import os
import json
import optparse
import time
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers.core import Dense
from keras import optimizers
from keras.models import model_from_json, load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import os.path
from matplotlib import style
from sklearn import decomposition
from scipy.spatial.distance import cdist

class Clustering:
    def fit(self, x, k, num_iterations=0):
        self.x = x
        self.num_dims = self.x.shape[1]
        self.k = k
        self.initialize_parameters()
        self.dist = 'euclidean'
        for i in range(num_iterations):
            self.e_step()
            self.m_step()
        return self.assignments, self.mues, self.covs
    def initialize_parameters(self):
        self.mues = self.x[np.random.randint(0,high=len(self.x),size=self.k)]
        self.covs = [np.identity(self.num_dims) for _ in range(self.k)]
    def e_step(self):
        distances = cdist(self.x, self.mues, self.dist)
        self.assignments = np.argmin(distances, axis=1)
    def m_step(self):
        for k in range(self.k):
            cluster = self.x[self.assignments==k]
            self.mues[k] = np.mean(cluster, axis=0)

def show_clusters(data, clusters):
    model = Clustering()
    for k in range(clusters, clusters+1):
#    for k in range(2,3):
        assignments, mues, covs = model.fit(data, k=k, num_iterations=5)
        print(assignments.shape)
        df_as = pd.DataFrame(assignments)
        for i in range(max(assignments)+1):
            cluster = data[assignments==i]
            mue = mues[i]
        return df_as

def addOptions(parser):
   parser.add_option("--PCAfile", default="",
             help="Config json file for the data to pass to the model")

parser = optparse.OptionParser()
addOptions(parser)

(options, args) = parser.parse_args()

if not options.PCAfile:
   print >> sys.stderr, "No configuration file specified\n"
   sys.exit(1)

with open(options.PCAfile, 'r') as cfg_file:
    cfg_data = json.load(cfg_file)

orig_folder = cfg_data['orig_folder']
dest_folder = cfg_data['dest_folder']

n_comp = cfg_data['n_components']
pca_labels = cfg_data['pca_labels']
days_info_file = cfg_data['days_info']
days_info = pd.read_csv(days_info_file)
day_length = days_info['length_day'][0]
days = days_info['number_train_days'][0]
seed = cfg_data['seed']
n_clusters = cfg_data['n_clusters']

out_folder = orig_folder + dest_folder
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

print('Loading dataframes...\n')
load_start = time.time()
x_original_train = pd.read_csv(orig_folder+'/X_tr_val.csv')
x_original_test = pd.read_csv(orig_folder+'X_test.csv')
y_original = pd.read_csv(orig_folder+'/Y_tr_val.csv')
y_test = pd.read_csv(orig_folder+'/Y_test.csv')

load_end = time.time()
load_time = load_end - load_start
load_min = int(load_time / 60)
load_sec = load_time % 60
print('Dataframes loaded in {} minutes {} seconds! Splitting for train and validation...\n'.format(load_min,load_sec))

## apply pca: over train to reduce, then separate train and test based on the resulting object
pca = decomposition.PCA(n_components=n_comp)
pca.fit(np.array(x_original_train)[:,:-2]) #remove elevation and azimuth maybe not necessary
x_train_pca = pca.transform(np.array(x_original_train)[:,:-2]) #x_original
x_test_pca = pca.transform(np.array(x_original_test)[:,:-2])
x_train_pca_df = pd.DataFrame(x_train_pca)
x_test_pca_df = pd.DataFrame(x_test_pca)
x_train_df = pd.DataFrame()
x_test_df = pd.DataFrame()
for i in range(len(pca_labels)):
    x_train_df[pca_labels[i]] = x_train_pca_df[x_train_pca_df.columns[i]]
    x_test_df[pca_labels[i]] = x_test_pca_df[x_test_pca_df.columns[i]]
#print('x_test_df {}'.format(x_test_df))
df_as_train = show_clusters(x_train_pca, n_clusters)
df_as_test = show_clusters(x_test_pca, n_clusters)
target_name = 'target'
x_train_df['cluster'] = df_as_train
x_test_df['cluster'] = df_as_test
x_train_df[target_name] = y_original
x_test_df[target_name] = y_test
x_train_df = x_train_df.join(x_original_train)
x_test_df = x_test_df.join(x_original_test)
#print(x_original_train.columns)
#print(x_original_test.columns)
#print(x_train_df.columns)
#print(x_test_df.columns)
print('Start saving......')
for i in range(n_clusters):
    out_folder_aux = out_folder + 'cluster_' + str(i)
    if not os.path.exists(out_folder_aux):
        os.makedirs(out_folder_aux)
    pca_folder = out_folder_aux + '/pca'
    if not os.path.exists(pca_folder):
        os.makedirs(pca_folder)
    df_train_i = x_train_df[x_train_df['cluster']==i]
    df_test_i = x_test_df[x_test_df['cluster']==i]
#    print(df_train_i.columns)
#    print(df_test_i.columns)
    print("Number of samples of cluster "+ str(i)+" in train set: ",df_train_i.shape[0])
    print("Number of samples of cluster "+ str(i)+" in test set: ",df_test_i.shape[0])
    ## Separate data into dataframes
    # Train 
    df_y_train = df_train_i[target_name] # Copy Y (result prediction)
    df_train_i = df_train_i.drop(target_name, axis=1) # Remove Y from X original
    df_train_i = df_train_i.drop('cluster', axis=1) # Remove cluster column
    df_train_pca = df_train_i[pca_labels] # Copy pca rows
    df_train_i = df_train_i.drop(pca_labels, axis=1) # Remove pca from X original
    # Test
    df_y_test = df_test_i[target_name] # Copy Y (result prediction)
    df_test_i = df_test_i.drop(target_name, axis=1) # Remove Y from X original
    df_test_i = df_test_i.drop('cluster', axis=1) # Remove cluster column
    df_test_pca = df_test_i[pca_labels] # Copy pca rows
    df_test_i = df_test_i.drop(pca_labels, axis=1) # Remove pca from X original
    ## Save data to files
    # Train
    df_train_i.to_csv(out_folder_aux+'/X_tr_val.csv',header=True, index=False)
    df_y_train.to_csv(out_folder_aux+'/Y_tr_val.csv',header=True, index=False)
    df_train_pca.to_csv(pca_folder+'/X_tr_val.csv',header=True,index=False)
    # Test
    df_test_i.to_csv(out_folder_aux+'/X_test.csv',header=True, index=False)
    df_y_test.to_csv(out_folder_aux+'/Y_test.csv',header=True, index=False)
    df_test_pca.to_csv(pca_folder+'/X_test.csv',header=True, index=False)
    lenfiles_test = len(df_test_i)
    lenfiles_train = len(df_train_i)
    lenfiles = lenfiles_train + lenfiles_test
    days_info = {'length_day':[day_length],'number_days':[lenfiles],'number_test_days':[lenfiles_test],'number_train_days':[lenfiles_train],'seed_used':[seed]}
    days_info_csv=pd.DataFrame(days_info)
    days_info_csv.to_csv(out_folder_aux+'/days_info.csv',header=True,index=False)

joblib.dump(pca, out_folder+"/pca_model.pkl")



