#!/usr/bin/env python3

#This file creates the trained models for a given neural network configuration

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
#from keras.optimizers import SGD

###
import matplotlib.dates as md
import pylab
import dateutil
from datetime import datetime
import math
from keras import optimizers
from keras.models import model_from_json, load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import os.path
from matplotlib import style

###
def addOptions(parser):
   parser.add_option("--NNfile", default="",
             help="Config json file for the data to pass to the model")

parser = optparse.OptionParser()
addOptions(parser)

(options, args) = parser.parse_args()

if not options.NNfile:
   print >> sys.stderr, "No configuration file specified\n"
   sys.exit(1)

#with open('config.json', 'r') as cfg_file:
with open(options.NNfile, 'r') as cfg_file:
    cfg_data = json.load(cfg_file)

orig_folder = cfg_data['orig_folder']
model_folder = cfg_data['model_folder']
dest_folder = cfg_data['dest_folder']

train_size = cfg_data['train_size'] # [1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 7/7]
hor_pred = cfg_data['hor_pred'] #folder_names
alpha_values = cfg_data['alpha'] #[0.0001, 0.001, 0.01, 0,1]
feature_values = cfg_data['features'] #[['dh3'], ['dh3','dh4','dh5','dh10','ap1'], ['all']]
hls = cfg_data['hls'] #we pass it as a list or int
testing_days = cfg_data['testing_days']
single_day = cfg_data['single_day']

days_info_file = cfg_data['days_info']
days_info = pd.read_csv(days_info_file)
day_length = days_info['length_day'][0]
days = days_info['number_test_days'][0]
test_days_file = cfg_data['test_days_file']
test_days = pd.read_csv(test_days_file).values
tg = cfg_data['time_granularity']
seed = cfg_data['seed']

target_station = cfg_data['target_station']


if isinstance(hls,list):
    hls=tuple(hls)



out_folder = orig_folder + dest_folder + '/' + target_station
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

print('Loading dataframes...\n')
load_start = time.time()
x_original = pd.read_csv(orig_folder+'/X_test.csv')
y_original = pd.read_csv(orig_folder+'/../../../Y_test.csv')
x_all = pd.read_csv(orig_folder+'/../../../X_test.csv')
###
#y_original = x_original[target_station + "_rel_ns0"]
###

load_end = time.time()
load_time = load_end - load_start
load_min = int(load_time / 60)
load_sec = load_time % 60
print('Dataframes loaded in {} minutes {} seconds! Splitting for train and validation...\n'.format(load_min,load_sec))

split_start = time.time()
#We get the number of days and split for train and validation
lenrow_original = len(x_original.values)

print('Days: {}\n'.format(days))

arr_days = np.array(x_original['index'])#np.arange(days)
ran_seed = seed #our seed to randomize data
np.random.seed(ran_seed)
np.random.shuffle(arr_days)
if testing_days == 'all':
    testing_days = days
days_test = arr_days[0:testing_days]
x_test = x_all#x_all.iloc[arr_days]
#Now we take random DAYS for train and validation:
#x_test = pd.DataFrame()
y_test = y_original#pd.DataFrame()
# maybe not necessary, since test takes everything in pca and clustering
# the concept of days is no longer real (samples separated among clusters)
##for day in days_test:
#    x_test = pd.concat([x_test,x_original.iloc[day*day_length:(day+1)*day_length]],ignore_index=True)
#    y_test = pd.concat([y_test,y_original.iloc[day*day_length:(day+1)*day_length]],ignore_index=True)
##    x_test = pd.concat([x_test,x_original.iloc[day:(day+1)]],ignore_index=True)
##    y_test = pd.concat([y_test,y_original.iloc[day:(day+1)]],ignore_index=True)
##x_test = x_test.drop('index',axis=1)
##y_test = y_test.drop('index',axis=1)

s_day = pd.read_csv(single_day)
day_time = s_day[0:len(s_day):tg].reset_index(drop=True)
lencol = len(x_test.columns) #number of columns for x
lenrow = len(x_test.values)
lenrow = len(x_original.values)
split_end = time.time()
split_time = split_end - split_start
split_min = int(split_time / 60)
split_sec = split_time % 60
print('Splitting completed in {} minutes {} seconds. Length for test: {}\n'.format(split_min,split_sec,len(y_test)))

forecast_prediction = []

nrmse_t_final = []
nrmse_v_final = []
skill_t_final = []
skill_v_final = []


#Since we configured our matrices with an offset we have to adjust to "jump" to the sample we want to actually predict

for hp in hor_pred:
    if hp.endswith("min"):
        hor_pred_indices = int(int(hp.replace('min','')) * 60 / tg)
    if hp.endswith("s"):
        hor_pred_indices = int(int(hp.replace('s','')) / tg)
    forecast_prediction.append(hp)
    day_length_forecast = day_length - hor_pred_indices

#TRAIN SIZE:

    for ts in train_size:

        n_rows = int(lenrow*ts)
        print('Taking less samples for train size = {}. y length: {} \n'.format(ts,n_rows))
        #y_t = y_train.sample(n_rows,random_state=seed)
        #y_t = y_train
        y_t_index = y_test.index.values
        y_t_index = x_test.sample(n_rows,random_state=seed).index.values
        y_t_index = arr_days#days_test
        y_t_index_valid = y_t_index[(y_t_index % day_length) < (day_length - hor_pred_indices)] #so we don't get values for the previous or next day
        y_t_indices_lost = len(y_t_index) - len(y_t_index_valid)
        print('Indices computed. {} indices lost \n.'.format(y_t_indices_lost))
        print('Building randomized y matrix with valid indices...\n')
        y_t = np.ravel(y_test.iloc[y_t_index_valid + hor_pred_indices])
        print('Building y matrix removing invalid indices for persistence model...\n')
        y_pred_persistence = np.ravel(y_test.iloc[y_t_index_valid])
        print('Building X matrix...Same thing as before...\n')
        x_t = x_test.iloc[y_t_index_valid] #like our randomization, just picking the same indices
        lencol_t = len(x_test.columns)
        lenrow_t = len(y_test.values)
        # time to correclty label the axis
        hours = []
        xs = []
        times = day_time[day_time.columns[0]].str[11:19]
        for i in range(day_length - hor_pred_indices):
            if times[i].endswith('00:00'):
                hours.append(times[i])
                xs.append(i)
        print('Length times: {}\n'.format(len(times)))
        print('Building completed. \nx columns: {}\nrows: {}\n'.format(lencol_t, lenrow_t))

#STATIONS TO SELECT:

        for ft in feature_values:
            X_t = pd.DataFrame()
            
            if ft[0] == 'all':
                X_t = x_t
            else:
                for n in range(len(ft)):

                    for i in range(lencol):

                        if x_test.columns[i].startswith(ft[n]):

                            X_t = pd.concat([X_t,x_t[x_t.columns[i]]],axis=1,ignore_index=True)

            ###
            date_list = []
            predict_mse_list = []
            predict_mae_list = []
            persistence_mse_list = []
            persistence_mae_list = []
            ###

            if isinstance(hls,tuple) == False:
                 if hls > 10:
                    neurons = (hls,)
                    len_hls = '1'

            if isinstance(hls,tuple) == False:
                if hls == 1:
                     neurons = int(len(X.columns)/2 + 1)
                     hls = (neurons,)
                     len_hls = '1'

            if isinstance(hls,tuple) == False:
                if hls == 2:
                     neurons = int(len(X.columns)/2 + 1)
                     hls = (neurons,neurons)
                     len_hls = '2'

            if isinstance(hls,tuple) == False:                    
                if hls == 3:
                     neurons = int(len(X.columns)/2 + 1)
                     hls = (neurons,neurons,neurons)
                     len_hls = '3'

            else:
                len_hls = str(len(hls))
            
            hls_str = str(hls).replace('(','_').replace(', ','_').replace(')','_')
            hls_neurons_str = ''
            for i in range(len(hls)):
                hls_neurons_str = hls_neurons_str + str(hls[i])+'_'

            for av in alpha_values:



                stations = ''
                if ft[0]=="all":
                    stations = "all "
                else:
                    for sta in ft:
                        stations = stations + sta + ' '
                sts = stations.replace(' ','_')
                prcnt = round(ts*0.7,2)



                output_text = 'stations_' + sts + 'for_' + hp + '_prediction_horizon_' + str(prcnt) + '_train_size_' + len_hls + '_hidden_layers_with_' + hls_neurons_str + 'neurons_time_granularity_' + str(tg) + '_'
                output_textb = '/stations_' + sts + 'for_' + hp + '_prediction_horizon_' + str(prcnt) + '_train_size_' + len_hls + '_hidden_layers_with_' + hls_neurons_str + 'neurons'


                loading_start = time.time()
                print('Loading MLPRegressor model equivalent in Keras\n')
                #####
                # Compilation
                #####
                stats=['MAE','mse']
                opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
                n_samples = lenrow
                batch_s = np.minimum(200, n_samples)
                steps_per_ep= n_samples // batch_s
                epchs = 200
                ###
                model_filename = model_folder + '/' + output_textb + 'model.h5'
                print(model_filename)

                if(os.path.exists(model_filename)):
                        nn_model = load_model(str(model_filename))
                        nn_model.compile(loss='mean_squared_error', optimizer=opt, metrics=stats)
                        print("Loaded model from disk")
                loading_end = time.time()
                print('Model loaded in {} seconds.\n'.format(int(loading_end-loading_start))+ 'Model name: ' + output_text + '\n')
                

                print('Predicting...\n')
                pred_start = time.time()
                y_pred_test = nn_model.predict(X_t)
                #print('Predict result type {} train\n'.format(type(y_pred_train)))
                pred_end = time.time()
                pred_time = pred_end - pred_start
                print('Predicted in {} seconds.\n'.format(int(pred_time)))

            print('Creating csv...\n')
            MSE_Final=mean_squared_error(y_t, y_pred_test)#acc_mse/(testing_days+1)
            RMSE=math.sqrt( (MSE_Final))
            print(RMSE)
            MSE_Final_persistence=mean_squared_error(y_t, y_pred_persistence)#acc_mse_persistence/(testing_days+1)
            RMSE_persistence=math.sqrt( (MSE_Final_persistence))
            print(RMSE_persistence)
            dl = pd.DataFrame(date_list)
            
            skill=100*(1-(RMSE/RMSE_persistence))
           
            MRSE_modelo_df = pd.DataFrame(np.array([RMSE]))
            MRSE_persistent_df = pd.DataFrame(np.array([RMSE_persistence]))
            skill_df = pd.DataFrame(np.array([skill]))
            
            scores_predict_mse = pd.DataFrame(predict_mse_list)
            scores_predict_mae = pd.DataFrame(predict_mae_list)
            scores_persistence_mse = pd.DataFrame(persistence_mse_list)
            scores_persistence_mae = pd.DataFrame(persistence_mae_list)
            out_folder = orig_folder + dest_folder+ '/' + target_station
            df_alphascores = pd.concat([dl, scores_predict_mse, scores_predict_mae, scores_persistence_mse, scores_persistence_mae,MRSE_modelo_df,MRSE_persistent_df,skill_df], axis=1, ignore_index=True)
#            df_alphascores.columns = ['Date', 'scores_predict_mse', 'scores_predict_mae', 'scores_persistence_mse', 'scores_persistence_mae','MRSE_modelo','MRSE_persistent','skill']
            df_alphascores.columns = ['MRSE_modelo','MRSE_persistent','skill']
            df_alphascores.to_csv(out_folder + output_text + 'daily_scores_' + '.csv', header=True, index=False)

