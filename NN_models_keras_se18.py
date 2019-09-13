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
dest_folder = cfg_data['dest_folder']

train_size = cfg_data['train_size'] 
hor_pred = cfg_data['hor_pred'] 
alpha_values = cfg_data['alpha'] 
feature_values = cfg_data['features'] 
hls = cfg_data['hls'] 
days_info_file = cfg_data['days_info']
days_info = pd.read_csv(days_info_file)
day_length = days_info['length_day'][0]
days = days_info['number_train_days'][0]
tg = cfg_data['time_granularity']
seed = cfg_data['seed']


station = cfg_data['station']


if isinstance(hls,list):
    hls=tuple(hls)



out_folder = orig_folder + dest_folder
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

model_folder = out_folder+'/models'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

csvs_folder = out_folder+'/csvs'
if not os.path.exists(csvs_folder):
    os.makedirs(csvs_folder)

graphs_folder = out_folder+'/graphs'
if not os.path.exists(graphs_folder):
    os.makedirs(graphs_folder)


print('Loading dataframes...\n')
load_start = time.time()
x_original = pd.read_csv(orig_folder+'/X_tr_val.csv')
#y_original = pd.read_csv(orig_folder+'/Y_tr_val.csv')
###
y_original = x_original[station + "_rel_ns0"]
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

arr_days = np.arange(days)
ran_seed = seed #our seed to randomize data
np.random.seed(ran_seed)
np.random.shuffle(arr_days)
len_days_validation = int(round(days * 0.176470588,0))
days_validation = arr_days[0:len_days_validation]
days_train = arr_days[len_days_validation:]

#Now we take random DAYS for train and validation:
x_train = pd.DataFrame()
y_train = pd.DataFrame()
x_val_original = pd.DataFrame()
y_val_original = pd.DataFrame()
for day in days_train:
    x_train = pd.concat([x_train,x_original.iloc[day*day_length:(day+1)*day_length]],ignore_index=True)
    y_train = pd.concat([y_train,y_original.iloc[day*day_length:(day+1)*day_length]],ignore_index=True)
for day in days_validation:
    x_val_original = pd.concat([x_val_original,x_original.iloc[day*day_length:(day+1)*day_length]],ignore_index=True)
    y_val_original = pd.concat([y_val_original,y_original.iloc[day*day_length:(day+1)*day_length]],ignore_index=True)

lencol = len(x_train.columns) #number of columns for x
lenrow = len(x_train.values)
split_end = time.time()
split_time = split_end - split_start
split_min = int(split_time / 60)
split_sec = split_time % 60
print('Splitting completed in {} minutes {} seconds. Length for train: {}\n'.format(split_min,split_sec,len(y_train)))

forecast_prediction = []


#Since we configured our matrices with an offset we have to adjust to "jump" to the sample we want to actually predict

for hp in hor_pred:
    if hp.endswith("min"):
        hor_pred_indices = int(int(hp.replace('min','')) * 60 / tg)
    if hp.endswith("s"):
        hor_pred_indices = int(int(hp.replace('s','')) / tg)
    forecast_prediction.append(hp)

#TRAIN SIZE:

    for ts in train_size:

        n_rows = int(lenrow*ts)
        print('Taking less samples for train size = {}. y length: {} \n'.format(ts,n_rows))
        y_t = y_train.sample(n_rows,random_state=seed)
#        y_t = y_train
        y_t_index = y_t.index.values
        y_t_index_valid = y_t_index[(y_t_index % day_length) < (day_length - hor_pred_indices)]
#        print(y_t_index_valid)
        y_t_indices_lost = len(y_t_index) - len(y_t_index_valid)
        print('Indices computed. {} indices lost \n.'.format(y_t_indices_lost))
#        print(y_train)
        print('Building randomized y matrix with valid indices...\n')
#        print(hor_pred_indices)
 #       hhhhhhhhh = y_t_index_valid + hor_pred_indices
#        print(hhhhhhhhh)
        y_t = np.ravel(y_train.iloc[y_t_index_valid + hor_pred_indices])
        print('Building y matrix removing invalid indices for persistence model...\n')
        y_pred_persistence = np.ravel(y_train.iloc[y_t_index_valid])

        y_val_index = y_val_original.index.values
        y_val_index_valid = y_val_index[(y_val_index % day_length) < (day_length - hor_pred_indices)]
        y_pred_persistence_val = np.ravel(y_val_original.iloc[y_val_index_valid])
        print('Building X matrix...Same thing as before...\n')
        x_t = x_train.iloc[y_t_index_valid]
        x_val = x_val_original.iloc[y_val_index_valid]
        y_val = np.ravel(y_val_original.iloc[y_val_index_valid + hor_pred_indices])
#STATIONS TO SELECT:

        for ft in feature_values:
            X_t = pd.DataFrame()
            X_val = pd.DataFrame()

            if ft[0] == 'all':
                X_t = x_t
                X_val = x_val
            else:
                for n in range(len(ft)):

                    for i in range(lencol):

                        if x.columns[i].startswith(ft[n]):

                            X_t = pd.concat([X,x[x.columns[i]]],axis=1,ignore_index=True)
                            X_val = pd.concat([X_val,x_val[x_val.columns[i]]],axis=1,ignore_index=True)

            persistence_mse_list_train = []
            persistence_mae_list_train = []
            persistence_mse_list_val = []
            persistence_mae_list_val = []


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



                output_text = '/stations_' + sts + 'for_' + hp + '_prediction_horizon_' + str(prcnt) + '_train_size_' + len_hls + '_hidden_layers_with_' + hls_neurons_str + 'neurons'
 
                print('Creating MLPRegressor equivalent in Keras\n')
                ####
                # Creation
                ####
                kern_reg= keras.regularizers.l2(av)
                nn_model = Sequential()
                nn_model.add(Dense(hls[0], input_shape=(lencol,), activation="relu", kernel_regularizer=kern_reg))
                for i in hls[1:]:
                    nn_model.add(Dense(i, activation="relu"))
                nn_model.add(Dense(1))
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
                print('Fitting...\n'+output_text+'\n')
                fit_start = time.time()
                model_filename = model_folder + output_text + 'model.h5'
                print(model_filename)

                if(os.path.exists(model_filename)):
                    nn_model = load_model(str(model_filename))
                    if(os.path.exists('file.json')):
                        with open('file.json', 'r', encoding='utf_8') as f:
                            H = json.loads(f.read())
                    nn_model.compile(loss='mean_squared_error', optimizer=opt, metrics=stats)
                    print("Loaded model from disk")
                else:
                    nn_model.compile(loss='mean_squared_error', optimizer=opt, metrics=stats)
                    H = nn_model.fit(X_t.values,y_t, validation_data=(x_val.values, y_val),batch_size=batch_s, epochs=epchs, callbacks=None, shuffle=True)
                    nn_model.save(str(model_filename))
                    print("Saved model to disk")
                with open('file.json', 'w') as f:
                    json.dump(H.history,f)
                ###
                fit_end = time.time()
                fit_time = fit_end - fit_start
                fit_min = int(fit_time / 60)
                fit_sec = fit_time % 60

                ###
                print(H.history.keys())
                # summarize history for mean_absolute_error
                plt.plot(H.history['mean_absolute_error'])
                plt.plot(H.history['val_mean_absolute_error'])
                plt.title('model mean_absolute_error')
                plt.ylabel('mean_absolute_error')
                plt.xlabel('epoch')
                plt.legend(['train_mean_absolute_error','validation_val_mean_absolute_error'],loc='upper left')
                plt.savefig(graphs_folder + output_text + station + 'mean_absolute_error.png')
                plt.close('all')
                # summarize history for mean_squared_error
                plt.plot(H.history['mean_squared_error'])
                plt.plot(H.history['val_mean_squared_error'])
                plt.title('model mean_squared_error')
                plt.ylabel('mean_squared_error')
                plt.xlabel('epoch')
                plt.legend(['train_mean_squared_error','validation_val_mean_squared_error'],loc='upper left')
                plt.savefig(graphs_folder + output_text + station + 'mean_squared_error.png')
                plt.close('all')
                # summarize history for loss
                plt.plot(H.history['loss'])
                plt.plot(H.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train_loss','validation_loss'],loc='upper left')
                plt.savefig(graphs_folder + output_text + station + 'loss.png')
                plt.close('all')
                ###
                print('Fitting completed in {} minutes {} seconds. Saving model to file \n'.format(fit_min,fit_sec))

                ###
                print('Getting scores\n')
                persistence_mse_train = mean_squared_error(y_t, y_pred_persistence)
                persistence_mae_train = mean_absolute_error(y_t, y_pred_persistence)
                persistence_mse_list_train.append(persistence_mse_train)
                persistence_mae_list_train.append(persistence_mae_train)
                persistence_mse_val = mean_squared_error(y_val, y_pred_persistence_val)
                persistence_mae_val = mean_absolute_error(y_val, y_pred_persistence_val)
                persistence_mse_list_val.append(persistence_mse_val)
                persistence_mae_list_val.append(persistence_mae_val)
                ###


            print('Saving figures and .csv file\n')

            #SAVING DATA AS .CSV
            ###
            persistence_mse_t = pd.DataFrame(persistence_mse_list_train)
            persistence_mae_t = pd.DataFrame(persistence_mae_list_train)
            persistence_mse_v = pd.DataFrame(persistence_mse_list_val)
            persistence_mae_v = pd.DataFrame(persistence_mae_list_val)
            loss_df = pd.DataFrame(H.history['loss'])
            val_loss_df = pd.DataFrame(H.history['val_loss'])
            mean_squared_error_df = pd.DataFrame(H.history['mean_squared_error'])
            val_mean_squared_error_df = pd.DataFrame(H.history['val_mean_squared_error'])
            mean_absolute_error_df = pd.DataFrame(H.history['mean_absolute_error'])
            val_mean_absolute_error_df = pd.DataFrame(H.history['val_mean_absolute_error'])

            df_alphascores = pd.concat([loss_df, val_loss_df, mean_squared_error_df, val_mean_squared_error_df,mean_absolute_error_df,val_mean_absolute_error_df,persistence_mse_t,persistence_mae_t, persistence_mse_v,persistence_mae_v],axis=1,ignore_index=True)
            df_alphascores.columns = ['loss', 'val_loss', 'mean_squared_error', 'val_mean_squared_error', 'mean_absolute_error', 'val_mean_absolute_error', 'persistence_mse_t', 'persistence_mae_t', 'persistence_mse_v', 'persistence_mae_v']

            df_alphascores.to_csv(csvs_folder + output_text + '.csv',header=True,index=False)



print('Figures and .csv generated!\n')
