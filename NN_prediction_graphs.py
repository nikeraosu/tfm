#!/usr/bin/env python3


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
import matplotlib.dates as md
import pylab
import dateutil
from datetime import datetime




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

#Train model parameters:

train_size = cfg_data['train_size'] # from 0.1 to 0.7. Ideally 0.7
hor_pred = cfg_data['hor_pred'] #prediction horizon
alpha_values = cfg_data['alpha'] #from 0.0001 to 0.1. Optimal is 0.001 in our ideal case.
feature_values = cfg_data['features'] #[['dh3'], ['dh3','dh4','dh5','dh10','ap1'], ['all']]. Ideally 'all'
hls = cfg_data['hls'] #we pass it as a list or int.
testing_days = cfg_data['testing_days'] #how many days do we want to test. int. Like 4/5 days.
single_day = cfg_data['single_day'] #to get the times

days_info_file = cfg_data['days_info']
days_info = pd.read_csv(days_info_file)
day_length = days_info['length_day'][0]
days = days_info['number_test_days'][0]
test_days_file = cfg_data['test_days_file']
test_days = pd.read_csv(test_days_file).values
seed = cfg_data['seed']
target_station = cfg_data['target_station']
tg = cfg_data['time_granularity']

if isinstance(hls,list):
    hls=tuple(hls)


out_folder = orig_folder + dest_folder

if not os.path.exists(out_folder):
    os.makedirs(out_folder)


print('Loading dataframes...\n')
load_start = time.time()
x_original = pd.read_csv(orig_folder+'/X_test.csv')
y_original = pd.read_csv(orig_folder+'/Y_test.csv')
load_end = time.time()
load_time = load_end - load_start
load_min = int(load_time / 60)
load_sec = load_time % 60
print('Dataframes loaded in {} minutes {} seconds!\n'.format(load_min,load_sec))

split_start = time.time()
#We get the number of days
lenrow_original = len(x_original.values)
print('Days: {}\n'.format(days))

arr_days = np.arange(days)
ran_seed = seed #our seed to randomize data
np.random.seed(ran_seed)
np.random.shuffle(arr_days)
if testing_days == 'all':
    testing_days = days
days_test = arr_days[0:testing_days]

x_test = pd.DataFrame()
y_test = pd.DataFrame()

for day in days_test:
    x_test = pd.concat([x_test,x_original.iloc[day*day_length:(day+1)*day_length]],ignore_index=True)
    y_test = pd.concat([y_test,y_original.iloc[day*day_length:(day+1)*day_length]],ignore_index=True)

s_day=pd.read_csv(single_day)
day_time = s_day[0:len(s_day):tg].reset_index(drop=True)


lencol = len(x_test.columns) #number of columns for x
lenrow = len(x_test.values)

print('x columns: {}\nx_rows: {}\n'.format(lencol,lenrow))


split_end = time.time()
split_time = split_end - split_start
split_min = int(split_time / 60)
split_sec = split_time % 60
print('Splitting completed in {} minutes {} seconds. Length for train: {}\n'.format(split_min,split_sec,len(y_test)))

forecast_prediction = []
nrmse_t_final = []
nrmse_v_final = []
skill_t_final = []
skill_v_final = []


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

        y_t_index = y_test.index.values
        y_t_index_valid = y_t_index[(y_t_index % day_length) < day_length_forecast] #so we don't get values for the previous or next day
        y_t_indices_lost = len(y_t_index) - len(y_t_index_valid)
        print('Indices computed. {} indices lost \n.'.format(y_t_indices_lost))
        print('Building randomized y matrix with valid indices...\n')
        y_t = np.ravel(y_test.iloc[y_t_index_valid + hor_pred_indices])
        ymax = y_t.max()
        print('Building y matrix removing invalid indices for persistence model...\n')
        y_pred_persistence = np.ravel(y_test.iloc[y_t_index_valid])

        print('Building X matrix...Same thing as before...\n')
        x_t = x_test.iloc[y_t_index_valid] #like our randomization, just picking the same indices
        
        lencol_t = len(x_test.columns) #number of columns for x
        lenrow_t = len(x_test.values)
        
        #Time to label the axis correctly:
        hours = []
        xs = []
        times = day_time[day_time.columns[0]].str[11:19] #Time
        for i in range(day_length_forecast):
            if times[i].endswith('00:00'):
                hours.append(times[i])
                xs.append(i)
       
        print('Length times: {}\n'.format(len(times)))
        print('Building completed.\nx columns: {}\nx_rows: {}\n'.format(lencol_t,lenrow_t))

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

            print('X rows: {}.\nY rows: {}.\n'.format(len(X_t),len(y_t)))
             
            scrs = []
            rmse_test_scores = []
            rmse_test_pers_scores = []
            skill_test_scores = []
            nrmse_test_scores = []

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
                model_filename = model_folder + output_text + '_and_alpha' + str(av) + '.pkl'

                loading_start = time.time()
                print('Loading MLPregressor model...\n')
                nn_model = joblib.load(model_filename)
                loading_end = time.time()
                print('Model loaded in {} seconds.\n'.format(int(loading_end - loading_start)) + 'Model name: ' + output_text + '\n')

                print('Predicting...\n')
                pred_start = time.time()
                y_pred_test = nn_model.predict(X_t)
                pred_end = time.time()
                pred_time = pred_end - pred_start
                print('Predicted in {} seconds.\n'.format(int(pred_time)))
                


            #WHAT I WANT TO PLOT:
            date_list = []
            nrmse_list = []
            #CREATE TESTING_DAYS NUMBER OF DAYS PREDICT FOR THAT AND CREATE Y_PERSISTENCE. AND PLOT FOR THOSE DAYS: Y, Y_PERSISTENCE AND Y_PREDICTED.
            #BEST OPTION, PROBABLY: SINCE IT'S ALREADY CREATED EVERYTHING JUST PICK THE INDICES NEEDED.
            y_df = pd.DataFrame()
            for d in range(testing_days):
                date = test_days[d][0][len(target_station)+1:len(target_station)+9]
                date_list.append(date)
                y_day = y_t[d*day_length_forecast:(d+1)*day_length_forecast]
                y_pred_day = y_pred_test[d*day_length_forecast:(d+1)*day_length_forecast]
                print('length of the day: {}\n'.format(len(y_day)))
                y_pers = y_pred_persistence[d*day_length_forecast:(d+1)*day_length_forecast]
                rmse_test = (np.mean((y_pred_day - y_day) **2)) ** 0.5
                rmse_pers = (np.mean((y_pers - y_day) **2)) ** 0.5 
                nrmse_test = rmse_test / ymax * 100
                nrmse_list.append(nrmse_test)
                skill = ( 1 - rmse_test / rmse_pers) * 100
                skill_test_scores.append(skill)
                print('Saving figures\n')
                #SAVING FIGURES FOR FIRST SCORE:
                plt.plot(y_day,label='Real')
                plt.plot(y_pred_day,label='Predicted')
                plt.legend(loc='best', fancybox=True, framealpha=0.5)
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                plt.title('Predicted and real relative values for day '+date+'\nForecasting prediction: ' + hp+ '. Skill: '+str(round(skill,2)))
                plt.xticks(xs,hours,rotation='vertical')
                plt.savefig(out_folder + output_text + '_day_' + date + '.png',bbox_inches = 'tight',dpi=300)
                plt.close()
                print('Building .csv')
                y_day_df = pd.DataFrame(y_day)
                y_pred_day_df = pd.DataFrame(y_pred_day)
                y_df = pd.concat([y_day_df,y_pred_day_df],axis=1)
                y_df.columns = ['real','predicted']
                y_df.to_csv(out_folder + output_text + '_day' + date + '.csv', header=True,index=False)
                print('Day finished!\n')
            print('Creating csv...\n')
            sts = pd.DataFrame(skill_test_scores)
            dl = pd.DataFrame(date_list)
            nrmse = pd.DataFrame(nrmse_list)
            skills_report = pd.concat([dl,nrmse,sts],axis=1,ignore_index=True)
            skills_report.columns = ['date','nrmse','skill']
            skills_report.to_csv(out_folder + '/daily_skills_'+hp+'.csv',header=True,index=False)
            print('csv generated!\n')
            print('Forecast prediction for {} completed!\n'.format(hp))
