#!/usr/bin/env python3

#This file creates matrices for every day ready for every ML algorithm.

import os
import pandas as pd
import json
import optparse
import sys
import time
import datetime

def addOptions(parser):
   parser.add_option("--configFile", default="",
             help="Config json file for the data to pass to the model")

parser = optparse.OptionParser()
addOptions(parser)

(options, args) = parser.parse_args()

if not options.configFile:
   print >> sys.stderr, "No configuration file specified\n"
   sys.exit(1)

#with open('config.json', 'r') as cfg_file:
with open(options.configFile, 'r') as cfg_file:
    cfg_data = json.load(cfg_file)

target_station = cfg_data['target_station']

orig_folder = cfg_data['orig_folder']
dest_folder = cfg_data['dest_folder']
dest_file_suffix = cfg_data['dest_file_suffix']

time_granularity = cfg_data['time_granularity']

params = cfg_data['params']
stations = [station for station in params if params[station]['nsamples'] > 0]
nstations = len(stations)

aggregation = cfg_data["aggregation"]
relative = cfg_data["relative"]
decimal_pos = cfg_data["decimal_pos"]
az_el = cfg_data['az_el']

if relative:
    rad_col = '_rel'
else:
    rad_col = '_ghi'

# Create output directory if it doesn't exist
if not os.path.exists(dest_folder):
    print('Creating folder ' + dest_folder)
    os.makedirs(dest_folder)

# Copy config.json to output directory
with open(options.configFile, 'r') as cfg_file:
    with open(dest_folder + options.configFile, 'w') as f:
        f.write(cfg_file.read())

# Get dates that are available for all stations
input_files=[]
input_files_dates=[]
in_files = os.listdir(orig_folder + stations[0])
for input_file in in_files:
    if input_file.endswith('.csv'):
        input_files.append(input_file)
#dates = set([date[:8] for date in input_files])
# winter
dates = set([date[:8] for date in input_files if int(date[4:8]) >= 1221 or int(date[4:8]) < 321])
for station in stations:
    nfiles=os.listdir(orig_folder + station)
    for input_file in nfiles:
        if input_file.endswith('.csv'):
            input_files_dates.append(input_file)

    station_dates = set([date[:8] for date in input_files_dates])
    dates.intersection_update(station_dates)
    input_files_dates=[]

dates = list(dates)
ndates = len(dates)

print('Days available: {}'.format(ndates))

# Find out what is the first possible prediction
first_prediction_index = 0
for station in cfg_data['params']:
    index = params[station]['nsamples'] + params[station]['offset']
    if index > first_prediction_index:
        first_prediction_index = index
first_prediction = first_prediction_index * (time_granularity)

# morning
mor_beg = datetime.datetime.strptime("07:30","%H:%M")
mor_mid = datetime.datetime.strptime("11:00","%H:%M")
mor_time = mor_mid - mor_beg
last_mor_index = mor_time.seconds

for date in dates:

    x = pd.DataFrame() # This matrix will include the features
    y = pd.DataFrame() # This matrix will include the target
    matrix = pd.DataFrame() # This matrix will store x and y

    print('[{}/{}] Processing {}...'.format(dates.index(date)+1, ndates, date))

    y = pd.read_csv(orig_folder + target_station + '/' + date + '_' + target_station + '.csv').round(decimal_pos)

    # print('$$$$$$$$$$ ' + target_station + ' $$$$$$$$$$')
    # print('TOTAL ROWS: {}\n'.format(len(y)))

    # We will skip intermediate samples for now
    
    y_temp = y[first_prediction+time_granularity::time_granularity] #We lose one sample but we don't have an error calculating the mean for the first sample
    ## morning
#    y_temp = y[first_prediction+time_granularity::last_mor_index-time_granularity] #We lose one sample but we don't have an error calculating the mean for the first sample


    last_prediction = y_temp.last_valid_index()
    if aggregation == 'skip':
        y = y_temp

    if aggregation == 'mean':
        _y = []
        samp = int((last_prediction - first_prediction + time_granularity) / time_granularity)
        for i in range(samp):
            a = round(y[first_prediction + (i)*time_granularity:first_prediction + (i+1) * time_granularity].mean(),decimal_pos)
            _y.append(a)
        y = pd.DataFrame(_y)
    
    # print('FIRST PREDICTION: {}\nLAST PREDICTION: {}\nROWS: {}\n'.format(y.first_valid_index(), y.last_valid_index(), len(y)))

    max_dist = 0 #maximum time from nsamples and offset

    for station in stations:
        nsamples = params[station]['nsamples']
        offset = params[station]['offset']
        df = pd.read_csv(orig_folder + station + '/' + date + '_' + station + '.csv').round(decimal_pos)
        gr = df[df.columns[-1]].values #GhiRel
        if max_dist < (nsamples + offset):
            max_dist = nsamples + offset

        # print('########## ' + station + ' ##########')
        # print('TOTAL ROWS: {}\n'.format(len(df)))

        for ns in range(nsamples):
            dist = (ns + offset) * time_granularity
            first_sample = first_prediction - dist + time_granularity
            last_sample = last_prediction - dist
            # Rename column to include ghi/rel and ns
            col_name = station + rad_col + '_ns' + str(ns)
            ghi_means = []

            if aggregation == 'skip':
                
               _x = df[df.columns[-1]][first_sample:last_sample + time_granularity:time_granularity]
               _x.columns = [col_name]
            # print('FIRST SAMPLE: {}\nLAST SAMPLE: {}\nROWS: {}\n'.format(_x.first_valid_index(), _x.last_valid_index(), len(_x)))
            if aggregation == 'mean':
                
                _x = pd.DataFrame()
                samp = int((last_sample - first_sample + time_granularity)/time_granularity)
                for i in range(samp):
                    a = round(gr[first_sample + (i-1)*time_granularity:first_sample + (i)*time_granularity].mean(),decimal_pos)
                    ghi_means.append(a)
                    
                _x = pd.DataFrame(ghi_means)
                _x.columns = [col_name]
                end_mean = time.time()
            # Only radiation is needed. We need to reset the index in order to concatenate columns properly
            x = pd.concat([x,_x.reset_index(drop=True)], axis=1)
    # Rename column to include ghi/rel and target
    
    if az_el:
        x = pd.concat([x,y['elevation'],y['azimut'].reset_index(drop=True)], axis=1)

    col_name = target_station + rad_col + '_target'
    y = y.rename(columns={target_station + rad_col: col_name})
    # Only radiation and sun position is needed. We need to reset the index in order to concatenate columns properly
    matrix = pd.concat([x,y[[col_name]].reset_index(drop=True)], axis=1)
    matrix = matrix[:-1] #Last x row has NaN, we just remove it, it's easier
    #WARNING: this will overwrite any existing CSV file with the same path and name
    matrix.to_csv(dest_folder + '/' + target_station + '_' + date + '_' + str(time_granularity) + '_' + str(max_dist) + '_'  + aggregation + '_' + str(dest_file_suffix), header=True, index=False)

